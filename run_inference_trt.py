import torch
import tensorrt as trt
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import numpy as np
import os

def main():
    """
    Runs SDXL inference using a TensorRT engine for the UNet.
    """
    # --- Configuration ---
    engine_file_path = "unet.engine"
    model_id = "socks22/sdxl-wai-nsfw-illustriousv14"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = "aqua_(konosuba), smiling, looking at the camera"
    negative_prompt = "lowres, bad anatomy, bad hands, blurry, text, watermark, signature"
    output_image_path = "output_trt.png"
    output_name = "conv2d_50" # From the onnx export log

    # --- File Check ---
    if not os.path.exists(engine_file_path):
        print(f"Error: Engine file not found at {engine_file_path}")
        print("Please run unet_onnx_tensorrt.py first.")
        return

    # --- Load Pipeline ---
    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    # Use Euler Ancestral scheduler
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    # Enable CPU offloading to avoid loading the original UNet into VRAM
    pipe.enable_model_cpu_offload()

    # --- Load TensorRT Engine ---
    print(f"Loading TensorRT engine from: {engine_file_path}")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()

    # --- Define Shapes and Allocate Buffers ---
    # We need to set the context's input shapes before we can allocate buffers.
    # The UNet input shapes are based on the inference settings.
    batch_size = 1 # This is the user-facing batch size.
    unet_batch_size = 2 # For classifier-free guidance, this is always 2*batch_size
    
    # Get image dimensions from the later pipe() call
    height=768
    width=1152
    latent_height = height // 8
    latent_width = width // 8
    
    # Manually define the input shapes, matching the names from the ONNX export
    # The UNet config is available from the loaded pipeline.
    unet_in_channels = pipe.unet.config.in_channels
    # SDXL uses a concatenated projection of two text encoders for its cross-attention context
    text_embed_dim = 2048 
    
    input_shapes = {
        "sample": (unet_batch_size, unet_in_channels, latent_height, latent_width),
        "timestep": (unet_batch_size,),
        "encoder_hidden_states": (unet_batch_size, 77, text_embed_dim),
        "text_embeds": (unet_batch_size, 1280),
        "time_ids": (unet_batch_size, 6)
    }
    
    for binding_name, shape in input_shapes.items():
        context.set_input_shape(binding_name, shape)
    
    # --- Prepare Buffers ---
    input_buffers = {}
    output_buffers = {}
    stream = torch.cuda.Stream()
    
    for binding in engine:
        # get_tensor_shape() will now return concrete shapes
        shape = tuple(context.get_tensor_shape(binding))
        dtype = torch.from_numpy(np.array([], dtype=trt.nptype(engine.get_tensor_dtype(binding)))).dtype
        
        buffer = torch.empty(shape, dtype=dtype, device=device).contiguous()
        
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            input_buffers[binding] = buffer
        else:
            output_buffers[binding] = buffer

    # --- Create Inference Wrapper ---
    # Keep the original forward method
    original_forward = pipe.unet.forward
    
    def trt_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        print(f"Sample tensor shape in trt_unet_forward: {sample.shape}")
        # Map pipeline inputs to our engine's buffers
        input_buffers["sample"].copy_(sample)
        
        # The timestep from the pipeline is a single value, but our engine expects a batch.
        timestep_buffer = input_buffers["timestep"]
        expanded_timestep = timestep.expand(timestep_buffer.shape[0]).to(timestep_buffer.dtype)
        timestep_buffer.copy_(expanded_timestep)

        input_buffers["encoder_hidden_states"].copy_(encoder_hidden_states)

        added_cond = kwargs["added_cond_kwargs"]
        input_buffers["text_embeds"].copy_(added_cond["text_embeds"])
        input_buffers["time_ids"].copy_(added_cond["time_ids"])
        
        # Set tensor addresses for execute_async_v3
        for name, buffer in input_buffers.items():
            context.set_tensor_address(name, buffer.data_ptr())
        for name, buffer in output_buffers.items():
            context.set_tensor_address(name, buffer.data_ptr())
        
        # Run inference
        context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()
        
        from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
        return UNet2DConditionOutput(sample=output_buffers[output_name])

    # --- Monkey-patch the UNet ---
    pipe.unet.forward = trt_unet_forward
    print("UNet has been replaced with TensorRT engine.")
    
    # --- Run Inference ---
    print(f"Running inference for prompt: '{prompt}'")
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=8,
        guidance_scale=1.2,
        height=768,
        width=1152,
    )
    print(f"Number of images returned by pipeline: {len(result.images)}")
    image = result.images[0]

    # --- Save Image ---
    print(f"Saving generated image to {output_image_path}")
    image.save(output_image_path)
    print("Inference complete.")
    
    # Restore original forward method for safety
    #pipe.unet.forward = original_forward

if __name__ == "__main__":
    main() 