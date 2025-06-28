import torch
import tensorrt as trt
from diffusers import StableDiffusionXLPipeline
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
    # Enable CPU offloading to avoid loading the original UNet into VRAM
    pipe.enable_model_cpu_offload()

    # --- Load TensorRT Engine ---
    print(f"Loading TensorRT engine from: {engine_file_path}")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # --- Prepare Buffers ---
    input_buffers = {}
    output_buffers = {}
    stream = torch.cuda.Stream()
    
    for binding in engine:
        shape = tuple(engine.get_tensor_shape(binding))
        dtype = torch.from_numpy(np.array([], dtype=trt.nptype(engine.get_binding_dtype(binding)))).dtype
        
        buffer = torch.empty(shape, dtype=dtype, device=device).contiguous()
        
        if engine.binding_is_input(binding):
            input_buffers[binding] = buffer
        else:
            output_buffers[binding] = buffer

    # --- Create Inference Wrapper ---
    # Keep the original forward method
    original_forward = pipe.unet.forward
    
    def trt_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        # Map pipeline inputs to our engine's buffers
        input_buffers["sample"].copy_(sample)
        input_buffers["timestep"].copy_(timestep)
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
    image = pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

    # --- Save Image ---
    print(f"Saving generated image to {output_image_path}")
    image.save(output_image_path)
    print("Inference complete.")
    
    # Restore original forward method for safety
    pipe.unet.forward = original_forward

if __name__ == "__main__":
    main() 