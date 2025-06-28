import torch
import tensorrt as trt
from diffusers import StableDiffusionXLPipeline, LCMScheduler
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
    prompt = "masterpiece,best quality,amazing quality, anime, aqua_(konosuba), smiling, looking at viewer, peace sign"
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
    # Use LCM scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    # Enable CPU offloading to avoid loading the original UNet into VRAM
    pipe.enable_model_cpu_offload()

    # --- Load TensorRT Engine ---
    print(f"Loading TensorRT engine from: {engine_file_path}")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()

    # --- Create Inference Wrapper ---
    # Keep the original forward method
    original_forward = pipe.unet.forward
    
    def trt_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        # --- Debug prints ---
        print("\n--- trt_unet_forward call ---")
        print(f"sample.shape: {sample.shape}, sample.dtype: {sample.dtype}, sample.device: {sample.device}")
        print(f"timestep: {timestep}, timestep.shape: {timestep.shape}, timestep.dtype: {timestep.dtype}, timestep.device: {timestep.device}")
        print(f"encoder_hidden_states.shape: {encoder_hidden_states.shape}, encoder_hidden_states.dtype: {encoder_hidden_states.dtype}, encoder_hidden_states.device: {encoder_hidden_states.device}")
        added_cond = kwargs["added_cond_kwargs"]
        print(f"text_embeds.shape: {added_cond['text_embeds'].shape}, text_embeds.dtype: {added_cond['text_embeds'].dtype}, text_embeds.device: {added_cond['text_embeds'].device}")
        print(f"time_ids.shape: {added_cond['time_ids'].shape}, time_ids.dtype: {added_cond['time_ids'].dtype}, time_ids.device: {added_cond['time_ids'].device}")
        print("--------------------------")

        # --- Prepare Buffers ---
        stream = torch.cuda.Stream()
        
        # Directly use input tensors, making them contiguous
        
        # Timestep from the pipeline is a scalar, but the engine expects a 1D tensor of size batch_size.
        batch_size = sample.shape[0]
        timestep = timestep.expand(batch_size).to(dtype=torch.float16)

        input_tensors = {
            "sample": sample.contiguous(),
            "timestep": timestep.contiguous(),
            "encoder_hidden_states": encoder_hidden_states.contiguous(),
            "text_embeds": added_cond["text_embeds"].contiguous(),
            "time_ids": added_cond["time_ids"].contiguous()
        }
        
        print("\n--- TensorRT Input Tensors ---")
        for name, tensor in input_tensors.items():
            print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
        print("------------------------------")
        
        # Set input shapes for the context
        for name, tensor in input_tensors.items():
            context.set_input_shape(name, tensor.shape)
            
        # Allocate output buffers
        output_buffers = {}
        for binding in engine:
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
                # The shape of the output tensor is determined by the shape of the input sample.
                shape = input_tensors["sample"].shape
                dtype = torch.from_numpy(np.array([], dtype=trt.nptype(engine.get_tensor_dtype(binding)))).dtype
                output_buffers[binding] = torch.empty(shape, dtype=dtype, device=device).contiguous()

        # Set tensor addresses for execute_async_v3
        for name, tensor in input_tensors.items():
            context.set_tensor_address(name, tensor.data_ptr())
        for name, buffer in output_buffers.items():
            context.set_tensor_address(name, buffer.data_ptr())
        
        # Run inference
        context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

        print("\n--- TensorRT Output ---")
        print(f"output shape: {output_buffers[output_name].shape}")
        print("-----------------------")
        
        from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
        return UNet2DConditionOutput(sample=output_buffers[output_name])

    # --- Monkey-patch the UNet ---
    pipe.unet.forward = trt_unet_forward
    print("UNet has been replaced with TensorRT engine.")
    
    # --- Run Inference ---
    print(f"Running inference for prompt: '{prompt}'")
    result = pipe(
        prompt=prompt,
        #negative_prompt=negative_prompt,
        num_inference_steps=8,
        guidance_scale=1.0,
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