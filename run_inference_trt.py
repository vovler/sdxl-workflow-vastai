import torch
import tensorrt as trt
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, UNet2DConditionModel
from accelerate import init_empty_weights
import numpy as np
import os
import argparse
from compel import Compel, ReturnedEmbeddingsType

def main():
    """
    Runs SDXL inference using a TensorRT engine for the UNet.
    """
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Run SDXL inference with a TensorRT UNet engine.")
    parser.add_argument("prompt", type=str, help="The base prompt for image generation.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for diffusion.")
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps.")
    args = parser.parse_args()

    # --- Configuration ---
    engine_file_path = "unet.engine"
    model_id = "socks22/sdxl-wai-nsfw-illustriousv14"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    prompt_prefix = "masterpiece,best quality,amazing quality, anime, aqua_(konosuba)"
    prompt = prompt_prefix + ", " + args.prompt
    
    negative_prompt = "lowres, bad anatomy, bad hands, blurry, text, watermark, signature"
    output_image_path = "output_trt.png"
    output_name = "conv2d_50" # From the onnx export log

    # --- File Check ---
    if not os.path.exists(engine_file_path):
        print(f"Error: Engine file not found at {engine_file_path}")
        print("Please run unet_onnx_tensorrt.py first.")
        return

    # --- Load Pipeline ---
    print("Creating dummy UNet to avoid loading original...")
    unet_config = UNet2DConditionModel.load_config(model_id, subfolder="unet")
    with init_empty_weights():
        # The UNet will be created with random weights on the 'meta' device,
        # consuming no RAM.
        unet = UNet2DConditionModel.from_config(unet_config)

    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        unet=unet,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # Enable CPU offloading to avoid loading the original UNet into VRAM
    # This is not strictly necessary for the UNet anymore, but good for other components
    # like VAE and text encoders if VRAM is limited.
    #pipe.enable_model_cpu_offload()

    # --- Set up Compel ---
    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True]
    )

    # Get positive prompt embeddings
    prompt_embeds, pooled_prompt_embeds = compel(prompt)

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
    
    generator = None
    if args.seed is not None:
        print(f"Using seed: {args.seed}")
        generator = torch.Generator(device=device).manual_seed(args.seed)
        
    # --- Timesteps & Inference Args ---
    pipe_kwargs = {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "num_inference_steps": args.steps,
        "guidance_scale": 1.0,
        "height": 768,
        "width": 1152,
        "generator": generator,
    }

    result = pipe(**pipe_kwargs)
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