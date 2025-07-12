#!/usr/bin/env python3
import torch
import numpy as np
from transformers import CLIPTokenizer
from pathlib import Path
import sys
from PIL import Image
import time
import argparse
import tensorrt_lean as trt
from cuda.bindings import runtime as cudart

def check_cudart_err(err):
    if isinstance(err, cudart.cudaError_t) and err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA Runtime Error: {cudart.cudaGetErrorString(err)}")
    # Some functions return a tuple (error, value)
    if isinstance(err, tuple) and isinstance(err[0], cudart.cudaError_t) and err[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA Runtime Error: {cudart.cudaGetErrorString(err[0])}")

def trt_dtype_to_torch(dtype: trt.DataType) -> torch.dtype:
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.int64:
        return torch.int64
    else:
        raise TypeError(f"Unsupported TensorRT data type: {dtype}")

def print_np_stats(name, arr):
    """Prints detailed statistics for a given numpy array on a single line."""
    if not isinstance(arr, np.ndarray):
        # If it's a tensor, convert to numpy for stats
        arr = arr.cpu().numpy()

    if arr is None:
        print(f"--- {name}: Array is None ---")
        return
    
    stats = f"Shape: {str(arr.shape):<20} | Dtype: {str(arr.dtype):<15}"
    if arr.size > 0:
        arr_float = arr.astype(np.float32)
        stats += f" | Mean: {arr_float.mean():<8.4f} | Min: {arr_float.min():<8.4f} | Max: {arr_float.max():<8.4f}"
        stats += f" | Has NaN: {str(np.isnan(arr_float).any()):<5} | Has Inf: {str(np.isinf(arr_float).any()):<5}"
    
    print(f"--- {name+':':<30} {stats} ---")

class TensorRTRunner:
    def __init__(self, engine_path):
        self.device = torch.device("cuda")
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        print("--- Loading TensorRT Engine ---")
        with open(self.engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine.")
            
        print(f"✓ Engine loaded from {self.engine_path}")
        
        self.context = self.engine.create_execution_context()
        err, self.stream = cudart.cudaStreamCreate()
        check_cudart_err(err)
        
        # Allocate device memory for bindings using PyTorch for better integration
        self.device_buffers = {}
        self.output_tensors = {}
        
        for binding_name in self.engine:
            shape = self.engine.get_tensor_shape(binding_name)
            dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(binding_name))
            
            # Allocate memory with PyTorch
            tensor = torch.empty(tuple(shape), dtype=dtype, device=self.device)
            self.device_buffers[binding_name] = tensor
            
            # Set address on context
            self.context.set_tensor_address(binding_name, tensor.data_ptr())
            
            if self.engine.get_tensor_mode(binding_name) == trt.TensorIOMode.OUTPUT:
                self.output_tensors[binding_name] = tensor

    def run(self, inputs: dict):
        # inputs is a dict of {name: torch.Tensor}
        
        # --- Copy inputs to device buffers ---
        for name, arr in inputs.items():
            expected_dtype = self.device_buffers[name].dtype
            # This copy handles device placement and dtype conversion automatically
            self.device_buffers[name].copy_(arr.to(self.device, dtype=expected_dtype))
            
        # --- Execute Model with enqueueV3 ---
        self.context.execute_async_v3(stream_handle=self.stream)
        check_cudart_err(cudart.cudaStreamSynchronize(self.stream))
        
        # --- Return clones of output buffers ---
        # Cloning is important to avoid returning a reference to an internal buffer
        # that might be overwritten in the next run.
        return {name: tensor.clone() for name, tensor in self.output_tensors.items()}

def main():
    parser = argparse.ArgumentParser(description="Run inference with the exported TensorRT monolithic model.")
    parser.add_argument("--prompt", type=str, default="masterpiece,best quality,amazing quality, general, 1girl, aqua_(konosuba), on a swing, looking at viewer, volumetric_lighting, park, night, shiny clothes, shiny skin, detailed_background", help="The prompt for image generation.")
    parser.add_argument("--engine_path", type=str, default="plan/monolith.plan", help="Path to the TensorRT engine file.")
    parser.add_argument("--height", type=int, default=832, help="Image height for generation.")
    parser.add_argument("--width", type=int, default=1216, help="Image width for generation.")
    parser.add_argument("--steps", type=int, default=8, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=1020094661, help="Seed for random generation.")
    parser.add_argument("--random_seed", action="store_true", help="Use a random seed.")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to generate in a loop.")
    args = parser.parse_args()

    # --- Configuration ---
    base_dir = Path("/lab/model")
    device = "cuda" # TRT runs on cuda
    dtype = torch.float16

    # --- Load TRT Runner and Tokenizers ---
    print("=== Loading TensorRT engine and tokenizers ===")
    
    if not Path(args.engine_path).exists():
        print(f"✗ Error: TensorRT engine not found at {args.engine_path}")
        sys.exit(1)

    runner = TensorRTRunner(args.engine_path)

    tokenizer_1 = CLIPTokenizer.from_pretrained(str(base_dir), subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(str(base_dir), subfolder="tokenizer_2")
    
    # --- Prepare Inputs ---
    print("\n=== Preparing inputs for TensorRT model ===")
    
    prep_start_time = time.time()
    # Tokenize prompts
    prompt_ids_1 = tokenizer_1(args.prompt, padding="max_length", max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt").input_ids
    prompt_ids_2 = tokenizer_2(args.prompt, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids

    # Seed
    if args.random_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    else:
        seed = args.seed
    print(f"\n--- Generating image with seed: {seed} ---")
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Latents and noise
    latents_shape = (1, 4, args.height // 8, args.width // 8)
    initial_latents = torch.randn(latents_shape, generator=generator, device=device, dtype=dtype)
    
    noise_shape = (args.steps, 1, 4, args.height // 8, args.width // 8)
    all_noises = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
    
    # Time IDs
    add_time_ids = torch.tensor([[args.height, args.width, 0, 0, args.height, args.width]], device=device, dtype=dtype)
    prep_end_time = time.time()
    print(f"✓ Input preparation took: {prep_end_time - prep_start_time:.4f} seconds")

    # --- Create input dictionary for TRT ---
    # We now pass torch tensors directly
    trt_inputs = {
        "prompt_ids_1": prompt_ids_1.to(torch.int32),
        "prompt_ids_2": prompt_ids_2.to(torch.int32),
        "initial_latents": initial_latents,
        "all_noises": all_noises,
        "add_time_ids": add_time_ids,
    }
    
    # --- Run Inference Loop ---
    print(f"\n=== Running TensorRT inference for {args.num_images} iterations ===")
    
    raw_image_tensor = None
    for i in range(args.num_images):
        start_time = time.time()
        
        outputs = runner.run(trt_inputs)
        raw_image_tensor = outputs['image']
        
        end_time = time.time()
        print(f"✓ Iteration {i+1}/{args.num_images} took: {end_time - start_time:.4f} seconds")

    if raw_image_tensor is None:
        print("No images were generated, exiting.")
        sys.exit(0)

    # --- Post-process and save final image ---
    print("\n--- Saving final image ---")
    image = (raw_image_tensor / 2 + 0.5).clamp(0, 1)
    image_uint8 = image.cpu().permute(0, 2, 3, 1).mul(255).round().to(torch.uint8)
    
    pil_image = Image.fromarray(image_uint8.numpy()[0])
    
    script_name = Path(__file__).stem
    image_idx = 0
    while True:
        output_path = f"output__{script_name}__{args.seed}_{image_idx:02d}.png"
        if not Path(output_path).exists():
            break
        image_idx += 1
        
    pil_image.save(output_path)
    
    print(f"✓ Image saved to {output_path}")
    print("Generation complete.")

if __name__ == "__main__":
    main() 