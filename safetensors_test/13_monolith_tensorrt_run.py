#!/usr/bin/env python3
import torch
import numpy as np
from transformers import CLIPTokenizer
from pathlib import Path
import sys
from PIL import Image
import time
import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def print_np_stats(name, arr):
    """Prints detailed statistics for a given numpy array on a single line."""
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
        self.stream = cuda.Stream()
        
        self.bindings = []
        self.output_shapes = {}
        self.output_dtypes = {}
        
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate memory
            size = trt.volume(shape) * np.dtype(dtype).itemsize
            mem = cuda.mem_alloc(size)
            self.bindings.append(int(mem))
            
            if not self.engine.binding_is_input(binding):
                self.output_shapes[binding] = shape
                self.output_dtypes[binding] = dtype

    def run(self, inputs: dict):
        # inputs should be a dict of {name: numpy_array}
        
        # --- Transfer inputs to GPU ---
        for name, arr in inputs.items():
            binding_idx = self.engine.get_binding_index(name)
            cuda.memcpy_htod_async(self.bindings[binding_idx], arr, self.stream)
            
        # --- Execute Model ---
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        self.stream.synchronize()
        
        # --- Transfer outputs from GPU ---
        outputs = {}
        for name, shape in self.output_shapes.items():
            binding_idx = self.engine.get_binding_index(name)
            dtype = self.output_dtypes[name]
            
            output_arr = np.empty(shape, dtype=dtype)
            cuda.memcpy_dtoh_async(output_arr, self.bindings[binding_idx], self.stream)
            outputs[name] = output_arr
            
        self.stream.synchronize()
        return outputs

def main():
    parser = argparse.ArgumentParser(description="Run inference with the exported TensorRT monolithic model.")
    parser.add_argument("--prompt", type=str, default="masterpiece, best quality, 1girl, aqua_(konosuba), on a swing, looking at viewer, detailed background", help="The prompt for image generation.")
    parser.add_argument("--engine_path", type=str, default="monolith.plan", help="Path to the TensorRT engine file.")
    parser.add_argument("--height", type=int, default=832, help="Image height for generation.")
    parser.add_argument("--width", type=int, default=1216, help="Image width for generation.")
    parser.add_argument("--steps", type=int, default=8, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=12345, help="Seed for random generation.")
    parser.add_argument("--random_seed", action="store_true", help="Use a random seed.")
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
    
    # Tokenize prompts
    prompt_ids_1 = tokenizer_1(args.prompt, padding="max_length", max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt").input_ids
    prompt_ids_2 = tokenizer_2(args.prompt, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids

    # Seed
    if args.random_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    else:
        seed = args.seed
    print(f"\n--- Generating image with seed: {seed} ---")
    generator = torch.Generator(device="cpu").manual_seed(seed) # Generate on CPU
    
    # Latents and noise
    latents_shape = (1, 4, args.height // 8, args.width // 8)
    initial_latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=dtype)
    
    noise_shape = (args.steps, 1, 4, args.height // 8, args.width // 8)
    all_noises = torch.randn(noise_shape, generator=generator, device="cpu", dtype=dtype)
    
    # Time IDs
    add_time_ids = torch.tensor([[args.height, args.width, 0, 0, args.height, args.width]], device="cpu", dtype=dtype)

    # --- Create input dictionary for TRT ---
    trt_inputs = {
        "prompt_ids_1": prompt_ids_1.numpy().astype(np.int64),
        "prompt_ids_2": prompt_ids_2.numpy().astype(np.int64),
        "initial_latents": initial_latents.numpy(),
        "all_noises": all_noises.numpy(),
        "add_time_ids": add_time_ids.numpy(),
    }
    
    print("\n--- Input Tensor Stats ---")
    for name, arr in trt_inputs.items():
        print_np_stats(name, arr)

    # --- Run Inference ---
    print("\n=== Running TensorRT inference (1st run) ===")
    start_time_1 = time.time()
    
    outputs_1 = runner.run(trt_inputs)
    raw_image_np = outputs_1['image']

    end_time_1 = time.time()
    print(f"✓ First TensorRT inference took: {end_time_1 - start_time_1:.4f} seconds")
    print_np_stats("TRT Output 1 (raw_image_np)", raw_image_np)

    print("\n=== Running TensorRT inference (2nd run) ===")
    start_time_2 = time.time()

    outputs_2 = runner.run(trt_inputs)
    raw_image_np = outputs_2['image']
    
    end_time_2 = time.time()
    print(f"✓ Second TensorRT inference took: {end_time_2 - start_time_2:.4f} seconds")
    print_np_stats("TRT Output 2 (raw_image_np)", raw_image_np)

    # --- Post-process and save final image ---
    print("\n--- Saving final image (from 2nd run) ---")
    image = torch.from_numpy(raw_image_np)
    image = (image / 2 + 0.5).clamp(0, 1)
    image_uint8 = image.permute(0, 2, 3, 1).mul(255).round().to(torch.uint8)
    
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