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
from cuda.bindings import runtime as cudart
import threading
import queue

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
    def __init__(self, engine_path, num_buffers=2):
        self.device = torch.device("cuda")
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        self.num_buffers = num_buffers
        self.buffer_idx = 0
        
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
        
        # --- Create events and buffers for pipelining ---
        self.events = []
        self.device_buffers = []
        self.output_tensors = []

        for i in range(num_buffers):
            # Events for each buffer
            err, input_consumed_event = cudart.cudaEventCreate()
            check_cudart_err(err)
            err, output_ready_event = cudart.cudaEventCreate()
            check_cudart_err(err)
            self.events.append({'input_consumed': input_consumed_event, 'output_ready': output_ready_event})

            # I/O buffers
            buffer_set = {}
            output_set = {}
            for binding_name in self.engine:
                shape = self.engine.get_tensor_shape(binding_name)
                dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(binding_name))
                tensor = torch.empty(tuple(shape), dtype=dtype, device=self.device)
                buffer_set[binding_name] = tensor
                if self.engine.get_tensor_mode(binding_name) == trt.TensorIOMode.OUTPUT:
                    output_set[binding_name] = tensor
            self.device_buffers.append(buffer_set)
            self.output_tensors.append(output_set)


    def run_async(self, inputs: dict):
        # Select the next available buffer set
        current_idx = self.buffer_idx
        self.buffer_idx = (self.buffer_idx + 1) % self.num_buffers

        # Get the events and buffers for the current run
        current_events = self.events[current_idx]
        current_buffers = self.device_buffers[current_idx]

        # On the very first use of a buffer, its input_consumed event is not yet meaningful.
        # For subsequent uses, we wait until the previous inference on this buffer has consumed the input.
        if current_idx in getattr(self, 'used_buffers', set()):
             check_cudart_err(cudart.cudaEventSynchronize(current_events['input_consumed']))
        else:
             self.used_buffers = getattr(self, 'used_buffers', set())
             self.used_buffers.add(current_idx)

        # Set events and tensor addresses for this specific run
        self.context.set_input_consumed_event(current_events['input_consumed'])
        for name, tensor in current_buffers.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        # Copy inputs to device buffers
        for name, arr in inputs.items():
            expected_dtype = current_buffers[name].dtype
            current_buffers[name].copy_(arr.to(self.device, dtype=expected_dtype))
            
        # Execute Model
        self.context.execute_async_v3(stream_handle=self.stream)
        
        # Record an event that will be signaled when execution is complete.
        check_cudart_err(cudart.cudaEventRecord(current_events['output_ready'], self.stream))

        return current_idx
            
    def synchronize(self, run_idx: int):
        """Waits for a specific run to complete and returns its output."""
        if not (0 <= run_idx < self.num_buffers):
            raise ValueError(f"Invalid run_idx: {run_idx}")

        check_cudart_err(cudart.cudaEventSynchronize(self.events[run_idx]['output_ready']))
        return {name: tensor.clone() for name, tensor in self.output_tensors[run_idx].items()}

def consumer(q: queue.Queue, runner: TensorRTRunner, args: argparse.Namespace):
    """
    This function runs in a separate thread.
    It waits for completed inference runs, post-processes the results, and saves the image.
    """
    script_name = Path(__file__).stem
    image_counter = 0
    previous_end_time = None

    while True:
        item = q.get()
        if item is None:
            # Sentinel value received, so exit the loop.
            q.task_done()
            break

        run_idx, start_time, seed = item

        print(f"\n--- [Consumer] Synchronizing run for seed {seed} ---")
        outputs = runner.synchronize(run_idx)
        current_end_time = time.time()
        
        raw_image_tensor = outputs['image']

        if previous_end_time is None:
            # For the first result, measure latency from submission to completion.
            latency = current_end_time - start_time
            print(f"✓ [Consumer] First run (seed {seed}) finished. Latency: {latency:.4f} seconds")
        else:
            # For subsequent results, measure the time since the previous result, which reflects throughput.
            throughput_time = current_end_time - previous_end_time
            print(f"✓ [Consumer] Subsequent run (seed {seed}) finished. Throughput time: {throughput_time:.4f} seconds")
        
        previous_end_time = current_end_time

        print_np_stats(f"TRT Output (seed {seed})", raw_image_tensor)

        print(f"\n--- [Consumer] Saving final image (seed {seed}) ---")
        image = (raw_image_tensor / 2 + 0.5).clamp(0, 1)
        image_uint8 = image.cpu().permute(0, 2, 3, 1).mul(255).round().to(torch.uint8)
        
        pil_image = Image.fromarray(image_uint8.numpy()[0])
        
        # Find a unique filename
        output_path = f"output__{script_name}__{seed}_{image_counter:02d}.png"
        while Path(output_path).exists():
            image_counter += 1
            output_path = f"output__{script_name}__{seed}_{image_counter:02d}.png"
            
        pil_image.save(output_path)
        
        print(f"✓ [Consumer] Image saved to {output_path}")
        image_counter += 1
        q.task_done()

def main():
    parser = argparse.ArgumentParser(description="Run inference with the exported TensorRT monolithic model.")
    parser.add_argument("--prompt", type=str, default="masterpiece,best quality,amazing quality, general, 1girl, aqua_(konosuba), on a swing, looking at viewer, volumetric_lighting, park, night, shiny clothes, shiny skin, detailed_background", help="The prompt for image generation.")
    parser.add_argument("--engine_path", type=str, default="plan/monolith.plan", help="Path to the TensorRT engine file.")
    parser.add_argument("--height", type=int, default=832, help="Image height for generation.")
    parser.add_argument("--width", type=int, default=1216, help="Image width for generation.")
    parser.add_argument("--steps", type=int, default=8, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=1020094661, help="Seed for random generation.")
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

    # With a consumer thread, we might want more buffers to keep the GPU busy
    runner = TensorRTRunner(args.engine_path, num_buffers=4)

    tokenizer_1 = CLIPTokenizer.from_pretrained(str(base_dir), subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(str(base_dir), subfolder="tokenizer_2")
    
    # --- Setup producer-consumer queue and thread ---
    results_queue = queue.Queue()
    consumer_thread = threading.Thread(target=consumer, args=(results_queue, runner, args))
    consumer_thread.start()

    # Measure the total time from the first submission until the last result is processed.
    total_start_time = time.time()

    # In this example, we'll submit two inference jobs.
    # In a real application, this could be a loop consuming requests from a web server.
    for i in range(2):
        print(f"\n=== [Producer] Preparing inputs for TensorRT model (run {i+1}) ===")
        
        # Tokenize prompts
        prompt_ids_1 = tokenizer_1(args.prompt, padding="max_length", max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt").input_ids
        prompt_ids_2 = tokenizer_2(args.prompt, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids

        # Seed - generate a new seed for each run to make it more realistic
        if args.random_seed:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            # Use a different seed for each run if not random
            seed = args.seed + i
        print(f"\n--- [Producer] Generating image with seed: {seed} ---")
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Latents and noise
        latents_shape = (1, 4, args.height // 8, args.width // 8)
        initial_latents = torch.randn(latents_shape, generator=generator, device=device, dtype=dtype)
        
        noise_shape = (args.steps, 1, 4, args.height // 8, args.width // 8)
        all_noises = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
        
        # Time IDs
        add_time_ids = torch.tensor([[args.height, args.width, 0, 0, args.height, args.width]], device=device, dtype=dtype)

        # --- Create input dictionary for TRT ---
        trt_inputs = {
            "prompt_ids_1": prompt_ids_1.to(torch.int32),
            "prompt_ids_2": prompt_ids_2.to(torch.int32),
            "initial_latents": initial_latents,
            "all_noises": all_noises,
            "add_time_ids": add_time_ids,
        }
        
        print("\n--- [Producer] Input Tensor Stats ---")
        for name, arr in trt_inputs.items():
            print_np_stats(name, arr)

        # --- Submit Inference Run ---
        print(f"--- [Producer] Submitting run {i+1} ---")
        start_time = time.time()
        run_idx = runner.run_async(trt_inputs)
        results_queue.put((run_idx, start_time, seed))

    # --- Signal consumer to exit and wait for it to finish ---
    print("\n--- [Producer] All inference runs submitted. Waiting for consumer to finish. ---")
    results_queue.put(None) # Sentinel to stop the consumer
    results_queue.join() # Wait for all tasks to be processed
    consumer_thread.join()
    
    total_end_time = time.time()
    print(f"\n--- Total execution time for all runs: {total_end_time - total_start_time:.4f} seconds ---")
    
    print("\nGeneration complete.")

if __name__ == "__main__":
    main() 