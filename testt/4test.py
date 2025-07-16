import torch
import numpy as np
import tensorrt as trt
from PIL import Image
import argparse
import os
from diffusers import AutoencoderKL
from typing import Dict, List
import time

# Only import tensorrt if it's available and needed
try:
    import tensorrt as trt
    from cuda import cudart
except ImportError:
    print("❌ TensorRT or CUDA bindings are not installed. Please ensure they are available.")
    trt = None
    cudart = None

# --- CUDA/TensorRT Helper Functions ---

def check_cudart_err(err):
    """Checks for and raises CUDA runtime errors."""
    if err is None:
        return
    if isinstance(err, cudart.cudaError_t) and err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA Runtime Error: {cudart.cudaGetErrorString(err)}")
    if isinstance(err, tuple) and err[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA Runtime Error: {cudart.cudaGetErrorString(err[0])}")

def trt_dtype_to_torch(dtype: trt.DataType) -> torch.dtype:
    """Converts a TensorRT a torch dtype."""
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError(f"Unsupported TensorRT data type: {dtype}")

# --- TensorRT Runner Class ---

class TensorRTRunner:
    """
    A class to run inference on a TensorRT engine, optimized for dynamic shapes.
    """
    def __init__(self, engine_path: str):
        if not trt:
            raise ImportError("TensorRT library is required for TensorRTRunner.")
            
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        print(f"--- Loading TensorRT Engine from {self.engine_path} ---")
        with open(self.engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize the TensorRT engine.")
        print("✓ Engine loaded successfully.")
        
        self.context = self.engine.create_execution_context()
        err, self.stream = cudart.cudaStreamCreate()
        check_cudart_err(err)
        
        err, self.start_event = cudart.cudaEventCreate()
        check_cudart_err(err)
        err, self.end_event = cudart.cudaEventCreate()
        check_cudart_err(err)

        # --- CORRECTED BUFFER ALLOCATION LOGIC ---
        print("--- Allocating Buffers for Engine Bindings ---")
        self.device_buffers = {}
        self.output_tensors = {}
        
        # 1. First, set the context's input shapes to the maximum from the profile
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                # Use profile 0, shape index 2 (max)
                max_shape = self.engine.get_tensor_profile_shape(name, 0)[2]
                self.context.set_input_shape(name, max_shape)

        # 2. Now, allocate memory for all bindings based on the context's (max) shapes
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            # Get shape from the context, which is now aware of max dimensions
            shape = self.context.get_tensor_shape(name)
            dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            
            if any(d < 0 for d in shape):
                 raise ValueError(f"Binding '{name}' has a dynamic dimension that could not be resolved. Shape: {shape}")

            tensor = torch.empty(tuple(shape), dtype=dtype, device="cuda")
            self.device_buffers[name] = tensor
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.output_tensors[name] = tensor
        # -----------------------------------------------
        
        print(f"✓ Buffers allocated for {len(self.device_buffers)} bindings.")


    def run(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run inference and measure GPU execution time accurately."""
        
        for name, tensor in inputs.items():
            if name not in self.device_buffers:
                raise ValueError(f"Input tensor '{name}' not found in engine bindings.")
            
            buffer = self.device_buffers[name]
            self.context.set_input_shape(name, tensor.shape)
            buffer.copy_(tensor) # Use copy_ for direct HtoD or DtoD transfer

        for name, buffer in self.device_buffers.items():
            self.context.set_tensor_address(name, buffer.data_ptr())
            
        check_cudart_err(cudart.cudaEventRecord(self.start_event, self.stream))
        self.context.execute_async_v3(self.stream)
        check_cudart_err(cudart.cudaEventRecord(self.end_event, self.stream))
        
        check_cudart_err(cudart.cudaEventSynchronize(self.end_event))
        
        err, ms = cudart.cudaEventElapsedTime(self.start_event, self.end_event)
        check_cudart_err(err)
        print(f"Time taken by GPU execution: {ms:.4f} ms")

        results = {}
        for name, tensor in self.output_tensors.items():
            actual_output_shape = self.context.get_tensor_shape(name)
            results[name] = tensor[:actual_output_shape[0],...].clone()
        
        return results

    def __del__(self):
        # Clean up CUDA resources on object deletion
        if hasattr(self, 'start_event'):
            check_cudart_err(cudart.cudaEventDestroy(self.start_event))
        if hasattr(self, 'end_event'):
            check_cudart_err(cudart.cudaEventDestroy(self.end_event))
        if hasattr(self, 'stream'):
            check_cudart_err(cudart.cudaStreamDestroy(self.stream))


# --- Image Processing Functions ---

def preprocess_images(image_paths: List[str]) -> torch.Tensor:
    """Loads, preprocesses, and batches a list of images."""
    preprocessed_tensors = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        img = Image.open(image_path).convert("RGB").resize((512, 512))
        arr = np.array(img).astype(np.float32)
        arr = (arr / 127.5) - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        preprocessed_tensors.append(tensor)
    return torch.stack(preprocessed_tensors)


def save_decoded_images(image_tensor: torch.Tensor, output_prefix: str):
    """Denormalizes and saves a batch of images."""
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_tensor = image_tensor.cpu().permute(0, 2, 3, 1)
    image_np = (image_tensor.float().numpy() * 255).round().astype("uint8")

    for i, img_arr in enumerate(image_np):
        img = Image.fromarray(img_arr)
        output_path = f"{output_prefix}{i+1}.png"
        img.save(output_path)
        print(f"✓ Saved decoded image to {output_path}")

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the VAE Decoder TensorRT engine.")
    parser.add_argument(
        "--engine_path", 
        type=str, 
        default="onnx/simple_vae_decoder.trt",
        help="Path to the TensorRT decoder engine file."
    )
    args = parser.parse_args()

    if not os.path.exists(args.engine_path):
        print(f"❌ Error: Engine file not found at '{args.engine_path}'.")
        print("Please build the engine first by running the export script with the --tensorrt flag.")
        exit(1)

    # --- Configuration ---
    VAE_PATH = "madebyollin/sdxl-vae-fp16-fix"
    DEVICE = "cuda"
    DTYPE = torch.float16
    IMAGE_FILES = [f"example{i}.png" for i in range(1, 5)]
    
    print("--- 1. Initializing Models ---")
    
    # Load Diffusers VAE for encoding
    print(f"Loading Diffusers VAE from {VAE_PATH}...")
    vae = AutoencoderKL.from_pretrained(VAE_PATH, torch_dtype=DTYPE).to(DEVICE).eval()
    
    # Initialize our TensorRT runner for decoding
    decoder_runner = TensorRTRunner(args.engine_path)

    print("\n--- 2. Preprocessing Input Images ---")
    try:
        image_batch = preprocess_images(IMAGE_FILES).to(DEVICE, dtype=DTYPE)
        print(f"✓ Loaded and processed a batch of {image_batch.shape[0]} images.")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please make sure 'example1.png', 'example2.png', 'example3.png', and 'example4.png' exist in the current directory.")
        exit(1)

    # --- WARMUP RUN ---
    print("\n--- 3. Performing a Warmup Run ---")
    with torch.no_grad():
        warmup_latents = torch.randn((1, 4, 64, 64), device=DEVICE, dtype=DTYPE)
        _ = decoder_runner.run({"latent_sample": warmup_latents})
    print("✓ Warmup complete.")

    print("\n--- 4. Encoding with Diffusers VAE ---")
    with torch.no_grad():
        latent_dist = vae.encode(image_batch).latent_dist
        latents = latent_dist.mode() * vae.config.scaling_factor
    print(f"✓ Encoded images into a latent tensor of shape: {latents.shape}")

    print("\n--- 5. Decoding with TensorRT Engine (Timed) ---")
    trt_inputs = {"latent_sample": latents}
    trt_outputs = decoder_runner.run(trt_inputs)
    reconstructed_batch = trt_outputs['sample']
    print(f"✓ Decoded latents into an image tensor of shape: {reconstructed_batch.shape}")

    print("\n--- 6. Saving Output Images ---")
    save_decoded_images(reconstructed_batch, "output")
    
    print("\n--- Test Complete! ---")