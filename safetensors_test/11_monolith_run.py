#!/usr/bin/env python3
import torch
import numpy as np
from transformers import CLIPTokenizer
from pathlib import Path
import sys
from PIL import Image
import time
import argparse
import onnxruntime as ort

def print_np_stats(name, arr):
    """Prints detailed statistics for a given numpy array on a single line."""
    if arr is None:
        print(f"--- {name}: Array is None ---")
        return
    
    stats = f"Shape: {str(arr.shape):<20} | Dtype: {str(arr.dtype):<15}"
    if arr.size > 0:
        # Prevent potential overflow with mixed types by converting to float32 for stats
        arr_float = arr.astype(np.float32)
        stats += f" | Mean: {arr_float.mean():<8.4f} | Min: {arr_float.min():<8.4f} | Max: {arr_float.max():<8.4f}"
        stats += f" | Has NaN: {str(np.isnan(arr_float).any()):<5} | Has Inf: {str(np.isinf(arr_float).any()):<5}"
    
    print(f"--- {name+':':<30} {stats} ---")

def main():
    """
    Runs inference with the exported monolithic ONNX model.
    """
    parser = argparse.ArgumentParser(description="Run inference with the exported ONNX monolithic model.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="masterpiece, best quality, 1girl, aqua_(konosuba), on a swing, looking at viewer, detailed background",
        help="The prompt for image generation."
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="monolith.onnx",
        help="Path to the ONNX model file."
    )
    parser.add_argument("--height", type=int, default=832, help="Image height for generation.")
    parser.add_argument("--width", type=int, default=1216, help="Image width for generation.")
    parser.add_argument("--steps", type=int, default=8, help="Number of inference steps. Must match the model if not dynamic.")
    parser.add_argument("--seed", type=int, default=12345, help="Seed for random generation.")
    parser.add_argument("--random_seed", action="store_true", help="Use a random seed instead of the one specified by --seed.")
    args = parser.parse_args()

    # --- Configuration ---
    base_dir = Path("/lab/model")
    # Device setting for input tensor creation. The actual inference device is determined by ONNX Runtime providers.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    # --- Load ONNX session and Tokenizers ---
    print("=== Loading ONNX session and tokenizers ===")
    
    if not Path(args.onnx_path).exists():
        print(f"✗ Error: ONNX model not found at {args.onnx_path}")
        sys.exit(1)

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        ort_session = ort.InferenceSession(args.onnx_path, providers=providers)
    except Exception as e:
        print(f"✗ Error loading ONNX session: {e}")
        sys.exit(1)
        
    print(f"✓ ONNX session loaded with providers: {ort_session.get_providers()}")

    # Check that CUDA is being used
    if "CUDAExecutionProvider" not in ort_session.get_providers():
        print("✗ Error: CUDAExecutionProvider is not available in ONNX Runtime.")
        print("  This can happen if 'onnxruntime' is installed alongside 'onnxruntime-gpu'.")
        print("  Please uninstall 'onnxruntime' and ensure 'onnxruntime-gpu' and CUDA are correctly installed.")
        sys.exit(1)

    tokenizer_1 = CLIPTokenizer.from_pretrained(str(base_dir), subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(str(base_dir), subfolder="tokenizer_2")
    
    # --- Prepare Inputs ---
    print("\n=== Preparing inputs for ONNX model ===")
    
    prompt = args.prompt
    height = args.height
    width = args.width
    num_inference_steps = args.steps
    batch_size = 1

    # Tokenize prompts
    prompt_ids_1 = tokenizer_1(prompt, padding="max_length", max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt").input_ids
    prompt_ids_2 = tokenizer_2(prompt, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids

    # Seed
    if args.random_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    else:
        seed = args.seed
    print(f"\n--- Generating image with seed: {seed} ---")
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Latents and noise
    unet_in_channels = 4 # SDXL's UNet has 4 input channels for latents
    latents_shape = (batch_size, unet_in_channels, height // 8, width // 8)
    initial_latents = torch.randn(latents_shape, generator=generator, device=device, dtype=dtype)
    
    # The number of noise tensors must match the number of steps in the scheduler
    noise_shape = (num_inference_steps, batch_size, unet_in_channels, height // 8, width // 8)
    all_noises = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
    
    # Time IDs
    add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=device, dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size, 1)

    # --- Create ONNX input dictionary ---
    ort_inputs = {
        "prompt_ids_1": prompt_ids_1.cpu().numpy(),
        "prompt_ids_2": prompt_ids_2.cpu().numpy(),
        "initial_latents": initial_latents.cpu().numpy(),
        "all_noises": all_noises.cpu().numpy(),
        "add_time_ids": add_time_ids.cpu().numpy(),
    }
    
    print("\n--- Input Tensor Stats ---")
    for name, arr in ort_inputs.items():
        print_np_stats(name, arr)

    # --- Run Inference ---
    print("\n=== Running ONNX inference (1st run) ===")
    start_time_1 = time.time()
    
    output_names = ["image"]
    outputs_1 = ort_session.run(output_names, ort_inputs)

    end_time_1 = time.time()
    print(f"✓ First ONNX inference took: {end_time_1 - start_time_1:.4f} seconds")
    print_np_stats("ONNX Output 1 (raw_image_np)", outputs_1[0])


    print("\n=== Running ONNX inference (2nd run) ===")
    start_time_2 = time.time()

    outputs_2 = ort_session.run(output_names, ort_inputs)
    raw_image_np = outputs_2[0]
    
    end_time_2 = time.time()
    print(f"✓ Second ONNX inference took: {end_time_2 - start_time_2:.4f} seconds")
    print_np_stats("ONNX Output 2 (raw_image_np)", raw_image_np)

    # --- Post-process and save final image ---
    # Convert numpy array back to torch tensor for easier post-processing
    raw_image_tensor = torch.from_numpy(raw_image_np).to(device)

    print("\n--- Saving final image (from 2nd run) ---")
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
