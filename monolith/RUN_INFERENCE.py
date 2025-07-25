#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
# from diffusers import (
#     AutoencoderKL,
# )
# --- MODIFICATION START ---
# 1. Import custom modules and safetensors loader
from unet import UNet2DConditionModel as CustomUNet2DConditionModel
from scheduler import EulerAncestralDiscreteScheduler
from monolith import MonolithicSDXL
from safetensors.torch import load_file
from vae import AutoEncoderKL as CustomAutoEncoderKL
from unet_loop import UNetLoop
import json
# --- MODIFICATION END ---
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from pathlib import Path
import sys
import gc
import os
from PIL import Image


def print_tensor_stats(name, tensor):
    """Prints detailed statistics for a given tensor on a single line."""
    # This function is a no-op during export to avoid side effects.
    return



# --- The Final, "Ready-to-Save" Monolithic Module ---



def main():
    """
    Runs inference using the MonolithicSDXL model.
    """
    with torch.no_grad():
        if not torch.cuda.is_available():
            print("Error: CUDA is not available. This script requires a GPU for model loading.")
            sys.exit(1)

        # --- Configuration ---
        base_dir = Path("/lab/model")
        device = "cuda"
        dtype = torch.float16

        # Pipeline settings
        num_inference_steps = 8
        height = 832
        width = 1216
        batch_size = 1
        seed = 1020094661
        
        # --- Load Model Components ---
        print("=== Loading models ===")
        
        # vae = AutoencoderKL.from_pretrained(base_dir / "vae", torch_dtype=dtype).to(device)
        print("Loading custom VAE...")
        with open(base_dir / "vae" / "config.json") as f:
            vae_config = json.load(f)
        vae = CustomAutoEncoderKL(vae_config).to(dtype).to(device)
        vae_weights_path = base_dir / "vae" / "diffusion_pytorch_model.safetensors"
        vae.load_state_dict(load_file(vae_weights_path, device=device))
        
        print("Enabling VAE tiling for memory-efficient decoding...")
        vae.enable_tiling()
        
        tokenizer_1 = CLIPTokenizer.from_pretrained(str(base_dir), subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(str(base_dir), subfolder="tokenizer_2")
        
        text_encoder_1 = CLIPTextModel.from_pretrained(
            str(base_dir), subfolder="text_encoder", torch_dtype=dtype, use_safetensors=True
        ).to(device)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            str(base_dir), subfolder="text_encoder_2", torch_dtype=dtype, use_safetensors=True
        ).to(device)
        
        # --- MODIFICATION START ---
        # 2. Efficiently load UNet weights using safetensors
        print("Instantiating custom UNet...")
        unet = CustomUNet2DConditionModel().to(device).to(dtype)

        unet_weights_path = base_dir / "unet" / "diffusion_pytorch_model.safetensors"
        if not unet_weights_path.exists():
            # Diffusers from_pretrained also checks for model.safetensors
            unet_weights_path = base_dir / "unet" / "model.safetensors"
            if not unet_weights_path.exists():
                 print(f"Error: UNet weights file (.safetensors) not found in {base_dir / 'unet'}")
                 sys.exit(1)

        print(f"Loading UNet weights directly from {unet_weights_path}...")
        state_dict = load_file(str(unet_weights_path), device=device)
        unet.load_state_dict(state_dict)
        print("Weights loaded into custom UNet successfully.")
        
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()
        # --- MODIFICATION END ---
        
        # --- Memory Optimization ---
        # print("Enabling memory-efficient attention...")
        # unet.enable_xformers_memory_efficient_attention()
        
        # --- Instantiate Monolithic Module ---
        print("Instantiating monolithic module...")
        
        onnx_scheduler = EulerAncestralDiscreteScheduler(
            num_inference_steps=num_inference_steps,
            dtype=dtype,
            timestep_spacing="linspace",
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012
        )
        
        unet_loop = UNetLoop(unet, onnx_scheduler)

        monolith = MonolithicSDXL(
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            unet=unet,
            vae=vae,
            unet_loop=unet_loop,
        ).to(device).eval()

        # --- Clean up memory ---
        print("Cleaning up memory before inference...")
        del text_encoder_1, text_encoder_2, vae, unet
        gc.collect()
        torch.cuda.empty_cache()

        # --- Prepare Inputs for Inference ---
        print("\n=== Preparing inputs for inference ===")
        prompt = "masterpiece,best quality,amazing quality, general, 1girl, aqua_(konosuba), on a swing, looking at viewer, volumetric_lighting, park, night, shiny clothes, shiny skin, detailed_background"

        print("Tokenizing prompt...")
        text_inputs_1 = tokenizer_1(
            prompt,
            padding="max_length",
            max_length=tokenizer_1.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_ids_1 = text_inputs_1.input_ids.to(device)

        text_inputs_2 = tokenizer_2(
            prompt,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_ids_2 = text_inputs_2.input_ids.to(device)

        del tokenizer_1, tokenizer_2
        gc.collect()

        # Prepare latents and noise
        print("Preparing latents and noise...")
        generator = torch.Generator(device=device).manual_seed(seed)
        latents_shape = (batch_size, CustomUNet2DConditionModel().config["in_channels"], height // 8, width // 8)
        initial_latents = torch.randn(latents_shape, generator=generator, device=device, dtype=dtype)

        noise_shape = (num_inference_steps, batch_size, CustomUNet2DConditionModel().config["in_channels"], height // 8, width // 8)
        all_noises = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)

        add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=device, dtype=dtype).repeat(batch_size, 1)

        # --- Run Inference ---
        print("\n=== Running inference ===")
        image_tensor = monolith(
            prompt_ids_1=prompt_ids_1,
            prompt_ids_2=prompt_ids_2,
            initial_latents=initial_latents,
            all_noises=all_noises,
            add_time_ids=add_time_ids,
        )

        # --- Post-process and Save Image ---
        print("Post-processing and saving image...")
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_np = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
        
        # Print stats before casting to uint8
        print(f"Image (min/max/mean): {image_np.min():.4f}, {image_np.max():.4f}, {image_np.mean():.4f}. Contains NaNs: {np.isnan(image_np).any()}")

        image_pil = Image.fromarray((image_np[0] * 255).round().astype("uint8"))

        output_image_path = "output_image.png"
        image_pil.save(output_image_path)
        print(f"✓ Image saved to {output_image_path}")

if __name__ == "__main__":
    main()



