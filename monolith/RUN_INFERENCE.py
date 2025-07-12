#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from diffusers import (
    AutoencoderKL,
)
# --- MODIFICATION START ---
# 1. Import your custom UNet and the original diffusers UNet with an alias
from sdxl import UNet2DConditionModel as CustomUNet2DConditionModel
from diffusers import UNet2DConditionModel as DiffusersUNet2DConditionModel
from scheduler import ONNXEulerAncestralDiscreteScheduler
from monolith import MonolithicSDXL

from utils.percentile_calibrator import PercentileCalibrator
# --- MODIFICATION END ---
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from pathlib import Path
import sys
import gc
import os
import modelopt.torch.opt as mto
from modelopt.torch.quantization.calib.max import MaxCalibrator
from modelopt.torch.quantization import utils as quant_utils
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
        
        # --- Load Model Components ---
        print("=== Loading models ===")
        
        vae = AutoencoderKL.from_pretrained(base_dir / "vae", torch_dtype=dtype).to(device)
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
        # 2. Load the original diffusers UNet to get the pretrained weights
        print("Loading base FP16 UNet from diffusers to extract weights...")
        diffusers_unet = DiffusersUNet2DConditionModel.from_pretrained(
            str(base_dir / "unet"), torch_dtype=dtype, use_safetensors=True
        ).to(device)

        int8_checkpoint_path = base_dir / "unet" / "model_int8.pth"
        if not os.path.exists(str(int8_checkpoint_path)):
            print(f"Error: Quantized UNet checkpoint not found at {int8_checkpoint_path}")
            print("Please run the quantization script first (e.g., 3_unet_quantization_int8.py).")
            sys.exit(1)
        
        print(f"Restoring INT8 weights from {int8_checkpoint_path} into diffusers UNet...")
        mto.restore(diffusers_unet, str(int8_checkpoint_path))
        print("INT8 weights restored into diffusers UNet successfully.")

        # 3. Instantiate your custom UNet and load the weights
        print("Instantiating custom UNet and loading weights...")
        unet = CustomUNet2DConditionModel().to(device).to(dtype)
        unet.load_state_dict(diffusers_unet.state_dict())
        print("Weights loaded into custom UNet successfully.")
        
        # Clean up the diffusers model
        del diffusers_unet
        gc.collect()
        torch.cuda.empty_cache()
        # --- MODIFICATION END ---
        
        # --- Memory Optimization ---
        # print("Enabling memory-efficient attention...")
        # unet.enable_xformers_memory_efficient_attention()
        
        # --- Instantiate Monolithic Module ---
        print("Instantiating monolithic module...")
        
        onnx_scheduler = ONNXEulerAncestralDiscreteScheduler(
            num_inference_steps=num_inference_steps,
            dtype=dtype,
            timestep_spacing="linspace",
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012
        )
        
        monolith = MonolithicSDXL(
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            unet=unet,
            vae=vae,
            scheduler_module=onnx_scheduler,
        ).to(device).eval()

        # --- Clean up memory ---
        print("Cleaning up memory before inference...")
        del text_encoder_1, text_encoder_2, vae, unet
        gc.collect()
        torch.cuda.empty_cache()

        # --- Prepare Inputs for Inference ---
        print("\n=== Preparing inputs for inference ===")
        prompt = "a photograph of an astronaut riding a horse on mars"

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
        latents_shape = (batch_size, CustomUNet2DConditionModel().config.in_channels, height // 8, width // 8)
        initial_latents = torch.randn(latents_shape, device=device, dtype=dtype)

        noise_shape = (num_inference_steps, batch_size, CustomUNet2DConditionModel().config.in_channels, height // 8, width // 8)
        all_noises = torch.randn(noise_shape, device=device, dtype=dtype)

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
        image_pil = Image.fromarray((image_np[0] * 255).round().astype("uint8"))

        output_image_path = "output_image.png"
        image_pil.save(output_image_path)
        print(f"✓ Image saved to {output_image_path}")

if __name__ == "__main__":
    main()



