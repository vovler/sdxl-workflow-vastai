#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from diffusers import (
    AutoencoderKL,
)
# --- MODIFICATION START ---
# 1. Import your custom UNet and the original diffusers UNet with an alias
from monolith.sdxl import UNet2DConditionModel as CustomUNet2DConditionModel
from diffusers import UNet2DConditionModel as DiffusersUNet2DConditionModel
from monolith.scheduler import ONNXEulerAncestralDiscreteScheduler
from monolith.monolith import MonolithicSDXL
# --- MODIFICATION END ---
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from pathlib import Path
import sys
import argparse
import gc
import os
import onnx
import glob
import shutil
import modelopt.torch.opt as mto
from modelopt.torch.quantization.calib.max import MaxCalibrator
from modelopt.torch.quantization import utils as quant_utils

def print_tensor_stats(name, tensor):
    """Prints detailed statistics for a given tensor on a single line."""
    # This function is a no-op during export to avoid side effects.
    return



# --- The Final, "Ready-to-Save" Monolithic Module ---



def main():
    """
    Exports the MonolithicSDXL model to ONNX format.
    """
    parser = argparse.ArgumentParser(description="Export the Monolithic SDXL model to ONNX.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="monolith.onnx",
        help="The path to save the exported ONNX model."
    )
    args = parser.parse_args()

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
        print("Cleaning up memory before export...")
        del text_encoder_1, text_encoder_2, vae
        gc.collect()
        torch.cuda.empty_cache()

        # --- Create Dummy Inputs for ONNX Export ---
        print("\n=== Creating dummy inputs for ONNX export ===")

        max_length_1 = tokenizer_1.model_max_length
        max_length_2 = tokenizer_2.model_max_length
        
        dummy_prompt_ids_1 = torch.randint(0, tokenizer_1.vocab_size, (batch_size, max_length_1), dtype=torch.int64, device=device)
        dummy_prompt_ids_2 = torch.randint(0, tokenizer_2.vocab_size, (batch_size, max_length_2), dtype=torch.int64, device=device)
        
        del tokenizer_1, tokenizer_2
        gc.collect()

        latents_shape = (batch_size, unet.config.in_channels, height // 8, width // 8)
        dummy_initial_latents = torch.randn(latents_shape, device=device, dtype=dtype)
        
        noise_shape = (num_inference_steps, batch_size, unet.config.in_channels, height // 8, width // 8)
        dummy_all_noises = torch.randn(noise_shape, device=device, dtype=dtype)
        
        dummy_add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=device, dtype=dtype).repeat(batch_size, 1)
        
        del unet
        gc.collect()
        torch.cuda.empty_cache()

        dummy_inputs = (
            dummy_prompt_ids_1,
            dummy_prompt_ids_2,
            dummy_initial_latents,
            dummy_all_noises,
            dummy_add_time_ids,
        )
        
        # --- Export to ONNX ---
        onnx_output_path = args.output_path
        # temp_dir = onnx_output_path + "_temp"
        # os.makedirs(temp_dir, exist_ok=True)
        # temp_onnx_path = os.path.join(temp_dir, "model.onnx")

        # print(f"\n=== Exporting model to temporary directory: {temp_dir} ===")
        print(f"\n=== Exporting model to: {onnx_output_path} ===")
        
        input_names = [
            "prompt_ids_1", "prompt_ids_2", "initial_latents",
            "all_noises", "add_time_ids"
        ]
        output_names = ["image"]
        
        dynamic_axes = {
            "prompt_ids_1": {0: "batch_size"},
            "prompt_ids_2": {0: "batch_size"},
            "initial_latents": {0: "batch_size", 2: "height_div_8", 3: "width_div_8"},
            "all_noises": {1: "batch_size", 3: "height_div_8", 4: "width_div_8"},
            "add_time_ids": {0: "batch_size"},
            "image": {0: "batch_size", 2: "height", 3: "width"},
        }

        try:
            torch.onnx.export(
                monolith,
                dummy_inputs,
                onnx_output_path,
                opset_version=18,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False,
                do_constant_folding=False,
                verify=False,
                optimize=False,
                # dynamo=True
            )
            print("✓ ONNX export complete.")

        except Exception as e:
            print(f"✗ ONNX export or consolidation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            # --- Clean up temporary directory ---
            # if os.path.exists(temp_dir):
            #     print(f"Cleaning up temporary directory: {temp_dir}")
            #     shutil.rmtree(temp_dir)
            pass

if __name__ == "__main__":
    main() 