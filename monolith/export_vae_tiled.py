#!/usr/bin/env python3
import torch
import torch.nn as nn
from pathlib import Path
import sys
import argparse
import json
import onnx
from vae import AutoEncoderKL as CustomAutoEncoderKL
from safetensors.torch import load_file

class VAETiledDecoder(nn.Module):
    def __init__(self, vae: CustomAutoEncoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.tiled_decode(z)

def main():
    """
    Exports the VAE Tiled Decoder to ONNX format.
    """
    parser = argparse.ArgumentParser(description="Export the VAE Tiled Decoder to ONNX.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="vae_tiled_decoder.onnx",
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

        # Pipeline settings from RUN_INFERENCE.py
        height = 832
        width = 1216
        batch_size = 1
        
        # --- Load VAE ---
        print("=== Loading VAE model ===")
        with open(base_dir / "vae" / "config.json") as f:
            vae_config = json.load(f)
        
        vae = CustomAutoEncoderKL(vae_config).to(dtype).to(device).eval()
        vae_weights_path = base_dir / "vae" / "diffusion_pytorch_model.safetensors"
        vae.load_state_dict(load_file(vae_weights_path, device=device))
        print("VAE loaded successfully.")

        # --- Instantiate Wrapper for Export ---
        vae_decoder_exportable = VAETiledDecoder(vae).to(device).eval()

        # --- Create Dummy Inputs for ONNX Export ---
        print("\n=== Creating dummy inputs for ONNX export ===")
        
        latent_channels = vae_config['latent_channels']
        dummy_latent_shape = (batch_size, latent_channels, height // vae.scale_factor, width // vae.scale_factor)
        dummy_latent = torch.randn(dummy_latent_shape, device=device, dtype=dtype)
        
        dummy_inputs = (dummy_latent,)

        # --- Export to ONNX ---
        onnx_output_path = args.output_path
        print(f"\n=== Exporting model to: {onnx_output_path} ===")
        
        input_names = ["latent"]
        output_names = ["image"]
        
        dynamic_axes = {
            "latent": {0: "batch_size", 2: "height_div_8", 3: "width_div_8"},
            "image": {0: "batch_size", 2: "height", 3: "width"},
        }

        try:
            torch.onnx.export(
                vae_decoder_exportable,
                dummy_inputs,
                onnx_output_path,
                opset_version=20,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False,
                do_constant_folding=False,
            )
            print(f"✓ ONNX export complete. Model saved to {onnx_output_path}")

        except Exception as e:
            print(f"✗ ONNX export failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main() 