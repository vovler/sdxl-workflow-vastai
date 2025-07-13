#!/usr/bin/env python3
import torch
import torch.nn as nn
from pathlib import Path
import sys
import argparse
import json
import onnx
import torch._dynamo as dynamo
from vae import AutoEncoderKL as CustomAutoEncoderKL
from safetensors.torch import load_file

class VAEExportTest(nn.Module):
    def __init__(self, vae: CustomAutoEncoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, input_tensor):
        return self.vae.test_export(input_tensor)

def main():
    """
    Exports the test_export method to ONNX format.
    """
    parser = argparse.ArgumentParser(description="Export the VAE test_export method to ONNX.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="vae_test_export.onnx",
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
        
        # --- Load VAE ---
        print("=== Loading VAE model ===")
        with open(base_dir / "vae" / "config.json") as f:
            vae_config = json.load(f)
        
        vae = CustomAutoEncoderKL(vae_config).to(dtype).to(device).eval()
        # We don't need weights for this test, but the class requires the config.

        # --- Instantiate Wrapper for Export ---
        vae_export_test = VAEExportTest(vae).to(device).eval()

        # --- Create Dummy Inputs for ONNX Export ---
        print("\n=== Creating dummy inputs for ONNX export ===")
        
        dummy_input = torch.randn((3, 5), device=device, dtype=dtype)
        dummy_inputs = (dummy_input,)

        # --- Export to ONNX ---
        onnx_output_path = args.output_path
        print(f"\n=== Exporting model to: {onnx_output_path} using Dynamo ===")
        
        input_names = ["input_tensor"]
        output_names = ["sum_output"]
        
        try:
            dynamo.config.capture_scalar_outputs = True
            torch.onnx.export(
                vae_export_test,
                dummy_inputs,
                onnx_output_path,
                opset_version=20,
                input_names=input_names,
                output_names=output_names,
                verbose=False,
                dynamo=True
            )
            print(f"✓ ONNX export complete. Model saved to {onnx_output_path}")

        except Exception as e:
            print(f"✗ ONNX export failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            dynamo.config.capture_scalar_outputs = False

if __name__ == "__main__":
    main() 