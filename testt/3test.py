import torch
import torch.nn as nn
import onnx
import os
from typing import List
from diffusers import AutoencoderKL
import traceback

class SimpleVaeDecoder(nn.Module):
    def __init__(self, traced_vae_decoder):
        super().__init__()
        self.vae_decoder = traced_vae_decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # --- Create a list to store the decoded samples ---
        decoded_samples: List[torch.Tensor] = []

        # --- Iterate over the batch dimension ---
        # Note: We use range(latent.shape[0]) to make it traceable by ONNX
        for i in range(latent.shape[0]):
            # --- Decode each latent individually ---
            # We add a batch dimension of 1 since the traced vae_decoder expects it
            decoded_sample = self.vae_decoder(latent[i:i+1])
            decoded_samples.append(decoded_sample)

        # --- Compile the results back into a single tensor ---
        return torch.cat(decoded_samples, dim=0)

def test_export_simple(vae: AutoencoderKL):
    class VaeDecodeWrapper(nn.Module):
        def __init__(self, vae_model):
            super().__init__()
            self.vae = vae_model
        def forward(self, latents):
            return self.vae.decode(latents).sample

    # --- Trace the VAE decoder with a single latent tile ---
    dummy_latent_tile = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        traced_vae_decoder = torch.jit.trace(VaeDecodeWrapper(vae), dummy_latent_tile)

    # --- Instantiate and script the simplified decoder ---
    simple_vae_decoder_instance = SimpleVaeDecoder(traced_vae_decoder)
    scripted_decoder = torch.jit.script(simple_vae_decoder_instance)

    # --- Prepare a sample batch of latents ---
    # We use a dynamic batch size for demonstration
    latent_sample = torch.randn(3, 4, 64, 64, device="cuda", dtype=torch.float16)

    onnx_path = "onnx/simple_vae_decoder.onnx"
    print("Exporting ONNX model with a simplified batch loop...")
    try:
        with torch.no_grad():
            torch.onnx.export(
                scripted_decoder,
                (latent_sample,),
                onnx_path,
                input_names=['latent_sample'],
                output_names=['sample'],
                dynamic_axes={
                    'latent_sample': {0: 'batch_size'},
                    'sample': {0: 'batch_size'}
                },
                opset_version=16
            )
            print(f"✅ Simplified VAE Decoder exported successfully to {onnx_path}")
    except Exception as e:
        print(f"❌ Simplified VAE Decoder export failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    os.makedirs("onnx", exist_ok=True)
    with torch.no_grad():
        print("Loading original VAE model from HuggingFace...")
        # --- Load the VAE model ---
        diffusers_vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        )
        diffusers_vae.to("cuda")
        diffusers_vae.eval()
        print("✅ Original VAE model loaded.")
        test_export_simple(diffusers_vae)