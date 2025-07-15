import torch
import torch.nn as nn
import onnx
import os
from diffusers import AutoencoderKL as DiffusersAutoencoderKL
from vae import AutoEncoderKL
import traceback

# VAE Decoder Wrapper for ONNX export
class VaeDecoder(nn.Module):
    def __init__(self, vae: AutoEncoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, latent_sample: torch.Tensor) -> torch.Tensor:
        # vae.py's decode returns a tensor directly
        return self.vae.decode(latent_sample)

# Test export
def test_export(vae: AutoEncoderKL):
    # VAE wrapper
    vae_decoder = VaeDecoder(vae)

    # Sample input
    latent_sample = torch.randn(1, 4, 128, 128, device="cuda", dtype=torch.float16)

    print("Testing ONNX export:")
    try:
        with torch.no_grad():
            torch.onnx.export(
                vae_decoder,
                (latent_sample,),
                "onnx/vae_decoder.onnx",
                input_names=['latent_sample'],
                output_names=['sample'],
                dynamic_axes={
                    'latent_sample': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'sample': {0: 'batch_size', 2: 'height_out', 3: 'width_out'}
                },
                opset_version=17
            )
            print("✅ VAE Decoder exported successfully to onnx/vae_decoder.onnx")
    except Exception as e:
        print(f"❌ VAE Decoder export failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    os.makedirs("onnx", exist_ok=True)
    
    with torch.no_grad():
        print("Loading original VAE model from HuggingFace...")
        # Use diffusers to load pretrained weights
        diffusers_vae = DiffusersAutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16
        )
        diffusers_vae.to("cuda")
        print("✅ Original VAE model loaded.")

        print("Initializing custom VAE and loading weights...")
        # Initialize our VAE and load weights
        # The config from diffusers is compatible with our vae.py AutoEncoderKL
        vae = AutoEncoderKL(diffusers_vae.config)
        vae.load_state_dict(diffusers_vae.state_dict())
        vae.to("cuda")
        vae.half() # Ensure all parameters are float16
        vae.eval()
        print("✅ Custom VAE initialized and weights loaded.")

        test_export(vae)