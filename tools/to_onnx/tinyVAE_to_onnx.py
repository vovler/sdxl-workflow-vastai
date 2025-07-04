import torch
from diffusers import AutoencoderTiny
from torch.export import Dim


class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae_decoder = vae.decoder

    def forward(self, latent_sample):
        sample = self.vae_decoder(latent_sample)
        return {"sample": sample}


def main():
    """
    Exports the Tiny VAE (TAESDXL) decoder to ONNX.
    """
    model_id = "cqyan/hybrid-sd-tinyvae-xl"
    output_path = "taesdxl_decoder.onnx"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading VAE from model: {model_id}")
    # We only need the VAE for this script.
    vae = AutoencoderTiny.from_pretrained(
        model_id, torch_dtype=torch.float16
    )

    decoder = vae.decoder
    decoder.to(device)
    decoder.eval()
    
    print("Preparing dummy inputs for VAE decoder export...")
    batch_size = 1
    # Standard latent space size for 1024x1024 SDXL.
    latent_channels = 4
    latent_height = 128
    latent_width = 128

    latent_sample_shape = (batch_size, latent_channels, latent_height, latent_width)
    latent_sample = torch.randn(latent_sample_shape, dtype=torch.float16).to(device)

    print("Wrapping VAE decoder for ONNX export.")
    decoder_wrapper = VAEDecoderWrapper(vae)

    model_args = (latent_sample,)

    print("Exporting VAE decoder to ONNX with TorchDynamo...")

    # Define dynamic axes for the model inputs.
    batch_dim = Dim("batch_size")
    height_dim = Dim("height")
    width_dim = Dim("width")
    dynamic_shapes = {
        "latent_sample": {
            0: batch_dim,
            2: height_dim,
            3: width_dim,
        },
    }

    onnx_program = torch.onnx.export(
        decoder_wrapper,
        model_args,
        input_names=["latent_sample"],
        output_names=["sample"],
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
        opset_version=18,
    )

    print("\n--- ONNX Model Inputs ---")
    for i, input_proto in enumerate(onnx_program.model_proto.graph.input):
        print(f"{i}: {input_proto.name}")

    print("\n--- ONNX Model Outputs ---")
    for i, output_proto in enumerate(onnx_program.model_proto.graph.output):
        print(f"{i}: {output_proto.name}")

    print(f"\nSaving ONNX model to {output_path}...")
    onnx_program.save(output_path)

    print(f"VAE decoder successfully exported to {output_path}")


if __name__ == "__main__":
    main() 