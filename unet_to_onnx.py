import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
import os

class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

def main():
    """
    Exports the UNet of an SDXL model to ONNX using torch.onnx.export with dynamo.
    """
    model_id = "socks22/sdxl-wai-nsfw-illustriousv14"
    output_path = "unet.onnx"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading SDXL model: {model_id}")
    # Load model and force it into FP16, then use that for all tensors.
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)
    unet = pipe.unet
    unet.to(device)
    unet.eval()
    unet_dtype = unet.dtype
    print(f"UNet dtype: {unet_dtype}")

    print("Preparing dummy inputs for UNet export...")
    # SDXL uses classifier-free guidance, so inputs are duplicated (one for conditional, one for unconditional)
    batch_size = 1
    eff_batch_size = batch_size * 2

    height = 1024
    width = 1024

    # Get model-specific dimensions
    unet_in_channels = unet.config.in_channels
    unet_latent_shape = (eff_batch_size, unet_in_channels, height // 8, width // 8)
    
    # SDXL has two text encoders, their embeddings are concatenated.
    # Text encoder 1: 768, Text encoder 2: 1280.
    # The UNet expects the concatenated projection of 2048.
    text_embed_dim = 2048 
    encoder_hidden_states_shape = (eff_batch_size, 77, text_embed_dim)

    # Additional conditioning from the second text encoder's pooled output
    add_text_embeds_shape = (eff_batch_size, 1280)

    # Additional conditioning for image size and cropping
    add_time_ids_shape = (eff_batch_size, 6)

    # Create dummy tensors with the same dtype as the UNet
    sample = torch.randn(unet_latent_shape, dtype=unet_dtype).to(device)
    timestep = torch.tensor([999] * eff_batch_size, dtype=unet_dtype).to(device)
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape, dtype=unet_dtype).to(device)
    text_embeds = torch.randn(add_text_embeds_shape, dtype=unet_dtype).to(device)
    time_ids = torch.randn(add_time_ids_shape, dtype=unet_dtype).to(device)

    model_args = (sample, timestep, encoder_hidden_states, text_embeds, time_ids)

    print("Wrapping UNet for ONNX export.")
    unet_wrapper = UNetWrapper(unet)
    
    print("Exporting UNet to ONNX with TorchDynamo...")
    export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
    onnx_program = torch.onnx.export(
        unet_wrapper,
        model_args,
        dynamo=True,
        export_options=export_options,
    )

    print("Optimizing ONNX model...")
    onnx_program.optimize()

    print("\n--- ONNX Model Inputs ---")
    for i, input_proto in enumerate(onnx_program.model_proto.graph.input):
        print(f"{i}: {input_proto.name}")

    print("\n--- ONNX Model Outputs ---")
    for i, output_proto in enumerate(onnx_program.model_proto.graph.output):
        print(f"{i}: {output_proto.name}\n")

    print(f"Saving ONNX model to {output_path}...")
    onnx_program.save(output_path)

    print(f"UNet successfully exported to {output_path}")

if __name__ == "__main__":
    main() 