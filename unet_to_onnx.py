import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from torch.export import Dim
import os
from torch.onnx import _flags

# Force the exporter to use the new, experimental logic that has better
# support for dynamic shapes, as deduced from the source code.
_flags.USE_EXPERIMENTAL_LOGIC = True
_flags.USE_EXPERIMENTAL_DYNAMIC_SHAPES = True


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
    
    print("Loading and fusing DMD2 LoRA...")
    pipe.load_lora_weights("tianweiy/DMD2", weight_name="dmd2_sdxl_4step_lora_fp16.safetensors")
    pipe.fuse_lora(lora_scale=0.8)
    
    unet = pipe.unet
    unet.to(device)
    unet.eval()
    unet_dtype = unet.dtype
    print(f"UNet dtype: {unet_dtype}")

    print("Preparing dummy inputs for UNet export...")
    # SDXL uses classifier-free guidance, so inputs are duplicated (one for conditional, one for unconditional)
    batch_size = 1
    eff_batch_size = batch_size * 2

    # These are latent space dimensions, not image dimensions.
    # The default for SDXL is 1024x1024, which corresponds to 128x128 in latent space.
    latent_height = 1024 // 8
    latent_width = 1024 // 8

    # Get model-specific dimensions
    unet_in_channels = unet.config.in_channels
    unet_latent_shape = (eff_batch_size, unet_in_channels, latent_height, latent_width)
    
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

    _flags.USE_EXPERIMENTAL_LOGIC = True
    _flags.USE_EXPERIMENTAL_DYNAMIC_SHAPES = True
    # The new experimental path works with the simple boolean flag.
    export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
    onnx_program = torch.onnx.export(
        unet_wrapper,
        model_args,
        dynamo=True,
        export_options=export_options,
    )

    print("Optimizing ONNX model...")
    # The new ONNXProgram object has an optimize method.
    onnx_program.optimize()

    print("\n--- ONNX Model Inputs ---")
    for i, input_proto in enumerate(onnx_program.model_proto.graph.input):
        print(f"{i}: {input_proto.name}")

    print("\n--- ONNX Model Outputs ---")
    for i, output_proto in enumerate(onnx_program.model_proto.graph.output):
        print(f"{i}: {output_proto.name}\n")

    print(f"Saving ONNX model to {output_path}...")
    # The new ONNXProgram object has a save method.
    onnx_program.save(output_path)

    print(f"UNet successfully exported to {output_path}")

if __name__ == "__main__":
    main() 