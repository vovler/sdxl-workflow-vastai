import torch
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download
import os

# Base model and LoRA configuration
base_model_id = "socks22/sdxl-wai-nsfw-illustriousv14"
lora_model_id = "tianweiy/DMD2"
lora_weight_name = "dmd2_sdxl_4step_lora_fp16.safetensors"

print(f"Loading base model: {base_model_id}")
# Load the main pipeline
pipeline = DiffusionPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading and fusing LoRA: {lora_model_id}")
# Load and fuse the LoRA weights
pipeline.load_lora_weights(lora_model_id, weight_name=lora_weight_name)
pipeline.fuse_lora()

print("Fusing complete. Saving the UNet.")

# Move UNet to CPU before saving to avoid device-specific issues
pipeline.unet.to("cpu")

# Get the cache path for the base model
model_path = snapshot_download(base_model_id)

# Define the path to save the fused UNet, overwriting the original
unet_path = os.path.join(model_path, "unet")

print(f"Saving fused UNet to: {unet_path}")
# Save the fused UNet
pipeline.unet.save_pretrained(unet_path)

print("Fused UNet saved successfully.")
