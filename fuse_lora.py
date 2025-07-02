import torch
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download
import os

# Base model and LoRA configuration
base_model_id = "vovler/w-tiny"
lora_model_id = "tianweiy/DMD2"
lora_weight_name = "dmd2_sdxl_4step_lora_fp16.safetensors"

print(f"Loading base model: {base_model_id}")
# Load the main pipeline on CPU to avoid meta device issues
pipeline = DiffusionPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
)

print(f"Loading and fusing LoRA: {lora_model_id}")
# Load and fuse the LoRA weights
pipeline.load_lora_weights(lora_model_id, weight_name=lora_weight_name)

# Move to GPU for fusion if available, as it can be memory-intensive
if torch.cuda.is_available():
    pipeline.to("cuda")

pipeline.fuse_lora()

print("Fusing complete. Unloading LoRA and moving to CPU for saving.")
# Unload LoRA weights from memory
pipeline.unload_lora_weights()

# Move UNet to CPU before saving to ensure portability
pipeline.unet.to("cpu")

# Get the cache path for the base model
model_path = snapshot_download(base_model_id)

# Define the path to save the fused UNet, overwriting the original
unet_path = os.path.join(model_path, "unet")

print(f"Saving fused UNet to: {unet_path}")
# Save the fused UNet
pipeline.unet.save_pretrained(unet_path)

print("Fused UNet saved successfully.")
