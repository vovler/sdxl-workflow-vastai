#!/usr/bin/env python3
import torch
import sys
from pathlib import Path
import shutil
from diffusers import StableDiffusionXLPipeline

def main():
    """
    This script fuses a LoRA with a base SDXL model.
    It renames the original UNet model to keep a backup.
    """
    base_dir = Path("/lab/model")
    lora_dir = base_dir / "lora"
    lora_filename = "dmd2_sdxl_4step_lora_fp16.safetensors"
    
    base_model_path = str(base_dir)
    lora_path = str(lora_dir)
    lora_file_path = lora_dir / lora_filename

    # Basic validation
    if not base_dir.is_dir():
        print(f"Error: Base model directory not found at '{base_dir}'")
        sys.exit(1)
    if not lora_file_path.is_file():
        print(f"Error: LoRA file not found at '{lora_file_path}'")
        sys.exit(1)

    print("=== Starting LoRA Fusion ===")
    
    try:
        # Check for GPU
        if torch.cuda.is_available():
            print("Loading pipeline to GPU for fusion...")
            device = "cuda"
            torch_dtype=torch.float16
        else:
            print("GPU not available, loading pipeline to CPU for fusion...")
            device = "cpu"
            torch_dtype=torch.float32

        # Load pipeline
        print(f"Loading base model from: {base_model_path}")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            low_cpu_mem_usage=False
        ).to(device)

        # Load and fuse LoRA
        print(f"Loading and fusing LoRA from: {lora_path} ({lora_filename})")
        pipeline.load_lora_weights(lora_path, weight_name=lora_filename)
        pipeline.fuse_lora()
        #print("Fusing complete. Unloading LoRA weights.")
        #pipeline.unload_lora_weights()

        # Move original UNet to a backup directory before saving the fused one
        unet_dir = base_dir / "unet"
        original_unet_safetensors = unet_dir / "diffusion_pytorch_model.safetensors"
        unfused_unet_dir = base_dir / "unet_unfused"
        
        if original_unet_safetensors.exists():
            unfused_unet_dir.mkdir(parents=True, exist_ok=True)
            backup_path = unfused_unet_dir / original_unet_safetensors.name
            print(f"Moving original UNet to: {backup_path}")
            shutil.move(str(original_unet_safetensors), backup_path)
        else:
            print(f"Warning: Original UNet not found at {original_unet_safetensors}. Cannot create backup.")

        # Save the fused pipeline
        print(f"Saving fused pipeline to: {base_model_path}")
        pipeline.save_pretrained(base_model_path, safe_serialization=True)
        print("✓ Fused pipeline saved successfully.")

        # Cleanup
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"✗ Error during LoRA fusion: {e}")
        sys.exit(1)

    print("\n=== Fusion Summary ===")
    print("✓ LoRA has been fused with the UNet model.")
    print(f"✓ Original UNet has been moved to the 'unet_unfused' directory as a backup.")
    print(f"✓ Fused model available at: {base_model_path}")

if __name__ == "__main__":
    # Check dependencies
    try:
        import torch
        import diffusers
    except ImportError as e:
        print(f"Error: Missing required Python package: {e}")
        print("Please install required packages:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("pip install diffusers transformers accelerate")
        sys.exit(1)
    
    main()
