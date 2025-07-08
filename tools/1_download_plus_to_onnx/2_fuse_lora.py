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
    with torch.no_grad():
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
                use_safetensors=True
            ).to(device)
    
            # Load and fuse LoRA
            print(f"Loading and fusing LoRA from: {lora_path} ({lora_filename})")
            pipeline.load_lora_weights(lora_path, weight_name=lora_filename)
            pipeline.fuse_lora()
            
            print("Fusing complete. Unloading LoRA weights.")
            pipeline.unload_lora_weights()
    
            # Move original UNet to a backup directory before saving the fused one
            unet_dir = base_dir / "unet"
            unfused_unet_dir = base_dir / "unet_unfused"
            unfused_unet_dir.mkdir(parents=True, exist_ok=True)
    
            # Move the safetensors model
            original_unet_safetensors = unet_dir / "diffusion_pytorch_model.safetensors"
            if original_unet_safetensors.exists():
                backup_path = unfused_unet_dir / "diffusion_pytorch_model.safetensors"
                print(f"Moving original UNet model to: {backup_path}")
                shutil.move(str(original_unet_safetensors), backup_path)
            else:
                print(f"Warning: Original UNet safetensors not found at {original_unet_safetensors}. Cannot create backup.")
            
            # Copy the config file
            original_config_json = unet_dir / "config.json"
            if original_config_json.exists():
                backup_path = unfused_unet_dir / "config.json"
                print(f"Copying original config.json to: {backup_path}")
                shutil.copy(str(original_config_json), backup_path)
            else:
                print(f"Warning: Original config.json not found at {original_config_json}. Cannot create backup.")
    
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
        print(f"✓ Original UNet model moved and config.json copied to the 'unet_unfused' directory.")
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
