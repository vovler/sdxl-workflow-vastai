#!/usr/bin/env python3
import requests
import subprocess
import os
import json
import torch
import shutil
from pathlib import Path
import sys
from diffusers import StableDiffusionXLPipeline

def get_model_files(model_id):
    """Get list of files from HuggingFace model API"""
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return [sibling['rfilename'] for sibling in data.get('siblings', [])]
    except Exception as e:
        print(f"Error fetching model files: {e}")
        sys.exit(1)

def download_with_aria2c(url, output_dir, filename=None):
    """Download file using aria2c with specified parameters, skipping if it already exists."""
    # Determine the effective filename to check for existence
    effective_filename = filename or os.path.basename(url.split("?")[0])
    output_path = Path(output_dir) / effective_filename

    # If the file already exists, skip the download
    if output_path.exists():
        print(f"✓ File already exists, skipping: {effective_filename}")
        return
        
    cmd = [
        "aria2c",
        "--max-connection-per-server=16",
        "--split=16", 
        "--min-split-size=10M",
        "--dir", str(output_dir),
        "--continue=true",
        "--auto-file-renaming=false"
    ]
    if filename:
        cmd.extend(["--out", filename])
    cmd.append(url)
    
    print(f"Downloading: {url}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Downloaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Download failed: {e}")
        sys.exit(1)

def fuse_lora_with_unet(base_model_path, lora_path, lora_filename):
    """Fuse LoRA weights with the UNet model"""
    print("\n=== Starting LoRA Fusion ===")
    
    try:
        # Check for GPU availability
        if torch.cuda.is_available():
            print("Loading pipeline to GPU for fusion...")
            device = "cuda"
        else:
            print("GPU not available, performing fusion on CPU...")
            device = "cpu"

        # Load the main pipeline, explicitly disabling low_cpu_mem_usage to avoid meta tensors.
        # This will load the full model onto the CPU first, then move it to the GPU.
        print(f"Loading base model from: {base_model_path}")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=False  # <<< This is the key change
        ).to(device)

        print(f"Loading and fusing LoRA from: {lora_path}")
        # Load and fuse the LoRA weights
        pipeline.load_lora_weights(lora_path, weight_name=lora_filename)

        print("Fusing LoRA weights...")
        pipeline.fuse_lora()

        print("Fusing complete. Unloading LoRA and moving to CPU for saving.")
        # Unload LoRA weights from memory
        pipeline.unload_lora_weights()

        # Save the entire pipeline back to disk.
        # This ensures all components, including the fused UNet and the text encoders,
        # are saved in a format that won't cause "meta tensor" issues in later scripts.
        print(f"Saving the entire pipeline to: {base_model_path}")
        pipeline.save_pretrained(base_model_path, safe_serialization=True)
        print("✓ Entire pipeline saved successfully.")
        
        # Clean up memory
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"✗ Error during LoRA fusion: {e}")
        sys.exit(1)

def main():
    # Base directory
    base_dir = Path("/lab/model")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Downloading SDXL Model Files ===")
    
    # Get main model files
    model_id = "John6666/wai-nsfw-illustrious-sdxl-v140-sdxl"
    print(f"Getting file list for {model_id}...")
    files = get_model_files(model_id)
    
    # Filter out VAE files and README.md (case insensitive)
    filtered_files = []
    for f in files:
        f_lower = f.lower()
        if 'vae' not in f_lower and 'readme.md' not in f_lower and ".gitattributes" not in f_lower:
            filtered_files.append(f)
    
    skipped_files = [f for f in files if f not in filtered_files]
    
    print(f"Found {len(files)} total files")
    print(f"Skipping {len(skipped_files)} files: {skipped_files}")
    print(f"Downloading {len(filtered_files)} files")
    print("Files to download:", filtered_files)
    
    # Download main model files
    print("\n=== Downloading Main Model Files ===")
    for i, filename in enumerate(filtered_files, 1):
        print(f"[{i}/{len(filtered_files)}] {filename}")
        url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"
        download_with_aria2c(url, base_dir, filename)
    
    # Create and download to lora directory
    print("\n=== Setting up LoRA ===")
    lora_dir = base_dir / "lora"
    lora_dir.mkdir(exist_ok=True)
    print(f"Created directory: {lora_dir}")
    
    lora_filename = "dmd2_sdxl_4step_lora_fp16.safetensors"
    lora_url = f"https://huggingface.co/tianweiy/DMD2/resolve/main/{lora_filename}"
    download_with_aria2c(lora_url, lora_dir, lora_filename)
    
    # Create and download to vae directory  
    print("\n=== Setting up VAE ===")
    vae_dir = base_dir / "vae"
    vae_dir.mkdir(exist_ok=True)
    print(f"Created directory: {vae_dir}")
    
    vae_files = [
        ("https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors", "diffusion_pytorch_model.safetensors"),
        ("https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/config.json", "config.json")
    ]
    
    for i, (vae_url, filename) in enumerate(vae_files, 1):
        print(f"[{i}/{len(vae_files)}] {filename}")
        download_with_aria2c(vae_url, vae_dir, filename)
    
    # Create and download to tagger directory
    print("\n=== Setting up Tagger Model ===")
    tagger_dir = base_dir / "tagger"
    tagger_dir.mkdir(exist_ok=True)
    print(f"Created directory: {tagger_dir}")

    tagger_files = [
        ("https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/model.onnx", "model.onnx"),
        ("https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/selected_tags.csv", "selected_tags.csv")
    ]

    for i, (tagger_url, filename) in enumerate(tagger_files, 1):
        print(f"[{i}/{len(tagger_files)}] {filename}")
        download_with_aria2c(tagger_url, tagger_dir, filename)

    # Create and download to upscaler directory
    print("\n=== Setting up Upscaler Model ===")
    upscaler_dir = base_dir / "upscaler"
    upscaler_dir.mkdir(exist_ok=True)
    print(f"Created directory: {upscaler_dir}")

    upscaler_url = "https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV4/2x-AnimeSharpV4_Fast_RCAN_PU_fp16_opset17.onnx"
    upscaler_filename = "model.onnx"
    download_with_aria2c(upscaler_url, upscaler_dir, upscaler_filename)

    # Create and download to YOLO directory
    print("\n=== Setting up YOLO Model ===")
    yolo_dir = base_dir / "yolo"
    yolo_dir.mkdir(exist_ok=True)
    print(f"Created directory: {yolo_dir}")

    yolo_url = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov9c.pt"
    yolo_filename = "model.pt"
    download_with_aria2c(yolo_url, yolo_dir, yolo_filename)

    # Create and download to SAM directory
    print("\n=== Setting up SAM Model ===")
    sam_dir = base_dir / "sam"
    sam_dir.mkdir(exist_ok=True)
    print(f"Created directory: {sam_dir}")

    sam_url = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/sams/sam_vit_b_01ec64.pth"
    sam_filename = "model.pth"
    download_with_aria2c(sam_url, sam_dir, sam_filename)

    print("\n=== Download Summary ===")
    print(f"✓ Main model files downloaded to: {base_dir}")
    print(f"✓ LoRA downloaded to: {lora_dir}")
    print(f"✓ VAE downloaded to: {vae_dir}")
    print(f"✓ Tagger model downloaded to: {tagger_dir}")
    print(f"✓ Upscaler model downloaded to: {upscaler_dir}")
    print(f"✓ YOLO model downloaded to: {yolo_dir}")
    print(f"✓ SAM model downloaded to: {sam_dir}")
    print(f"✓ Skipped files: {', '.join(skipped_files)}")
    
    # Fuse LoRA with UNet
    lora_file_path = lora_dir / lora_filename
    if lora_file_path.exists():
        fuse_lora_with_unet(str(base_dir), str(lora_dir), lora_filename)
        
        # Delete the original LoRA directory after successful fusion
        print("\n=== Cleaning up LoRA files ===")
        try:
            shutil.rmtree(lora_dir)
            print(f"✓ Deleted original LoRA directory: {lora_dir}")
        except Exception as e:
            print(f"⚠ Warning: Could not delete LoRA directory: {e}")
    else:
        print(f"✗ LoRA file not found: {lora_file_path}")
        sys.exit(1)
    
    print("\n=== Final Summary ===")
    print("✓ All downloads completed successfully!")
    print("✓ LoRA has been fused with the UNet model!")
    print("✓ Original LoRA files have been cleaned up!")
    print(f"✓ Fused model available at: {base_dir}")

if __name__ == "__main__":
    # Check dependencies
    try:
        subprocess.run(["aria2c", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: aria2c is not installed or not in PATH")
        print("Please install aria2c first: sudo apt install aria2 (on Ubuntu/Debian)")
        sys.exit(1)
    
    # Check Python packages
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
