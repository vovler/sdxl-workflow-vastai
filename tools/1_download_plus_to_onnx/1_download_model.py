#!/usr/bin/env python3
import requests
import subprocess
import os
import json
import shutil
from pathlib import Path
import sys

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
        "--max-connection-per-server=10",
        "--split=10", 
        "--min-split-size=10M",
        "--lowest-speed-limit=1M",
        "--user-agent='unknown/None; hf_hub/0.33.0.dev0; python/3.12.0'",
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

    # Delete VAE directory
    if vae_dir.exists():
        shutil.rmtree(vae_dir)
        print(f"Deleted directory: {vae_dir}")

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
    
    print("\n=== Final Summary ===")
    print("✓ All downloads completed successfully!")
    print(f"✓ Models available at: {base_dir}")

if __name__ == "__main__":
    # Check dependencies
    try:
        subprocess.run(["aria2c", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: aria2c is not installed or not in PATH")
        print("Please install aria2c first: sudo apt install aria2 (on Ubuntu/Debian)")
        sys.exit(1)
    
    main()
