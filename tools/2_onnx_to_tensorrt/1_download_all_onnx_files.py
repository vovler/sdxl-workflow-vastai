#!/usr/bin/env python3
import requests
import subprocess
import os
import sys
from pathlib import Path

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

def main():
    # Base directory
    base_dir = Path("/lab/model")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Downloading ONNX KL Model Files ===")
    
    # Get model files
    model_id = "vovler/KL-Onnx"
    print(f"Getting file list for {model_id}...")
    files = get_model_files(model_id)
    
    # Filter out files
    filtered_files = [f for f in files if ".gitattributes" not in f.lower()]
    
    skipped_files = [f for f in files if f not in filtered_files]
    
    print(f"Found {len(files)} total files")
    print(f"Skipping {len(skipped_files)} files: {skipped_files}")
    print(f"Downloading {len(filtered_files)} files")
    print("Files to download:", filtered_files)
    
    # Download model files
    print("\n=== Downloading ONNX KL Files ===")
    for i, filename in enumerate(filtered_files, 1):
        print(f"[{i}/{len(filtered_files)}] {filename}")
        url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"
        download_with_aria2c(url, base_dir, filename)

    print("\n=== Download Summary ===")
    print(f"✓ ONNX KL model files downloaded to: {base_dir}")
    print(f"✓ Skipped files: {', '.join(skipped_files)}")
    
    print("\n=== Final Summary ===")
    print("✓ All downloads completed successfully!")


if __name__ == "__main__":
    # Check dependencies
    try:
        subprocess.run(["aria2c", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: aria2c is not installed or not in PATH")
        print("Please install aria2c first: sudo apt install aria2 (on Ubuntu/Debian)")
        sys.exit(1)
    
    main()
