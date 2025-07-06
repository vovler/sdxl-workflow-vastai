#!/usr/bin/env python3
import subprocess
import sys
import os
import shutil
from pathlib import Path

def check_optimum_cli():
    """Check if optimum-cli is available"""
    try:
        result = subprocess.run(["optimum-cli", "--help"], 
                              capture_output=True, check=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def delete_unet_onnx(model_path):
    """Delete the exported UNet ONNX file"""
    model_path = Path(model_path)
    unet_onnx_path = model_path / "unet" / "model.onnx"
    
    try:
        if unet_onnx_path.exists():
            # Get file size before deletion for reporting
            file_size = unet_onnx_path.stat().st_size / (1024**3)  # Size in GB
            
            # Delete the ONNX file
            unet_onnx_path.unlink()
            print(f"✓ Deleted UNet ONNX file: {unet_onnx_path.name} ({file_size:.2f} GB)")
        else:
            print(f"⚠ Warning: UNet ONNX file not found at expected location: {unet_onnx_path}")
    
    except Exception as e:
        print(f"⚠ Warning: Could not delete UNet ONNX file: {e}")

def delete_vae_directories(model_path):
    """Delete VAE and VAE encoder directories"""
    model_path = Path(model_path)
    
    directories_to_delete = ["vae", "vae_encoder"]
    
    print("=== Cleaning up VAE Directories ===")
    
    for dir_name in directories_to_delete:
        dir_path = model_path / dir_name
        
        if dir_path.exists() and dir_path.is_dir():
            try:
                # Calculate total size before deletion
                total_size = 0
                file_count = 0
                for file in dir_path.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size
                        file_count += 1
                
                total_size_gb = total_size / (1024**3)
                
                # Delete the directory
                shutil.rmtree(dir_path)
                print(f"✓ Deleted {dir_name}/ directory ({file_count} files, {total_size_gb:.2f} GB)")
                
            except Exception as e:
                print(f"✗ Failed to delete {dir_name}/ directory: {e}")
        else:
            print(f"⚠ {dir_name}/ directory not found")

def preserve_important_configs(model_path):
    """Report on preserved important config files"""
    model_path = Path(model_path)
    
    important_configs = [
        "text_encoder/config.json",
        "text_encoder_2/config.json", 
        "vae_decoder/config.json",
        "unet/config.json"
    ]
    
    print("=== Preserving Important Config Files ===")
    
    preserved_configs = []
    for config_path in important_configs:
        full_path = model_path / config_path
        if full_path.exists():
            preserved_configs.append(config_path)
            file_size = full_path.stat().st_size / 1024  # Size in KB
            print(f"✓ Preserved: {config_path} ({file_size:.1f} KB)")
        else:
            print(f"⚠ Not found: {config_path}")
    
    return preserved_configs

def delete_non_unet_safetensors(model_path):
    """Delete all safetensors files except those in the UNet directory"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"⚠ Warning: Model path does not exist: {model_path}")
        return
    
    print("=== Cleaning up Non-UNet Safetensors Files ===")
    
    # Find all safetensors files
    all_safetensors = list(model_path.glob("**/*.safetensors"))
    
    # Filter out UNet safetensors (keep those)
    non_unet_safetensors = []
    unet_safetensors = []
    
    for file in all_safetensors:
        # Check if the file is in a unet directory (case insensitive)
        file_parts_lower = [part.lower() for part in file.parts]
        if 'unet' in file_parts_lower:
            unet_safetensors.append(file)
        else:
            non_unet_safetensors.append(file)
    
    print(f"Found {len(all_safetensors)} total safetensors files:")
    print(f"  - {len(unet_safetensors)} UNet files (keeping)")
    print(f"  - {len(non_unet_safetensors)} non-UNet files (deleting)")
    
    if unet_safetensors:
        print(f"\nKeeping UNet safetensors:")
        for file in unet_safetensors:
            relative_path = file.relative_to(model_path)
            file_size = file.stat().st_size / (1024**3)  # Size in GB
            print(f"  ✓ {relative_path} ({file_size:.2f} GB)")
    
    if non_unet_safetensors:
        print(f"\nDeleting non-UNet safetensors:")
        total_deleted_size = 0
        deleted_count = 0
        
        for file in non_unet_safetensors:
            try:
                relative_path = file.relative_to(model_path)
                file_size = file.stat().st_size / (1024**3)  # Size in GB
                
                file.unlink()
                total_deleted_size += file_size
                deleted_count += 1
                print(f"  ✓ Deleted {relative_path} ({file_size:.2f} GB)")
                
            except Exception as e:
                print(f"  ✗ Failed to delete {relative_path}: {e}")
        
        print(f"\n✓ Deleted {deleted_count} safetensors files")
        print(f"✓ Freed up {total_deleted_size:.2f} GB of disk space")
        
        # Clean up empty directories (but preserve those with config files)
        cleanup_empty_directories(model_path, non_unet_safetensors)
    else:
        print("No non-UNet safetensors files found to delete")

def cleanup_empty_directories(model_path, deleted_files):
    """Remove empty directories after deleting safetensors files"""
    # Get unique parent directories of deleted files
    parent_dirs = set()
    for file in deleted_files:
        parent_dirs.add(file.parent)
    
    # Sort by depth (deepest first) to avoid removing parent before child
    sorted_dirs = sorted(parent_dirs, key=lambda p: len(p.parts), reverse=True)
    
    removed_dirs = []
    preserved_dirs = []
    
    for dir_path in sorted_dirs:
        try:
            # Only remove if directory is empty and not the model root
            if dir_path != model_path and dir_path.exists():
                remaining_files = list(dir_path.iterdir())
                if not remaining_files:
                    dir_path.rmdir()
                    relative_path = dir_path.relative_to(model_path)
                    removed_dirs.append(str(relative_path))
                else:
                    # Directory has remaining files (likely config.json)
                    relative_path = dir_path.relative_to(model_path)
                    remaining_names = [f.name for f in remaining_files]
                    preserved_dirs.append(f"{relative_path} (contains: {', '.join(remaining_names)})")
        except Exception as e:
            # Directory might not be empty or other issues, ignore
            pass
    
    if removed_dirs:
        print(f"✓ Removed {len(removed_dirs)} empty directories: {', '.join(removed_dirs)}")
    
    if preserved_dirs:
        print(f"✓ Preserved {len(preserved_dirs)} directories with config files:")
        for dir_info in preserved_dirs:
            print(f"  - {dir_info}")

def export_to_onnx(model_path):
    """Export the model to ONNX format using optimum-cli to the same directory"""
    
    # Verify input model exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"✗ Error: Model path does not exist: {model_path}")
        sys.exit(1)
    
    # Check for required model files
    required_files = ["model_index.json"]
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ Error: Missing required model files: {missing_files}")
        print(f"Make sure {model_path} contains a complete diffusion model")
        sys.exit(1)
    
    print("=== Starting ONNX Export ===")
    print(f"Model path: {model_path}")
    print("Export settings:")
    print("  - Device: CUDA")
    print("  - ONNX Opset: 18")
    print("  - Data type: FP16")
    print("  - Framework: PyTorch")
    print("  - Task: text-to-image")
    print("  - Post-processing: Disabled")
    print("  - Constant folding: Disabled")
    print("  - Output: Same directory as input model")
    
    # Build the optimum-cli command - export to the same directory
    cmd = [
        "optimum-cli", "export", "onnx",
        "--device", "cuda",
        "--opset", "18",
        "--dtype", "fp16",
        "--no-post-process",
        "--no-constant-folding",
        "--framework", "pt",
        "--model", str(model_path),
        "--task", "text-to-image",
        str(model_path)  # Output to the same directory
    ]
    
    print(f"\nExecuting command:")
    print(" ".join(cmd))
    print("\n" + "="*50)
    
    try:
        # Run the export command
        result = subprocess.run(cmd, check=True, text=True)
        print("\n" + "="*50)
        print("✓ ONNX export completed successfully!")
        print(f"✓ ONNX files exported to: {model_path}")
        
        # Check if output files were created
        if model_path.exists():
            onnx_files = list(model_path.glob("**/*.onnx"))
            if onnx_files:
                print(f"✓ Found {len(onnx_files)} ONNX model files:")
                total_size = 0
                for file in sorted(onnx_files):
                    file_size = file.stat().st_size / (1024**3)  # Size in GB
                    total_size += file_size
                    relative_path = file.relative_to(model_path)
                    print(f"  - {relative_path} ({file_size:.2f} GB)")
                print(f"  Total ONNX size: {total_size:.2f} GB")
            else:
                print("⚠ Warning: No .onnx files found in output directory")
        
        # Delete the UNet ONNX file
        print("\n=== Cleaning up UNet ONNX ===")
        delete_unet_onnx(model_path)
        
        # Delete VAE directories
        print()
        delete_vae_directories(model_path)
        
        # Preserve important config files (report on them)
        print()
        preserved_configs = preserve_important_configs(model_path)
        
        # Delete non-UNet safetensors files
        print()
        delete_non_unet_safetensors(model_path)
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ONNX export failed with return code: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⚠ Export interrupted by user")
        sys.exit(1)

def main():
    # Default path
    default_model_path = "/lab/model"
    
    print("=== SDXL to ONNX Converter ===")
    print("This script will export your local SDXL model to ONNX format in the same directory")
    print("Config files for text encoders and VAE decoder will be preserved")
    
    # Allow custom path via command line argument
    if len(sys.argv) >= 2:
        model_path = sys.argv[1]
    else:
        model_path = default_model_path
    
    print(f"\nConfiguration:")
    print(f"Model path (input & output): {model_path}")
    
    # Check if optimum-cli is available
    if not check_optimum_cli():
        print("\n✗ Error: optimum-cli is not installed or not in PATH")
        print("Please install it with:")
        print("pip install optimum[exporters,onnxruntime-gpu]")
        sys.exit(1)
    
    # Check if CUDA is available
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠ Warning: CUDA is not available. Export may fail or be very slow.")
            response = input("Continue anyway? (y/N): ").strip().lower()
            if response != 'y':
                print("Export cancelled.")
                sys.exit(0)
    except ImportError:
        print("⚠ Warning: Could not check CUDA availability (torch not installed)")
    
    # Confirm before starting
    print(f"\nReady to export model to ONNX format in the same directory.")
    print("This process may take a significant amount of time.")
    print("\nAfter export, the following will be deleted:")
    print("  - UNet ONNX file (keeping UNet safetensors)")
    print("  - vae/ directory (all VAE safetensors)")
    print("  - vae_encoder/ directory (VAE encoder ONNX)")
    print("  - All other non-UNet safetensors files")
    print("\nWhat will be preserved:")
    print("  - UNet safetensors (your fused LoRA model)")
    print("  - UNet config.json")
    print("  - text_encoder/ ONNX files + config.json")
    print("  - text_encoder_2/ ONNX files + config.json") 
    print("  - vae_decoder/ ONNX files + config.json")
    print("  - model_index.json")
    response = input("Continue? (Y/n): ").strip().lower()
    if response in ['n', 'no']:
        print("Export cancelled.")
        sys.exit(0)
    
    # Start the export
    export_to_onnx(model_path)
    
    print("\n=== Export Complete ===")
    print("Your SDXL model has been successfully converted!")
    print("✓ ONNX files created for text encoders and VAE decoder")
    print("✓ Config files preserved for all components")
    print("✓ UNet ONNX file removed (UNet safetensors preserved)")
    print("✓ VAE directories removed")
    print("✓ Non-UNet safetensors files cleaned up")
    print(f"✓ Optimized model available at: {model_path}")
    
    # Show final structure
    model_path = Path(model_path)
    if model_path.exists():
        print(f"\nFinal optimized model structure:")
        
        # Organize by component
        components = ["unet", "text_encoder", "text_encoder_2", "vae_decoder"]
        
        for component in components:
            comp_dir = model_path / component
            if comp_dir.exists():
                print(f"  {component}/")
                for file in sorted(comp_dir.iterdir()):
                    if file.is_file():
                        file_size = file.stat().st_size
                        if file.suffix == '.onnx':
                            size_str = f"{file_size / (1024**3):.2f} GB"
                        elif file.suffix == '.safetensors':
                            size_str = f"{file_size / (1024**3):.2f} GB"
                        else:
                            size_str = f"{file_size / 1024:.1f} KB"
                        print(f"    {file.name} ({size_str})")
        
        # Show root files
        root_files = [f for f in model_path.iterdir() if f.is_file()]
        if root_files:
            print("  Root files:")
            for file in sorted(root_files):
                file_size = file.stat().st_size / 1024  # KB
                print(f"    {file.name} ({file_size:.1f} KB)")
        
        # Show total size
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / (1024**3)
        print(f"\nTotal optimized model size: {total_size:.2f} GB")

if __name__ == "__main__":
    main()
