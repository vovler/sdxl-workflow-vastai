import os
import argparse
import subprocess
import sys
from pathlib import Path

def check_onnxslim():
    """Check if onnxslim is installed."""
    try:
        subprocess.run(["onnxslim", "--help"], capture_output=True, check=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Recursively find and optimize all 'model.onnx' files using onnxslim."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/lab/model",
        help="The root directory to search for 'model.onnx' files.",
    )
    args = parser.parse_args()
    model_path = Path(args.model_path)

    if not check_onnxslim():
        print("✗ Error: onnxslim is not installed or not in PATH.")
        print("Please install it with: pip install onnxslim")
        sys.exit(1)

    if not model_path.is_dir():
        print(f"✗ Error: The specified model path is not a directory: {model_path}")
        sys.exit(1)

    print(f"--- Starting ONNX Slimming Process in {model_path} ---")
    
    onnx_files = list(model_path.rglob("model.onnx"))

    if not onnx_files:
        print("⚠ No 'model.onnx' files found to optimize.")
        sys.exit(0)

    print(f"Found {len(onnx_files)} 'model.onnx' files to process.")

    for onnx_file in onnx_files:
        relative_path = onnx_file.relative_to(model_path)
        output_log_path = onnx_file.parent / "onnxslim_output.txt"
        
        print(f"\nProcessing: {relative_path}")
        
        command = ["onnxslim", str(onnx_file), str(onnx_file)]
        
        try:
            print(f"  Running: {' '.join(command)}")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8'
            )
            
            output = result.stdout
            if result.stderr:
                output += "\n--- STDERR ---\n" + result.stderr

            with open(output_log_path, "w", encoding='utf-8') as f:
                f.write(output)
                
            print(f"  ✓ Successfully optimized model.")
            print(f"  ✓ Output log saved to: {output_log_path.relative_to(model_path)}")

        except subprocess.CalledProcessError as e:
            error_message = f"✗ onnxslim failed for {relative_path} with exit code {e.returncode}.\n"
            error_message += "--- STDOUT ---\n"
            error_message += e.stdout
            error_message += "\n--- STDERR ---\n"
            error_message += e.stderr
            
            print(error_message)
            with open(output_log_path, "w", encoding='utf-8') as f:
                f.write(error_message)
        
        except Exception as e:
            print(f"✗ An unexpected error occurred while processing {relative_path}: {e}")
            with open(output_log_path, "w", encoding='utf-8') as f:
                f.write(f"An unexpected error occurred: {e}")

    print("\n--- ONNX Slimming Process Complete ---")

if __name__ == "__main__":
    main()
