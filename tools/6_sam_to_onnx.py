import os
import argparse
import sys

def export_sam_to_onnx(pt_path, onnx_path):
    """Export SAM .pth model to .onnx format."""
    if os.path.exists(onnx_path):
        print(f"ONNX file already exists at {onnx_path}, skipping export.")
        return True
    
    if not os.path.exists(pt_path):
        print(f"Error: SAM .pth file not found at {pt_path}")
        return False

    print(f"Exporting {pt_path} to {onnx_path}...")
    try:
        from ultralytics import SAM
        
        # Load the SAM model
        model = SAM(pt_path)
        
        # Export the model to ONNX format
        # The export will create a file like `model.onnx` if the input is `model.pth`
        model.export(format="onnx", imgsz=1024, opset=17, dynamic=True, simplify=False, device='0')
        
        # Ensure the exported file is named correctly
        exported_name = os.path.splitext(os.path.basename(pt_path))[0] + ".onnx"
        default_onnx_path = os.path.join(os.path.dirname(pt_path), exported_name)

        if os.path.exists(default_onnx_path):
            if default_onnx_path != onnx_path:
                os.rename(default_onnx_path, onnx_path)
            print("✓ SAM ONNX export successful.")
            return True
        else:
            print(f"✗ Error: Expected ONNX file not found at {default_onnx_path}")
            return False

    except ImportError:
        print("✗ Error: `ultralytics` package not found.")
        print("Please install it with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"✗ An error occurred during SAM ONNX export: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Exports a SAM .pth model to ONNX format."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/lab/model",
        help="Path to the model directory containing the SAM sam_b.pt file.",
    )
    args = parser.parse_args()
    model_path = args.model_path

    print("--- Exporting SAM sam_b.pt to ONNX ---")
    
    subfolder = "sam"
    pt_path = os.path.join(model_path, subfolder, "sam_b.pt")
    onnx_path = os.path.join(model_path, subfolder, "model.onnx")

    if export_sam_to_onnx(pt_path, onnx_path):
        # Cleanup original .pth file
        print(f"\nCleaning up original PyTorch model: {os.path.basename(pt_path)}")
        try:
            os.remove(pt_path)
            print(f"✓ Removed {os.path.basename(pt_path)}")
        except OSError as e:
            print(f"✗ Error deleting .pth file: {e}")
        return 0
    else:
        print(f"✗ Failed to export SAM model.")
        return 1

if __name__ == "__main__":
    exit(main())
