import os
import argparse
import sys

def export_yolo_to_onnx(pt_path, onnx_path):
    """Export YOLO .pt model to .onnx format."""
    if os.path.exists(onnx_path):
        print(f"ONNX file already exists at {onnx_path}, skipping export.")
        return True
    
    if not os.path.exists(pt_path):
        print(f"Error: YOLO .pt file not found at {pt_path}")
        return False

    print(f"Exporting {pt_path} to {onnx_path}...")
    try:
        from ultralytics import YOLO
        model = YOLO(pt_path)
        model.export(format="onnx", imgsz=640, opset=18, dynamic=True, simplify=False, device='0')
        
        # The export command saves it with the original basename, e.g., `model.onnx` if `pt_path` is `/path/to/model.pt`
        exported_name = os.path.splitext(os.path.basename(pt_path))[0] + ".onnx"
        default_onnx_path = os.path.join(os.path.dirname(pt_path), exported_name)

        if os.path.exists(default_onnx_path):
            if default_onnx_path != onnx_path:
                os.rename(default_onnx_path, onnx_path)
            print("✓ YOLO ONNX export successful.")
            return True
        else:
            print(f"✗ Error: Expected ONNX file not found at {default_onnx_path}")
            return False

    except ImportError:
        print("✗ Error: `ultralytics` package not found.")
        print("Please install it with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"✗ An error occurred during YOLO ONNX export: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Exports a YOLO .pt model to ONNX format."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/lab/model",
        help="Path to the model directory containing the YOLO .pt file.",
    )
    args = parser.parse_args()
    model_path = args.model_path

    print("--- Exporting YOLO .pt to ONNX ---")
    
    subfolder = "yolo"
    pt_path = os.path.join(model_path, subfolder, "model.pt")
    onnx_path = os.path.join(model_path, subfolder, "model.onnx")

    if export_yolo_to_onnx(pt_path, onnx_path):
        # Cleanup original .pt file
        print(f"\nCleaning up original PyTorch model: {os.path.basename(pt_path)}")
        try:
            os.remove(pt_path)
            print(f"✓ Removed {os.path.basename(pt_path)}")
        except OSError as e:
            print(f"✗ Error deleting .pt file: {e}")
        return 0
    else:
        print(f"✗ Failed to export YOLO model.")
        return 1

if __name__ == "__main__":
    exit(main())
