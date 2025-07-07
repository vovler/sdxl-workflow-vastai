import os
import argparse
from tensorrt_config import SAM_PROFILES
from tensorrt_exporter import build_engine

def build_sam_engine(model_path, model_name, input_profiles, shape_input_profiles={}):
    """A helper function to build a TensorRT engine for the SAM model."""
    print(f"--- Building SAM Engine ---")
    
    subfolder = "sam"
    onnx_path = os.path.join(model_path, subfolder, f"{model_name}.onnx")
    engine_path = os.path.join(model_path, subfolder, f"{model_name}.plan")

    if not os.path.exists(onnx_path):
        print(f"Error: SAM ONNX file not found at {onnx_path}")
        return False

    print(f"\nInput ONNX: {onnx_path}")
    print(f"Output engine: {engine_path}")
    for name, (min_shape, opt_shape, max_shape) in input_profiles.items():
        print(f"  Profile for '{name}': min={min_shape}, opt={opt_shape}, max={max_shape}")
    print()

    try:
        build_engine(
            engine_path=engine_path,
            onnx_path=onnx_path,
            input_profiles=input_profiles,
            shape_input_profiles=shape_input_profiles,
        )
        print(f"✓ Successfully built engine for SAM")
        
        print(f"\nCleaning up ONNX file: {os.path.basename(onnx_path)}")
        try:
            os.remove(onnx_path)
            print(f"✓ Removed {os.path.basename(onnx_path)}")
        except OSError as e:
            print(f"✗ Error deleting ONNX file: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to build engine for SAM: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Builds a TensorRT engine for the combined SAM model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/lab/model",
        help="Path to the model directory containing the SAM ONNX file.",
    )
    args = parser.parse_args()
    model_path = args.model_path

    min_p, opt_p, max_p = SAM_PROFILES["min"], SAM_PROFILES["opt"], SAM_PROFILES["max"]
    
    profiles = {
        "input_image": (
            (min_p["bs"], 3, 1024, 1024),
            (opt_p["bs"], 3, 1024, 1024),
            (max_p["bs"], 3, 1024, 1024),
        ),
        "point_coords": (
            (min_p["bs"], 1, 2),
            (opt_p["bs"], 1, 2),
            (max_p["bs"], 1, 2),
        ),
        "point_labels": (
            (min_p["bs"], 1),
            (opt_p["bs"], 1),
            (max_p["bs"], 1),
        ),
        "mask_input": (
            (min_p["bs"], 1, 256, 256),
            (opt_p["bs"], 1, 256, 256),
            (max_p["bs"], 1, 256, 256),
        ),
        "has_mask_input": ((min_p["bs"],), (opt_p["bs"],), (max_p["bs"],)),
    }
    
    shape_input_profiles = {
        "orig_im_size": ((1024, 1024), (1024, 1024), (1024, 1024)),
    }

    success = build_sam_engine(
        model_path, "model", profiles, shape_input_profiles
    )

    if success:
        print("\nSAM engine built successfully.")
        return 0
    else:
        print("\nSAM engine failed to build.")
        return 1

if __name__ == "__main__":
    exit(main())
