import os
import argparse
from tensorrt_config import (
    WD_TAGGER_PROFILES,
)
from tensorrt_exporter import build_engine

def main():
    parser = argparse.ArgumentParser(
        description="Builds a TensorRT engine for the WD-Tagger model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/lab/model",
        help="Path to the model directory containing the Tagger ONNX file.",
    )
    args = parser.parse_args()
    model_path = args.model_path

    print("--- Building WD-Tagger Engine ---")
    
    # Define paths
    subfolder = "tagger"
    onnx_path = os.path.join(model_path, subfolder, "model.onnx")
    engine_path = os.path.join(model_path, subfolder, "model.plan")

    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found at {onnx_path}. Skipping.")
        return 1
    
    # Define input profiles for the tagger
    min_p = WD_TAGGER_PROFILES["min"]
    opt_p = WD_TAGGER_PROFILES["opt"]
    max_p = WD_TAGGER_PROFILES["max"]

    tagger_input_profiles = {
        "input": (
            (min_p["bs"], min_p["height"], min_p["width"], 3),
            (opt_p["bs"], opt_p["height"], opt_p["width"], 3),
            (max_p["bs"], max_p["height"], max_p["width"], 3),
        ),
    }

    print(f"Input ONNX: {onnx_path}")
    print(f"Output engine: {engine_path}")
    for name, (min_shape, opt_shape, max_shape) in tagger_input_profiles.items():
        print(f"  Profile for '{name}': min={min_shape}, opt={opt_shape}, max={max_shape}")
    print()

    try:
        build_engine(
            engine_path=engine_path,
            onnx_path=onnx_path,
            input_profiles=tagger_input_profiles,
        )
        print(f"✓ Successfully built engine for WD-Tagger")
        
        # Cleanup ONNX file
        print(f"\nCleaning up ONNX file: {os.path.basename(onnx_path)}")
        try:
            os.remove(onnx_path)
            print(f"✓ Removed {os.path.basename(onnx_path)}")
        except OSError as e:
            print(f"✗ Error deleting ONNX file: {e}")
        
        return 0

    except Exception as e:
        print(f"✗ Failed to build engine for WD-Tagger: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
