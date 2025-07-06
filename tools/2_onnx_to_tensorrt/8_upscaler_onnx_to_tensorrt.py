import os
import argparse
from tensorrt_config import (
    UPSCALER_PROFILES,
)
from tensorrt_exporter import build_engine

def main():
    parser = argparse.ArgumentParser(
        description="Builds a TensorRT engine for the Upscaler model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/lab/model",
        help="Path to the model directory containing the Upscaler ONNX file.",
    )
    args = parser.parse_args()
    model_path = args.model_path

    print("--- Building Upscaler Engine ---")
    
    subfolder = "upscaler"
    onnx_path = os.path.join(model_path, subfolder, "model.onnx")
    engine_path = os.path.join(model_path, subfolder, "model.plan")

    if not os.path.exists(onnx_path):
        print(f"Error: Upscaler ONNX file not found at {onnx_path}")
        print("Please run the Upscaler ONNX export script first.")
        return 1
        
    min_p = UPSCALER_PROFILES["min"]
    opt_p = UPSCALER_PROFILES["opt"]
    max_p = UPSCALER_PROFILES["max"]

    upscaler_input_profiles = {
        "input": (
            (min_p["bs"], 3, min_p["height"], min_p["width"]),
            (opt_p["bs"], 3, opt_p["height"], opt_p["width"]),
            (max_p["bs"], 3, max_p["height"], max_p["width"]),
        ),
    }

    print(f"\nInput ONNX: {onnx_path}")
    print(f"Output engine: {engine_path}")
    for name, (min_shape, opt_shape, max_shape) in upscaler_input_profiles.items():
        print(f"  Profile for '{name}': min={min_shape}, opt={opt_shape}, max={max_shape}")
    print()

    try:
        build_engine(
            engine_path=engine_path,
            onnx_path=onnx_path,
            input_profiles=upscaler_input_profiles,
        )
        print(f"✓ Successfully built engine for Upscaler")
        
        print(f"\nCleaning up ONNX file: {os.path.basename(onnx_path)}")
        try:
            os.remove(onnx_path)
            print(f"✓ Removed {os.path.basename(onnx_path)}")
        except OSError as e:
            print(f"✗ Error deleting ONNX file: {e}")
        
        return 0

    except Exception as e:
        print(f"✗ Failed to build engine for Upscaler: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 