import os
import argparse
from tensorrt_config import (
    UNET_PROFILES,
    UNET_INT8_CONFIG,
)
from tensorrt_exporter import build_engine

def get_default_unet_profiles():
    """Get UNet profiles from the config."""
    min_p = UNET_PROFILES["min"]
    opt_p = UNET_PROFILES["opt"]
    max_p = UNET_PROFILES["max"]

    # Latent space dimensions
    min_h, min_w = min_p["height"] // 8, min_p["width"] // 8
    opt_h, opt_w = opt_p["height"] // 8, opt_p["width"] // 8
    max_h, max_w = max_p["height"] // 8, max_p["width"] // 8
    
    # Batch sizes
    min_bs, opt_bs, max_bs = min_p["bs"], opt_p["bs"], max_p["bs"]
    
    # Sequence lengths
    min_sl, opt_sl, max_sl = min_p["seq_len"], opt_p["seq_len"], max_p["seq_len"]

    return {
        "sample": (
            (min_bs, 4, min_h, min_w),
            (opt_bs, 4, opt_h, opt_w),
            (max_bs, 4, max_h, max_w),
        ),
        "timestep": ((), (), ()),
        "encoder_hidden_states": ((min_bs, min_sl, 2048), (opt_bs, opt_sl, 2048), (max_bs, max_sl, 2048)),
        "text_embeds": ((min_bs, 1280), (opt_bs, 1280), (max_bs, 1280)),
        "time_ids": ((min_bs, 6), (opt_bs, 6), (max_bs, 6)),
    }

def main():
    parser = argparse.ArgumentParser(
        description="Convert quantized ONNX UNet to a TensorRT engine."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/lab/model",
        help="Path to the model directory containing the ONNX file.",
    )
    args = parser.parse_args()

    model_path = args.model_path
    onnx_path = os.path.join(model_path, "unet", "model_int8.onnx")
    engine_path = os.path.join(model_path, "unet", "model_int8.plan")

    # Validate input file
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found: {onnx_path}")
        print("Please run the ONNX export script (4_unet_quantized_to_onnx.py) first.")
        return 1

    print(f"Input ONNX: {onnx_path}")
    print(f"Output engine: {engine_path}")

    # Get default UNet profiles
    input_profiles = get_default_unet_profiles()
    
    print("Using default SDXL UNet input profiles:")
    for name, (min_shape, opt_shape, max_shape) in input_profiles.items():
        print(f"  {name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
    print()

    try:
        build_engine(
            engine_path=engine_path,
            onnx_path=onnx_path,
            input_profiles=input_profiles,
            model_flags=UNET_INT8_CONFIG["flags"],
        )
        print(f"\n✓ Conversion completed successfully!")
        print(f"✓ TensorRT engine saved to: {engine_path}")

        # After successful conversion, clean up the ONNX model files
        print(f"\nCleaning up ONNX model: {os.path.basename(onnx_path)}")
        onnx_data_path = onnx_path.replace(".onnx", ".data")
        
        cleaned_files = 0
        if os.path.exists(onnx_path):
            try:
                os.remove(onnx_path)
                cleaned_files += 1
                print(f"✓ Removed ONNX file: {os.path.basename(onnx_path)}")
            except OSError as e:
                print(f"✗ Error deleting ONNX file: {e}")
        
        if os.path.exists(onnx_data_path):
            try:
                os.remove(onnx_data_path)
                cleaned_files += 1
                print(f"✓ Removed ONNX data file: {os.path.basename(onnx_data_path)}")
            except OSError as e:
                print(f"✗ Error deleting ONNX data file: {e}")

        if cleaned_files > 0:
            print("✓ Cleanup successful.")
        else:
            print("⚠ ONNX files not found for cleanup.")
            
        return 0
        
    except Exception as e:
        print(f"\nError during conversion: {e}")
        return 1

if __name__ == "__main__":
    exit(main())