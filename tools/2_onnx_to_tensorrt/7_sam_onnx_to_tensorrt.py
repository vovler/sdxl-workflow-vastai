import os
import argparse
from tensorrt_config import SAM_ENCODER_PROFILES, SAM_DECODER_PROFILES
from tensorrt_exporter import build_engine

def build_sam_engine(model_path, model_name, input_profiles):
    """A helper function to build a TensorRT engine for a given SAM model."""
    print(f"--- Building SAM {model_name.capitalize()} Engine ---")
    
    subfolder = "sam"
    onnx_path = os.path.join(model_path, subfolder, f"{model_name}.onnx")
    engine_path = os.path.join(model_path, subfolder, f"{model_name}.plan")

    if not os.path.exists(onnx_path):
        print(f"Error: SAM {model_name} ONNX file not found at {onnx_path}")
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
        )
        print(f"✓ Successfully built engine for SAM {model_name.capitalize()}")
        
        print(f"\nCleaning up ONNX file: {os.path.basename(onnx_path)}")
        try:
            os.remove(onnx_path)
            print(f"✓ Removed {os.path.basename(onnx_path)}")
        except OSError as e:
            print(f"✗ Error deleting ONNX file: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to build engine for SAM {model_name.capitalize()}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Builds TensorRT engines for the SAM encoder and decoder models."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/lab/model",
        help="Path to the model directory containing the SAM ONNX files.",
    )
    args = parser.parse_args()
    model_path = args.model_path

    # --- Encoder ---
    min_enc = SAM_ENCODER_PROFILES["min"]
    opt_enc = SAM_ENCODER_PROFILES["opt"]
    max_enc = SAM_ENCODER_PROFILES["max"]
    
    encoder_profiles = {
        "input_image": (
            (min_enc["bs"], 3, min_enc["height"], min_enc["width"]),
            (opt_enc["bs"], 3, opt_enc["height"], opt_enc["width"]),
            (max_enc["bs"], 3, max_enc["height"], max_enc["width"]),
        ),
    }
    
    # --- Decoder ---
    min_dec = SAM_DECODER_PROFILES["min"]
    opt_dec = SAM_DECODER_PROFILES["opt"]
    max_dec = SAM_DECODER_PROFILES["max"]

    decoder_profiles = {
        "image_embeddings": ((1, 256, 64, 64), (1, 256, 64, 64), (1, 256, 64, 64)),
        "point_coords": (
            (min_dec["bs"], min_dec["num_points"], 2),
            (opt_dec["bs"], opt_dec["num_points"], 2),
            (max_dec["bs"], max_dec["num_points"], 2),
        ),
        "point_labels": (
            (min_dec["bs"], min_dec["num_points"]),
            (opt_dec["bs"], opt_dec["num_points"]),
            (max_dec["bs"], max_dec["num_points"]),
        ),
        "mask_input": ((1, 1, 256, 256), (1, 1, 256, 256), (1, 1, 256, 256)),
        "has_mask_input": ((1,), (1,), (1,)),
        "orig_im_size": ((2,), (2,), (2,)),
    }

    # Build engines
    encoder_ok = build_sam_engine(model_path, "encoder", encoder_profiles)
    decoder_ok = build_sam_engine(model_path, "decoder", decoder_profiles)

    if encoder_ok and decoder_ok:
        print("\nAll SAM engines built successfully.")
        return 0
    else:
        print("\nOne or more SAM engines failed to build.")
        return 1

if __name__ == "__main__":
    exit(main())
