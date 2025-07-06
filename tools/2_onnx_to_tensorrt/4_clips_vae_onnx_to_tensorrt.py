import os
import argparse
from collections import OrderedDict
from tensorrt_config import (
    VAE_DECODER_PROFILES,
    CLIP_PROFILES,
)
from tensorrt_exporter import build_engine

def get_abs_path(path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

def get_engine_path(onnx_path: str):
    """
    Returns the TensorRT engine path from an ONNX model path.
    e.g. /path/to/model.onnx -> /path/to/model.plan
    """
    return os.path.splitext(onnx_path)[0] + ".plan"

def main():
    parser = argparse.ArgumentParser(
        description="Builds TensorRT engines for SDXL VAE and CLIP models."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/lab/model",
        help="Path to the model directory containing the ONNX files.",
    )
    args = parser.parse_args()
    model_path = args.model_path

    print("Building TensorRT engines for VAE and CLIP models...")

    # VAE Profiles
    vae_min_p = VAE_DECODER_PROFILES["min"]
    vae_opt_p = VAE_DECODER_PROFILES["opt"]
    vae_max_p = VAE_DECODER_PROFILES["max"]
    
    vae_min_bs, vae_opt_bs, vae_max_bs = vae_min_p["bs"], vae_opt_p["bs"], vae_max_p["bs"]
    vae_min_h, vae_min_w = vae_min_p["height"] // 8, vae_min_p["width"] // 8
    vae_opt_h, vae_opt_w = vae_opt_p["height"] // 8, vae_opt_p["width"] // 8
    vae_max_h, vae_max_w = vae_max_p["height"] // 8, vae_max_p["width"] // 8

    # CLIP Profiles
    clip_min_p = CLIP_PROFILES["min"]
    clip_opt_p = CLIP_PROFILES["opt"]
    clip_max_p = CLIP_PROFILES["max"]
    
    clip_min_bs, clip_opt_bs, clip_max_bs = clip_min_p["bs"], clip_opt_p["bs"], clip_max_p["bs"]
    clip_min_sl, clip_opt_sl, clip_max_sl = clip_min_p["seq_len"], clip_opt_p["seq_len"], clip_max_p["seq_len"]

    # Define components to build
    components = OrderedDict([
        ("VAE Decoder", {
            "subfolder": "vae_decoder",
            "profiles": {
                "latent_sample": (
                    (vae_min_bs, 4, vae_min_h, vae_min_w),
                    (vae_opt_bs, 4, vae_opt_h, vae_opt_w),
                    (vae_max_bs, 4, vae_max_h, vae_max_w),
                ),
            }
        }),
        ("CLIP-L Text Encoder", {
            "subfolder": "text_encoder",
            "profiles": {
                "input_ids": (
                    (clip_min_bs, clip_min_sl), 
                    (clip_opt_bs, clip_opt_sl), 
                    (clip_max_bs, clip_max_sl),
                ),
            }
        }),
        ("CLIP-G Text Encoder", {
            "subfolder": "text_encoder_2",
            "profiles": {
                "input_ids": (
                    (clip_min_bs, clip_min_sl), 
                    (clip_opt_bs, clip_opt_sl), 
                    (clip_max_bs, clip_max_sl),
                ),
            }
        })
    ])

    for name, data in components.items():
        subfolder = data["subfolder"]
        profiles = data["profiles"]
        
        print(f"\n--- Building {name} Engine ---")
        onnx_path = os.path.join(model_path, subfolder, "model.onnx")
        engine_path = os.path.join(model_path, subfolder, "model.plan")

        if not os.path.exists(onnx_path):
            print(f"Error: ONNX file not found at {onnx_path}. Skipping.")
            continue
        
        try:
            build_engine(
                engine_path=engine_path,
                onnx_path=onnx_path,
                input_profiles=profiles,
            )
            print(f"✓ Successfully built engine for {name}")
            
            # Cleanup ONNX file
            print(f"Cleaning up ONNX file: {os.path.basename(onnx_path)}")
            try:
                os.remove(onnx_path)
                print(f"✓ Removed {os.path.basename(onnx_path)}")
            except OSError as e:
                print(f"✗ Error deleting ONNX file: {e}")

        except Exception as e:
            print(f"✗ Failed to build engine for {name}: {e}")

    print("\nAll specified TensorRT engines built.")

if __name__ == "__main__":
    main() 