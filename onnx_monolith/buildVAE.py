import json
import sys
from safetensors.numpy import load_file
import spoxVAE

# --- Configuration ---
SAFETENSORS_FILE_PATH = "/lab/model/vae/diffusion_pytorch_model.safetensors"
CONFIG_FILE_PATH = "/lab/model/vae/config.json"
ENCODER_OUTPUT_PATH = "encoder.onnx"
DECODER_OUTPUT_PATH = "decoder.onnx"

if __name__ == '__main__':
    try:
        print("--- Building VAE ONNX Models ---")
        
        print(f"Loading config from {CONFIG_FILE_PATH}...")
        with open(CONFIG_FILE_PATH, 'r') as f:
            config = json.load(f)
            
        print(f"Loading weights from {SAFETENSORS_FILE_PATH}...")
        state_dict = load_file(SAFETENSORS_FILE_PATH)

        # Build and save the encoder
        print(f"\nBuilding encoder...")
        encoder_proto = spoxVAE.build_encoder_onnx_model(state_dict, config)
        with open(ENCODER_OUTPUT_PATH, "wb") as f:
            f.write(encoder_proto.SerializeToString())
        print(f"Saved encoder model to {ENCODER_OUTPUT_PATH}")

        # Build and save the decoder
        print(f"\nBuilding decoder...")
        decoder_proto = spoxVAE.build_decoder_onnx_model(state_dict, config)
        with open(DECODER_OUTPUT_PATH, "wb") as f:
            f.write(decoder_proto.SerializeToString())
        print(f"Saved decoder model to {DECODER_OUTPUT_PATH}")
        
        print("\n--- Build process completed successfully! ---")

    except FileNotFoundError as e:
        print(f"\nERROR: Could not find a required file: {e.filename}", file=sys.stderr)
    except KeyError as e:
        print(f"\n--- MODEL BUILDING FAILED ---", file=sys.stderr)
        print(f"A required weight/bias was not found. Missing Key -> {e}\n", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during build: {e}", file=sys.stderr)