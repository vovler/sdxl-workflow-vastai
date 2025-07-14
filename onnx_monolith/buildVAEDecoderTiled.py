import json
import sys
import traceback
from safetensors.numpy import load_file
import spoxVAE
import onnxruntime as rt
import onnxoptimizer
import onnx

# --- Configuration ---
SAFETENSORS_FILE_PATH = "/lab/model/vae/diffusion_pytorch_model.safetensors"
CONFIG_FILE_PATH = "/lab/model/vae/config.json"
DECODER_OUTPUT_PATH = "tiled_decoder.onnx"
OPTIMIZED_OUTPUT_PATH1 = "tiled_decoder_optimized_constant_folding.onnx"
OPTIMIZED_OUTPUT_PATH2 = "tiled_decoder_optimized_runtime_extended.onnx"

if __name__ == '__main__':
    try:
        print("--- Building VAE ONNX Models ---")
        
        print(f"Loading config from {CONFIG_FILE_PATH}...")
        with open(CONFIG_FILE_PATH, 'r') as f:
            config = json.load(f)
            
        print(f"Loading weights from {SAFETENSORS_FILE_PATH}...")
        state_dict = load_file(SAFETENSORS_FILE_PATH)

        # Build and save the encoder
        print(f"\nBuilding tiled decoder...")
        encoder_proto = spoxVAE.build_tiled_decoder_onnx_model_with_loop(state_dict, config)
        with open(DECODER_OUTPUT_PATH, "wb") as f:
            f.write(encoder_proto.SerializeToString())
        print(f"Saved encoder model to {DECODER_OUTPUT_PATH}")

        # Optimize the model using onnxruntime
        print(f"\nOptimizing model with onnxruntime...")
        
        # First, create a version with ONLY constant folding using ONNX Optimizer
        print(f"Creating constant folding optimized version...")
        
        model = onnx.load(DECODER_OUTPUT_PATH)
            
        # Apply only constant folding optimization
        optimized_model = onnxoptimizer.optimize(model, ['eliminate_unused_initializer', 'eliminate_constant_of_shape', 'constant_folding'])
            
        # Save the optimized model
        onnx.save(optimized_model, OPTIMIZED_OUTPUT_PATH1)
    
           
        # Fallback to ONNXRuntime if onnxoptimizer is not available
        sess_options_basic = rt.SessionOptions()
        sess_options_basic.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options_basic.optimized_model_filepath = OPTIMIZED_OUTPUT_PATH2
        
        # Create session for basic optimization (constant folding focused)
        session_basic = rt.InferenceSession(DECODER_OUTPUT_PATH, sess_options_basic, providers=['CPUExecutionProvider'])
        print(f"Saved constant folding optimized model to {OPTIMIZED_OUTPUT_PATH2}")
        
        # Now create the extended optimization version
        print(f"Creating extended optimized version...")
        sess_options = rt.SessionOptions()
        
        # Set graph optimization level
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        
        # To enable model serialization after graph optimization set this
        sess_options.optimized_model_filepath = OPTIMIZED_OUTPUT_PATH2

        # Create session which will trigger optimization and save the optimized model
        session = rt.InferenceSession(DECODER_OUTPUT_PATH, sess_options, providers=['CPUExecutionProvider'])
        print(f"Saved optimized model to {OPTIMIZED_OUTPUT_PATH2}")

        print("\n--- Build process completed successfully! ---")

    except FileNotFoundError as e:
        print(f"\nERROR: Could not find a required file: {e.filename}", file=sys.stderr)
        print("\nFull traceback:", file=sys.stderr)
        traceback.print_exc()
    except KeyError as e:
        print(f"\n--- MODEL BUILDING FAILED ---", file=sys.stderr)
        print(f"A required weight/bias was not found. Missing Key -> {e}\n", file=sys.stderr)
        print("Full traceback:", file=sys.stderr)
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred during build: {e}", file=sys.stderr)
        print("\nFull traceback:", file=sys.stderr)
        traceback.print_exc()