import onnx
import onnxruntime as ort
import numpy as np
import urllib.request
import os

# --- Configuration ---
MODEL_DIR = "onnx"
MODEL_FILENAME = "simple_vae_decoder_direct_optimized_scan.onnx"
MODEL_PATH = "simple_vae_decoder_direct_optimized_scan.onnx"

def run_checks():
    """Runs a series of strict checks on the specified ONNX model."""
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}. Cannot run checks.")
        return

    print("\n--- Starting ONNX Model Checks ---")

    # --- Level 1: Strictest ONNX Specification Check ---
    print("\n[LEVEL 1] Running ONNX Specification Checker...")
    try:
        onnx_model = onnx.load(MODEL_PATH)
        # full_check=True performs a more thorough check, including large tensor data.
        onnx.checker.check_model(onnx_model, full_check=True)
        print("✅ SUCCESS: Model passed the ONNX specification check.")
    except onnx.checker.ValidationError as e:
        print(f"❌ FAILED: Model is invalid according to the ONNX checker: {e}")
        return # Stop if the basic check fails
    except Exception as e:
        print(f"❌ FAILED: An unexpected error occurred while loading or checking the model: {e}")
        return
        
    # --- Level 2: Strictest Shape and Type Inference ---
    print("\n[LEVEL 2] Running Shape & Type Inference...")
    try:
        # strict_mode=True will cause an error if an operator is unknown
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=True)
        print("✅ SUCCESS: Shape inference ran successfully.")
        
        # As an extra strict step, check the model AGAIN after shape inference
        print("    > Running post-inference model check...")
        onnx.checker.check_model(inferred_model, full_check=True)
        print("✅ SUCCESS: Model passed the post-inference ONNX specification check.")

    except Exception as e:
        print(f"❌ FAILED: Shape inference or the post-inference check failed: {e}")
        return # Stop if shape inference introduces an error

    # --- Level 3: Runtime Engine Verification (ONNX Runtime) ---
    print("\n[LEVEL 3] Running Runtime Engine Verification...")
    try:
        # The check is simply attempting to load the model into an InferenceSession.
        # This verifies that the runtime supports all nodes, initializers, and attributes.
        # We explicitly choose a provider for a more targeted check.
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3 # Suppress verbose logs
        
        session = ort.InferenceSession(
            MODEL_PATH, 
            sess_options=sess_options,
            providers=['CUDAExecutionProvider']
        )
        print("✅ SUCCESS: Model was successfully loaded by ONNX Runtime (CPUExecutionProvider).")
        
        # As a bonus, let's print some model info from the session
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        print(f"    > Model Input:  '{input_meta.name}', Shape: {input_meta.shape}, Type: {input_meta.type}")
        print(f"    > Model Output: '{output_meta.name}', Shape: {output_meta.shape}, Type: {output_meta.type}")

    except Exception as e:
        print(f"❌ FAILED: ONNX Runtime could not load the model: {e}")

    print("\n--- All checks complete. ---")

if __name__ == "__main__":
    # Ensure necessary libraries are installed
    try:
        import onnx, onnxruntime
    except ImportError:
        print("Please install required libraries: pip install onnx onnxruntime numpy")
        exit()

    run_checks()