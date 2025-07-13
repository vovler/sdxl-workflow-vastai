import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
import os
import torch._dynamo as dynamo
import sys

# --- 1. Define the PyTorch Model using torch.while_loop ---

class IterativeDenoisingModel(nn.Module):
    """
    A model that iteratively applies a denoising filter to an image.
    The loop continues for a fixed number of iterations.
    """
    def __init__(self, max_iterations=5):
        super().__init__()
        self.denoising_filter = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False)
        self.max_iterations = torch.tensor(max_iterations)

        # Initialize weights for demonstration
        torch.nn.init.xavier_uniform_(self.denoising_filter.weight)

    def forward(self, x):
        # Initial loop variables: (iteration_count, image_tensor)
        initial_loop_vars = (torch.tensor(0), x)

        def cond(iter_count, image):
            # Loop until the maximum number of iterations is reached
            return iter_count < self.max_iterations

        def body(iter_count, image):
            # Apply the denoising filter
            denoised_image = self.denoising_filter(image)
            # Increment the iteration counter
            return iter_count + 1, denoised_image

        # Execute the while loop
        final_iter_count, final_image = torch.while_loop(cond, body, initial_loop_vars)
        return final_image

# --- 2. Instantiate the Model ---

model = IterativeDenoisingModel(max_iterations=5)
model.eval() # Set model to evaluation mode

# --- 3. Export the Model to ONNX with Dynamic Shapes ---

print("--- Exporting to ONNX with Dynamic Shapes using Dynamo ---")
onnx_file_path = "iterative_denoising_model_dynamic.onnx"
input_names = ["input"]
output_names = ["output"]

# Define a dummy input for tracing
dummy_input = torch.randn(1, 3, 32, 32)

# With Dynamo, dynamic axes are usually detected automatically.
# We no longer need to specify `dynamic_axes` manually.

try:
    dynamo.config.capture_scalar_outputs = True
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_file_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=20, # The Loop operator is well-supported in recent opsets
        dynamo=True
    )
    print(f"Model successfully exported to {onnx_file_path}")
    print("You can inspect the model with Netron to see the Loop operator.")

except Exception as e:
    print(f"✗ ONNX export with Dynamo failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    dynamo.config.capture_scalar_outputs = False


print("-" * 45 + "\n")


# --- 4. Verify the Exported ONNX Model with ONNX Runtime ---

print("--- Verifying the Dynamic ONNX Model ---")

try:
    # Create an ONNX Runtime inference session
    ort_session = onnxruntime.InferenceSession(onnx_file_path)
    print("ONNX Runtime session created successfully.")

    # --- Test Case 1: Input with shape (1, 3, 64, 64) ---
    print("\n--- Test Case 1 ---")
    input_data_1 = np.random.randn(1, 3, 64, 64).astype(np.float32)
    print(f"Input shape: {input_data_1.shape}")

    ort_inputs_1 = {ort_session.get_inputs()[0].name: input_data_1}
    ort_outs_1 = ort_session.run(None, ort_inputs_1)
    output_tensor_1 = ort_outs_1[0]
    print(f"Output shape: {output_tensor_1.shape}")
    assert input_data_1.shape == output_tensor_1.shape
    print("Shape check passed.")

    # --- Test Case 2: Input with a different shape (2, 3, 48, 48) ---
    print("\n--- Test Case 2 ---")
    input_data_2 = np.random.randn(2, 3, 48, 48).astype(np.float32)
    print(f"Input shape: {input_data_2.shape}")

    ort_inputs_2 = {ort_session.get_inputs()[0].name: input_data_2}
    ort_outs_2 = ort_session.run(None, ort_inputs_2)
    output_tensor_2 = ort_outs_2[0]
    print(f"Output shape: {output_tensor_2.shape}")
    assert input_data_2.shape == output_tensor_2.shape
    print("Shape check passed.")

    print("\nVerification successful! The model with torch.while_loop correctly handles dynamic shapes.")

except Exception as e:
    print(f"An error occurred during verification: {e}")
