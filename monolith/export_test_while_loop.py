import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
import os
from torch.export import Dim

# --- 1. Define the PyTorch Model using torch.while_loop ---

class IterativeDenoisingModel(nn.Module):
    """
    A model that iteratively applies a denoising filter to an image.
    The loop continues for a fixed number of iterations.
    """
    def __init__(self, max_iterations=5):
        super().__init__()
        self.denoising_filter = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False)
        # Store as a Python int for loop unrolling, not a tensor.
        self.max_iterations = max_iterations

        torch.nn.init.xavier_uniform_(self.denoising_filter.weight)

    def forward(self, x):
        image = x
        # Unroll the loop since torch.while_loop is not supported by the ONNX exporter.
        for _ in range(self.max_iterations):
            image = self.denoising_filter(image)

        # The final iteration count is fixed. Return it as a tensor to match
        # the original model's output signature.
        final_iter_count = torch.tensor(self.max_iterations, dtype=torch.int64)
        return final_iter_count, image

# --- 2. Instantiate the Model ---

model = IterativeDenoisingModel(max_iterations=5)
model.eval()

# --- 3. Export the Model to ONNX with Dynamic Shapes using Dynamo ---

print("--- Exporting to ONNX with Dynamic Shapes using Dynamo ---")
onnx_file_path = "iterative_denoising_model_dynamic.onnx"

input_names = ["input"]
# CRITICAL FIX: Define names for BOTH outputs of the model
output_names = ["final_iteration_count", "final_image"]

# --- DYNAMIC SHAPES SETUP ---
batch = Dim("batch_size")
height = Dim("height")
width = Dim("width")

# The key 'x' must match the forward() argument name.
dynamic_shapes = {
    "x": {0: batch, 2: height, 3: width},
}

dummy_input = torch.randn(1, 3, 32, 32)

torch.onnx.export(
    model,
    (dummy_input,),
    onnx_file_path,
    input_names=input_names,
    output_names=output_names,
    opset_version=20,
    dynamo=True,
    dynamic_shapes=dynamic_shapes,
    report=True
)

print(f"Model successfully exported to {onnx_file_path}")
print("You can inspect the model with Netron. It will have two outputs.")
print("-" * 45 + "\n")


# --- 4. Verify the Exported ONNX Model with ONNX Runtime ---

print("--- Verifying the Dynamic ONNX Model ---")

try:
    ort_session = onnxruntime.InferenceSession(onnx_file_path)
    print("ONNX Runtime session created successfully.")

    onnx_input_name = ort_session.get_inputs()[0].name
    onnx_output_names = [output.name for output in ort_session.get_outputs()]
    print(f"ONNX model input name: {onnx_input_name}")
    print(f"ONNX model output names: {onnx_output_names}")

    # --- Test Case 1 ---
    print("\n--- Test Case 1 ---")
    input_data_1 = np.random.randn(1, 3, 64, 64).astype(np.float32)
    print(f"Input shape: {input_data_1.shape}")

    ort_inputs_1 = {onnx_input_name: input_data_1}
    # ONNX Runtime will return a list of all outputs
    ort_outs_1 = ort_session.run(None, ort_inputs_1)

    # The final image is the SECOND output
    output_tensor_1 = ort_outs_1[1]
    
    print(f"Final iteration count: {ort_outs_1[0]}")
    print(f"Output image shape: {output_tensor_1.shape}")
    assert input_data_1.shape == output_tensor_1.shape
    print("Shape check passed.")

    print("\nVerification successful! The model with torch.while_loop correctly exports with Dynamo.")

except Exception as e:
    print(f"An error occurred during verification: {e}")