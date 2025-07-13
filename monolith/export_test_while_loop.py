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
        self.max_iterations = torch.tensor(max_iterations, dtype=torch.int64)

        # Initialize weights for demonstration
        torch.nn.init.xavier_uniform_(self.denoising_filter.weight)

    # The input argument here is named 'x'
    def forward(self, x):
        initial_loop_vars = (torch.tensor(0, dtype=torch.int64), x)

        def cond(iter_count, image):
            return iter_count < self.max_iterations

        def body(iter_count, image):
            denoised_image = self.denoising_filter(image)
            return iter_count + 1, denoised_image

        _, final_image = torch.while_loop(cond, body, initial_loop_vars)
        
        return (final_image,)

# --- 2. Instantiate the Model ---

model = IterativeDenoisingModel(max_iterations=5)
model.eval()

# --- 3. Export the Model to ONNX with Dynamic Shapes using Dynamo ---

print("--- Exporting to ONNX with Dynamic Shapes using Dynamo ---")
onnx_file_path = "iterative_denoising_model_dynamic.onnx"

# The name for the ONNX graph's input node will be 'input'
input_names = ["input"]
output_names = ["output"]

# --- DYNAMIC SHAPES SETUP ---
batch = Dim("batch_size")
height = Dim("height")
width = Dim("width")

# CRITICAL FIX: The key 'x' must match the forward() argument name.
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
    dynamic_shapes=dynamic_shapes
)

print(f"Model successfully exported to {onnx_file_path}")
print("You can inspect the model with Netron to see the Loop operator and dynamic dimensions.")
print("-" * 45 + "\n")


# --- 4. Verify the Exported ONNX Model with ONNX Runtime ---

print("--- Verifying the Dynamic ONNX Model ---")

try:
    ort_session = onnxruntime.InferenceSession(onnx_file_path)
    print("ONNX Runtime session created successfully.")

    # Get the actual input name from the loaded ONNX model
    onnx_input_name = ort_session.get_inputs()[0].name
    print(f"ONNX model input name: {onnx_input_name}")
    assert onnx_input_name == "input"

    # --- Test Case 1 ---
    print("\n--- Test Case 1 ---")
    input_data_1 = np.random.randn(1, 3, 64, 64).astype(np.float32)
    print(f"Input shape: {input_data_1.shape}")

    ort_inputs_1 = {onnx_input_name: input_data_1}
    ort_outs_1 = ort_session.run(None, ort_inputs_1)
    output_tensor_1 = ort_outs_1[0]
    print(f"Output shape: {output_tensor_1.shape}")
    assert input_data_1.shape == output_tensor_1.shape
    print("Shape check passed.")

    # --- Test Case 2 ---
    print("\n--- Test Case 2 ---")
    input_data_2 = np.random.randn(2, 3, 48, 48).astype(np.float32)
    print(f"Input shape: {input_data_2.shape}")

    ort_inputs_2 = {onnx_input_name: input_data_2}
    ort_outs_2 = ort_session.run(None, ort_inputs_2)
    output_tensor_2 = ort_outs_2[0]
    print(f"Output shape: {output_tensor_2.shape}")
    assert input_data_2.shape == output_tensor_2.shape
    print("Shape check passed.")

    print("\nVerification successful! The model with torch.while_loop correctly exports with Dynamo.")

except Exception as e:
    print(f"An error occurred during verification: {e}")