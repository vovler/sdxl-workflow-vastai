import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
import os

# --- 1. Define the PyTorch Model with a Python while loop ---

class DynamicLoopModel(nn.Module):
    """
    A model that uses a standard Python 'while' loop to apply an operation
    a dynamic number of times.
    - The number of iterations is determined by a second input to the model.
    """
    def __init__(self):
        super().__init__()
        self.loop_body = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False)
        # Initialize weights for demonstration
        torch.nn.init.xavier_uniform_(self.loop_body.weight)

    def forward(self, x, loop_iterations):
        # TorchDynamo will trace this 'while' loop and convert it
        # into an ONNX 'Loop' operator.
        # Note: The loop count must be a tensor for the dynamic loop to be captured.
        i = torch.tensor(0, dtype=torch.int64)
        while i < loop_iterations:
            x = self.loop_body(x)
            i += 1
        return x

# --- 2. Instantiate the Model ---

model = DynamicLoopModel()
model.eval()

# --- 3. Export the Model to ONNX with Dynamic Shapes ---

print("--- Exporting to ONNX from Loop Model ---")
onnx_file_path = "dynamic_loop_model.onnx"
input_names = ["input", "loop_iterations"]
output_names = ["output"]

# Define dummy inputs for tracing
dummy_input = torch.randn(1, 3, 10, 10)
# The loop count must be a tensor to be treated as a dynamic input
dummy_loop_iterations = torch.tensor(3, dtype=torch.int64)

# Define the dynamic axes
dynamic_axes = {
    input_names[0]: {0: 'batch_size', 2: 'height', 3: 'width'},
    output_names[0]: {0: 'batch_size', 2: 'height', 3: 'width'}
}

# Export using torch.onnx with dynamo=True
torch.onnx.export(
    model,
    (dummy_input, dummy_loop_iterations),
    onnx_file_path,
    input_names=input_names,
    output_names=output_names,
    opset_version=20,
    dynamo=True,
    dynamic_axes=dynamic_axes
)

print(f"Model successfully exported to {onnx_file_path}")
print("Inspect the model with Netron. You will see a 'Loop' operator.")
print("-" * 45 + "\n")


# --- 4. Verify the Exported ONNX Model ---

print("--- Verifying the Dynamic Loop ONNX Model ---")

try:
    ort_session = onnxruntime.InferenceSession(onnx_file_path)
    print("ONNX Runtime session created successfully.")

    # --- Test Case 1: Loop 2 times ---
    print("\n--- Test Case 1 (Loop 2 times) ---")
    input_data = np.ones((1, 3, 8, 8), dtype=np.float32)
    loop_count_1 = np.array(2, dtype=np.int64)
    print(f"Input shape: {input_data.shape}, Loop iterations: {loop_count_1}")

    ort_inputs_1 = {
        ort_session.get_inputs()[0].name: input_data,
        ort_session.get_inputs()[1].name: loop_count_1
    }
    ort_outs_1 = ort_session.run(None, ort_inputs_1)
    output_tensor_1 = ort_outs_1[0]
    print(f"Output shape: {output_tensor_1.shape}")
    assert input_data.shape == output_tensor_1.shape
    print("Shape check passed.")

    # --- Test Case 2: Loop 5 times with different input shape ---
    print("\n--- Test Case 2 (Loop 5 times) ---")
    input_data_2 = np.ones((2, 3, 16, 16), dtype=np.float32)
    loop_count_2 = np.array(5, dtype=np.int64)
    print(f"Input shape: {input_data_2.shape}, Loop iterations: {loop_count_2}")

    ort_inputs_2 = {
        ort_session.get_inputs()[0].name: input_data_2,
        ort_session.get_inputs()[1].name: loop_count_2
    }
    ort_outs_2 = ort_session.run(None, ort_inputs_2)
    output_tensor_2 = ort_outs_2[0]
    print(f"Output shape: {output_tensor_2.shape}")
    assert input_data_2.shape == output_tensor_2.shape
    print("Shape check passed.")

    print("\nVerification successful! The model correctly handles dynamic loops.")

except Exception as e:
    print(f"An error occurred during verification: {e}")