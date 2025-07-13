import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
import os

# --- 1. Define the PyTorch Model using a Python if/else statement ---

class LoopEquivalentModel(nn.Module):
    """
    An equivalent model that uses a standard Python if/else statement for
    conditional processing. This is often more readable than torch.cond.
    - If the sum of the input tensor's elements is positive, it applies a 3x3 convolution.
    - Otherwise, it applies a 1x1 convolution (pointwise).
    """
    def __init__(self):
        super().__init__()
        self.path_true = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False)
        self.path_false = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=False)

        # Initialize weights for demonstration purposes
        torch.nn.init.ones_(self.path_true.weight)
        torch.nn.init.zeros_(self.path_false.weight)

    def forward(self, x):
        # The predicate is the same as before
        predicate = torch.sum(x) > 0

        # Use a standard Python if/else block instead of torch.cond
        # TorchDynamo (enabled via dynamo=True) will trace this control flow
        # and convert it to an ONNX 'If' operator.
        if predicate:
            output = self.path_true(x)
        else:
            output = self.path_false(x)

        return output

# --- 2. Instantiate the Model ---

model = LoopEquivalentModel()
model.eval() # Set model to evaluation mode

# --- 3. Export the Model to ONNX with Dynamic Shapes ---

print("--- Exporting to ONNX from if/else Model ---")
onnx_file_path = "loop_model_dynamic.onnx"
input_names = ["input"]
output_names = ["output"]

# Define a dummy input for tracing
dummy_input = torch.randn(1, 3, 10, 10)

# Define the dynamic axes, same as before
dynamic_axes = {
    input_names[0]: {0: 'batch_size', 2: 'height', 3: 'width'},
    output_names[0]: {0: 'batch_size', 2: 'height', 3: 'width'}
}

# Export using torch.onnx with dynamo=True
# Dynamo is essential for capturing the Python control flow.
torch.onnx.export(
    model,
    (dummy_input,),
    onnx_file_path,
    input_names=input_names,
    output_names=output_names,
    opset_version=20,
    dynamo=True,
    dynamic_axes=dynamic_axes
)

print(f"Model successfully exported to {onnx_file_path}")
print("Inspect the model with Netron. You will see an 'If' operator, just like the torch.cond version.")
print("-" * 45 + "\n")


# --- 4. Verify the Exported ONNX Model with ONNX Runtime ---
# This verification is identical to the one for the torch.cond model,
# proving that the resulting ONNX file is functionally equivalent.

print("--- Verifying the Dynamic ONNX Model (from if/else) ---")

try:
    # Create an ONNX Runtime inference session
    ort_session = onnxruntime.InferenceSession(onnx_file_path)
    print("ONNX Runtime session created successfully.")

    # --- Test Case 1: Trigger TRUE branch with shape (1, 3, 20, 20) ---
    print("\n--- Test Case 1 (TRUE Branch) ---")
    input_data_1 = np.ones((1, 3, 20, 20), dtype=np.float32) # sum > 0
    print(f"Input shape: {input_data_1.shape}")

    ort_inputs_1 = {ort_session.get_inputs()[0].name: input_data_1}
    ort_outs_1 = ort_session.run(None, ort_inputs_1)
    output_tensor_1 = ort_outs_1[0]
    print(f"Output shape: {output_tensor_1.shape}")
    assert input_data_1.shape == output_tensor_1.shape
    print("Shape check passed.")

    print(f"Sample output value: {output_tensor_1[0, 0, 10, 10]:.1f} (Expected ~27.0 for the true branch)")
    assert np.allclose(output_tensor_1, 27.0)


    # --- Test Case 2: Trigger FALSE branch with a different shape (2, 3, 32, 18) ---
    print("\n--- Test Case 2 (FALSE Branch) ---")
    input_data_2 = np.full((2, 3, 32, 18), -1.0, dtype=np.float32) # sum < 0
    print(f"Input shape: {input_data_2.shape}")

    ort_inputs_2 = {ort_session.get_inputs()[0].name: input_data_2}
    ort_outs_2 = ort_session.run(None, ort_inputs_2)
    output_tensor_2 = ort_outs_2[0]
    print(f"Output shape: {output_tensor_2.shape}")
    assert input_data_2.shape == output_tensor_2.shape
    print("Shape check passed.")

    print(f"Sample output value: {output_tensor_2[0, 0, 10, 10]:.1f} (Expected 0.0 for the false branch)")
    assert np.allclose(output_tensor_2, 0.0)

    print("\nVerification successful! The model with if/else correctly handles dynamic shapes and conditional logic.")

except Exception as e:
    print(f"An error occurred during verification: {e}")