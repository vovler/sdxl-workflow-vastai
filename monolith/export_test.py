import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
import os

# --- 1. Define the PyTorch Model using torch.cond ---
# (The model definition is identical to the previous example)

class ConditionalProcessingModel(nn.Module):
    """
    A model that applies different processing based on a tensor's content.
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
        predicate = torch.sum(x) > 0

        def true_branch(tensor):
            return self.path_true(tensor)

        def false_branch(tensor):
            return self.path_false(tensor)

        output = torch.cond(predicate, true_branch, false_branch, (x,))
        return output

# --- 2. Instantiate the Model ---

model = ConditionalProcessingModel()
model.eval() # Set model to evaluation mode

# --- 3. Export the Model to ONNX with Dynamic Shapes ---

print("--- Exporting to ONNX with Dynamic Shapes ---")
onnx_file_path = "conditional_model_dynamic.onnx"
input_names = ["input"]
output_names = ["output"]

# Define a dummy input for tracing. The actual shape values don't matter
# as much as the rank (number of dimensions).
dummy_input = torch.randn(1, 3, 10, 10)

# Define the dynamic axes. Here, we are marking the batch size (axis 0),
# height (axis 2), and width (axis 3) as dynamic for both the input and output.
dynamic_axes = {
    input_names[0]: {0: 'batch_size', 2: 'height', 3: 'width'},
    output_names[0]: {0: 'batch_size', 2: 'height', 3: 'width'}
}

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
print("You can inspect the model with Netron to see the dynamic dimensions (e.g., 'batch_size', 'height').")
print("-" * 45 + "\n")


# --- 4. Verify the Exported ONNX Model with ONNX Runtime ---

print("--- Verifying the Dynamic ONNX Model ---")

try:
    # Create an ONNX Runtime inference session
    ort_session = onnxruntime.InferenceSession(onnx_file_path)
    print("ONNX Runtime session created successfully.")

    # --- Test Case 1: Trigger TRUE branch with shape (1, 3, 20, 20) ---
    print("\n--- Test Case 1 (TRUE Branch) ---")
    # Input data is all positive, sum > 0
    input_data_1 = np.ones((1, 3, 20, 20), dtype=np.float32)
    print(f"Input shape: {input_data_1.shape}")

    ort_inputs_1 = {ort_session.get_inputs()[0].name: input_data_1}
    ort_outs_1 = ort_session.run(None, ort_inputs_1)
    output_tensor_1 = ort_outs_1[0]
    print(f"Output shape: {output_tensor_1.shape}")
    assert input_data_1.shape == output_tensor_1.shape
    print("Shape check passed.")

    # Optional: Check if the output is numerically correct (it should be the result of the 3x3 conv)
    # The 3x3 conv with all-ones weights sums a 3x3 neighborhood of channels.
    # For a 3-channel input of all ones, the output should be 9 * 3 = 27.
    print(f"Sample output value: {output_tensor_1[0, 0, 10, 10]:.1f} (Expected ~27.0 for the true branch)")
    assert np.allclose(output_tensor_1, 27.0)


    # --- Test Case 2: Trigger FALSE branch with a different shape (2, 3, 32, 18) ---
    print("\n--- Test Case 2 (FALSE Branch) ---")
    # Input data is all negative, sum < 0
    input_data_2 = np.full((2, 3, 32, 18), -1.0, dtype=np.float32)
    print(f"Input shape: {input_data_2.shape}")

    ort_inputs_2 = {ort_session.get_inputs()[0].name: input_data_2}
    ort_outs_2 = ort_session.run(None, ort_inputs_2)
    output_tensor_2 = ort_outs_2[0]
    print(f"Output shape: {output_tensor_2.shape}")
    assert input_data_2.shape == output_tensor_2.shape
    print("Shape check passed.")

    # Optional: Check if the output is numerically correct (it should be the result of the 1x1 conv)
    # The 1x1 conv was initialized with all-zero weights.
    print(f"Sample output value: {output_tensor_2[0, 0, 10, 10]:.1f} (Expected 0.0 for the false branch)")
    assert np.allclose(output_tensor_2, 0.0)

    print("\nVerification successful! The model correctly handles dynamic shapes and conditional logic.")

except Exception as e:
    print(f"An error occurred during verification: {e}")