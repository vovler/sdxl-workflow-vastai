import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
import os
# torch.scan is an experimental H.O.P.
from torch.experimental.higher_order_ops import scan

# --- 1. Define the PyTorch Model using torch.scan ---

class ScanDenoisingModel(nn.Module):
    """
    A model that iteratively applies a denoising filter using torch.scan.
    """
    def __init__(self, max_iterations=5):
        super().__init__()
        self.denoising_filter = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False)
        self.max_iterations = max_iterations
        torch.nn.init.xavier_uniform_(self.denoising_filter.weight)

    def forward(self, x):
        # Define the function to be applied at each step of the scan.
        # It takes the 'carry' (the image from the last step) and a 'slice' from xs.
        def combine_fn(carry, x_slice):
            # We ignore x_slice as it's from a dummy tensor.
            # The core logic is applying the filter to the carry.
            next_carry = self.denoising_filter(carry)
            
            # The second return value is the output for this specific iteration.
            # We return the denoised image. It's crucial to clone to avoid
            # aliasing issues, as required by the scan operator.
            y_slice = next_carry.clone()
            
            return next_carry, y_slice

        # The initial carry for the scan is the input image itself.
        init = x

        # Create a dummy input tensor for 'xs'. The scan will iterate
        # over its first dimension. Its length determines the number of loops.
        dummy_xs = torch.zeros(self.max_iterations, 1)

        # Perform the scan. This returns the final carry and a tensor where
        # all the y_slices have been stacked along a new dimension (dim 0).
        final_carry, stacked_ys = scan(combine_fn, init, dummy_xs)

        # The result of the final iteration is the final_carry.
        return final_carry

# --- 2. Instantiate the Model ---

model = ScanDenoisingModel(max_iterations=5)
model.eval()

# --- 3. Export the Model using the TorchScript Exporter ---

print("--- Exporting to ONNX using the TorchScript Exporter (dynamo=False) ---")
onnx_file_path = "scan_denoising_model_dynamic.onnx"
input_names = ["input"]
output_names = ["output"]

dummy_input = torch.randn(1, 3, 32, 32)

# Use the 'dynamic_axes' API for the TorchScript backend
dynamic_axes = {
    input_names[0]: {0: 'batch_size', 2: 'height', 3: 'width'},
    output_names[0]: {0: 'batch_size', 2: 'height', 3: 'width'}
}

# NOTE: We do NOT use 'dynamo=True'. This makes PyTorch fall back
# to the TorchScript exporter which has support for torch.scan.
torch.onnx.export(
    model,
    (dummy_input,),
    onnx_file_path,
    input_names=input_names,
    output_names=output_names,
    opset_version=20,
    dynamic_axes=dynamic_axes
)

print(f"Model successfully exported to {onnx_file_path}")
print("You can inspect the model with Netron to see the 'Scan' operator.")
print("-" * 45 + "\n")


# --- 4. Verify the Exported ONNX Model ---

print("--- Verifying the ONNX Model ---")
try:
    ort_session = onnxruntime.InferenceSession(onnx_file_path)
    print("ONNX Runtime session created successfully.")

    onnx_input_name = ort_session.get_inputs()[0].name
    
    # --- Test Case 1: Dynamic Shape (2, 3, 64, 64) ---
    print("\n--- Test Case 1 ---")
    input_data_1 = np.random.randn(2, 3, 64, 64).astype(np.float32)
    ort_inputs_1 = {onnx_input_name: input_data_1}
    ort_outs_1 = ort_session.run(None, ort_inputs_1)
    output_tensor_1 = ort_outs_1[0]
    
    print(f"Input shape: {input_data_1.shape}")
    print(f"Output shape: {output_tensor_1.shape}")
    assert input_data_1.shape == output_tensor_1.shape
    print("Shape check passed.")
    print("\nVerification successful!")

except Exception as e:
    print(f"An error occurred during verification: {e}")```