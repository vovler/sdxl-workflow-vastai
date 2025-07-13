import torch
import onnxscript
# Import 'const' for creating constant tensors idiomatically
from onnxscript import script, const
from onnxscript.values import Opset, OnnxFunction
from torch.export import Dim
# Import numpy to create a typed scalar for our constant
import numpy as np

# Ensure you have the necessary libraries installed:
# pip install torch>=2.7.0 onnxscript onnx onnxruntime numpy

# --- Step 1: Create a Custom PyTorch Operator ---
# We define a custom operator to encapsulate our logic. This gives us a specific
# target in the PyTorch graph to replace with our custom ONNX implementation. [2]
@torch.library.custom_op("mylibrary::row_sum_loop", mutates_args=())
def row_sum_loop(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    This is the reference implementation of our custom operator in PyTorch.
    It will be replaced during ONNX export.
    """
    return torch.sum(input_tensor, dim=1, keepdim=True)


@row_sum_loop.register_fake
def _row_sum_loop_fake(input_tensor):
    """
    A fake implementation is required for torch.export and dynamo. [2]
    """
    output_shape = (input_tensor.shape[0], 1)
    return torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)


# --- Step 2: Define a PyTorch Model Using the Custom Operator ---
class RowSumModel(torch.nn.Module):
    def forward(self, x):
        return row_sum_loop(x)


# --- Step 3: Define the ONNX Loop Body using onnxscript ---
# The ONNX `Loop` operator requires a 'body' graph that is executed for each
# iteration. [0]

op = Opset('', 20)

@script()
def row_sum_loop_body(iteration_num, condition_in, input_tensor):
    """
    Defines the graph for a single iteration of the ONNX Loop. [0]
    """
    row = op.Gather(input_tensor, iteration_num, axis=0)
    row_sum = op.ReduceSum(row, keepdims=False)
    # FIX: Use `const()` to create the boolean constant.
    condition_out = const(True)
    return condition_out, row_sum


# --- Step 4: Implement the Custom ONNX Translation Function ---
def onnx_row_sum_loop(input_tensor: OnnxFunction):
    """
    This function provides the custom ONNX implementation for our PyTorch op. [1]
    It translates the operation into an ONNX Loop.
    """
    shape = op.Shape(input_tensor)
    
    # FIX: Use `const()` to create the integer constant for the Gather index.
    # We use a numpy array to ensure the dtype is int64, as required by ONNX.
    gather_index = const(np.array(0, dtype=np.int64))
    trip_count = op.Gather(shape, gather_index)
    
    loop_node = op.Loop(trip_count, None, body=row_sum_loop_body, new_inputs=[input_tensor])
    
    scan_output_sums = loop_node
    final_output = op.Unsqueeze(scan_output_sums, axes=[1])

    return final_output


# --- Step 5: Export the Model and Verify the Output ---
if __name__ == "__main__":
    model = RowSumModel().eval()
    batch_size = 3
    dummy_input = torch.randint(0, 10, (batch_size, 5), dtype=torch.float32)

    # Define dynamic shapes using the recommended `dynamic_shapes` argument.
    batch_dim = Dim("batch_size", min=2)
    dynamic_shapes = ({0: batch_dim},)

    print("--- Starting ONNX Export with Dynamo, Dynamic Shapes, and Reporting ---")
    
    onnx_program = torch.onnx.export(
        model,
        (dummy_input,),
        dynamo=True,
        opset_version=20,
        custom_translation_table={
            torch.ops.mylibrary.row_sum_loop.default: onnx_row_sum_loop,
        },
        dynamic_shapes=dynamic_shapes,
        report=True,
    )

    print("\n--- ONNX Export Successful ---")
    print("Export report has been saved to the current directory.")

    print("\n--- ONNX Model Graph (with dynamic axes) ---")
    print(onnx_program.model)

    # --- Verification ---
    print("\n--- Verifying Outputs ---")
    
    pytorch_output = model(dummy_input)
    print(f"Input Tensor (shape {dummy_input.shape}):\n{dummy_input}")
    print(f"\nPyTorch Model Output:\n{pytorch_output}")

    onnx_output = onnx_program(dummy_input)[0]
    print(f"\nONNX Model Output:\n{onnx_output}")

    torch.testing.assert_close(pytorch_output, onnx_output)
    print("\n✅ Verification successful: The outputs of the PyTorch and ONNX models are identical.")