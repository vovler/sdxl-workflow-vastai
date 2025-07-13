import torch
import onnxscript
from onnxscript import script
from onnxscript.values import Opset, OnnxFunctionArgument

# Ensure you have the necessary libraries installed:
# pip install torch>=2.7.0 onnxscript onnx onnxruntime

# --- Step 1: Create a Custom PyTorch Operator ---
# We define a custom operator to encapsulate our logic. This gives us a specific
# target in the PyTorch graph to replace with our custom ONNX implementation.
# The `register_fake` implementation is crucial for compatibility with
# torch.export and Dynamo, as it allows PyTorch to understand the
# metadata (shape, dtype) of the operator's output without running it,
# especially with dynamic shapes. [2]

@torch.library.custom_op("mylibrary::row_sum_loop", mutates_args=())
def row_sum_loop(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    This is the reference implementation of our custom operator in PyTorch.
    It sums each row of the input tensor. This implementation is for eager
    execution in PyTorch and will be replaced during ONNX export.
    """
    return torch.sum(input_tensor, dim=1, keepdim=True)


@row_sum_loop.register_fake
def _row_sum_loop_fake(input_tensor):
    """
    A fake implementation is required for torch.export and dynamo. [2]
    It defines the metadata (e.g., shape, dtype) of the output tensor
    without performing actual computation. This correctly propagates symbolic
    dimensions for dynamic shapes.
    """
    # input_tensor.shape[0] will be a symbolic dimension (e.g., 'batch')
    # when exporting with dynamic shapes.
    output_shape = (input_tensor.shape[0], 1)
    return torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)


# --- Step 2: Define a PyTorch Model Using the Custom Operator ---
# This model incorporates our custom operator, which will be targeted for
# custom translation during the ONNX export process.

class RowSumModel(torch.nn.Module):
    def forward(self, x):
        return row_sum_loop(x)


# --- Step 3: Define the ONNX Loop Body using onnxscript ---
# The ONNX `Loop` operator requires a 'body' graph that is executed for each
# iteration. This function defines that graph. [0]

# Use the target ONNX opset version 20.
op = Opset('', 20)

@script()
def row_sum_loop_body(iteration_num, condition_in, input_tensor):
    """
    Defines the graph for a single iteration of the ONNX Loop. [0]
    """
    row = op.Gather(input_tensor, iteration_num, axis=0)
    row_sum = op.ReduceSum(row, keepdims=False)
    # The body must always return a boolean condition value. [0]
    condition_out = op.Constant(value=torch.tensor(True, dtype=torch.bool))
    # Return signature: (condition, loop_carried_dependencies..., scan_outputs...). [0]
    return condition_out, row_sum


# --- Step 4: Implement the Custom ONNX Translation Function ---
# This function maps our custom PyTorch operator to an ONNX `Loop` operator.

# FIX: The function signature is updated.
# It no longer takes a graph 'g' argument. The arguments must match the
# PyTorch operator's inputs. Type hints are added to resolve warnings.
def onnx_row_sum_loop(input_tensor: OnnxFunctionArgument):
    """
    This function provides the custom ONNX implementation for our PyTorch op. [1]
    It translates the operation into an ONNX Loop.
    """
    shape = op.Shape(input_tensor)
    gather_index = op.Constant(value=torch.tensor(0, dtype=torch.int64))
    trip_count = op.Gather(shape, gather_index)
    
    loop_node = op.Loop(trip_count, None, body=row_sum_loop_body, new_inputs=[input_tensor])
    
    scan_output_sums = loop_node
    final_output = op.Unsqueeze(scan_output_sums, axes=[1])

    return final_output


# --- Step 5: Export the Model and Verify the Output ---
if __name__ == "__main__":
    model = RowSumModel().eval()
    # Use a different batch size to demonstrate dynamic shape handling
    batch_size = 3
    dummy_input = torch.randint(0, 10, (batch_size, 5), dtype=torch.float32)

    # Define dynamic axes for the input and output.
    # The names 'x' and 'row_sum_loop' are taken from the exported program graph.
    dynamic_axes = {
        "x": {0: "batch_size"},
        "row_sum_loop": {0: "batch_size"},
    }

    print("--- Starting ONNX Export with Dynamo, Dynamic Shapes, and Reporting ---")
    
    onnx_program = torch.onnx.export(
        model,
        (dummy_input,),
        dynamo=True,
        opset_version=20,
        custom_translation_table={
            torch.ops.mylibrary.row_sum_loop.default: onnx_row_sum_loop,
        },
        dynamic_axes=dynamic_axes,
        report=True, # Enable generation of the export report
    )

    print("\n--- ONNX Export Successful ---")
    print("Export report has been saved to the current directory.")

    print("\n--- ONNX Model Graph (with dynamic axes) ---")
    print(onnx_program.model)

    # --- Verification ---
    print("\n--- Verifying Outputs ---")
    
    # 1. Run the original PyTorch model
    pytorch_output = model(dummy_input)
    print(f"Input Tensor (shape {dummy_input.shape}):\n{dummy_input}")
    print(f"\nPyTorch Model Output:\n{pytorch_output}")

    # 2. Run the exported ONNX model
    onnx_output = onnx_program(dummy_input)[0]
    print(f"\nONNX Model Output:\n{onnx_output}")

    # 3. Compare the results
    torch.testing.assert_close(pytorch_output, onnx_output)
    print("\n✅ Verification successful: The outputs of the PyTorch and ONNX models are identical.")