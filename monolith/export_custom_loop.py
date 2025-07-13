import torch
import onnxscript
from onnxscript import script
from onnxscript.values import Opset
from torch.export import Dim
import onnx

# Ensure you have the necessary libraries installed:
# pip install torch>=2.7.0 onnxscript onnx onnxruntime

# --- Step 1: Create a Custom PyTorch Operator ---
# [2]
@torch.library.custom_op("mylibrary::row_sum_loop", mutates_args=())
def row_sum_loop(input_tensor: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch implementation."""
    return torch.sum(input_tensor, dim=1, keepdim=True)


@row_sum_loop.register_fake
def _row_sum_loop_fake(input_tensor):
    """Fake implementation for torch.export. [2]"""
    output_shape = (input_tensor.shape[0], 1)
    return torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)


# --- Step 2: Define a PyTorch Model Using the Custom Operator ---
class RowSumModel(torch.nn.Module):
    def forward(self, x):
        return row_sum_loop(x)


# --- Step 3: Define the ONNX Loop Body using onnxscript ---
op = Opset('', 20)

# The body signature for ONNX Loop without loop-carried dependencies
# takes iteration_num, condition

def row_sum_loop_body(iteration_num, condition_in):
    """
    Defines the graph for a single iteration of the ONNX Loop.
    No loop-carried dependencies - captures input_tensor from outer scope
    Returns: (condition_out, *scan_outputs)
    """
    row = op.Gather(input_tensor, iteration_num, axis=0)
    row_sum = op.ReduceSum(row, keepdims=False)
    
    condition_out = op.Constant(value=onnx.helper.make_tensor(
        name='const_true', data_type=onnx.TensorProto.BOOL, dims=[], vals=[1]))
    
    # Return condition and scan outputs only
    return condition_out, row_sum


# --- Step 4: Implement the Custom ONNX Translation Function ---
def onnx_row_sum_loop(input_tensor):
    """
    This function provides the custom ONNX implementation for our PyTorch op.
    """
    shape = op.Shape(input_tensor)
    
    gather_index = op.Constant(value=onnx.helper.make_tensor(
        name='const_zero', data_type=onnx.TensorProto.INT64, dims=[], vals=[0]))
    trip_count = op.Gather(shape, gather_index)
    
    # Create initial condition (True to start the loop)
    condition = op.Constant(value=onnx.helper.make_tensor(
        name='loop_condition', data_type=onnx.TensorProto.BOOL, dims=[], vals=[1]))
    
    # Pass input_tensor as loop-carried dependency (it remains constant through iterations)
    # The Loop returns: final_loop_carried_deps..., scan_outputs...
    # Since we have 1 loop-carried dependency (input_tensor) and 1 scan output (row_sum),
    # we expect 2 outputs: final_input_tensor, scan_output_sums
    final_input_tensor, scan_output_sums = op.Loop(
        trip_count, condition, input_tensor,
        body=row_sum_loop_body)

    final_output = op.Unsqueeze(scan_output_sums, axes=[1])

    return final_output


# --- Step 5: Export the Model and Verify the Output ---
if __name__ == "__main__":
    model = RowSumModel().eval()
    batch_size = 3
    dummy_input = torch.randint(0, 10, (batch_size, 5), dtype=torch.float32)

    batch_dim = Dim("batch_size", min=2)
    dynamic_shapes = ({0: batch_dim},)

    print("--- Starting ONNX Export with Dynamo, Dynamic Shapes, and Reporting ---")
    
    # Get the custom operator reference using getattr to avoid linter issues
    custom_op = getattr(torch.ops.mylibrary.row_sum_loop, 'default')
    
    onnx_program = torch.onnx.export(
        model,
        (dummy_input,),
        dynamo=True,
        opset_version=20,
        custom_translation_table={
            custom_op: onnx_row_sum_loop,
        },
        dynamic_shapes=dynamic_shapes,
        report=True,
    )

    if onnx_program is not None:
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
    else:
        print("\n--- ONNX Export Failed ---")
        print("onnx_program is None. Check the export process.")