import torch
import onnxscript
from onnxscript import script
from onnxscript.values import Opset, OnnxFunction
from torch.export import Dim
# Import the main 'onnx' library to use its helper functions
import onnx

# Ensure you have the necessary libraries installed:
# pip install torch>=2.7.0 onnxscript onnx onnxruntime numpy

# --- Step 1: Create a Custom PyTorch Operator ---
# We define a custom operator to encapsulate our logic. This gives us a specific
# target in the PyTorch graph to replace with our custom ONNX implementation. [2]
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

# The loop body's signature is updated to accept and return the dummy state.
@script()
def row_sum_loop_body(iteration_num, condition_in, dummy_state_in, input_tensor):
    """
    Defines the graph for a single iteration of the ONNX Loop. [0]
    It now accepts and returns a dummy loop-carried state to resolve type inference.
    """
    row = op.Gather(input_tensor, iteration_num, axis=0)
    row_sum = op.ReduceSum(row, keepdims=False)
    
    condition_out = op.Constant(value=onnx.helper.make_tensor(
        name='const_true', data_type=onnx.TensorProto.BOOL, dims=[], vals=[1]))
    
    # Return the dummy state unchanged as the first loop-carried dependency output.
    # The return signature is (condition, loop_carried_dependencies..., scan_outputs...).
    return condition_out, dummy_state_in, row_sum


# --- Step 4: Implement the Custom ONNX Translation Function ---
def onnx_row_sum_loop(input_tensor: OnnxFunction):
    """
    This function provides the custom ONNX implementation for our PyTorch op. [1]
    It now uses a dummy state to resolve type inference issues.
    """
    shape = op.Shape(input_tensor)
    
    gather_index = op.Constant(value=onnx.helper.make_tensor(
        name='const_zero', data_type=onnx.TensorProto.INT64, dims=[], vals=[0]))
    trip_count = op.Gather(shape, gather_index)
    
    # Create an initial dummy state to pass as a loop-carried dependency.
    # This gives `v_initial` a concrete type and resolves the ambiguity.
    dummy_initial_state = op.Constant(value=onnx.helper.make_tensor(
        name='dummy_state', data_type=onnx.TensorProto.INT64, dims=[], vals=[0]))
    
    # Call the Loop with the dummy state. It will now have two outputs:
    # 1. The final value of the dummy state.
    # 2. The concatenated scan output.
    final_dummy_state, scan_output_sums = op.Loop(
        trip_count, None, dummy_initial_state,
        body=row_sum_loop_body, new_inputs=[input_tensor])

    # We ignore `final_dummy_state` and use `scan_output_sums` as before.
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