import torch
import onnxscript
from onnxscript import script
from onnxscript.values import Opset

# Ensure you have the necessary libraries installed:
# pip install torch>=2.7.0 onnxscript onnx onnxruntime

# --- Step 1: Create a Custom PyTorch Operator ---
# We define a custom operator to encapsulate our logic. This gives us a specific
# target in the PyTorch graph to replace with our custom ONNX implementation.
# The `register_fake` implementation is crucial for compatibility with
# torch.export and Dynamo, as it allows PyTorch to understand the
# metadata (shape, dtype) of the operator's output without running it. [2]

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
    without performing actual computation.
    """
    output_shape = (input_tensor.shape[0], 1)
    return torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)


# --- Step 2: Define a PyTorch Model Using the Custom Operator ---
# This model incorporates our custom operator, which will be targeted for
# custom translation during the ONNX export process.

class RowSumModel(torch.nn.Module):
    def forward(self, x):
        # When this model is exported, the call to 'row_sum_loop' will be
        # intercepted and translated using our custom ONNX implementation.
        return row_sum_loop(x)


# --- Step 3: Define the ONNX Loop Body using onnxscript ---
# The ONNX `Loop` operator requires a 'body' graph that is executed for each
# iteration. This function defines that graph. The body graph must return a
# termination condition, any updated loop-carried dependencies, and any scan outputs. [0]

# Use the target ONNX opset version 20.
op = Opset('', 20)

@script()
def row_sum_loop_body(iteration_num, condition_in, input_tensor):
    """
    Defines the graph for a single iteration of the ONNX Loop. [0]

    Args:
        iteration_num: The current loop iteration number (e.g., 0, 1, 2...). [0]
        condition_in: The loop termination condition from the previous iteration. [0]
        input_tensor: The full input tensor, captured from the outer scope. [0]
    """
    # Extract the current row from the input tensor using the iteration number.
    row = op.Gather(input_tensor, iteration_num, axis=0)

    # Calculate the sum of the row's elements. This will be our "scan output".
    # A scan output is a value from each iteration that gets collected and
    # concatenated together. [0]
    row_sum = op.ReduceSum(row, keepdims=False)

    # The body must always return a boolean condition value. [0]
    # **FIX:** Create the constant using a tensor value.
    condition_out = op.Constant(value=torch.tensor(True, dtype=torch.bool))

    # The body's return signature is (condition, loop_carried_dependencies..., scan_outputs...). [0]
    # We have no loop-carried dependencies, but we have one scan output (row_sum).
    return condition_out, row_sum


# --- Step 4: Implement the Custom ONNX Translation Function ---
# This function maps our custom PyTorch operator to an ONNX `Loop` operator.
# It is the bridge between the PyTorch graph and the desired ONNX representation.

def onnx_row_sum_loop(g, input_tensor):
    """
    This function provides the custom ONNX implementation for our PyTorch op. [1]
    It translates the operation into an ONNX Loop.
    """
    # The trip count 'M' is the number of rows in the input tensor (dimension 0). [0]
    shape = op.Shape(input_tensor)
    # **FIX:** Create the constant index using a tensor value. ONNX indices should be int64.
    gather_index = op.Constant(value=torch.tensor(0, dtype=torch.int64))
    trip_count = op.Gather(shape, gather_index)

    # The Loop operator signature is Loop(M, cond, v_initial, body=...). [0]
    # We provide the trip count 'M'.
    # We do not use a termination condition 'cond', so we pass 'None'.
    # We have no loop-carried dependencies 'v_initial', so onnxscript handles this.
    # The `new_inputs` argument makes `input_tensor` available inside the loop body.
    loop_node = op.Loop(trip_count, None, body=row_sum_loop_body, new_inputs=[input_tensor])

    # The Loop operator returns final loop-carried dependency values, then scan outputs. [0]
    # Since we have no dependencies, the first output is our concatenated scan output.
    # This gives a 1D tensor of shape (X,).
    scan_output_sums = loop_node

    # Reshape the output from (X,) to (X, 1) to match the model's expected output shape.
    final_output = op.Unsqueeze(scan_output_sums, axes=[1])

    return final_output


# --- Step 5: Export the Model and Verify the Output ---
if __name__ == "__main__":
    # Instantiate the model and create a dummy input tensor of shape (X, 5).
    model = RowSumModel().eval()
    batch_size = 4
    dummy_input = torch.randint(0, 10, (batch_size, 5), dtype=torch.float32)

    print("--- Starting ONNX Export with Dynamo ---")
    
    # Export the model using `torch.onnx.export`.
    # `dynamo=True` enables the new Dynamo-based exporter.
    # `custom_translation_table` maps our PyTorch op to our ONNX translation function. [1]
    onnx_program = torch.onnx.export(
        model,
        (dummy_input,),
        dynamo=True,
        opset_version=20, # Specify the target opset
        custom_translation_table={
            torch.ops.mylibrary.row_sum_loop.default: onnx_row_sum_loop,
        },
        report=True,
    )

    print("\n--- ONNX Model Graph ---")
    print("The graph contains a 'Loop' operator as defined in our custom translation.")
    print(onnx_program.model)

    # --- Verification ---
    print("\n--- Verifying Outputs ---")
    
    # 1. Run the original PyTorch model
    pytorch_output = model(dummy_input)
    print(f"Input Tensor (shape {dummy_input.shape}):\n{dummy_input}")
    print(f"\nPyTorch Model Output:\n{pytorch_output}")

    # 2. Run the exported ONNX model
    # The onnx_program object is directly callable for inference.
    onnx_output = onnx_program(dummy_input)[0]
    print(f"\nONNX Model Output:\n{onnx_output}")

    # 3. Compare the results
    torch.testing.assert_close(pytorch_output, onnx_output)
    print("\n✅ Verification successful: The outputs of the PyTorch and ONNX models are identical.")