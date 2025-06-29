import onnx
from onnx import TensorProto, helper
import onnx.shape_inference

def patch_concat_nodes(input_model_path: str, output_model_path: str):
    """
    Loads an ONNX model, finds Concat nodes with any mismatched input types,
    and inserts Cast nodes to conform all inputs to the type of the first input.
    """
    try:
        print(f"Loading model from {input_model_path}...")
        model = onnx.load(input_model_path)
        print("Running shape and type inference to populate tensor types...")
        model = onnx.shape_inference.infer_shapes(model)
        print("Inference complete.")
    except Exception as e:
        print(f"Error loading or processing ONNX model: {e}")
        return

    # Build a comprehensive map of tensor names to their types
    tensor_types = {vi.name: vi.type for vi in model.graph.value_info}
    for i in model.graph.input:
        tensor_types[i.name] = i.type
    for init in model.graph.initializer:
        tensor_types[init.name] = helper.make_tensor_type_proto(init.data_type, init.dims)

    graph_modified = False
    new_nodes = []
    
    print("Scanning for mismatched Concat nodes to patch...")
    for node in model.graph.node:
        if node.op_type != "Concat" or len(node.input) < 2:
            new_nodes.append(node)
            continue

        # Get the data type of the first input. This will be our target type.
        first_input_type_proto = tensor_types.get(node.input[0])
        if not first_input_type_proto or not first_input_type_proto.tensor_type:
            # Cannot determine the type, cannot patch.
            new_nodes.append(node)
            continue
        
        target_type = first_input_type_proto.tensor_type.elem_type
        
        # Check if any other input has a different type
        is_mismatched = False
        for input_name in node.input[1:]:
            current_type_proto = tensor_types.get(input_name)
            if current_type_proto and current_type_proto.tensor_type and current_type_proto.tensor_type.elem_type != target_type:
                is_mismatched = True
                break

        if not is_mismatched:
            new_nodes.append(node)
            continue

        # If we get here, a mismatch was found.
        print(f"  - Found problematic Concat node: '{node.name}'")
        graph_modified = True
        target_type_name = TensorProto.DataType.Name(target_type)
        print(f"    - Target type for casting is {target_type_name} (from input '{node.input[0]}')")

        new_concat_inputs = []
        for input_name in node.input:
            current_type_proto = tensor_types.get(input_name)
            current_type = current_type_proto.tensor_type.elem_type if current_type_proto and current_type_proto.tensor_type else None

            if current_type is not None and current_type != target_type:
                # This input needs to be cast.
                current_type_name = TensorProto.DataType.Name(current_type)
                print(f"    - Input '{input_name}' is {current_type_name}. Creating a Cast node.")
                
                casted_tensor_name = f"{input_name}_casted_to_{target_type_name.lower()}"
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[input_name],
                    outputs=[casted_tensor_name],
                    to=target_type,
                    name=f"Cast_{node.name}_{input_name}"
                )
                new_nodes.append(cast_node)
                new_concat_inputs.append(casted_tensor_name)
            else:
                new_concat_inputs.append(input_name)
        
        # Re-create the concat node with the new (potentially casted) inputs
        new_concat_node = helper.make_node(
            "Concat",
            inputs=new_concat_inputs,
            outputs=node.output,
            name=node.name,
            axis=node.attribute[0].i
        )
        new_nodes.append(new_concat_node)

    if not graph_modified:
        print("No mismatched Concat nodes were found. The model appears to be correct.")
        return

    # Rebuild the graph with the new list of nodes
    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)

    try:
        print("Checking patched model for correctness...")
        onnx.checker.check_model(model)
        print("Model check passed.")
    except Exception as e:
        print(f"Model check failed after patching: {e}")
        return

    print(f"Saving patched model to {output_model_path}...")
    onnx.save(model, output_model_path)
    print("Patching complete.")

if __name__ == "__main__":
    patch_concat_nodes("unet.onnx", "unet.patched.onnx") 