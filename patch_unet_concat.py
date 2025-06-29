import onnx
from onnx import TensorProto, helper

def patch_concat_nodes(input_model_path: str, output_model_path: str):
    """
    Loads an ONNX model, finds Concat nodes with any mismatched input types using
    the same logic as inspect_onnx.py, and inserts Cast nodes to fix them.
    """
    try:
        print(f"Loading model from {input_model_path}...")
        model = onnx.load(input_model_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    # Build the tensor type map exactly as in the working inspect_onnx.py script
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

        # Use the exact detection logic from inspect_onnx.py
        input_type_protos = [tensor_types.get(name) for name in node.input]
        
        # Filter out inputs where type info is missing
        valid_inputs = [(name, proto) for name, proto in zip(node.input, input_type_protos) if proto and proto.tensor_type]
        if len(valid_inputs) < 2:
            new_nodes.append(node)
            continue

        first_input_name, first_type_proto = valid_inputs[0]
        target_type = first_type_proto.tensor_type.elem_type

        # Check for any mismatch against the first input's type
        is_mismatched = any(
            proto.tensor_type.elem_type != target_type for _, proto in valid_inputs[1:]
        )

        if not is_mismatched:
            new_nodes.append(node)
            continue
            
        # If we get here, a mismatch was found.
        print(f"  - Found problematic Concat node: '{node.name}'")
        graph_modified = True
        target_type_name = TensorProto.DataType.Name(target_type)
        print(f"    - Target type for casting is {target_type_name} (from input '{first_input_name}')")

        new_concat_inputs = []
        for input_name, current_type_proto in zip(node.input, input_type_protos):
            # Also check if type info was missing for this input
            if not current_type_proto or not current_type_proto.tensor_type:
                print(f"    - WARNING: Could not determine type for input '{input_name}'. It will not be cast.")
                new_concat_inputs.append(input_name)
                continue

            current_type = current_type_proto.tensor_type.elem_type
            if current_type != target_type:
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
        
        # Re-create the concat node
        new_concat_node = helper.make_node(
            "Concat",
            inputs=new_concat_inputs,
            outputs=node.output,
            name=node.name,
            axis=node.attribute[0].i
        )
        new_nodes.append(new_concat_node)

    if not graph_modified:
        print("No mismatched Concat nodes were found.")
        return

    # Rebuild the graph
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