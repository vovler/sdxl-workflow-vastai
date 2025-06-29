import onnx
from onnx import TensorProto, helper
import onnx.shape_inference
from collections import Counter

def patch_concat_nodes(input_model_path: str, output_model_path: str):
    """
    Loads an ONNX model, finds Concat nodes with any mismatched input types,
    and inserts Cast nodes to conform all inputs to the most common type.
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

    tensor_types = {vi.name: vi.type for vi in model.graph.value_info}
    for i in model.graph.input:
        tensor_types[i.name] = i.type
    for init in model.graph.initializer:
        tensor_types[init.name] = helper.make_tensor_type_proto(init.data_type, init.dims)

    graph_modified = False
    new_nodes = []
    
    print("Scanning for mismatched Concat nodes to patch...")
    for node in model.graph.node:
        if node.op_type != "Concat":
            new_nodes.append(node)
            continue

        # Get the element type for each input, filtering out any that can't be found
        input_elem_types = []
        for name in node.input:
            tt = tensor_types.get(name)
            if tt and tt.tensor_type and tt.tensor_type.elem_type:
                input_elem_types.append(tt.tensor_type.elem_type)
            else:
                input_elem_types.append(None) # Keep placeholder for index mapping

        # Check if there is more than one unique data type among the valid inputs
        valid_elem_types = [t for t in input_elem_types if t is not None]
        if len(set(valid_elem_types)) <= 1:
            new_nodes.append(node)
            continue

        # Mismatch found!
        print(f"  - Found problematic Concat node: '{node.name}'")
        graph_modified = True
        
        # Determine the target type (the most common one)
        target_type = Counter(valid_elem_types).most_common(1)[0][0]
        target_type_name = TensorProto.DataType.Name(target_type)
        print(f"    - Target type for casting is {target_type_name}")

        new_concat_inputs = []
        for i, input_name in enumerate(node.input):
            current_type = input_elem_types[i]
            if current_type is not None and current_type != target_type:
                # This is an input that needs to be cast.
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