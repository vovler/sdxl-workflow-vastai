import onnx
from onnx import TensorProto, helper
import onnx.shape_inference

def patch_concat_nodes(input_model_path: str, output_model_path: str):
    """
    Loads an ONNX model, finds Concat nodes with any mismatched input types,
    inserts Cast nodes to fix them, and re-runs shape inference to update the graph.
    """
    try:
        print(f"Loading model from {input_model_path}...")
        model = onnx.load(input_model_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    # Build the tensor type map
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

        valid_inputs = [(name, tensor_types.get(name)) for name in node.input if tensor_types.get(name) and tensor_types.get(name).tensor_type]
        if len(valid_inputs) < 2:
            new_nodes.append(node)
            continue

        first_input_name, first_type_proto = valid_inputs[0]
        target_type = first_type_proto.tensor_type.elem_type

        is_mismatched = any(
            proto.tensor_type.elem_type != target_type for _, proto in valid_inputs[1:]
        )

        if not is_mismatched:
            new_nodes.append(node)
            continue
            
        print(f"  - Found problematic Concat node: '{node.name}'")
        graph_modified = True
        target_type_name = TensorProto.DataType.Name(target_type)
        print(f"    - Target type for casting is {target_type_name} (from input '{first_input_name}')")

        new_concat_inputs = []
        for input_name in node.input:
            current_type_proto = tensor_types.get(input_name)
            if not current_type_proto or not current_type_proto.tensor_type:
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

    # Create a new graph with the patched nodes and original metadata
    new_graph = helper.make_graph(
        new_nodes,
        model.graph.name,
        model.graph.input,
        model.graph.output,
        model.graph.initializer
    )
    
    # Create a new model from the new graph, ensuring to copy the opset version
    new_model = helper.make_model(new_graph, producer_name='unet-patcher')
    new_model.opset_import.CopyFrom(model.opset_import)


    try:
        print("Re-running shape inference to update output types...")
        new_model = onnx.shape_inference.infer_shapes(new_model)
        print("Shape inference complete.")
    except Exception as e:
        print(f"An error occurred during shape inference after patching: {e}")
        return

    print(f"Saving patched model to {output_model_path} with external data...")
    onnx.save(
        new_model,
        output_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{output_model_path}.data",
    )

    try:
        print("Final model check from file path...")
        onnx.checker.check_model(output_model_path)
        print("Final model check passed.")
    except Exception as e:
        print(f"Final model check from file path failed: {e}")
        return

    print("Patching complete.")

if __name__ == "__main__":
    patch_concat_nodes("unet.onnx", "unet.patched.onnx") 