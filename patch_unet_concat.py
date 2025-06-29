import onnx
from onnx import TensorProto, helper
import onnx.shape_inference

def patch_concat_nodes(input_model_path: str, output_model_path: str):
    """
    Loads an ONNX model, finds Concat nodes with mixed float16/float32 inputs,
    and inserts a Cast node to convert the float32 input to float16.
    """
    try:
        print(f"Loading model from {input_model_path}...")
        model = onnx.load(input_model_path)
        print("Running shape and type inference to populate tensor types...")
        # It's crucial to run shape inference first to get type info for all tensors.
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

        input_dtypes = [tensor_types.get(name, None) for name in node.input]
        elem_types = [t.tensor_type.elem_type if t else None for t in input_dtypes]

        if TensorProto.FLOAT in elem_types and TensorProto.FLOAT16 in elem_types:
            print(f"  - Found problematic Concat node: '{node.name}'")
            graph_modified = True
            
            new_concat_inputs = []
            for i, input_name in enumerate(node.input):
                if elem_types[i] == TensorProto.FLOAT:
                    # This is the one we need to cast.
                    print(f"    - Input '{input_name}' is float32. Creating a Cast node.")
                    
                    casted_tensor_name = f"{input_name}_casted_to_fp16"
                    cast_node = helper.make_node(
                        "Cast",
                        inputs=[input_name],
                        outputs=[casted_tensor_name],
                        to=TensorProto.FLOAT16,
                        name=f"Cast_{node.name}_{input_name}"
                    )
                    new_nodes.append(cast_node)
                    new_concat_inputs.append(casted_tensor_name)
                else:
                    new_concat_inputs.append(input_name)
            
            # Re-create the concat node with the new inputs
            new_concat_node = helper.make_node(
                "Concat",
                inputs=new_concat_inputs,
                outputs=node.output,
                name=node.name,
                axis=node.attribute[0].i
            )
            new_nodes.append(new_concat_node)
        else:
            new_nodes.append(node)

    if not graph_modified:
        print("No mismatched Concat nodes were found. The model appears to be correct.")
        return

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