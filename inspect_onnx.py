import onnx
from onnx import TensorProto
import onnx.shape_inference

def get_type_name(type_proto):
    """Returns a string representation of a type and its raw enum value."""
    if type_proto and type_proto.tensor_type.elem_type in TensorProto.DataType.DESCRIPTOR.values:
        elem_type = type_proto.tensor_type.elem_type
        return f"{TensorProto.DataType.Name(elem_type)} (enum: {elem_type})"
    return "UNKNOWN"

def find_mismatched_types(model_path: str):
    """
    Loads an ONNX model, runs type inference, and checks for nodes with mismatched input data types.
    """
    try:
        print("Loading ONNX model and running type inference...")
        model = onnx.load(model_path)
        # Run shape and type inference to populate value_info for all tensors.
        model = onnx.shape_inference.infer_shapes(model)
        print("Type inference complete.")

    except Exception as e:
        print(f"Error loading or processing ONNX model at {model_path}: {e}")
        return

    # Now that we've run inference, value_info should be much more complete.
    tensor_types = {vi.name: vi.type for vi in model.graph.value_info}
    for i in model.graph.input:
        tensor_types[i.name] = i.type
    for init in model.graph.initializer:
        tensor_types[init.name] = onnx.helper.make_tensor_type_proto(init.data_type, init.dims)

    print(f"--- Checking model for type mismatches: {model_path} ---")
    found_mismatch = False
    
    for node in model.graph.node:
        if node.op_type != "Concat":
            continue

        input_names = node.input
        if len(input_names) < 2:
            continue

        first_input_name = input_names[0]
        if first_input_name not in tensor_types:
            continue 
        
        first_type_proto = tensor_types[first_input_name]
        first_type = first_type_proto.tensor_type.elem_type
        
        is_mismatched_node = False
        for input_name in input_names:
            if input_name not in tensor_types:
                continue
            current_type = tensor_types[input_name].tensor_type.elem_type
            if current_type != first_type:
                is_mismatched_node = True
                break
        
        if is_mismatched_node:
            found_mismatch = True
            print(f"\n[MISMATCH FOUND] in node '{node.name}' (Type: {node.op_type})")
            for i, name in enumerate(input_names):
                input_type = get_type_name(tensor_types.get(name))
                print(f"  - Input {i} ('{name}') has type: {input_type}")
            print("-" * 20)

    if not found_mismatch:
        print("\nNo Concat nodes with mismatched input types were found.")
    
    print(f"--- Check complete ---")


if __name__ == "__main__":
    # Assuming the script is run from the root of the sdxl-workflow project
    find_mismatched_types("unet.onnx") 