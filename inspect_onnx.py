import onnx
from onnx import TensorProto

def get_type_name(type_proto):
    """Returns a string representation of a type."""
    if type_proto.tensor_type.elem_type in TensorProto.DataType.DESCRIPTOR.values:
        return TensorProto.DataType.Name(type_proto.tensor_type.elem_type)
    return "UNKNOWN"

def find_mismatched_types(model_path: str):
    """
    Loads an ONNX model and checks for nodes with mismatched input data types.
    Specifically targets Concat nodes as per the error message.
    """
    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"Error loading ONNX model at {model_path}: {e}")
        return

    # Create a map of all tensor names to their data types for easy lookup.
    # This includes initializers (weights) and value_info (activations).
    value_info = {vi.name: vi.type for vi in model.graph.value_info}
    inputs = {i.name: i.type for i in model.graph.input}
    initializers = {init.name: init for init in model.graph.initializer}

    tensor_types = {**value_info, **inputs}

    # For initializers, we need to get their type from the tensor itself
    for name, init in initializers.items():
        # Create a TypeProto for the initializer
        type_proto = onnx.helper.make_tensor_type_proto(init.data_type, init.dims)
        tensor_types[name] = type_proto

    print(f"--- Checking model: {model_path} ---")
    found_mismatch = False
    
    for node in model.graph.node:
        if node.op_type != "Concat":
            continue

        input_types = []
        for input_name in node.input:
            if input_name in tensor_types:
                input_types.append(tensor_types[input_name])
            else:
                input_types.append(None)

        if not input_types or len(input_types) < 2:
            continue

        first_type_proto = input_types[0]
        if first_type_proto is None:
            continue
            
        first_type = first_type_proto.tensor_type.elem_type
        
        for i, type_proto in enumerate(input_types[1:], 1):
            if type_proto is None:
                continue
                
            current_type = type_proto.tensor_type.elem_type
            if current_type != first_type:
                found_mismatch = True
                print(f"\n[MISMATCH FOUND] in node '{node.name}' (Type: {node.op_type})")
                print(f"  Input {node.input[0]} has type {current_type} {get_type_name(input_types[0])}")
                print(f"  Input {node.input[i]} has type {first_type} {get_type_name(type_proto)}")
                print("-" * 20)


    if not found_mismatch:
        print("\nNo Concat nodes with mismatched input types were found.")
    
    print(f"--- Check complete ---")


if __name__ == "__main__":
    # Assuming the script is run from the root of the sdxl-workflow project
    find_mismatched_types("unet.onnx") 