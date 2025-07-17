import onnx
from onnx import helper

def patch_vae_decoder_loop(input_onnx_path: str, output_onnx_path: str):
    """
    Patches a specific ONNX model structure where a Loop with a ScatterND
    update is used to simulate a batch-wise operation. This pattern is not
    supported by TensorRT.

    The script transforms the loop from using a loop-carried dependency (state variable)
    to using a scan output, which is the correct and supported way to model this.

    Args:
        input_onnx_path: Path to the original, problematic ONNX file.
        output_onnx_path: Path where the patched ONNX file will be saved.
    """
    print(f"Loading model from {input_onnx_path}...")
    model = onnx.load(input_onnx_path)
    graph = model.graph

    loop_node = None
    body_graph = None
    scatter_node = None

    # 1. Find the specific Loop node that contains a ScatterND
    print("Searching for the problematic Loop node...")
    for node in graph.node:
        if node.op_type == "Loop":
            # The 'body' of the loop is a subgraph stored in an attribute
            current_body_graph = next((attr.g for attr in node.attribute if attr.name == "body"), None)
            if not current_body_graph:
                continue
            
            # Check if this loop's body contains a ScatterND node
            if any(sub_node.op_type == "ScatterND" for sub_node in current_body_graph.node):
                loop_node = node
                body_graph = current_body_graph
                scatter_node = next(sub_node for sub_node in body_graph.node if sub_node.op_type == "ScatterND")
                print(f"Found Loop node '{loop_node.name}' containing ScatterND node '{scatter_node.name}'.")
                break

    if not loop_node:
        print("Could not find a Loop with a ScatterND to patch. Exiting.")
        return

    # 2. Identify the key tensors and nodes to be modified or removed
    
    # The large tensor being updated slice-by-slice (the state variable)
    state_variable_name_internal = scatter_node.input[0]
    
    # The small, decoded slice that we want to turn into a scan output
    update_tensor_name = scatter_node.input[2] # This is '/Reshape_output_0' in your graph
    
    # Find the node that produces this update tensor (it's the '/Expand' node)
    expand_node = next(n for n in body_graph.node if n.output[0] == update_tensor_name)
    
    # The *actual* raw output from the VAE decoder is the input to this Expand node
    new_scan_output_name = expand_node.input[0] # This is '/vae_decoder/decoder/conv_out/Conv_output_0'
    
    print(f"Identified state variable: '{state_variable_name_internal}'")
    print(f"Identified new scan output: '{new_scan_output_name}'")

    # 3. Patch the Loop's main inputs and the body's inputs
    
    # Find the index of the state variable in the body's input list
    state_var_body_index = -1
    for i, inp in enumerate(body_graph.input):
        if inp.name == state_variable_name_internal:
            state_var_body_index = i
            break
            
    # Remove the state variable from the loop's main input list
    main_loop_input_to_remove = loop_node.input[state_var_body_index]
    loop_node.input.remove(main_loop_input_to_remove)
    print(f"Removed '{main_loop_input_to_remove}' from the main Loop's inputs.")
    
    # Remove the state variable from the body's input list
    del body_graph.input[state_var_body_index]
    print(f"Removed '{state_variable_name_internal}' from the loop body's inputs.")

    # 4. Patch the Loop body's outputs to create a scan output
    
    # Find the body's output that comes from ScatterND
    scatter_output_name = scatter_node.output[0]
    for i, out in enumerate(body_graph.output):
        if out.name == scatter_output_name:
            # Change its name to point to our desired scan output tensor
            print(f"Changing body output '{out.name}' to '{new_scan_output_name}'.")
            out.name = new_scan_output_name
            # Clear type info, ONNX will infer it
            out.ClearField("type")
            break

    # 5. Remove the unnecessary nodes from the loop body
    nodes_to_remove = {scatter_node.name, expand_node.name}
    new_nodes = [n for n in body_graph.node if n.name not in nodes_to_remove]
    
    body_graph.ClearField("node")
    body_graph.node.extend(new_nodes)
    print(f"Removed the following nodes from the loop body: {nodes_to_remove}")

    # 6. Save the patched model
    print(f"Saving patched model to {output_onnx_path}...")
    onnx.save(model, output_onnx_path)
    print("Patching complete.")

# --- USAGE ---
if __name__ == "__main__":
    # Replace with the path to your ONNX file
    INPUT_MODEL_PATH = "simple_vae_decoder_optimized.onnx" 
    # The patched model will be saved here
    OUTPUT_MODEL_PATH = "simple_vae_decoder_patched_patched.onnx" 

    patch_vae_decoder_loop(INPUT_MODEL_PATH, OUTPUT_MODEL_PATH)