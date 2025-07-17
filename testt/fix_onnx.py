import onnx
from onnx import helper, shape_inference

def patch_loop_scatter_to_scan_output(input_onnx_path: str, output_onnx_path: str):
    """
    Generalizes the patching of ONNX models for TensorRT compatibility.

    This script finds any ONNX Loop operator that uses a state variable 
    (a loop-carried dependency) as the target for a ScatterND update. This
    pattern is inefficient and often unsupported in TensorRT.

    The script transforms the loop by:
    1. Identifying the 'updates' tensor for the ScatterND.
    2. Tracing back to find the node producing these updates (e.g., an Expand op)
       and identifying its input as the TRUE scan output.
    3. Rerouting any metadata nodes (like Shape ops) that depended on the old 
       state variable to depend on this new true scan output tensor instead.
    4. Removing the large state variable from the loop's inputs/outputs.
    5. Making the true 'updates' tensor a proper 'scan_output'.
    6. Removing the now-redundant ScatterND and its producer node from the loop's body.

    This results in a much more efficient and compliant model for TensorRT.

    Args:
        input_onnx_path: Path to the original ONNX file.
        output_onnx_path: Path where the patched ONNX file will be saved.
    """
    print(f"Loading model from {input_onnx_path}...")
    try:
        model = onnx.load(input_onnx_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    graph = model.graph
    loops_to_process = []

    print("Searching for Loop nodes with the ScatterND state-update pattern...")
    for node in graph.node:
        if node.op_type == "Loop":
            body_graph = next((attr.g for attr in node.attribute if attr.name == "body"), None)
            if not body_graph:
                continue

            for sub_node in body_graph.node:
                if sub_node.op_type == "ScatterND":
                    scatter_data_input = sub_node.input[0]
                    body_state_inputs = [inp.name for inp in body_graph.input[2:]]
                    
                    if scatter_data_input in body_state_inputs:
                        loops_to_process.append((node, sub_node))
                        print(f"  [+] Found candidate for patching: Loop '{node.name}' contains ScatterND '{sub_node.name}' updating state variable '{scatter_data_input}'.")

    if not loops_to_process:
        print("No loops matching the ScatterND pattern were found. No patching is needed.")
        return

    for loop_node, scatter_node in loops_to_process:
        print(f"\n--- Patching Loop '{loop_node.name}' ---")
        body_graph = next(attr.g for attr in loop_node.attribute if attr.name == "body")

        state_var_internal_name = scatter_node.input[0]
        scatter_updates_name = scatter_node.input[2]
        scatter_output_name = scatter_node.output[0]

        # Find the node that produces the 'updates' tensor for the ScatterND
        producer_of_updates = next((n for n in body_graph.node if scatter_updates_name in n.output), None)
        if not producer_of_updates:
            print(f"  [!] ERROR: Could not find the producer of the ScatterND updates tensor '{scatter_updates_name}'. Aborting patch.")
            continue
        
        # The TRUE scan output is the input to this producer node.
        new_scan_output_name = producer_of_updates.input[0]

        print(f"  State variable to remove: '{state_var_internal_name}'")
        print(f"  Identified producer of updates: '{producer_of_updates.name}' (Op: {producer_of_updates.op_type})")
        print(f"  New scan output tensor will be: '{new_scan_output_name}'")

        print("  Checking for other nodes that depend on the state variable...")
        for body_node in body_graph.node:
            if body_node.name == scatter_node.name:
                continue
            for i, input_name in enumerate(body_node.input):
                if input_name == state_var_internal_name:
                    # Reroute this dependency to the *true* scan output tensor
                    print(f"    Rerouting input for node '{body_node.name}': '{input_name}' -> '{new_scan_output_name}'")
                    body_node.input[i] = new_scan_output_name

        state_var_body_index = next((i for i, inp in enumerate(body_graph.input) if inp.name == state_var_internal_name), -1)
        
        if state_var_body_index == -1:
            print(f"  [!] Warning: Could not find state variable '{state_var_internal_name}' in body inputs. Skipping.")
            continue
            
        main_loop_input_to_remove = loop_node.input[state_var_body_index]
        loop_node.input.remove(main_loop_input_to_remove)
        print(f"  Removed '{main_loop_input_to_remove}' from the main Loop's inputs.")

        del body_graph.input[state_var_body_index]
        print(f"  Removed '{state_var_internal_name}' from the loop body's inputs.")

        found_output = False
        for out in body_graph.output:
            if out.name == scatter_output_name:
                print(f"  Changing body output '{out.name}' to become a scan output pointing to '{new_scan_output_name}'.")
                out.name = new_scan_output_name
                out.ClearField("type")
                found_output = True
                break
        
        if not found_output:
            print(f"  [!] Warning: Could not find the corresponding state output '{scatter_output_name}'. Skipping.")
            continue

        nodes_to_remove = {scatter_node.name, producer_of_updates.name}
        new_nodes = [n for n in body_graph.node if n.name not in nodes_to_remove]
        body_graph.ClearField("node")
        body_graph.node.extend(new_nodes)
        print(f"  Removed the following redundant nodes from the loop body: {nodes_to_remove}")

    print("\nFinalizing model...")
    try:
        model = shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        print("  Model check passed.")
        onnx.save(model, output_onnx_path)
        print(f"Successfully saved patched model to {output_onnx_path}")
    except Exception as e:
        print(f"An error occurred during final model validation or saving: {e}")
        print(f"Attempting to save the model without shape inference for manual inspection to: {output_onnx_path.replace('.onnx', '_failed_inference.onnx')}")
        onnx.save(model, output_onnx_path.replace('.onnx', '_failed_inference.onnx'))

if __name__ == "__main__":
    INPUT_MODEL_PATH = "onnx/simple_vae_decoder_optimized.onnx"
    OUTPUT_MODEL_PATH = "onnx/simple_vae_decoder_patched_scan_output.onnx" 

    patch_loop_scatter_to_scan_output(INPUT_MODEL_PATH, OUTPUT_MODEL_PATH)