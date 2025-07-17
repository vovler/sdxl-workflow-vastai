import onnx
from onnx import helper, shape_inference

def patch_loop_scatter_to_scan_output(input_onnx_path: str, output_onnx_path: str):
    """
    Generalizes the patching of ONNX models for TensorRT compatibility.

    This script finds any ONNX Loop operator that uses a state variable 
    (a loop-carried dependency) as the target for a ScatterND update. This
    pattern is inefficient and often unsupported in TensorRT.

    The script transforms the loop by:
    1. Finding all nodes that use the large state variable for metadata (like Shape ops).
    2. Rerouting those nodes to use the smaller 'updates' tensor instead.
    3. Removing the large state variable from the loop's inputs/outputs.
    4. Making the 'updates' tensor a proper 'scan_output'.
    5. Removing the now-redundant ScatterND node from the loop's body.

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

    # --- Step 1: Find all Loop nodes that match the problematic pattern ---
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

    # --- Step 2: Process each identified loop ---
    for loop_node, scatter_node in loops_to_process:
        print(f"\n--- Patching Loop '{loop_node.name}' ---")
        body_graph = next(attr.g for attr in loop_node.attribute if attr.name == "body")

        state_var_internal_name = scatter_node.input[0]
        slice_update_tensor_name = scatter_node.input[2] # The 'updates' tensor IS the new scan output
        scatter_output_name = scatter_node.output[0]

        print(f"  State variable to remove: '{state_var_internal_name}'")
        print(f"  New scan output tensor will be: '{slice_update_tensor_name}'")

        # --- Step 3: Reroute metadata dependencies ---
        # Find any node (like a Shape op) that depends on the state variable and isn't the ScatterND itself.
        # Reroute it to depend on the slice update tensor instead.
        print("  Checking for other nodes that depend on the state variable...")
        for body_node in body_graph.node:
            if body_node.name == scatter_node.name:
                continue
            
            for i, input_name in enumerate(body_node.input):
                if input_name == state_var_internal_name:
                    print(f"    Rerouting input for node '{body_node.name}': '{input_name}' -> '{slice_update_tensor_name}'")
                    body_node.input[i] = slice_update_tensor_name

        # --- Step 4: Rewire the loop's main inputs and the body's inputs ---
        state_var_body_index = next((i for i, inp in enumerate(body_graph.input) if inp.name == state_var_internal_name), -1)
        
        if state_var_body_index == -1:
            print(f"  [!] Warning: Could not find state variable '{state_var_internal_name}' in body inputs. Skipping this loop.")
            continue
            
        main_loop_input_to_remove = loop_node.input[state_var_body_index]
        loop_node.input.remove(main_loop_input_to_remove)
        print(f"  Removed '{main_loop_input_to_remove}' from the main Loop's inputs.")

        del body_graph.input[state_var_body_index]
        print(f"  Removed '{state_var_internal_name}' from the loop body's inputs.")

        # --- Step 5: Rewire the loop body's outputs ---
        found_output = False
        for out in body_graph.output:
            if out.name == scatter_output_name:
                print(f"  Changing body output '{out.name}' to become a scan output pointing to '{slice_update_tensor_name}'.")
                out.name = slice_update_tensor_name
                out.ClearField("type")
                found_output = True
                break
        
        if not found_output:
            print(f"  [!] Warning: Could not find the corresponding state output '{scatter_output_name}'. Skipping this loop.")
            continue

        # --- Step 6: Remove the now-redundant ScatterND node ---
        new_nodes = [n for n in body_graph.node if n.name != scatter_node.name]
        body_graph.ClearField("node")
        body_graph.node.extend(new_nodes)
        print(f"  Removed ScatterND node '{scatter_node.name}' from the loop body.")

    # --- Step 7: Clean, check, and save the final model ---
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
        onnx.save(model, output_onnx_path.replace(".onnx", "_failed_inference.onnx"))

if __name__ == "__main__":
    INPUT_MODEL_PATH = "onnx/simple_vae_decoder_optimized.onnx"
    OUTPUT_MODEL_PATH = "onnx/simple_vae_decoder_patched_scan_output.onnx" 

    patch_loop_scatter_to_scan_output(INPUT_MODEL_PATH, OUTPUT_MODEL_PATH)