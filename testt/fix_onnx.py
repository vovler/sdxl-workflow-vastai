import onnx
from onnx import helper, shape_inference

def patch_loop_scatter_to_scan_output(input_onnx_path: str, output_onnx_path: str):
    """
    Generalizes the patching of ONNX models for TensorRT compatibility.

    This script finds any ONNX Loop operator that uses a state variable 
    (a loop-carried dependency) as the target for a ScatterND update. This
    pattern is inefficient and often unsupported in TensorRT.

    The script transforms the loop by:
    1. Removing the large state variable from the loop's inputs/outputs.
    2. Making the 'updates' tensor (the small slice) a proper 'scan_output'.
    3. Removing the now-redundant ScatterND node from the loop's body.

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
    nodes_to_process = []

    # --- Step 1: Find all Loop nodes that match the problematic pattern ---
    print("Searching for Loop nodes with the ScatterND state-update pattern...")
    for node in graph.node:
        if node.op_type == "Loop":
            body_graph = next((attr.g for attr in node.attribute if attr.name == "body"), None)
            if not body_graph:
                continue

            for sub_node in body_graph.node:
                if sub_node.op_type == "ScatterND":
                    # This is the pattern: the tensor being updated by ScatterND...
                    scatter_data_input = sub_node.input[0]
                    # ...is also a state variable (a loop-carried input to the body).
                    # (Note: Loop inputs 0 and 1 are special: 'iteration_num' and 'condition')
                    body_state_inputs = [inp.name for inp in body_graph.input[2:]]
                    
                    if scatter_data_input in body_state_inputs:
                        nodes_to_process.append((node, sub_node))
                        print(f"  [+] Found candidate for patching: Loop '{node.name}' contains ScatterND '{sub_node.name}' updating state variable '{scatter_data_input}'.")

    if not nodes_to_process:
        print("No loops matching the ScatterND pattern were found. No patching is needed.")
        # Optionally save a copy anyway if you want the flow to be consistent
        # onnx.save(model, output_onnx_path)
        return

    # --- Step 2: Process each identified loop ---
    for loop_node, scatter_node in nodes_to_process:
        print(f"\n--- Patching Loop '{loop_node.name}' ---")
        body_graph = next(attr.g for attr in loop_node.attribute if attr.name == "body")

        # Identify key tensor names
        state_var_internal_name = scatter_node.input[0]
        new_scan_output_name = scatter_node.input[2] # The 'updates' tensor IS the new scan output
        scatter_output_name = scatter_node.output[0]

        print(f"  State variable to remove: '{state_var_internal_name}'")
        print(f"  New scan output tensor: '{new_scan_output_name}'")

        # --- Step 3: Rewire the loop's main inputs and the body's inputs ---
        
        # Find the index of the state variable in the body's input list
        state_var_body_index = -1
        for i, inp in enumerate(body_graph.input):
            if inp.name == state_var_internal_name:
                state_var_body_index = i
                break
        
        if state_var_body_index == -1:
            print(f"  [!] Warning: Could not find state variable '{state_var_internal_name}' in body inputs. Skipping this loop.")
            continue
            
        # The corresponding input on the main Loop node is at the same index
        main_loop_input_to_remove = loop_node.input[state_var_body_index]
        loop_node.input.remove(main_loop_input_to_remove)
        print(f"  Removed '{main_loop_input_to_remove}' from the main Loop's inputs.")

        del body_graph.input[state_var_body_index]
        print(f"  Removed '{state_var_internal_name}' from the loop body's inputs.")

        # --- Step 4: Rewire the loop body's outputs ---
        
        # The output of ScatterND was previously a state output. Now, we change it
        # to be our new scan output by renaming it to the 'updates' tensor name.
        found_output = False
        for out in body_graph.output:
            if out.name == scatter_output_name:
                print(f"  Changing body output '{out.name}' to become a scan output pointing to '{new_scan_output_name}'.")
                out.name = new_scan_output_name
                # Clear the type info. It will be re-inferred later.
                out.ClearField("type")
                found_output = True
                break
        
        if not found_output:
            print(f"  [!] Warning: Could not find the corresponding state output '{scatter_output_name}'. Skipping this loop.")
            continue

        # --- Step 5: Remove the now-redundant ScatterND node ---
        new_nodes = [n for n in body_graph.node if n.name != scatter_node.name]
        body_graph.ClearField("node")
        body_graph.node.extend(new_nodes)
        print(f"  Removed ScatterND node '{scatter_node.name}' from the loop body.")

    # --- Step 6: Clean, check, and save the final model ---
    print("\nFinalizing model...")
    try:
        # It's good practice to run shape inference after graph modifications
        model = shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        print("  Model check passed.")
        onnx.save(model, output_onnx_path)
        print(f"Successfully saved patched model to {output_onnx_path}")
    except Exception as e:
        print(f"An error occurred during final model validation or saving: {e}")
        print("Attempting to save the model without shape inference for manual inspection...")
        # Reload the un-inferred model and save
        #model_to_save = onnx.load(input_onnx_path)
        # Re-apply the changes (this is simplified, a better approach would be to work on a copy)
        # For this script, we'll just save the in-memory model that failed inference
        onnx.save(model, output_onnx_path.replace(".onnx", "_failed_inference.onnx"))

if __name__ == "__main__":
    # --- USAGE ---
    # Point this to your original ONNX file
    INPUT_MODEL_PATH = "onnx/simple_vae_decoder_optimized.onnx"
    # The correctly patched model will be saved here
    OUTPUT_MODEL_PATH = "onnx/simple_vae_decoder_patched_scan_output.onnx" 

    patch_loop_scatter_to_scan_output(INPUT_MODEL_PATH, OUTPUT_MODEL_PATH)