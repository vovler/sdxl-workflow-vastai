import onnx
from onnx import helper, shape_inference

def topologically_sort_nodes(nodes: list) -> list:
    """Performs a topological sort on a list of ONNX nodes."""
    node_map = {node.name: node for node in nodes}
    tensor_producer = {out: node.name for node in nodes for out in node.output}
    in_degree = {node.name: 0 for node in nodes}
    dependencies = {node.name: [] for node in nodes}

    for node in nodes:
        for input_name in node.input:
            if input_name in tensor_producer:
                producer_name = tensor_producer[input_name]
                if producer_name in dependencies:
                    dependencies[producer_name].append(node.name)
                    in_degree[node.name] += 1

    queue = [name for name, degree in in_degree.items() if degree == 0]
    sorted_names = []
    while queue:
        name = queue.pop(0)
        sorted_names.append(name)
        for dependent in dependencies.get(name, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(sorted_names) != len(nodes):
        raise RuntimeError("Cycle detected or disconnected components in the graph, topological sort failed.")
        
    return [node_map[name] for name in sorted_names]

def patch_loop_scatter_to_scan_output(input_onnx_path: str, output_onnx_path: str):
    """
    Generalizes the patching of ONNX models for TensorRT compatibility by converting
    inefficient ScatterND-based state updates into efficient scan outputs.
    """
    print(f"Loading model from {input_onnx_path}...")
    model = onnx.load(input_onnx_path)
    graph = model.graph
    
    loop_node = next((n for n in graph.node if n.op_type == 'Loop'), None)
    if not loop_node:
        print("No Loop node found. Exiting.")
        return

    print(f"--- Patching Loop '{loop_node.name}' ---")
    body_graph = next(attr.g for attr in loop_node.attribute if attr.name == "body")

    # --- Step 1: Identify key tensors and nodes ---
    scatter_node = next(n for n in body_graph.node if n.op_type == 'ScatterND')
    state_var_internal_name = scatter_node.input[0]
    scatter_updates_name = scatter_node.input[2]
    scatter_output_name = scatter_node.output[0]
    producer_of_updates = next(n for n in body_graph.node if scatter_updates_name in n.output)
    new_scan_output_name = producer_of_updates.input[0]
    
    # Identify the node that produces the dummy 'true' condition output
    cond_body_output_name = body_graph.output[0].name
    cond_producer_node = next(n for n in body_graph.node if cond_body_output_name in n.output)
    
    print(f"  State variable to remove: '{state_var_internal_name}'")
    print(f"  Producer of slice: '{producer_of_updates.name}' (Op: {producer_of_updates.op_type})")
    print(f"  New scan output tensor: '{new_scan_output_name}'")

    # --- Step 2: Remove state variable and condition from main Loop inputs ---
    # The Loop inputs are: M, cond, v_initial. We want to keep only M.
    new_loop_inputs = [loop_node.input[0]] # Keep only M (trip count)
    loop_node.ClearField('input')
    loop_node.input.extend(new_loop_inputs)
    print(f"  Configured main Loop to only have trip count input '/ReduceMin_output_0'.")

    # --- Step 3: Remove state variable and condition from body graph inputs ---
    # The body inputs are: iter_num, cond, v_initial. We keep only iter_num.
    new_body_inputs = [body_graph.input[0]]
    body_graph.ClearField('input')
    body_graph.input.extend(new_body_inputs)
    print("  Configured loop body to only have 'iteration_num' input.")
    
    # --- Step 4: Reroute metadata dependencies ---
    print("  Rerouting metadata dependencies...")
    for body_node in body_graph.node:
        for i, input_name in enumerate(body_node.input):
            if input_name == state_var_internal_name:
                print(f"    Rerouting input for node '{body_node.name}': '{input_name}' -> '{new_scan_output_name}'")
                body_node.input[i] = new_scan_output_name

    # --- Step 5: Set the correct body outputs [cond_out, scan_out] ---
    # The first output MUST be the condition. The second is our new scan output.
    new_body_outputs = [
        helper.make_tensor_value_info(cond_body_output_name, onnx.TensorProto.BOOL, []),
        helper.make_tensor_value_info(new_scan_output_name, onnx.TensorProto.FLOAT16, None) # Let shape inference fix the shape
    ]
    body_graph.ClearField('output')
    body_graph.output.extend(new_body_outputs)
    print(f"  Configured loop body outputs: [cond='{cond_body_output_name}', scan_out='{new_scan_output_name}']")

    # --- Step 6: Remove redundant nodes and re-sort ---
    # We keep the cond_producer_node (the Identity node) but remove the others.
    nodes_to_remove = {scatter_node.name, producer_of_updates.name}
    remaining_nodes = [n for n in body_graph.node if n.name not in nodes_to_remove]
    print(f"  Removed redundant nodes: {nodes_to_remove}")

    print("  Topologically re-sorting nodes within the loop body...")
    sorted_nodes = topologically_sort_nodes(remaining_nodes)
    
    body_graph.ClearField('node')
    body_graph.node.extend(sorted_nodes)
    print("  Node re-sorting complete.")

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
        print(f"Attempting to save the model without shape inference for manual inspection to: {output_onnx_path.replace('.onnx', '_failed_inference.onnx')}")
        onnx.save(model, output_onnx_path.replace('.onnx', '_failed_inference.onnx'))

if __name__ == "__main__":
    INPUT_MODEL_PATH = "onnx/simple_vae_decoder_optimized.onnx"
    OUTPUT_MODEL_PATH = "onnx/simple_vae_decoder_patched_scan_output.onnx" 

    patch_loop_scatter_to_scan_output(INPUT_MODEL_PATH, OUTPUT_MODEL_PATH)