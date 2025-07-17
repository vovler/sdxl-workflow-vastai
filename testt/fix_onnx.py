import onnx
from onnx import helper, shape_inference

def topologically_sort_nodes(nodes, graph_inputs):
    """
    Performs a topological sort on a list of ONNX nodes.

    Args:
        nodes: A list of onnx.NodeProto objects.
        graph_inputs: A list of tensor names that are inputs to this graph/subgraph.

    Returns:
        A new list of onnx.NodeProto objects in topologically sorted order.
    """
    # 1. Build the graph representation
    node_map = {node.name: node for node in nodes}
    dependencies = {node.name: [] for node in nodes}
    in_degree = {node.name: 0 for node in nodes}
    tensor_producer = {}

    for node in nodes:
        for output_name in node.output:
            tensor_producer[output_name] = node.name
    
    for node in nodes:
        for input_name in node.input:
            # Check if the input is produced by another node in this list
            if input_name in tensor_producer:
                producer_node_name = tensor_producer[input_name]
                # Add dependency: producer must come before current node
                dependencies[producer_node_name].append(node.name)
                in_degree[node.name] += 1

    # 2. Initialize the queue with nodes that have no internal dependencies
    # (their inputs are either graph inputs or initializers)
    queue = [name for name, degree in in_degree.items() if degree == 0]
    
    # 3. Kahn's algorithm for topological sorting
    sorted_nodes = []
    while queue:
        node_name = queue.pop(0)
        sorted_nodes.append(node_map[node_name])
        
        if node_name in dependencies:
            for dependent_node_name in dependencies[node_name]:
                in_degree[dependent_node_name] -= 1
                if in_degree[dependent_node_name] == 0:
                    queue.append(dependent_node_name)

    if len(sorted_nodes) != len(nodes):
        raise RuntimeError("Cycle detected in the graph, topological sort failed.")
        
    return sorted_nodes


def patch_loop_scatter_to_scan_output(input_onnx_path: str, output_onnx_path: str):
    """
    Generalizes the patching of ONNX models for TensorRT compatibility by converting
    inefficient ScatterND-based state updates into efficient scan outputs.
    """
    print(f"Loading model from {input_onnx_path}...")
    model = onnx.load(input_onnx_path)
    graph = model.graph
    loops_to_process = []

    print("Searching for Loop nodes with the ScatterND state-update pattern...")
    for node in graph.node:
        if node.op_type == "Loop":
            body_graph = next((attr.g for attr in node.attribute if attr.name == "body"), None)
            if not body_graph: continue

            for sub_node in body_graph.node:
                if sub_node.op_type == "ScatterND":
                    scatter_data_input = sub_node.input[0]
                    body_state_inputs = [inp.name for inp in body_graph.input[2:]]
                    if scatter_data_input in body_state_inputs:
                        loops_to_process.append((node, sub_node))
                        print(f"  [+] Found candidate: Loop '{node.name}' uses ScatterND '{sub_node.name}' to update state '{scatter_data_input}'.")

    if not loops_to_process:
        print("No loops matching the pattern were found. No patching needed.")
        return

    for loop_node, scatter_node in loops_to_process:
        print(f"\n--- Patching Loop '{loop_node.name}' ---")
        body_graph = next(attr.g for attr in loop_node.attribute if attr.name == "body")

        state_var_internal_name = scatter_node.input[0]
        scatter_updates_name = scatter_node.input[2]
        scatter_output_name = scatter_node.output[0]

        producer_of_updates = next((n for n in body_graph.node if scatter_updates_name in n.output), None)
        if not producer_of_updates:
            print(f"  [!] ERROR: Could not find the producer of '{scatter_updates_name}'. Aborting patch for this loop.")
            continue
        
        new_scan_output_name = producer_of_updates.input[0]
        print(f"  State variable to remove: '{state_var_internal_name}'")
        print(f"  Identified producer of slice: '{producer_of_updates.name}' (Op: {producer_of_updates.op_type})")
        print(f"  New scan output tensor will be: '{new_scan_output_name}'")

        print("  Rerouting metadata dependencies...")
        for body_node in body_graph.node:
            if body_node.name == scatter_node.name: continue
            for i, input_name in enumerate(body_node.input):
                if input_name == state_var_internal_name:
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

        for out in body_graph.output:
            if out.name == scatter_output_name:
                print(f"  Changing body output '{out.name}' to point to scan output '{new_scan_output_name}'.")
                out.name = new_scan_output_name
                out.ClearField("type")
                break
        
        nodes_to_remove = {scatter_node.name, producer_of_updates.name}
        remaining_nodes = [n for n in body_graph.node if n.name not in nodes_to_remove]
        print(f"  Removed redundant nodes: {nodes_to_remove}")

        print("  Topologically re-sorting nodes within the loop body...")
        sorted_nodes = topologically_sort_nodes(remaining_nodes, [inp.name for inp in body_graph.input])

        body_graph.ClearField("node")
        body_graph.node.extend(sorted_nodes)
        print("  Node re-sorting complete.")

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