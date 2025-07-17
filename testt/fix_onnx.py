import onnx
from onnx import helper, shape_inference, numpy_helper
import numpy as np

def topologically_sort_nodes(nodes: list) -> list:
    """
    Performs a topological sort on a list of ONNX nodes.
    This is crucial after manual graph modifications.
    """
    # Create a map from tensor name to the node that produces it
    tensor_producer = {}
    for node in nodes:
        for output_name in node.output:
            tensor_producer[output_name] = node.name

    # Build dependency graph and in-degree count for each node
    in_degree = {node.name: 0 for node in nodes}
    # Adjacency list: for a given node, which nodes depend on it?
    dependencies = {node.name: [] for node in nodes}

    for node in nodes:
        for input_name in node.input:
            if input_name in tensor_producer:
                producer_node_name = tensor_producer[input_name]
                if producer_node_name in dependencies:
                    dependencies[producer_node_name].append(node.name)
                    in_degree[node.name] += 1

    # Kahn's algorithm for topological sorting
    node_map = {node.name: node for node in nodes}
    queue = [name for name, degree in in_degree.items() if degree == 0]
    sorted_node_names = []
    
    while queue:
        node_name = queue.pop(0)
        sorted_node_names.append(node_name)
        
        if node_name in dependencies:
            for dependent_node_name in dependencies[node_name]:
                in_degree[dependent_node_name] -= 1
                if in_degree[dependent_node_name] == 0:
                    queue.append(dependent_node_name)

    if len(sorted_node_names) != len(nodes):
        raise RuntimeError("Cycle detected in the graph or disconnected components, topological sort failed.")
        
    return [node_map[name] for name in sorted_node_names]

def patch_loop_scatter_to_scan_output(input_onnx_path: str, output_onnx_path: str):
    """
    Generalizes the patching of ONNX models for TensorRT compatibility.
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
            if any(sub_node.op_type == "ScatterND" and sub_node.input[0] in [inp.name for inp in body_graph.input[2:]] for sub_node in body_graph.node):
                scatter_node = next(sub_node for sub_node in body_graph.node if sub_node.op_type == "ScatterND" and sub_node.input[0] in [inp.name for inp in body_graph.input[2:]])
                loops_to_process.append((node, scatter_node))
                print(f"  [+] Found candidate: Loop '{node.name}' uses ScatterND '{scatter_node.name}' to update state '{scatter_node.input[0]}'.")

    if not loops_to_process:
        print("No loops matching the pattern were found. No patching needed.")
        return

    for loop_node, scatter_node in loops_to_process:
        print(f"\n--- Patching Loop '{loop_node.name}' ---")
        body_graph = next(attr.g for attr in loop_node.attribute if attr.name == "body")

        # --- Step 1: Identify key tensors and nodes ---
        state_var_internal_name = scatter_node.input[0]
        scatter_updates_name = scatter_node.input[2]
        scatter_output_name = scatter_node.output[0]
        producer_of_updates = next(n for n in body_graph.node if scatter_updates_name in n.output)
        new_scan_output_name = producer_of_updates.input[0]

        # --- Step 2: Remove the redundant boolean condition ---
        cond_main_input_name = loop_node.input[1]
        cond_body_input_name = body_graph.input[1].name
        cond_body_output_name = body_graph.output[0].name
        cond_producer_node = next(n for n in body_graph.node if cond_body_output_name in n.output)

        print(f"  Removing redundant boolean condition '{cond_main_input_name}'.")
        loop_node.input[1] = "" # Blank the optional condition input
        del body_graph.input[1]
        del body_graph.output[0]

        # --- Step 3: Reroute metadata dependencies and remove state variable ---
        print(f"  Rerouting metadata dependencies from '{state_var_internal_name}' to '{new_scan_output_name}'.")
        for body_node in body_graph.node:
            if body_node.name == scatter_node.name: continue
            for i, input_name in enumerate(body_node.input):
                if input_name == state_var_internal_name:
                    print(f"    Rerouting input for node '{body_node.name}': '{input_name}' -> '{new_scan_output_name}'")
                    body_node.input[i] = new_scan_output_name

        state_var_body_index = next((i for i, inp in enumerate(body_graph.input) if inp.name == state_var_internal_name), -1)
        
        if state_var_body_index == -1: continue
            
        main_loop_input_to_remove = loop_node.input[state_var_body_index]
        loop_node.input.remove(main_loop_input_to_remove)
        print(f"  Removed state variable input '{main_loop_input_to_remove}' from the main Loop.")
        del body_graph.input[state_var_body_index]
        print(f"  Removed state variable '{state_var_internal_name}' from the loop body's inputs.")

        # --- Step 4: Rewire the loop body's output to be the scan output ---
        for out in body_graph.output:
            if out.name == scatter_output_name:
                print(f"  Changing body output '{out.name}' to point to scan output '{new_scan_output_name}'.")
                out.name = new_scan_output_name
                out.ClearField("type")
                break
        
        # --- Step 5: Remove redundant nodes and re-sort the graph ---
        nodes_to_remove = {scatter_node.name, producer_of_updates.name, cond_producer_node.name}
        remaining_nodes = [n for n in body_graph.node if n.name not in nodes_to_remove]
        print(f"  Removed redundant nodes: {nodes_to_remove}")

        print("  Topologically re-sorting nodes within the loop body...")
        sorted_nodes = topologically_sort_nodes(remaining_nodes)
        
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
