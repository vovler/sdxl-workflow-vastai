import onnx
from onnx import helper, checker, shape_inference
import argparse
import sys
import os

# --- Helper Functions (to navigate the ONNX graph) ---

def find_node_by_output(graph, output_name):
    """Finds a node in a graph by its output name."""
    for node in graph.node:
        if output_name in node.output:
            return node
    return None

def find_initializer_by_name(graph, name):
    """Finds an initializer in a graph by its name."""
    for init in graph.initializer:
        if init.name == name:
            return init
    return None

def find_value_info(graph, name):
    """Finds a value_info entry in a graph by its name, searching all relevant lists."""
    for vi in graph.value_info:
        if vi.name == name:
            return vi
    for vi in graph.input:
        if vi.name == name:
            return vi
    for vi in graph.output:
        if vi.name == name:
            return vi
    # In some models, the required value_info might be in the body graph's value_info
    for node in graph.node:
        if node.op_type == "Loop":
            for attr in node.attribute:
                if attr.name == "body":
                    for bvi in attr.g.value_info:
                        if bvi.name == name:
                            return bvi
    return None


# --- Step 1: Detection Function ---

def detect_loops_concat_scan_to_native_scan(model):
    """
    Detects loops that use a Concat-based scan pattern. This pattern is
    common but not efficiently handled by the TensorRT ONNX parser.

    Args:
        model (onnx.ModelProto): The ONNX model to inspect.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              details needed to transform one loop. Returns an empty list
              if no such loops are found.
    """
    graph = model.graph
    loops_to_transform = []

    for node in graph.node:
        if node.op_type != "Loop":
            continue

        body_graph_attr = [attr for attr in node.attribute if attr.name == "body"]
        if not body_graph_attr:
            continue
        body_graph = body_graph_attr[0].g

        for body_output_idx, body_output in enumerate(body_graph.output):
            if body_output_idx == 0:  # Skip the condition output
                continue

            producer_node = find_node_by_output(body_graph, body_output.name)
            if not (producer_node and producer_node.op_type == "Concat"):
                continue

            body_inputs_names = [inp.name for inp in body_graph.input]
            found_pattern_for_this_output = False

            for body_input_idx, body_input_name in enumerate(body_inputs_names):
                if body_input_name in producer_node.input:
                    per_iteration_tensor_name = [
                        inp for inp in producer_node.input if inp != body_input_name
                    ][0]

                    transformation_details = {
                        "loop_node_name": node.name,
                        "concat_node": producer_node,
                        "state_body_input_name": body_input_name,
                        "state_body_input_idx": body_input_idx,
                        "state_body_output_name": body_output.name,
                        "state_body_output_idx": body_output_idx,
                        "per_iteration_tensor_name": per_iteration_tensor_name,
                    }
                    loops_to_transform.append(transformation_details)
                    found_pattern_for_this_output = True
                    break

            if found_pattern_for_this_output:
                break

    return loops_to_transform


# --- Step 2: Processing Function ---

def process_loops_concat_scan_to_native_scan(model, loops_to_transform):
    """
    Processes the model to transform the detected loops into a native
    scan-output pattern.

    Args:
        model (onnx.ModelProto): The ONNX model to transform.
        loops_to_transform (list): The list of transformation details from the
                                   detection function.

    Returns:
        onnx.ModelProto: A new, transformed ONNX model.
    """
    if not loops_to_transform:
        return model

    graph = model.graph
    nodes_to_remove = set()
    nodes_to_add = []
    initializers_to_remove = set()

    for details in loops_to_transform:
        loop_node = next((n for n in graph.node if n.name == details['loop_node_name']), None)
        if not loop_node: continue

        nodes_to_remove.add(loop_node.name)
        body_graph = next(attr.g for attr in loop_node.attribute if attr.name == "body")

        # A. Modify the body graph
        new_body_nodes = [n for n in body_graph.node if n.name != details["concat_node"].name]
        new_body_inputs = [inp for i, inp in enumerate(body_graph.input) if i != details["state_body_input_idx"]]
        
        # We need the full ValueInfo of the per-iteration tensor. Search the entire model.
        per_iter_tensor_vi = find_value_info(model.graph, details["per_iteration_tensor_name"])
        if not per_iter_tensor_vi:
            print(f"ERROR: Could not find ValueInfo for tensor '{details['per_iteration_tensor_name']}'. Aborting.", file=sys.stderr)
            return None

        new_body_outputs = []
        original_state_output_name = body_graph.output[details["state_body_output_idx"]].name
        for i, out in enumerate(body_graph.output):
            if i == 0:
                new_body_outputs.append(out)
            elif i == details["state_body_output_idx"]:
                new_scan_output = onnx.ValueInfoProto()
                new_scan_output.CopyFrom(per_iter_tensor_vi)
                new_scan_output.name = original_state_output_name
                new_body_outputs.append(new_scan_output)

        new_body_graph = helper.make_graph(
            nodes=new_body_nodes, name=body_graph.name + "_transformed",
            inputs=new_body_inputs, outputs=new_body_outputs,
            initializer=body_graph.initializer, value_info=body_graph.value_info
        )

        # B. Modify the main Loop node
        main_loop_state_input_name = loop_node.input[details["state_body_input_idx"]]
        new_loop_inputs = [inp for i, inp in enumerate(loop_node.input) if i != details["state_body_input_idx"]]
        
        main_loop_output_idx = details["state_body_output_idx"] - 1
        new_loop_outputs = [loop_node.output[main_loop_output_idx]]
        
        new_loop_node = helper.make_node('Loop', inputs=new_loop_inputs, outputs=new_loop_outputs, name=loop_node.name)
        new_loop_node.attribute.append(helper.make_attribute("body", new_body_graph))
        nodes_to_add.append(new_loop_node)

        # C. Mark the initial state provider for removal
        initial_state_producer = find_node_by_output(graph, main_loop_state_input_name)
        if initial_state_producer:
            nodes_to_remove.add(initial_state_producer.name)
        elif find_initializer_by_name(graph, main_loop_state_input_name):
            initializers_to_remove.add(main_loop_state_input_name)

    # Reconstruct the main graph
    final_nodes = [node for node in graph.node if node.name not in nodes_to_remove]
    final_nodes.extend(nodes_to_add)
    final_initializers = [init for init in graph.initializer if init.name not in initializers_to_remove]

    new_graph = helper.make_graph(
        nodes=final_nodes, name=graph.name + "_transformed",
        inputs=graph.input, outputs=graph.output,
        initializer=final_initializers, value_info=graph.value_info
    )

    new_model = helper.make_model(new_graph, producer_name='onnx-loop-transformer')
    new_model.opset_import.extend(model.opset_import)
    return new_model


# --- Main Orchestration and Saving Logic ---

def patch_loops_output(input_onnx_path, output_onnx_path):
    """
    Main function to detect, process, and save an ONNX model with transformed loops.
    """
    print(f"Loading model from: {input_onnx_path}")
    try:
        model = onnx.load(input_onnx_path)
    except Exception as e:
        print(f"ERROR: Failed to load ONNX model. {e}", file=sys.stderr)
        return

    # 1. Detect loops that need transformation
    transform_list = detect_loops_concat_scan_to_native_scan(model)

    if not transform_list:
        print("INFO: No loops with the target Concat-scan pattern were found. Model is unchanged.")
        return

    # 2. Print what was found
    print("\nFound the following loops to transform:")
    for details in transform_list:
        print(f"  - Loop Node: '{details['loop_node_name']}'")
        print(f"    - Inner Concat Node: '{details['concat_node'].name}'")
        print(f"    - Per-iteration tensor: '{details['per_iteration_tensor_name']}'\n")

    # 3. Process the model
    print("Processing model...")
    transformed_model = process_loops_concat_scan_to_native_scan(model, transform_list)
    if not transformed_model:
        print("ERROR: Transformation failed during processing.", file=sys.stderr)
        return

    # 4. Perform health check and save
    print("Performing final health check and saving...")
    try:
        model_for_saving = shape_inference.infer_shapes(transformed_model)
        onnx.checker.check_model(model_for_saving)
        print("  Model check passed.")
        onnx.save(model_for_saving, output_onnx_path)
        print(f"Successfully saved patched model to {output_onnx_path}")
    except Exception as e:
        print(f"An error occurred during final model validation or saving: {e}")
        output_failed_path = output_onnx_path.replace('.onnx', '_failed_inference.onnx')
        print(f"Attempting to save the model without shape inference for manual inspection to: {output_failed_path}")
        onnx.save(transformed_model, output_failed_path)


def main():
    """
    Main entry point for the script. Parses command-line arguments and
    orchestrates the transformation.
    """
    parser = argparse.ArgumentParser(
        description="Transforms ONNX loops from a Concat-based scan to a native scan-output pattern for TensorRT compatibility.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_onnx_path",
        nargs='?',
        default="onnx/simple_vae_decoder_direct_optimized.onnx",
        help="Path to the input ONNX file.\n(default: onnx/simple_vae_decoder_direct_optimized.onnx)"
    )
    args = parser.parse_args()

    input_path = args.input_onnx_path

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found at '{input_path}'", file=sys.stderr)
        print("Please provide a valid path to an ONNX model.", file=sys.stderr)
        sys.exit(1)
    
    # Generate the output path by adding '_patched' before the extension
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_patched{ext}"
    
    print("-" * 50)
    print(f"Input model:  {input_path}")
    print(f"Output model: {output_path}")
    print("-" * 50)

    patch_loops_output(input_path, output_path)

if __name__ == "__main__":
    main()