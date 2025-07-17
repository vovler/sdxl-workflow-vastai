import onnx
from onnx import helper, checker, shape_inference
import argparse
import sys
import os

# --- Helper Functions ---

def find_node_by_output(graph, output_name):
    """Finds a node in a graph by its output name."""
    for node in graph.node:
        if output_name in node.output:
            return node
    return None

def find_value_info(graph, name):
    """Finds a value_info entry in a graph by its name, searching all relevant lists."""
    for vi in graph.value_info:
        if vi.name == name: return vi
    for vi in graph.input:
        if vi.name == name: return vi
    for vi in graph.output:
        if vi.name == name: return vi
    for node in graph.node:
        if node.op_type in ['Loop', 'Scan', 'If']:
            for attr in node.attribute:
                if attr.name in ["body", "then_branch", "else_branch"]:
                    result = find_value_info(attr.g, name)
                    if result: return result
    return None

# --- Step 1: Detection Function ---

def detect_loop_for_scan_conversion(model):
    """
    Detects a Loop that follows a batch-processing pattern, making it a candidate
    for conversion to a more robust Scan operator.
    """
    loops_to_transform = []
    for node in model.graph.node:
        if node.op_type != "Loop":
            continue
        body_graph = next((attr.g for attr in node.attribute if attr.name == "body"), None)
        if not body_graph:
            continue
        
        # Pattern: A Loop with an inner Gather and a final Concat for state accumulation.
        inner_gather = next((n for n in body_graph.node if n.op_type == 'Gather'), None)
        # The second output of the loop body is the accumulated state.
        final_concat = find_node_by_output(body_graph, body_graph.output[1].name)

        if inner_gather and final_concat and final_concat.op_type == 'Concat':
            details = {
                "loop_node": node,
                "inner_gather_node": inner_gather,
                "final_concat_node": final_concat,
            }
            loops_to_transform.append(details)
            print(f"INFO: Detected Loop '{node.name}' as a candidate for Scan conversion.")

    return loops_to_transform

# --- Step 2: Processing Function ---

def process_loop_to_scan(model, loops_to_transform):
    """
    Processes the model to transform the detected Loop operators into Scan operators.
    """
    if not loops_to_transform:
        return model

    graph = model.graph
    details = loops_to_transform[0] # Assuming one loop for this specific model
    loop_node = details["loop_node"]
    body_graph = next(attr.g for attr in loop_node.attribute if attr.name == "body")

    # 1. Identify key tensors
    inner_gather_node = details["inner_gather_node"]
    main_scan_tensor_name = inner_gather_node.input[0]
    per_item_tensor_name = inner_gather_node.output[0]
    
    final_concat_node = details["final_concat_node"]
    state_input_name = body_graph.input[2].name
    final_scan_output_name = next(inp for inp in final_concat_node.input if inp != state_input_name)
    
    # 2. Create the new body graph for the Scan operator
    scan_slice_input = onnx.ValueInfoProto()
    scan_slice_input.name = main_scan_tensor_name + "_slice"
    original_tensor_info = find_value_info(graph, main_scan_tensor_name)
    scan_slice_input.type.CopyFrom(original_tensor_info.type)
    del scan_slice_input.type.tensor_type.shape.dim[0]
    new_body_inputs = [scan_slice_input]

    final_scan_output_info = find_value_info(body_graph, final_scan_output_name)
    new_body_outputs = [final_scan_output_info]

    nodes_to_remove_from_body = {
        inner_gather_node.name,
        final_concat_node.name,
        find_node_by_output(body_graph, body_graph.output[0].name).name
    }
    new_body_nodes = [n for n in body_graph.node if n.name not in nodes_to_remove_from_body]

    for node in new_body_nodes:
        for i, inp in enumerate(node.input):
            if inp == per_item_tensor_name:
                node.input[i] = scan_slice_input.name

    new_body_graph = helper.make_graph(
        nodes=new_body_nodes, name=body_graph.name + "_scan_body",
        inputs=new_body_inputs, outputs=new_body_outputs,
        initializer=body_graph.initializer,
        value_info=body_graph.value_info  # ### THIS IS THE CRITICAL FIX ###
    )
    
    # 3. Create the new Scan node
    scan_node = helper.make_node(
        op_type='Scan',
        inputs=[main_scan_tensor_name],
        outputs=[graph.output[0].name],
        name=loop_node.name + "_as_scan",
        num_scan_inputs=1,
        body=new_body_graph,
        scan_input_axes=[0],
        scan_output_axes=[0]
    )
    
    # 4. Build the new main graph
    nodes_to_remove_from_main = {loop_node.name}
    trip_count_producer = find_node_by_output(graph, loop_node.input[0])
    if trip_count_producer:
        nodes_to_remove_from_main.add(trip_count_producer.name)
        shape_producer = find_node_by_output(graph, trip_count_producer.input[0])
        if shape_producer:
            nodes_to_remove_from_main.add(shape_producer.name)
            
    final_nodes = [n for n in graph.node if n.name not in nodes_to_remove_from_main]
    final_nodes.append(scan_node)
    
    new_graph = helper.make_graph(
        nodes=final_nodes, name=graph.name + "_scanned",
        inputs=graph.input, outputs=graph.output,
        initializer=graph.initializer, value_info=graph.value_info
    )
    
    new_model = helper.make_model(new_graph, producer_name='onnx-loop-to-scan-transformer')
    new_model.opset_import.extend(model.opset_import)
    
    return new_model

# --- Main Orchestration and Saving Logic ---

def patch_loop_to_scan(input_onnx_path, output_onnx_path):
    """Main function to orchestrate the detection, processing, and saving of the model."""
    print(f"Loading model from: {input_onnx_path}")
    try:
        model = onnx.load(input_onnx_path)
        print("INFO: Performing initial shape inference to gather metadata...")
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"ERROR: Failed to load or run initial shape inference on ONNX model. {e}", file=sys.stderr)
        return

    transform_list = detect_loop_for_scan_conversion(model)
    if not transform_list:
        print("INFO: No candidate loops for Scan conversion were found. Model is unchanged.")
        return

    print("\nFound the following loop to transform to a Scan operator:")
    print(f"  - Loop Node: '{transform_list[0]['loop_node'].name}'\n")

    print("Processing model to convert Loop to Scan...")
    transformed_model = process_loop_to_scan(model, transform_list)
    if not transformed_model:
        print("ERROR: Transformation to Scan operator failed during processing.", file=sys.stderr)
        return

    print("Performing final health check and saving...")
    try:
        model_for_saving = shape_inference.infer_shapes(transformed_model)
        checker.check_model(model_for_saving)
        print("  Model check passed.")
        onnx.save(model_for_saving, output_onnx_path)
        print(f"Successfully saved Scan-based model to {output_onnx_path}")
    except Exception as e:
        print(f"An error occurred during final model validation or saving: {e}", file=sys.stderr)
        output_failed_path = output_onnx_path.replace('.onnx', '_failed_scan.onnx')
        print(f"Attempting to save the model without shape inference for manual inspection to: {output_failed_path}")
        onnx.save(transformed_model, output_failed_path)

def main():
    parser = argparse.ArgumentParser(
        description="Transforms a batch-processing ONNX Loop into a robust Scan operator for TensorRT.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_onnx_path",
        nargs='?',
        default="onnx/simple_vae_decoder_direct_optimized.onnx",
        help="Path to the original ONNX file with the Loop.\n(default: onnx/simple_vae_decoder_direct_optimized.onnx)"
    )
    args = parser.parse_args()
    input_path = args.input_onnx_path

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found at '{input_path}'", file=sys.stderr)
        sys.exit(1)
    
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_scan{ext}"
    
    print("-" * 50)
    print(f"Input model:  {input_path}")
    print(f"Output model: {output_path}")
    print("-" * 50)

    patch_loop_to_scan(input_path, output_path)

if __name__ == "__main__":
    main()