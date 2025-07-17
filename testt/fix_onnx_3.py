import onnx
from onnx import helper, checker, shape_inference
import argparse
import sys
import os

def find_node_by_output(graph, output_name):
    """Finds a node in a graph by its output name."""
    for node in graph.node:
        if output_name in node.output:
            return node
    return None

def transform_loop_to_scan(model):
    """
    Transforms a batch-processing Loop into a more explicit and robust Scan operator,
    which is better supported by the ONNX-TensorRT parser.
    """
    graph = model.graph
    
    # 1. Find the Loop node
    loop_node = next((n for n in graph.node if n.op_type == 'Loop'), None)
    if not loop_node:
        print("INFO: No Loop node found. Nothing to transform.")
        return None, "No Loop found"

    print(f"INFO: Found Loop node '{loop_node.name}'. Transforming to Scan operator.")
    body_graph = next(attr.g for attr in loop_node.attribute if attr.name == "body")
    
    # 2. Identify key tensors from the original Loop structure
    # The tensor being iterated over is the input to the inner Gather node
    inner_gather_node = next((n for n in body_graph.node if n.op_type == 'Gather'), None)
    if not inner_gather_node:
        return None, "Could not find inner Gather node in loop body."
        
    main_scan_tensor_name = inner_gather_node.input[0]
    per_item_tensor_name = inner_gather_node.output[0]
    
    # The final computed result is the input to the old Concat/Identity node
    final_body_op = find_node_by_output(body_graph, body_graph.output[1].name)
    state_input_name = body_graph.input[2].name
    final_scan_output_name = next(inp for inp in final_body_op.input if inp != state_input_name)
    
    # 3. Create the new, simplified body graph for the Scan operator
    
    # New body inputs: The Scan op provides the sliced tensor directly.
    # We must copy the type and shape info from the original tensor.
    scan_slice_input = onnx.ValueInfoProto()
    scan_slice_input.name = main_scan_tensor_name + "_slice"
    
    # The slice will have the same type, but one less dimension.
    original_tensor_info = next(i for i in model.graph.input if i.name == main_scan_tensor_name)
    scan_slice_input.type.CopyFrom(original_tensor_info.type)
    # Remove the batch dimension (axis 0) from the shape
    del scan_slice_input.type.tensor_type.shape.dim[0]
    
    new_body_inputs = [scan_slice_input]
    
    # New body outputs: Just the final computed tensor
    final_scan_output_info = next(vi for vi in body_graph.value_info if vi.name == final_scan_output_name)
    new_body_outputs = [final_scan_output_info]
    
    # New body nodes: Copy all nodes, but remove the machinery
    nodes_to_remove_from_body = {
        inner_gather_node.name,
        final_body_op.name,
        find_node_by_output(body_graph, body_graph.output[0].name).name
    }
    new_body_nodes = [n for n in body_graph.node if n.name not in nodes_to_remove_from_body]
    
    # Rewire the first node in the sequence to use the new sliced input
    for node in new_body_nodes:
        for i, inp in enumerate(node.input):
            if inp == per_item_tensor_name:
                node.input[i] = scan_slice_input.name
                print(f"INFO: Rewired input of node '{node.name}' to use new slice '{scan_slice_input.name}'")

    new_body_graph = helper.make_graph(
        nodes=new_body_nodes,
        name=body_graph.name + "_scan_body",
        inputs=new_body_inputs,
        outputs=new_body_outputs,
        initializer=body_graph.initializer
    )
    
    # 4. Create the new Scan node for the main graph
    scan_node = helper.make_node(
        op_type='Scan',
        inputs=[main_scan_tensor_name],  # Only the tensor to be scanned
        outputs=[graph.output[0].name], # The final output of the whole graph
        name=loop_node.name + "_as_scan",
        num_scan_inputs=1,
        body=new_body_graph,
        scan_input_axes=[0],
        scan_output_axes=[0]
    )
    
    # 5. Build the new main graph, removing all old loop machinery
    nodes_to_remove_from_main = {loop_node.name}
    # Find and remove the Shape->Gather chain that calculated the trip count
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
        initializer=graph.initializer
    )
    
    new_model = helper.make_model(new_graph, producer_name='onnx-loop-to-scan-transformer')
    new_model.opset_import.extend(model.opset_import)
    
    return new_model, "Success"

def main():
    parser = argparse.ArgumentParser(
        description="Transforms a batch-processing ONNX Loop into a robust Scan operator for TensorRT.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_onnx_path",
        help="Path to the original ONNX file with the Loop."
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

    try:
        model = onnx.load(input_path)
        # Run shape inference to populate value_info, which is crucial for the script
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"ERROR: Failed to load or run initial shape inference on ONNX model. {e}", file=sys.stderr)
        sys.exit(1)

    transformed_model, status = transform_loop_to_scan(model)

    if not transformed_model:
        print(f"ERROR: Transformation failed. Reason: {status}")
        sys.exit(1)
    
    print("INFO: Transformation to Scan complete. Performing final health check and saving...")
    try:
        model_for_saving = shape_inference.infer_shapes(transformed_model)
        checker.check_model(model_for_saving)
        print("  Model check passed.")
        onnx.save(model_for_saving, output_path)
        print(f"Successfully saved Scan-based model to {output_path}")
    except Exception as e:
        print(f"An error occurred during final model validation or saving: {e}", file=sys.stderr)
        output_failed_path = output_path.replace('.onnx', '_failed_inference.onnx')
        print(f"Attempting to save the model without shape inference for manual inspection to: {output_failed_path}")
        onnx.save(transformed_model, output_failed_path)


if __name__ == "__main__":
    main()