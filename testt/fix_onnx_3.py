import onnx
from onnx import helper, checker, shape_inference
import argparse
import sys
import os
import numpy as np
import onnxruntime

# --- Helper Functions ---

def find_node_by_output(graph, output_name):
    """Finds a node in a graph by its output name."""
    for node in graph.node:
        if output_name in node.output:
            return node
    return None

def find_value_info(graph, name):
    """Finds a value_info entry in a graph by its name, searching all relevant lists."""
    all_value_infos = list(graph.value_info) + list(graph.input) + list(graph.output)
    for vi in all_value_infos:
        if vi.name == name: return vi
    for node in graph.node:
        if node.op_type in ['Loop', 'Scan', 'If']:
            for attr in node.attribute:
                if attr.name in ["body", "then_branch", "else_branch"]:
                    result = find_value_info(attr.g, name)
                    if result: return result
    return None

# --- Detection Function ---

def detect_loop_for_scan_conversion(model):
    """Detects a Loop that is a candidate for conversion to a Scan operator."""
    loops_to_transform = []
    for node in model.graph.node:
        if node.op_type != "Loop":
            continue
        body_graph = next((attr.g for attr in node.attribute if attr.name == "body"), None)
        if not body_graph:
            continue
        
        inner_gather = next((n for n in body_graph.node if n.op_type == 'Gather'), None)
        final_concat = find_node_by_output(body_graph, body_graph.output[1].name)

        if inner_gather and final_concat and final_concat.op_type == 'Concat':
            details = {"loop_node": node, "inner_gather_node": inner_gather, "final_concat_node": final_concat}
            loops_to_transform.append(details)
            print(f"INFO: Detected Loop '{node.name}' as a candidate for Scan conversion.")
    return loops_to_transform

# --- Processing Function ---

def process_loop_to_scan(model, loops_to_transform, opset_version, ir_version):
    """Transforms the detected Loop operators into Scan operators and adds a Reshape."""
    if not loops_to_transform: return model

    graph = model.graph
    details = loops_to_transform[0]
    loop_node, body_graph = details["loop_node"], next(attr.g for attr in details["loop_node"].attribute if attr.name == "body")

    inner_gather_node, final_concat_node = details["inner_gather_node"], details["final_concat_node"]
    main_scan_tensor_name, per_item_tensor_name = inner_gather_node.input[0], inner_gather_node.output[0]
    state_input_name = body_graph.input[2].name
    final_scan_output_name = next(inp for inp in final_concat_node.input if inp != state_input_name)
    
    scan_slice_input = onnx.ValueInfoProto(name=main_scan_tensor_name + "_slice")
    original_tensor_info = find_value_info(graph, main_scan_tensor_name)
    scan_slice_input.type.CopyFrom(original_tensor_info.type)
    del scan_slice_input.type.tensor_type.shape.dim[0]
    new_body_inputs = [scan_slice_input]

    final_scan_output_info = find_value_info(body_graph, final_scan_output_name)
    new_body_outputs = [final_scan_output_info]

    nodes_to_remove_from_body = {
        inner_gather_node.name, final_concat_node.name,
        find_node_by_output(body_graph, body_graph.output[0].name).name
    }
    new_body_nodes = [n for n in body_graph.node if n.name not in nodes_to_remove_from_body]

    for node in new_body_nodes:
        for i, inp in enumerate(node.input):
            if inp == per_item_tensor_name: node.input[i] = scan_slice_input.name
    
    body_tensors = {inp.name for inp in new_body_inputs}
    for node in new_body_nodes:
        body_tensors.update(node.input); body_tensors.update(node.output)
    new_body_value_info = [vi for vi in body_graph.value_info if vi.name in body_tensors]

    new_body_graph = helper.make_graph(
        nodes=new_body_nodes, name=body_graph.name + "_scan_body",
        inputs=new_body_inputs, outputs=new_body_outputs,
        initializer=body_graph.initializer, value_info=new_body_value_info
    )
    
    # ### CHANGE: Add a Reshape node after the Scan ###
    scan_output_raw_name = loop_node.name + "_scan_output_raw"
    
    scan_node = helper.make_node(
        op_type='Scan', inputs=[main_scan_tensor_name], outputs=[scan_output_raw_name],
        name=loop_node.name + "_as_scan", num_scan_inputs=1, body=new_body_graph,
        scan_input_axes=[0], scan_output_axes=[0]
    )

    # Get the target shape from the original graph's output, e.g., (3, 512, 512)
    original_output_dims = [d.dim_value for d in graph.output[0].type.tensor_type.shape.dim]
    target_shape_dims = [-1] + original_output_dims[1:] # e.g., [-1, 3, 512, 512]
    
    shape_const_name = "final_reshape_shape"
    shape_const_node = helper.make_node(
        'Constant', inputs=[], outputs=[shape_const_name],
        value=helper.make_tensor(name='value', data_type=onnx.TensorProto.INT64,
                                 dims=[len(target_shape_dims)], vals=target_shape_dims)
    )

    reshape_node = helper.make_node(
        'Reshape', inputs=[scan_output_raw_name, shape_const_name],
        outputs=[graph.output[0].name], name="final_output_reshape"
    )
    
    nodes_to_remove_from_main = {loop_node.name}
    trip_count_producer = find_node_by_output(graph, loop_node.input[0])
    if trip_count_producer:
        nodes_to_remove_from_main.add(trip_count_producer.name)
        shape_producer = find_node_by_output(graph, trip_count_producer.input[0])
        if shape_producer: nodes_to_remove_from_main.add(shape_producer.name)
            
    final_nodes = [n for n in graph.node if n.name not in nodes_to_remove_from_main]
    final_nodes.extend([scan_node, shape_const_node, reshape_node])
    
    new_graph = helper.make_graph(
        nodes=final_nodes, name=graph.name + "_scanned",
        inputs=graph.input, outputs=graph.output,
        initializer=graph.initializer, value_info=graph.value_info
    )
    
    new_model = helper.make_model(
        new_graph, producer_name='onnx-loop-to-scan-transformer',
        opset_imports=[helper.make_opsetid("", opset_version)]
    )
    new_model.ir_version = ir_version
    
    return new_model

# --- Inference Test & Main Orchestration --- (Unchanged from previous version)

def run_inference_test(onnx_path):
    print("\n" + "-" * 20)
    print("--- Running Inference Test ---")
    print("-" * 20)
    try:
        session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_meta, output_meta = session.get_inputs()[0], session.get_outputs()[0]

        print(f"Input Name: {input_meta.name}, Shape: {input_meta.shape}, Type: {input_meta.type}")
        input_shape = [2 if isinstance(dim, str) else dim for dim in input_meta.shape]
        dtype_map = {'tensor(float16)': np.float16, 'tensor(float)': np.float32}
        input_dtype = dtype_map.get(input_meta.type, np.float32)
        dummy_input = np.random.rand(*input_shape).astype(input_dtype)

        print(f"Created dummy input with shape: {dummy_input.shape}, dtype: {dummy_input.dtype}")
        result = session.run([output_meta.name], {input_meta.name: dummy_input})
        
        print(f"Inference successful!")
        print(f"Output Name: {output_meta.name}, Shape: {result[0].shape}, Dtype: {result[0].dtype}")
        print("-" * 20)
    except Exception as e:
        print(f"\nERROR: ONNXRuntime inference test failed: {e}", file=sys.stderr)
        print("-" * 20)

def patch_model_main(input_onnx_path, output_onnx_path, opset_version, ir_version, test_inference):
    """Main function to orchestrate the detection, processing, saving, and testing."""
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

    print(f"\nFound Loop '{transform_list[0]['loop_node'].name}' to transform to a Scan operator.\n")
    print("Processing model...")
    transformed_model = process_loop_to_scan(model, transform_list, opset_version, ir_version)
    if not transformed_model:
        print("ERROR: Transformation failed during processing.", file=sys.stderr)
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
        if test_inference:
            run_inference_test(output_failed_path)
        return

    if test_inference:
        run_inference_test(output_onnx_path)

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
    parser.add_argument("--opset", type=int, default=20, help="The ONNX opset version to set. (default: 20)")
    parser.add_argument("--ir", type=int, default=9, help="The ONNX IR version to set. (default: 9)")
    parser.add_argument("--test-inference", action="store_true", help="Run a test inference on the exported model using ONNXRuntime.")
    args = parser.parse_args()
    input_path = args.input_onnx_path

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found at '{input_path}'", file=sys.stderr)
        sys.exit(1)
    
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_scan{ext}"
    
    print("-" * 50)
    print(f"Input model:    {input_path}")
    print(f"Output model:   {output_path}")
    print(f"Target Opset:   {args.opset}")
    print(f"Target IR Ver:  {args.ir}")
    print(f"Test Inference: {'Yes' if args.test_inference else 'No'}")
    print("-" * 50)

    patch_model_main(input_path, output_path, args.opset, args.ir, args.test_inference)

if __name__ == "__main__":
    main()