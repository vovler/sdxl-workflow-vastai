import onnx
from onnx import helper, checker, shape_inference
import argparse
import sys
import os

# --- Helper Functions (unchanged) ---
def find_node_by_output(graph, output_name):
    for node in graph.node:
        if output_name in node.output:
            return node
    return None

def find_initializer_by_name(graph, name):
    for init in graph.initializer:
        if init.name == name:
            return init
    return None

def find_value_info(graph, name):
    for vi in graph.value_info:
        if vi.name == name: return vi
    for vi in graph.input:
        if vi.name == name: return vi
    for vi in graph.output:
        if vi.name == name: return vi
    for node in graph.node:
        if node.op_type == "Loop":
            for attr in node.attribute:
                if attr.name == "body":
                    for bvi in attr.g.value_info:
                        if bvi.name == name: return bvi
                    for binp in attr.g.input:
                        if binp.name == name: return binp
    return None

# --- Detection Function (unchanged) ---
def detect_loops_concat_scan_to_native_scan(model):
    graph = model.graph
    loops_to_transform = []
    for node in graph.node:
        if node.op_type != "Loop": continue
        body_graph_attr = [attr for attr in node.attribute if attr.name == "body"]
        if not body_graph_attr: continue
        body_graph = body_graph_attr[0].g
        for body_output_idx, body_output in enumerate(body_graph.output):
            if body_output_idx == 0: continue
            producer_node = find_node_by_output(body_graph, body_output.name)
            if not (producer_node and producer_node.op_type == "Concat"): continue
            body_inputs_names = [inp.name for inp in body_graph.input]
            found_pattern_for_this_output = False
            for body_input_idx, body_input_name in enumerate(body_inputs_names):
                if body_input_name in producer_node.input:
                    per_iteration_tensor_name = [inp for inp in producer_node.input if inp != body_input_name][0]
                    transformation_details = {
                        "loop_node": node, "loop_node_name": node.name, "concat_node": producer_node,
                        "state_body_input_name": body_input_name, "state_body_input_idx": body_input_idx,
                        "state_body_output_name": body_output.name, "state_body_output_idx": body_output_idx,
                        "per_iteration_tensor_name": per_iteration_tensor_name,
                    }
                    loops_to_transform.append(transformation_details)
                    found_pattern_for_this_output = True
                    break
            if found_pattern_for_this_output: break
    return loops_to_transform


# --- Processing Function (Corrected) ---
def process_loops_concat_scan_to_native_scan(model, loops_to_transform):
    if not loops_to_transform:
        return model

    graph = model.graph
    nodes_to_remove = set()
    nodes_to_add = []
    initializers_to_remove = set()

    for details in loops_to_transform:
        loop_node = details["loop_node"]
        nodes_to_remove.add(loop_node.name)
        body_graph = next(attr.g for attr in loop_node.attribute if attr.name == "body")

        # --- A. Modify the body graph ---

        # ### CHANGE ###: This is the robust fix.
        # We will completely replace the old condition-producing node with a new, self-contained Constant.

        # 1. Identify the name of the body's first output (the one to be replaced by a dummy).
        dummy_cond_output_name = body_graph.output[0].name

        # 2. Identify the node that originally produced it, so we can remove it from the list.
        condition_producer_node_to_remove = find_node_by_output(body_graph, dummy_cond_output_name)

        # 3. Create a new Constant(true) that directly produces the required output name.
        dummy_cond_const_node = helper.make_node(
            'Constant',
            inputs=[],
            outputs=[dummy_cond_output_name],
            name=dummy_cond_output_name.replace('/', '_').strip('_') + "_dummy_const",
            value=helper.make_tensor(name='value', data_type=onnx.TensorProto.BOOL, dims=(), vals=[True])
        )

        # 4. Replace the Concat node with an Identity node to rename the scan tensor.
        renamer_identity_node = helper.make_node(
            'Identity',
            inputs=[details['per_iteration_tensor_name']],
            outputs=[details['state_body_output_name']],
            name=details['concat_node'].name + "_renamer"
        )
        
        # 5. Build the new list of nodes for the body graph.
        new_body_nodes = [dummy_cond_const_node] # Start with our new dummy constant for topological sort.
        for n in body_graph.node:
            if n.name == details["concat_node"].name:
                new_body_nodes.append(renamer_identity_node)
            # Skip the original node that produced the condition
            elif condition_producer_node_to_remove and n.name == condition_producer_node_to_remove.name:
                continue
            else:
                new_body_nodes.append(n)
        
        # 6. Remove the now-unused body inputs (state and external condition).
        condition_body_input_name = body_graph.input[1].name
        new_body_inputs = [
            inp for inp in body_graph.input 
            if inp.name not in [details["state_body_input_name"], condition_body_input_name]
        ]
        
        # The body output list is now correct, as the new nodes produce the expected names.
        new_body_outputs = body_graph.output
        
        new_body_graph = helper.make_graph(
            nodes=new_body_nodes, name=body_graph.name + "_transformed",
            inputs=new_body_inputs, outputs=new_body_outputs,
            initializer=body_graph.initializer, value_info=body_graph.value_info
        )

        # --- B. Modify the main Loop node ---
        
        main_loop_state_input_name = loop_node.input[details["state_body_input_idx"]]
        main_loop_cond_input_name = loop_node.input[1]
        
        new_loop_inputs = list(loop_node.input)
        new_loop_inputs[1] = "" # Set condition to empty to signal a pure counted loop
        new_loop_inputs = [inp for inp in new_loop_inputs if inp != main_loop_state_input_name]
        
        main_loop_output_idx = details["state_body_output_idx"] - 1
        new_loop_outputs = [loop_node.output[main_loop_output_idx]]
        
        new_loop_node = helper.make_node('Loop', inputs=new_loop_inputs, outputs=new_loop_outputs, name=loop_node.name)
        new_loop_node.attribute.append(helper.make_attribute("body", new_body_graph))
        nodes_to_add.append(new_loop_node)

        # --- C. Mark initial state and condition providers for removal ---
        for input_name_to_remove in [main_loop_state_input_name, main_loop_cond_input_name]:
            if not input_name_to_remove: continue
            producer = find_node_by_output(graph, input_name_to_remove)
            if producer:
                nodes_to_remove.add(producer.name)
            elif find_initializer_by_name(graph, input_name_to_remove):
                initializers_to_remove.add(input_name_to_remove)

    # Reconstruct the main graph
    final_nodes = [node for node in graph.node if node.name not in nodes_to_remove]
    final_nodes.extend(nodes_to_add)
    final_initializers = [init for init in graph.initializer if init.name not in initializers_to_remove]
    
    all_used_inputs = {inp for n in final_nodes for inp in n.input if inp}
    final_graph_inputs = [inp for inp in graph.input if inp.name in all_used_inputs]

    new_graph = helper.make_graph(
        nodes=final_nodes, name=graph.name + "_transformed",
        inputs=final_graph_inputs, outputs=graph.output,
        initializer=final_initializers, value_info=graph.value_info
    )

    new_model = helper.make_model(new_graph, producer_name='onnx-loop-transformer')
    new_model.opset_import.extend(model.opset_import)
    return new_model


# --- Main Orchestration and Saving Logic (unchanged) ---
def patch_loops_output(input_onnx_path, output_onnx_path):
    print(f"Loading model from: {input_onnx_path}")
    try:
        model = onnx.load(input_onnx_path)
    except Exception as e:
        print(f"ERROR: Failed to load ONNX model. {e}", file=sys.stderr)
        return

    transform_list = detect_loops_concat_scan_to_native_scan(model)

    if not transform_list:
        print("INFO: No loops with the target Concat-scan pattern were found. Model is unchanged.")
        return

    print("\nFound the following loops to transform:")
    for details in transform_list:
        print(f"  - Loop Node: '{details['loop_node_name']}'")
        print(f"    - Inner Concat Node: '{details['concat_node'].name}'")
        print(f"    - Per-iteration tensor: '{details['per_iteration_tensor_name']}'\n")

    print("Processing model...")
    transformed_model = process_loops_concat_scan_to_native_scan(model, transform_list)
    if not transformed_model:
        print("ERROR: Transformation failed during processing.", file=sys.stderr)
        return

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
        sys.exit(1)
    
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_patched{ext}"
    
    print("-" * 50)
    print(f"Input model:  {input_path}")
    print(f"Output model: {output_path}")
    print("-" * 50)

    patch_loops_output(input_path, output_path)

if __name__ == "__main__":
    main()