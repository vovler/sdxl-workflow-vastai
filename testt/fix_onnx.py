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

# --- Step 1: Detection Function (unchanged) ---
def detect_loops_concat_scan_to_native_scan(model):
    # This function is still correct as it identifies the right pattern
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


# --- Step 2: Processing Function (Updated) ---
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

        # ### NEW CHANGE ###: Identify and remove the condition-related nodes/IOs.
        # The condition output is always the first output of the body.
        condition_body_output_name = body_graph.output[0].name
        condition_producer_node = find_node_by_output(body_graph, condition_body_output_name)
        
        # The condition input is always the second input of the body.
        condition_body_input_name = body_graph.input[1].name

        # Replace the Concat node with an Identity node to rename the scan tensor.
        renamer_identity_node = helper.make_node(
            'Identity',
            inputs=[details['per_iteration_tensor_name']],
            outputs=[details['state_body_output_name']],
            name=details['concat_node'].name + "_renamer"
        )
        
        # Build the new list of nodes for the body graph
        new_body_nodes = []
        for n in body_graph.node:
            if n.name == details["concat_node"].name:
                new_body_nodes.append(renamer_identity_node)
            # Remove the node that produces the loop condition (e.g., an Identity node)
            elif condition_producer_node and n.name == condition_producer_node.name:
                continue
            else:
                new_body_nodes.append(n)
        
        # Build new body inputs, removing state and condition inputs
        new_body_inputs = [
            inp for inp in body_graph.input 
            if inp.name not in [details["state_body_input_name"], condition_body_input_name]
        ]
        
        # Build new body outputs, removing the condition output
        new_body_outputs = [
            out for out in body_graph.output 
            if out.name != condition_body_output_name
        ]
        
        new_body_graph = helper.make_graph(
            nodes=new_body_nodes, name=body_graph.name + "_transformed",
            inputs=new_body_inputs, outputs=new_body_outputs,
            initializer=body_graph.initializer, value_info=body_graph.value_info
        )

        # --- B. Modify the main Loop node ---
        
        # ### NEW CHANGE ###: Remove both the condition and the state inputs from the loop.
        # The loop inputs are: M, cond, loop-carried-states...
        # We need to remove input 1 (cond) and the state input.
        main_loop_state_input_name = loop_node.input[details["state_body_input_idx"]]
        main_loop_cond_input_name = loop_node.input[1]
        
        new_loop_inputs = [loop_node.input[0]] # Start with trip count (M)
        new_loop_inputs.append("") # Add an empty string for the optional 'cond' input
        # Add the remaining inputs, skipping the one we are removing
        for i in range(2, len(loop_node.input)):
            if loop_node.input[i] != main_loop_state_input_name:
                new_loop_inputs.append(loop_node.input[i])
        
        # The loop now only has one output: the scan output.
        # The index in the original output list is (state_body_output_idx - 1)
        main_loop_output_idx = details["state_body_output_idx"] - 1
        new_loop_outputs = [loop_node.output[main_loop_output_idx]]
        
        new_loop_node = helper.make_node('Loop', inputs=new_loop_inputs, outputs=new_loop_outputs, name=loop_node.name)
        new_loop_node.attribute.append(helper.make_attribute("body", new_body_graph))
        nodes_to_add.append(new_loop_node)

        # --- C. Mark initial state and condition providers for removal ---
        for input_name_to_remove in [main_loop_state_input_name, main_loop_cond_input_name]:
            producer = find_node_by_output(graph, input_name_to_remove)
            if producer:
                nodes_to_remove.add(producer.name)
            elif find_initializer_by_name(graph, input_name_to_remove):
                initializers_to_remove.add(input_name_to_remove)

    # Reconstruct the main graph
    final_nodes = [node for node in graph.node if node.name not in nodes_to_remove]
    final_nodes.extend(nodes_to_add)
    final_initializers = [init for init in graph.initializer if init.name not in initializers_to_remove]
    
    # Remove unused inputs from the graph if they are no longer needed
    all_used_inputs = {inp for n in final_nodes for inp in n.input}
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
    # (This function remains unchanged)
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
    # (This function remains unchanged)
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
    
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_patched{ext}"
    
    print("-" * 50)
    print(f"Input model:  {input_path}")
    print(f"Output model: {output_path}")
    print("-" * 50)

    patch_loops_output(input_path, output_path)

if __name__ == "__main__":
    main()