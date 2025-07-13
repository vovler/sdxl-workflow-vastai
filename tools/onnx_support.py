import inspect
from typing import get_type_hints

import torch
from torch.onnx import _constants
from torch.onnx._internal import registration

def analyze_onnx_supported_ops_robust():
    """
    Analyzes and prints detailed information about all PyTorch operators
    with registered ONNX symbolic conversion functions.
    This version is more robust to changes in PyTorch's internal APIs.
    """
    print("--- PyTorch ONNX Exporter Supported Operators (Robust Analysis) ---")
    print("This list shows PyTorch operators that have a direct conversion rule to ONNX.\n")

    all_registered_ops = sorted(registration.registry.all_functions())
    total_ops = len(all_registered_ops)
    print(f"Found {total_ops} registered ONNX conversion rules.\n")
    print("-" * 80)

    for i, op_name in enumerate(all_registered_ops):
        try:
            func_group = registration.registry.get_function_group(op_name)
            if not func_group:
                continue

            # --- ROBUST FIX STARTS HERE ---
            # Instead of relying on get_min/max_supported(), we inspect the internal
            # dictionary that holds the opset -> function mapping.
            if not hasattr(func_group, "_functions") or not func_group._functions:
                # Skip if the group is empty or doesn't have the expected structure
                continue
            
            supported_opsets = list(func_group._functions.keys())
            min_opset = min(supported_opsets)
            max_opset = max(supported_opsets)

            # Get the symbolic function for the highest available opset in the group
            symbolic_func = func_group.get(max_opset)
            # --- ROBUST FIX ENDS HERE ---
            
            if symbolic_func is None:
                # This case should be rare with the new logic but is kept for safety
                print(f"{i+1:04d}/{total_ops}: {op_name} (Could not retrieve symbolic function)")
                continue

            sig = inspect.signature(symbolic_func)
            params = sig.parameters
            
            arg_list = []
            for name, param in params.items():
                if name in {"g", "_outputs"}:
                    continue
                arg_list.append(str(param))

            args_str = ", ".join(arg_list)

            print(f"{i+1:04d}/{total_ops}: {op_name}")
            print(f"    Signature: symbolic({args_str})")
            print(f"    Opset Range: {min_opset} - {max_opset}")
            print("-" * 80)

        except Exception as e:
            # Provide a more informative error message for debugging
            print(f"{i+1:04d}/{total_ops}: {op_name} -> FAILED TO ANALYZE: {type(e).__name__}: {e}")
            print("-" * 80)

if __name__ == "__main__":
    analyze_onnx_supported_ops_robust()