import torch
import torch.nn as nn
import onnx
import tensorrt as trt
import numpy as np
import os
import json
from tqdm import tqdm
from collections import OrderedDict

# Corrected model that uses a Tensor for loop control
class ComplexLoop(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_add = nn.Linear(10, 10)
        self.linear_sub = nn.Linear(10, 10)

    def forward(self, x: torch.Tensor, num_steps: torch.Tensor) -> torch.Tensor:
        # Loop from 0 to the number of elements in the num_steps tensor.
        # This pattern is convertible to an ONNX Loop operator.
        for i in range(num_steps.size(0)):
            # Get the step value for the current iteration
            #step_val = num_steps[i]
            x = self.linear_add(x) + i
        return x

# Progress bar for TensorRT engine building (unchanged)
class TQDMProgressMonitor(trt.IProgressMonitor):
    def __init__(self):
        trt.IProgressMonitor.__init__(self)
        self._active_phases = {}
        self._step_result = True
        self.max_indent = 5

    def phase_start(self, phase_name, parent_phase, num_steps):
        leave = False
        try:
            if parent_phase is not None:
                nbIndents = (self._active_phases.get(parent_phase, {}).get("nbIndents", self.max_indent) + 1)
                if nbIndents >= self.max_indent: return
            else:
                nbIndents = 0
                leave = True
            self._active_phases[phase_name] = {
                "tq": tqdm(total=num_steps, desc=phase_name, leave=leave, position=nbIndents),
                "nbIndents": nbIndents,
                "parent_phase": parent_phase,
            }
        except KeyboardInterrupt:
            self._step_result = False

    def phase_finish(self, phase_name):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    self._active_phases[phase_name]["tq"].total - self._active_phases[phase_name]["tq"].n
                )
                parent_phase = self._active_phases[phase_name].get("parent_phase", None)
                while parent_phase is not None:
                    self._active_phases[parent_phase]["tq"].refresh()
                    parent_phase = self._active_phases[parent_phase].get("parent_phase", None)
                if self._active_phases[phase_name]["parent_phase"] in self._active_phases.keys():
                    self._active_phases[self._active_phases[phase_name]["parent_phase"]]["tq"].refresh()
                del self._active_phases[phase_name]
        except KeyboardInterrupt:
            self._step_result = False

    def step_complete(self, phase_name, step):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(step - self._active_phases[phase_name]["tq"].n)
            return self._step_result
        except KeyboardInterrupt:
            return False

def test_exports():
    """Tests the ONNX export for the corrected model."""
    scripted_model = torch.jit.script(ComplexLoop())
    
    # Sample inputs are now all tensors
    x = torch.randn(1, 10)
    # Convert list to a tensor. This will be a dynamic input.
    num_steps = torch.tensor(list(range(0, 5, 1)), dtype=torch.int32)
    use_add = torch.tensor(True, dtype=torch.bool)
    
    # Define output path
    onnx_path = "scripted_loop.onnx"
    
    print("Testing TorchScript to ONNX export:")
    try:
        torch.onnx.export(
            scripted_model,
            (x, num_steps),
            onnx_path,
            input_names=['x', 'num_steps'],
            output_names=['output'],
            dynamic_axes={
                'x': {0: 'batch_size'},
                # Mark dimension 0 of num_steps as dynamic
                'num_steps': {0: 'num_loop_steps'}
            },
            opset_version=13  # Opset 13+ is recommended for better control flow support
        )
        print(f"✅ Model exported successfully to {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"❌ Model export failed: {e}")
        return None

def build_tensorrt_engine(
    onnx_file: str,
    engine_file: str,
    input_profiles: dict,
    fp16: bool = True,
    timing_cache_path: str | None = None
):
    """Builds a TensorRT engine from an ONNX model."""
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print("="*50)
    print(f"Building TensorRT Engine from: {onnx_file}")
    
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # --- Create Profile ---
    profile = builder.create_optimization_profile()
    for name, dims in input_profiles.items():
        min_shape, opt_shape, max_shape = dims['min'], dims['opt'], dims['max']
        print(f"  Setting profile for '{name}': min={min_shape}, opt={opt_shape}, max={max_shape}")
        profile.set_shape(name, min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)

    # --- Parse ONNX ---
    if not os.path.exists(onnx_file):
        raise FileNotFoundError(f"ONNX file not found: {onnx_file}")
        
    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("Failed to parse the ONNX file.")

    # --- Build and Save Engine ---
    print("Building engine... (This may take a moment)")
    serialized_engine = builder.build_serialized_network(network, config)
    if not serialized_engine:
        raise RuntimeError("Failed to build the TensorRT engine.")

    with open(engine_file, "wb") as f:
        f.write(serialized_engine)
    
    print(f"✅ TensorRT engine saved to: {engine_file}")
    print("="*50)
    return engine_file

def test_tensorrt_engines(onnx_path: str):
    """Builds and analyzes the TensorRT engine."""
    if not onnx_path:
        print("Skipping TensorRT build because ONNX export failed.")
        return

    print("\n" + "="*50)
    print("TENSORRT ENGINE BUILDING")
    print("="*50)

    # Corrected profile for a dynamic 1D tensor `num_steps`
    input_profiles = OrderedDict([
        ("x", {
            "min": (1, 10),
            "opt": (4, 10),
            "max": (16, 10),
        }),
        # Profile for num_steps allowing 1 to 20 loop iterations
        ("num_steps", {
            "min": (1,),
            "opt": (5,),
            "max": (20,),
        }),
    ])
    
    engine_path = "scripted_loop.engine"
    try:
        build_tensorrt_engine(
            onnx_path, 
            engine_path,
            input_profiles=input_profiles
        )
        print(f"\nEngine analysis for {engine_path}:")
        # You can add the engine inspection logic here if needed
        
    except Exception as e:
        print(f"❌ TensorRT engine build failed: {e}")

if __name__ == "__main__":
    onnx_file_path = test_exports()
    test_tensorrt_engines(onnx_file_path)