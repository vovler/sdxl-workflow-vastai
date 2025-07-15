import torch
import torch.nn as nn
import onnx
import tensorrt as trt
import numpy as np
import os
from tqdm import tqdm
from collections import OrderedDict

# Simple model with a loop
class SimpleLoop(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        
    def forward(self, x: torch.Tensor, num_steps: int) -> torch.Tensor:
        for i in range(num_steps):
            x = self.linear(x) + 1.0
        return x

# Regular model with loop (non-scripted)
class RegularLoop(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        
    def forward(self, x, num_steps):
        for i in range(num_steps):
            x = self.linear(x) + 1.0
        return x

# Progress bar for TensorRT engine building
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
                nbIndents = (
                    self._active_phases.get(parent_phase, {}).get(
                        "nbIndents", self.max_indent
                    )
                    + 1
                )
                if nbIndents >= self.max_indent:
                    return
            else:
                nbIndents = 0
                leave = True
            self._active_phases[phase_name] = {
                "tq": tqdm(
                    total=num_steps, desc=phase_name, leave=leave, position=nbIndents
                ),
                "nbIndents": nbIndents,
                "parent_phase": parent_phase,
            }
        except KeyboardInterrupt:
            self._step_result = False

    def phase_finish(self, phase_name):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    self._active_phases[phase_name]["tq"].total
                    - self._active_phases[phase_name]["tq"].n
                )
                parent_phase = self._active_phases[phase_name].get("parent_phase", None)
                while parent_phase is not None:
                    self._active_phases[parent_phase]["tq"].refresh()
                    parent_phase = self._active_phases[parent_phase].get(
                        "parent_phase", None
                    )
                if (
                    self._active_phases[phase_name]["parent_phase"]
                    in self._active_phases.keys()
                ):
                    self._active_phases[
                        self._active_phases[phase_name]["parent_phase"]
                    ]["tq"].refresh()
                del self._active_phases[phase_name]
            pass
        except KeyboardInterrupt:
            self._step_result = False

    def step_complete(self, phase_name, step):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    step - self._active_phases[phase_name]["tq"].n
                )
            return self._step_result
        except KeyboardInterrupt:
            return False

# Test both versions
def test_exports():
    # Create models
    regular_model = SimpleLoop()
    scripted_model = torch.jit.script(regular_model)  # Script the instance
    
    # Sample input
    x = torch.randn(1, 10)
    num_steps = 5
    
    print("Testing TorchScript version:")
    try:
        torch.onnx.export(
            scripted_model,
            (x, num_steps),
            "scripted_loop.onnx",
            input_names=['x', 'num_steps'],
            output_names=['output'],
            dynamic_axes={'x': {0: 'batch_size'}},
            opset_version=11
        )
        print("✅ Scripted model exported successfully")
    except Exception as e:
        print(f"❌ Scripted model failed: {e}")
    
    print("\nTesting regular model:")
    try:
        torch.onnx.export(
            regular_model,
            (x, num_steps),
            "regular_loop.onnx", 
            input_names=['x', 'num_steps'],
            output_names=['output'],
            dynamic_axes={'x': {0: 'batch_size'}},
            opset_version=11
        )
        print("✅ Regular model exported successfully")
    except Exception as e:
        print(f"❌ Regular model failed: {e}")
    
def build_tensorrt_engine(
    onnx_file: str,
    engine_file: str,
    input_profiles: dict,
    fp16: bool = True,
    timing_cache_path: str = None
):
    """Builds a TensorRT engine from an ONNX model, following a standardized configuration."""
    
    logger = trt.Logger(trt.Logger.INFO)
    logger.min_severity = trt.Logger.Severity.INFO

    print("="*50)
    print(f"Exporting ONNX to TensorRT Engine")
    print(f"  ONNX Path: {onnx_file}")
    print(f"  Engine Path: {engine_file}")
    print(f"  FP16: {fp16}")
    print("="*50)

    builder = trt.Builder(logger)
    builder.max_threads = torch.get_num_threads()
    config = builder.create_builder_config()
    
    # Apply settings based on reference scripts
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # --- Timing Cache ---
    if timing_cache_path:
        if os.path.exists(timing_cache_path):
            print(f"Loading timing cache from: {timing_cache_path}")
            with open(timing_cache_path, "rb") as f:
                cache_data = f.read()
            timing_cache = config.create_timing_cache(cache_data)
        else:
            print("Creating a new timing cache.")
            timing_cache = config.create_timing_cache(b"")
        
        if timing_cache:
            config.set_timing_cache(timing_cache, ignore_mismatch=False)

    # --- Create Profile ---
    profile = builder.create_optimization_profile()
    for name, dims in input_profiles.items():
        min_shape, opt_shape, max_shape = dims['min'], dims['opt'], dims['max']
        print(f"  Setting profile for input: {name} with min={min_shape}, opt={opt_shape}, max={max_shape}")
        profile.set_shape(name, min=min_shape, opt=opt_shape, max=max_shape)
    
    config.add_optimization_profile(profile)
    config.progress_monitor = TQDMProgressMonitor()

    # --- Create Network & Parse ONNX ---
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(onnx_file):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise ValueError(f"Failed to parse ONNX file: {onnx_file}")

    # --- Build and Save Engine ---
    print("Building TensorRT engine. This may take a while...")
    plan = builder.build_serialized_network(network, config)
    if not plan:
        raise RuntimeError("Failed to build TensorRT engine.")

    # Save the timing cache
    if timing_cache_path:
        new_timing_cache = config.get_timing_cache()
        if new_timing_cache:
            with open(timing_cache_path, "wb") as f:
                f.write(new_timing_cache.serialize())
            print(f"Timing cache saved to: {timing_cache_path}")

    print(f"Writing TensorRT engine to: {engine_file}")
    with open(engine_file, "wb") as f:
        f.write(plan)
    
    print("✓ TensorRT engine exported successfully.")
    print("="*50)
    
    # Return path to engine file
    return engine_file

def test_tensorrt_engines():
    """Test TensorRT engines"""
    print("\n" + "="*50)
    print("TENSORRT ENGINE BUILDING")
    print("="*50)

    input_profiles = OrderedDict([
        ("x", {
            "min": (1, 10),
            "opt": (4, 10),
            "max": (16, 10),
        }),
        ("num_steps", {
            "min": (1,),
            "opt": (1,),
            "max": (1,),
        }),
    ])
    
    scripted_engine_path = None
    regular_engine_path = None

    # Build engines
    try:
        scripted_engine_path = build_tensorrt_engine(
            "scripted_loop.onnx", 
            "scripted_loop.trt",
            input_profiles=input_profiles,
            timing_cache_path="scripted.cache"
        )
    except Exception as e:
        print(f"❌ Scripted TensorRT engine build failed: {e}")

    try:
        regular_engine_path = build_tensorrt_engine(
            "regular_loop.onnx", 
            "regular_loop.trt",
            input_profiles=input_profiles,
            timing_cache_path="regular.cache"
        )
    except Exception as e:
        print(f"❌ Regular TensorRT engine build failed: {e}")

    # Analyze engine details
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    if scripted_engine_path and os.path.exists(scripted_engine_path):
        print("\n--- Analysis of Scripted Engine ---")
        with open(scripted_engine_path, 'rb') as f:
            scripted_engine = runtime.deserialize_cuda_engine(f.read())
        
        if scripted_engine:
            print("✅ Scripted TensorRT engine loaded successfully")
            print(f"   Engine size: {os.path.getsize(scripted_engine_path)} bytes")
            print(f"   Engine layers: {scripted_engine.num_layers}")
            print("   Layer info:")
            for i in range(scripted_engine.num_layers):
                layer = scripted_engine.get_layer(i)
                print(f"     Layer {i}: {layer.name} | Type: {layer.type}")
    else:
        print("\n❌ Scripted TensorRT engine not found or failed to build.")
    
    if regular_engine_path and os.path.exists(regular_engine_path):
        print("\n--- Analysis of Regular Engine ---")
        with open(regular_engine_path, 'rb') as f:
            regular_engine = runtime.deserialize_cuda_engine(f.read())

        if regular_engine:
            print("✅ Regular TensorRT engine loaded successfully")
            print(f"   Engine size: {os.path.getsize(regular_engine_path)} bytes")
            print(f"   Regular engine layers: {regular_engine.num_layers}")
            print("   Layer info:")
            for i in range(regular_engine.num_layers):
                layer = regular_engine.get_layer(i)
                print(f"     Layer {i}: {layer.name} | Type: {layer.type}")
    else:
        print("\n❌ Regular TensorRT engine not found or failed to build.")

if __name__ == "__main__":
    os.makedirs("onnx", exist_ok=True)
    test_exports()
    test_tensorrt_engines()