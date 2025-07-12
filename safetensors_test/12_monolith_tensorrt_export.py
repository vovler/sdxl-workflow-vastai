#!/usr/bin/env python3
import tensorrt as trt
import torch
import argparse
import os
import sys
from collections import OrderedDict
from tqdm import tqdm

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

def build_engine(
    engine_path: str,
    onnx_path: str,
    input_profiles: dict,
    fp16: bool = True,
    workspace_size: int = 10240, # in MB (VRAM)
    timing_cache_path: str = None
):
    """Builds a TensorRT engine from an ONNX model, following a standardized configuration."""
    
    logger = trt.Logger(trt.Logger.INFO)
    logger.min_severity = trt.Logger.Severity.INFO

    print("="*50)
    print(f"Exporting ONNX to TensorRT Engine")
    print(f"  ONNX Path: {onnx_path}")
    print(f"  Engine Path: {engine_path}")
    print(f"  FP16: {fp16}")
    print("="*50)

    builder = trt.Builder(logger)
    
    if hasattr(builder, 'max_threads'):
        builder.max_threads = torch.get_num_threads()

    config = builder.create_builder_config()
    
    # --- Apply Builder Config ---
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * 1024 * 1024)
    
    # Apply settings based on reference scripts
    config.set_flag(trt.BuilderFlag.FP16) # Ensure FP16 is enabled by default
    config.set_flag(trt.BuilderFlag.INT8)
    config.builder_optimization_level = 4
    config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.NONE
    config.tiling_optimization_level = trt.TilingOptimizationLevel.NONE
    
    print(f"Builder Optimization Level: {config.builder_optimization_level}")
    print(f"Hardware Compatibility Level: {config.hardware_compatibility_level}")
    print(f"Tiling Optimization Level: {config.tiling_optimization_level}")

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

    if not parser.parse_from_file(onnx_path):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise ValueError(f"Failed to parse ONNX file: {onnx_path}")

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

    print(f"Writing TensorRT engine to: {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(plan)
    
    print("✓ TensorRT engine exported successfully.")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Export the monolithic ONNX model to a TensorRT engine.")
    parser.add_argument("--onnx_path", type=str, default="onnx/monolith_opt.onnx", help="Path to the input ONNX model file.")
    parser.add_argument("--engine_path", type=str, default="plan/monolith.plan", help="Path to save the output TensorRT engine.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mode for the engine (enabled by default).")
    parser.add_argument("--steps", type=int, default=8, help="Number of inference steps used in the model.")
    parser.add_argument("--timing_cache_path", type=str, default="monolith.cache", help="Path to save and load the timing cache.")
    args = parser.parse_args()

    # Define the optimization profile. We'll start with a static profile for 
    # batch size 1 and the optimal resolution to simplify the initial conversion.
    batch_size = 1
    height = 832
    width = 1216
    height_div_8 = height // 8
    width_div_8 = width // 8
    num_inference_steps = args.steps
    
    input_profiles = OrderedDict([
        ("prompt_ids_1", {
            "min": (batch_size, 77),
            "opt": (batch_size, 77),
            "max": (batch_size, 77),
        }),
        ("prompt_ids_2", {
            "min": (batch_size, 77),
            "opt": (batch_size, 77),
            "max": (batch_size, 77),
        }),
        ("initial_latents", {
            "min": (batch_size, 4, height_div_8, width_div_8),
            "opt": (batch_size, 4, height_div_8, width_div_8),
            "max": (batch_size, 4, height_div_8, width_div_8),
        }),
        ("all_noises", {
            "min": (num_inference_steps, batch_size, 4, height_div_8, width_div_8),
            "opt": (num_inference_steps, batch_size, 4, height_div_8, width_div_8),
            "max": (num_inference_steps, batch_size, 4, height_div_8, width_div_8),
        }),
        ("add_time_ids", {
            "min": (batch_size, 6),
            "opt": (batch_size, 6),
            "max": (batch_size, 6),
        }),
    ])
    
    # Following the pattern of the other scripts, we will build with FP16 by default.
    build_engine(
        engine_path=args.engine_path,
        onnx_path=args.onnx_path,
        input_profiles=input_profiles,
        fp16=True, 
        timing_cache_path=args.timing_cache_path,
    )


if __name__ == "__main__":
    main() 