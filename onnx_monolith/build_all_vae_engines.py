import tensorrt as trt
import torch
import argparse
import os
from collections import OrderedDict
from tqdm import tqdm
from typing import Optional

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
    timing_cache_path: Optional[str] = None
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
    builder.max_threads = torch.get_num_threads()
    config = builder.create_builder_config()
    
    # --- Apply Builder Config ---
    config.set_flag(trt.BuilderFlag.FP16) # Ensure FP16 is enabled by default
    config.builder_optimization_level = 4
    config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY
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
    parser = argparse.ArgumentParser(description="Export VAE ONNX models to TensorRT engines.")
    parser.add_argument("--onnx_dir", type=str, default=".", help="Directory containing the input ONNX model files.")
    parser.add_argument("--engine_dir", type=str, default="engine", help="Directory to save the output TensorRT engines.")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 mode for the engine (enabled by default).")
    parser.add_argument("--timing_cache_dir", type=str, default="cache", help="Directory to save and load timing caches.")
    args = parser.parse_args()

    # Create the output directories if they don't exist
    os.makedirs(args.engine_dir, exist_ok=True)
    os.makedirs(args.timing_cache_dir, exist_ok=True)
    
    onnx_files = [
        "decoder_opt.onnx",
        "tiled_decoder_optimized_constant_folding.onnx",
        "tiled_decoder_optimized_onnxslim_full.onnx",
        "tiled_decoder_optimized_runtime_extended.onnx"
    ]

    batch_size = 1
    height = 832
    width = 1216
    height_div_8 = height // 8
    width_div_8 = width // 8
    
    input_profiles = OrderedDict([
        ("latent_sample", {
            "min": (batch_size, 4, height_div_8, width_div_8),
            "opt": (batch_size, 4, height_div_8, width_div_8),
            "max": (batch_size, 4, height_div_8, width_div_8),
        }),
    ])
    
    for onnx_file in onnx_files:
        onnx_path = os.path.join(args.onnx_dir, onnx_file)
        if not os.path.exists(onnx_path):
            print(f"ONNX file not found: {onnx_path}. Skipping.")
            continue
            
        engine_filename = os.path.splitext(onnx_file)[0] + ".engine"
        engine_path = os.path.join(args.engine_dir, engine_filename)
        
        if os.path.exists(engine_path):
            print(f"Engine file already exists, skipping: {engine_path}")
            continue

        cache_filename = os.path.splitext(onnx_file)[0] + ".cache"
        timing_cache_path = os.path.join(args.timing_cache_dir, cache_filename)

        build_engine(
            engine_path=engine_path,
            onnx_path=onnx_path,
            input_profiles=input_profiles,
            fp16=args.fp16,
            timing_cache_path=timing_cache_path,
        )


if __name__ == "__main__":
    main()