import torch
import tensorrt as trt
import os
from tqdm import tqdm
from collections import OrderedDict
import json
import traceback

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

def build_tensorrt_engine(
    onnx_file: str,
    engine_file: str,
    input_profiles: dict,
    fp16: bool = True,
    timing_cache_path: str | None = None
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

def main():
    """Test TensorRT engines"""
    print("\n" + "="*50)
    print("TENSORRT ENGINE BUILDING")
    print("="*50)

    # Profile for VAE Decoder
    input_profiles = OrderedDict([
        ("latent_sample", {
            "min": (1, 4, 64, 64),
            "opt": (1, 4, 128, 128),
            "max": (2, 4, 128, 128),
        }),
    ])
    
    engine_path = None

    # Build engines
    try:
        engine_path = build_tensorrt_engine(
            "onnx/vae_decoder_dynamic_loop_opt.onnx", 
            "vae_decoder_dynamic_loop.trt",
            input_profiles=input_profiles,
            timing_cache_path="vae_decoder_dynamic_loop.cache"
        )
    except Exception as e:
        print(f"❌ TensorRT engine build failed: {e}")
        traceback.print_exc()

    # Analyze engine details
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    if engine_path and os.path.exists(engine_path):
        print("\n--- Analysis of Scripted Engine ---")
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine:
            print("✅ TensorRT engine loaded successfully")
            print(f"   Engine size: {os.path.getsize(engine_path)} bytes")
            
            inspector = engine.create_engine_inspector()
            engine_info_str = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
            engine_info = json.loads(engine_info_str)
            
            print(f"   Engine layers: {len(engine_info.get('Layers', []))}")
            
            with open("engine_info.json", 'w') as f:
                json.dump(engine_info, f, indent=2)
            print("   ✅ Engine info saved to engine_info.json")
    else:
        print("\n❌ TensorRT engine not found or failed to build.")
        

if __name__ == "__main__":
    os.makedirs("onnx", exist_ok=True)
    main() 