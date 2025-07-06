import os
import argparse
import torch
import tensorrt as trt
from collections import OrderedDict
from pipeline import defaults
from tqdm import tqdm

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

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
            # The phase_start callback cannot directly cancel the build, so request the cancellation from within step_complete.
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
            # There is no need to propagate this exception to TensorRT. We can simply cancel the build.
            return False

def get_abs_path(path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

def build_engine(
    engine_path: str,
    onnx_path: str,
    input_profiles: dict,
    fp16: bool = True,
):
    print(f"Building TensorRT engine for {onnx_path}: {engine_path}")
    
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    if os.path.exists(engine_path):
        print("Engine already exists, skipping build.")
        return

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    success = parser.parse_from_file(onnx_path)
    if not success:
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
        raise RuntimeError(f"Failed to parse ONNX file: {onnx_path}")

    config = builder.create_builder_config()
    config.builder_optimization_level = 5
    config.set_preview_feature(trt.PreviewFeature.RUNTIME_ACTIVATION_RESIZE_10_10, True)
    config.hardware_compatibility_level = (
        trt.HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY
    )
    config.tiling_optimization_level = trt.TilingOptimizationLevel.MODERATE
    #config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)

    profile = builder.create_optimization_profile()
    for name, (min_shape, opt_shape, max_shape) in input_profiles.items():
        profile.set_shape(name, min=min_shape, opt=opt_shape, max=max_shape)

    config.add_optimization_profile(profile)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.progress_monitor = TQDMProgressMonitor()

    print("Building engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine.")

    print("Engine built successfully.")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved to {engine_path}")

def get_engine_path(onnx_path: str):
    """
    Returns the TensorRT engine path from an ONNX model path.
    e.g. /path/to/model.onnx -> /path/to/model.plan
    """
    return os.path.splitext(onnx_path)[0] + ".plan"

def main():
    parser = argparse.ArgumentParser(
        description="Builds TensorRT engines for SDXL VAE and CLIP models."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/lab/model",
        help="Path to the model directory containing the ONNX files.",
    )
    args = parser.parse_args()
    model_path = args.model_path

    print("Building TensorRT engines for VAE and CLIP models...")

    # Define dynamic shapes for VAE
    latent_heights = [768 // 8, 1152 // 8, 960 // 8]  # 96, 144, 120
    latent_widths = [1152 // 8, 768 // 8, 960 // 8]   # 144, 96, 120
    min_h, max_h = min(latent_heights), max(latent_heights)
    min_w, max_w = min(latent_widths), max(latent_widths)
    opt_h, opt_w = 960 // 8, 960 // 8

    # Define components to build
    components = OrderedDict([
        ("VAE Decoder", {
            "subfolder": "vae_decoder",
            "profiles": {
                "latent_sample": (
                    (1, 4, min_h, min_w),
                    (1, 4, opt_h, opt_w),
                    (1, 4, max_h, max_w),
                ),
            }
        }),
        ("CLIP-L Text Encoder", {
            "subfolder": "text_encoder",
            "profiles": {
                "input_ids": ((1, 77), (1, 77), (1, 77)),
            }
        }),
        ("CLIP-G Text Encoder", {
            "subfolder": "text_encoder_2",
            "profiles": {
                "input_ids": ((1, 77), (1, 77), (1, 77)),
            }
        })
    ])

    for name, data in components.items():
        subfolder = data["subfolder"]
        profiles = data["profiles"]
        
        print(f"\n--- Building {name} Engine ---")
        onnx_path = os.path.join(model_path, subfolder, "model.onnx")
        engine_path = os.path.join(model_path, subfolder, "model.plan")

        if not os.path.exists(onnx_path):
            print(f"Error: ONNX file not found at {onnx_path}. Skipping.")
            continue
        
        try:
            build_engine(
                engine_path=engine_path,
                onnx_path=onnx_path,
                input_profiles=profiles,
            )
            print(f"✓ Successfully built engine for {name}")
            
            # Cleanup ONNX file
            print(f"Cleaning up ONNX file: {os.path.basename(onnx_path)}")
            try:
                os.remove(onnx_path)
                print(f"✓ Removed {os.path.basename(onnx_path)}")
            except OSError as e:
                print(f"✗ Error deleting ONNX file: {e}")

        except Exception as e:
            print(f"✗ Failed to build engine for {name}: {e}")

    print("\nAll specified TensorRT engines built.")

if __name__ == "__main__":
    main() 