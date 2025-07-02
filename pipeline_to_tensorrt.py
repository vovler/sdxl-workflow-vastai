import os
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
    
    if os.path.exists(engine_path):
        print("Engine already exists, skipping build.")
        return

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        | (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
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
    """
    Builds TensorRT engines for all SDXL pipeline models.
    """
    print("Building TensorRT engines for SDXL pipeline...")
    device = torch.device("cuda:0")

    # UNet
    print("\n--- Building UNet Engine ---")
    latent_heights = [768 // 8, 1152 // 8, 960 // 8] # 96, 144, 120
    latent_widths = [1152 // 8, 768 // 8, 960 // 8]  # 144, 96, 120
    
    min_h, max_h = min(latent_heights), max(latent_heights)
    min_w, max_w = min(latent_widths), max(latent_widths)
    opt_h, opt_w = 960 // 8, 960 // 8

    unet_input_profiles = {
        "sample": (
            (1, 4, min_h, min_w),
            (1, 4, opt_h, opt_w),
            (1, 4, max_h, max_w),
        ),
        "timestep": ((), (), ()),
        "encoder_hidden_states": ((1, 77, 2048), (1, 77, 2048), (1, 77, 2048)),
        "text_embeds": ((1, 1280), (1, 1280), (1, 1280)),
        "time_ids": ((1, 6), (1, 6), (1, 6)),
    }
    build_engine(
        engine_path=get_engine_path(defaults.UNET_PATH),
        onnx_path=defaults.UNET_PATH,
        input_profiles=unet_input_profiles,
    )

    # VAE Decoder
    print("\n--- Building VAE Decoder Engine ---")
    vae_input_profiles = {
        "latent_sample": (
            (1, 4, min_h, min_w),
            (1, 4, opt_h, opt_w),
            (1, 4, max_h, max_w),
        ),
    }
    build_engine(
        engine_path=get_engine_path(defaults.VAE_DECODER_PATH),
        onnx_path=defaults.VAE_DECODER_PATH,
        input_profiles=vae_input_profiles,
    )

    # CLIP Text Encoder 1 (CLIP-L)
    print("\n--- Building CLIP-L Text Encoder Engine ---")
    clip_l_input_profiles = {
        "input_ids": ((1, 77), (1, 77), (1, 77)),
    }
    build_engine(
        engine_path=get_engine_path(defaults.CLIP_TEXT_ENCODER_1_PATH),
        onnx_path=defaults.CLIP_TEXT_ENCODER_1_PATH,
        input_profiles=clip_l_input_profiles,
    )

    # CLIP Text Encoder 2 (CLIP-G)
    print("\n--- Building CLIP-G Text Encoder Engine ---")
    clip_g_input_profiles = {
        "input_ids": ((1, 77), (1, 77), (1, 77)),
    }
    build_engine(
        engine_path=get_engine_path(defaults.CLIP_TEXT_ENCODER_2_PATH),
        onnx_path=defaults.CLIP_TEXT_ENCODER_2_PATH,
        input_profiles=clip_g_input_profiles,
    )

    print("\nAll TensorRT engines built successfully!")

if __name__ == "__main__":
    main() 