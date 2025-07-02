import os
import torch
import tensorrt as trt
from collections import OrderedDict
from pipeline import defaults
from tqdm import tqdm
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ----------------- INT8 Calibrator -----------------
# Based on: https://github.com/NVIDIA/TensorRT/blob/main/demo/BERT/python/bert_builder.py

class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file, batch_size=1):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.inputs = None
        self.device_inputs = []
        self.current_idx = 0
        self.total_samples = 0

    def set_inputs(self, inputs: list[torch.Tensor]):
        """
        Set the calibration inputs.
        Args:
            inputs: A list of PyTorch tensors.
        """
        self.inputs = inputs
        self.current_idx = 0
        self.total_samples = len(self.inputs[0]) if self.inputs else 0
        # Allocate GPU memory for the inputs
        self.device_inputs = [inp.cuda() for inp in self.inputs]

    def get_batch_size(self):
        """
        Get the batch size for calibration.
        """
        return self.batch_size

    def get_batch(self, names):
        """
        Get the next batch for calibration.
        Args:
            names: The names of the inputs.
        """
        if self.current_idx >= self.total_samples:
            return None

        end_idx = min(self.current_idx + self.batch_size, self.total_samples)
        batch = [inp[self.current_idx:end_idx].contiguous() for inp in self.device_inputs]
        
        self.current_idx += self.batch_size

        return [b.data_ptr() for b in batch]

    def read_calibration_cache(self):
        """
        Read the calibration cache.
        """
        if os.path.exists(self.cache_file):
            print(f"Reading calibration cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """
        Write the calibration cache.
        """
        print(f"Writing calibration cache: {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

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
    calibrator: EngineCalibrator,
):
    print(f"Building TensorRT engine for {onnx_path}: {engine_path}")
    
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
    #config.builder_optimization_level = 5
    #config.set_preview_feature(trt.PreviewFeature.RUNTIME_ACTIVATION_RESIZE_10_10, True)
    #config.hardware_compatibility_level = (
    #    trt.HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY
    #)
    #config.tiling_optimization_level = trt.TilingOptimizationLevel.MODERATE
    #config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)

    profile = builder.create_optimization_profile()
    for name, (min_shape, opt_shape, max_shape) in input_profiles.items():
        profile.set_shape(name, min=min_shape, opt=opt_shape, max=max_shape)

    config.add_optimization_profile(profile)

    # Configure for INT8 mode
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_calibration_profile(profile)
    config.int8_calibrator = calibrator

    config.progress_monitor = TQDMProgressMonitor()

    print("Building INT8 engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine.")

    print("Engine built successfully.")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved to {engine_path}")

def get_engine_path(onnx_path: str):
    """
    Returns the TensorRT INT8 engine path from an ONNX model path.
    e.g. /path/to/model.onnx -> /path/to/model_int8.plan
    """
    return os.path.splitext(onnx_path)[0] + "_int8.plan"

def main():
    """
    Builds INT8 TensorRT engines for all SDXL pipeline models.
    """
    print("Building INT8 TensorRT engines for SDXL pipeline...")
    device = torch.device("cuda:0")
    
    # Create a directory for calibration caches
    if not os.path.exists("calibration_cache"):
        os.makedirs("calibration_cache")
        
    # Calibration settings
    CALIBRATION_BATCH_SIZE = 1
    NUM_CALIBRATION_SAMPLES = 16

    # UNet
    print("\n--- Building UNet Engine (INT8) ---")
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
    
    # Generate calibration data for UNet
    unet_calib_data = [
        torch.randn(NUM_CALIBRATION_SAMPLES, 4, opt_h, opt_w, dtype=torch.float16),
        torch.randint(0, 1000, (NUM_CALIBRATION_SAMPLES,), dtype=torch.int64),
        torch.randn(NUM_CALIBRATION_SAMPLES, 77, 2048, dtype=torch.float16),
        torch.randn(NUM_CALIBRATION_SAMPLES, 1280, dtype=torch.float16),
        torch.randn(NUM_CALIBRATION_SAMPLES, 6, dtype=torch.float16),
    ]
    unet_calibrator = EngineCalibrator(
        "calibration_cache/unet_calib.cache", 
        batch_size=CALIBRATION_BATCH_SIZE
    )
    unet_calibrator.set_inputs(unet_calib_data)

    build_engine(
        engine_path=get_engine_path(defaults.UNET_PATH),
        onnx_path=defaults.UNET_PATH,
        input_profiles=unet_input_profiles,
        calibrator=unet_calibrator,
    )

    # VAE Decoder
    print("\n--- Building VAE Decoder Engine (INT8) ---")
    vae_input_profiles = {
        "latent_sample": (
            (1, 4, min_h, min_w),
            (1, 4, opt_h, opt_w),
            (1, 4, max_h, max_w),
        ),
    }

    # Generate calibration data for VAE Decoder
    vae_calib_data = [
        torch.randn(NUM_CALIBRATION_SAMPLES, 4, opt_h, opt_w, dtype=torch.float16),
    ]
    vae_calibrator = EngineCalibrator(
        "calibration_cache/vae_decoder_calib.cache",
        batch_size=CALIBRATION_BATCH_SIZE
    )
    vae_calibrator.set_inputs(vae_calib_data)

    build_engine(
        engine_path=get_engine_path(defaults.VAE_DECODER_PATH),
        onnx_path=defaults.VAE_DECODER_PATH,
        input_profiles=vae_input_profiles,
        calibrator=vae_calibrator,
    )

    # CLIP Text Encoder 1 (CLIP-L)
    print("\n--- Building CLIP-L Text Encoder Engine (INT8) ---")
    clip_l_input_profiles = {
        "input_ids": ((1, 77), (1, 77), (1, 77)),
    }
    
    # Generate calibration data for CLIP-L
    clip_l_calib_data = [
        torch.randint(0, 49408, (NUM_CALIBRATION_SAMPLES, 77), dtype=torch.int64),
    ]
    clip_l_calibrator = EngineCalibrator(
        "calibration_cache/clip_l_calib.cache",
        batch_size=CALIBRATION_BATCH_SIZE
    )
    clip_l_calibrator.set_inputs(clip_l_calib_data)
    
    build_engine(
        engine_path=get_engine_path(defaults.CLIP_TEXT_ENCODER_1_PATH),
        onnx_path=defaults.CLIP_TEXT_ENCODER_1_PATH,
        input_profiles=clip_l_input_profiles,
        calibrator=clip_l_calibrator,
    )

    # CLIP Text Encoder 2 (CLIP-G)
    print("\n--- Building CLIP-G Text Encoder Engine (INT8) ---")
    clip_g_input_profiles = {
        "input_ids": ((1, 77), (1, 77), (1, 77)),
    }
    
    # Generate calibration data for CLIP-G
    clip_g_calib_data = [
        torch.randint(0, 49408, (NUM_CALIBRATION_SAMPLES, 77), dtype=torch.int64),
    ]
    clip_g_calibrator = EngineCalibrator(
        "calibration_cache/clip_g_calib.cache",
        batch_size=CALIBRATION_BATCH_SIZE
    )
    clip_g_calibrator.set_inputs(clip_g_calib_data)
    
    build_engine(
        engine_path=get_engine_path(defaults.CLIP_TEXT_ENCODER_2_PATH),
        onnx_path=defaults.CLIP_TEXT_ENCODER_2_PATH,
        input_profiles=clip_g_input_profiles,
        calibrator=clip_g_calibrator,
    )

    print("\nAll TensorRT engines built successfully!")

if __name__ == "__main__":
    main() 