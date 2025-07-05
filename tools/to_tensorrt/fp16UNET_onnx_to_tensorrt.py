import os
import argparse
import tensorrt as trt
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

def get_default_unet_profiles():
    """Get the same UNet profiles used in the original script"""
    # Same values as in the original script
    latent_heights = [768 // 8, 1152 // 8, 960 // 8]  # [96, 144, 120]
    latent_widths = [1152 // 8, 768 // 8, 960 // 8]   # [144, 96, 120]
    
    min_h, max_h = min(latent_heights), max(latent_heights)  # 96, 144
    min_w, max_w = min(latent_widths), max(latent_widths)    # 96, 144
    opt_h, opt_w = 960 // 8, 960 // 8                       # 120, 120
    bs = 1

    return {
        "sample": (
            (bs, 4, min_h, min_w),
            (bs, 4, opt_h, opt_w),
            (bs, 4, max_h, max_w),
        ),
        "timestep": ((), (), ()),
        "encoder_hidden_states": ((bs, 77, 2048), (bs, 77, 2048), (bs, 77, 2048)),
        "text_embeds": ((bs, 1280), (bs, 1280), (bs, 1280)),
        "time_ids": ((bs, 6), (bs, 6), (bs, 6)),
    }

def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX UNet model to TensorRT engine with FP16"
    )
    parser.add_argument(
        "onnx_path",
        help="Path to input ONNX file"
    )

    args = parser.parse_args()

    onnx_path = args.onnx_path
    
    # Generate engine path by replacing .onnx with .plan
    if onnx_path.endswith('.onnx'):
        engine_path = onnx_path.replace('.onnx', '.plan')
    else:
        engine_path = onnx_path + '.plan'

    # Validate input file
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found: {onnx_path}")
        return 1

    print(f"Input ONNX: {onnx_path}")
    print(f"Output engine: {engine_path}")
    print("Using FP16")

    # Get default UNet profiles
    input_profiles = get_default_unet_profiles()
    
    print("Using default SDXL UNet input profiles:")
    for name, (min_shape, opt_shape, max_shape) in input_profiles.items():
        print(f"  {name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
    print()

    try:
        build_engine(
            engine_path=engine_path,
            onnx_path=onnx_path,
            input_profiles=input_profiles,
            fp16=True,
        )
        print(f"\nConversion completed successfully!")
        print(f"TensorRT engine saved to: {engine_path}")
        return 0
        
    except Exception as e:
        print(f"\nError during conversion: {e}")
        return 1

if __name__ == "__main__":
    exit(main())