import tensorrt as trt
import os
import json
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
            # The phase_start callback cannot directly cancel the build, so request the cancellation from within step_complete.
            self._step_result = False
            
    def phase_finish(self, phase_name):
        try:
            if phase_name in self._active_phases:
                self._active_phases[phase_name]["tq"].close()
                del self._active_phases[phase_name]
        except KeyboardInterrupt:
            self._step_result = False

    def step_complete(self, phase_name, step):
        try:
            if self._step_result and phase_name in self._active_phases:
                self._active_phases[phase_name]["tq"].update()
        except KeyboardInterrupt:
            self._step_result = False
        return self._step_result

# --- Configuration for multiple optimization profiles ---
# Defines the discrete values for dynamic dimensions.
# Each combination of these values will result in a specific optimization profile.
BATCH_SIZES = [1, 2, 4]
IMAGE_SIZES = [(768, 1152), (1152, 768), (1024, 1024)]
# From 77 (1 chunk) to 231 (3 chunks)
PROMPT_LENGTHS = [77 * i for i in range(1, 5)]
PROFILE_MAP_PATH = "profile_map.json"


def build_engine(onnx_path, engine_path, use_fp16=True):
    """
    Builds a TensorRT engine from an ONNX file with multiple optimization profiles.
    """
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    
    # Check if the ONNX file exists
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found at {onnx_path}")
        print("Please run unet_to_onnx.py first to generate the ONNX file.")
        return

    # Initialize TensorRT builder, network, and parser
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    config.progress_monitor = TQDMProgressMonitor()

    # Set memory constraints
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 10 * (1024 ** 3)) # 10 GiB

    # Enable FP16 if supported and requested
    if use_fp16 and builder.platform_has_fast_fp16:
        print("Enabling FP16 mode.")
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print("FP16 mode not enabled.")

    # Parse the ONNX model
    print(f"Parsing ONNX file: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("Completed parsing ONNX file.")

    # --- Create Multiple Optimization Profiles ---
    profile_map = []
    profile_count = len(BATCH_SIZES) * len(IMAGE_SIZES) * len(PROMPT_LENGTHS)
    print(f"Generating {profile_count} optimization profiles...")

    for batch_size in BATCH_SIZES:
        for height, width in IMAGE_SIZES:
            for num_tokens in PROMPT_LENGTHS:
                profile = builder.create_optimization_profile()
                
                # Store the configuration for this profile to look up later
                profile_map.append({
                    "batch_size": batch_size,
                    "height": height,
                    "width": width,
                    "num_tokens": num_tokens
                })

                # Define shapes for this specific profile. min=opt=max for peak performance.
                h_div_8 = height // 8
                w_div_8 = width // 8
                
                profile.set_shape('sample', (batch_size, 4, h_div_8, w_div_8), (batch_size, 4, h_div_8, w_div_8), (batch_size, 4, h_div_8, w_div_8))
                profile.set_shape('timestep', (batch_size,), (batch_size,), (batch_size,))
                profile.set_shape('encoder_hidden_states', (batch_size, num_tokens, 2048), (batch_size, num_tokens, 2048), (batch_size, num_tokens, 2048))
                profile.set_shape('text_embeds', (batch_size, 1280), (batch_size, 1280), (batch_size, 1280))
                profile.set_shape('time_ids', (batch_size, 6), (batch_size, 6), (batch_size, 6))
                
                config.add_optimization_profile(profile)

    print("All profiles created.")

    # Build the engine
    print("Building TensorRT engine... (This may take a few minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("ERROR: Failed to build the engine.")
        return None
    print("Successfully built TensorRT engine.")

    # Save the engine
    print(f"Saving engine to file: {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print("Engine saved.")

    # Save the profile map for runtime use
    with open(PROFILE_MAP_PATH, "w") as f:
        json.dump(profile_map, f)
    print(f"Profile map saved to {PROFILE_MAP_PATH}")

def main():
    onnx_file_path = "unet.onnx"
    engine_file_path = "unet.engine"
    build_engine(onnx_file_path, engine_file_path, use_fp16=True)

if __name__ == "__main__":
    main() 