import torch
import torch.nn as nn
import onnx
import os
import argparse
import subprocess
import traceback
import json
from typing import List
from diffusers import AutoencoderKL
from tqdm import tqdm
from collections import OrderedDict

# Only import tensorrt if it's available and needed
try:
    import tensorrt as trt
except ImportError:
    trt = None


class SimpleVaeDecoder(nn.Module):
    def __init__(self, traced_vae_decoder, out_channels: int, out_height: int, out_width: int):
        super().__init__()
        self.vae_decoder = traced_vae_decoder
        self.out_channels = out_channels
        self.out_height = out_height
        self.out_width = out_width

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.shape[0]
        #output_tensor = torch.zeros(
        #    (batch_size, self.out_channels, self.out_height, self.out_width),
        #    dtype=latent.dtype,
        #    device=latent.device
        #)

        decoded_slice = torch.zeros(batch_size, self.out_channels, self.out_height, self.out_width, dtype=latent.dtype, device=latent.device)

        for i in range(batch_size):
            latent_slice = latent[i:i+1]
            decoded_slice_2 = self.vae_decoder(latent_slice)
            decoded_slice[i:i+1] = decoded_slice_2
            #output_tensor[i:i+1] = decoded_slice

        return decoded_slice


def export_onnx_model(vae: AutoencoderKL, onnx_path: str):
    class VaeDecodeWrapper(nn.Module):
        def __init__(self, vae_model):
            super().__init__()
            self.vae = vae_model
        def forward(self, latents):
            return self.vae.decode(latents).sample

    dummy_latent_tile = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        temp_decoder = VaeDecodeWrapper(vae)
        dummy_output = temp_decoder(dummy_latent_tile)
        _, out_channels, out_height, out_width = dummy_output.shape
        print(f"Detected VAE output shape for a single sample: ({out_channels}, {out_height}, {out_width})")

    with torch.no_grad():
        traced_vae_decoder = torch.jit.trace(VaeDecodeWrapper(vae), dummy_latent_tile)

    simple_vae_decoder_instance = SimpleVaeDecoder(
        traced_vae_decoder, out_channels, out_height, out_width
    )
    scripted_decoder = torch.jit.script(simple_vae_decoder_instance)

    latent_sample = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)

    print("Exporting ONNX model with a pre-allocated output tensor...")
    try:
        with torch.no_grad():
            torch.onnx.export(
                scripted_decoder,
                (latent_sample,),
                onnx_path,
                input_names=['latent_sample'],
                output_names=['sample'],
                dynamic_axes={
                    'latent_sample': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'sample': {0: 'batch_size', 2: 'height', 3: 'width'}
                },
                opset_version=16
            )
            print(f"✅ Simplified VAE Decoder exported successfully to {onnx_path}")
            return True
    except Exception as e:
        print(f"❌ Simplified VAE Decoder export failed: {e}")
        traceback.print_exc()
        return False

def inspect_onnx(onnx_path: str):
    """Inspects an ONNX model using onnxslim."""
    print(f"--- Inspecting {onnx_path} ---")
    try:
        result = subprocess.run(["onnxslim", onnx_path, onnx_path], capture_output=True, text=True, check=True)
        print(result.stdout)
    except FileNotFoundError:
        print("❌ Error: 'onnxslim' command not found. Please ensure onnx-slim is installed (`pip install onnx-slim`).")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during inspection:")
        print(e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- TensorRT Engine Building Utilities ---

class TQDMProgressMonitor(trt.IProgressMonitor if trt else object):
    """A TensorRT progress monitor that uses TQDM to display build progress."""
    def __init__(self):
        if trt:
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
                self._active_phases[phase_name]["tq"].close()
                del self._active_phases[phase_name]
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
    """Builds a TensorRT engine from an ONNX model."""
    if not trt:
        raise ImportError("TensorRT library is not installed. Please install it to build engines.")

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
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
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
    if timing_cache_path and config.get_timing_cache():
        new_timing_cache = config.get_timing_cache()
        if new_timing_cache:
            with open(timing_cache_path, "wb") as f:
                f.write(new_timing_cache.serialize())
            print(f"Timing cache saved to: {timing_cache_path}")

    print(f"Writing TensorRT engine to: {engine_file}")
    with open(engine_file, "wb") as f:
        f.write(plan)

    print("✅ TensorRT engine exported successfully.")
    print("="*50)
    return engine_file

# --- VAE Decoder Definition and Export Logic ---
# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and build VAE models.")
    parser.add_argument("--onnx", action='store_true', help="Export the model to ONNX format. (Default)")
    parser.add_argument("--tensorrt", action='store_true', help="Build a TensorRT engine from the ONNX model.")
    parser.add_argument("--optimize", action='store_true', help="Optimize an existing ONNX model.")
    parser.add_argument("--onnx_path", type=str, default="onnx/simple_vae_decoder.onnx", help="Path for the ONNX file.")
    
    args = parser.parse_args()
    
    # Default to ONNX export if no other action is specified
    if not args.tensorrt and not args.optimize:
        args.onnx = True

    os.makedirs("onnx", exist_ok=True)

    if args.optimize:
        if os.path.exists(args.onnx_path):
            inspect_onnx(args.onnx_path)
        else:
            print(f"❌ ONNX file not found at {args.onnx_path}. Please export it first.")
    
    if args.onnx:
        with torch.no_grad():
            print("Loading original VAE model from HuggingFace...")
            diffusers_vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=torch.float16
            ).to("cuda").eval()
            print("✅ Original VAE model loaded.")
            
            onnx_export_success = export_onnx_model(diffusers_vae, args.onnx_path)

            if onnx_export_success and args.tensorrt:
                print("Successfully exported ONNX model.")

    if args.tensorrt:
        engine_path = args.onnx_path.replace(".onnx", ".trt")
        cache_path = args.onnx_path.replace(".onnx", ".cache")

        # Define the optimization profile for the VAE decoder
        input_profiles = OrderedDict([
            ("latent_sample", {
                "min": (1, 4, 64, 64),   # Batch 1, SD 1.5 latent size
                "opt": (2, 4, 64, 64),  # Batch 2, SDXL latent size
                "max": (4, 4, 64, 64),  # Max batch 4, SDXL latent size
            }),
        ])

        try:
            build_tensorrt_engine(
                args.onnx_path,
                engine_path,
                input_profiles=input_profiles,
                timing_cache_path=cache_path
            )
            print("✅ TensorRT engine built successfully.")
        except Exception as e:
            print(f"❌ TensorRT engine build failed: {e}")
            traceback.print_exc()