import torch
import torch.nn as nn
import onnx
import os
import argparse
import subprocess
import traceback
import json
from typing import List, Dict, Any
from diffusers import AutoencoderKL
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
from torch.export.dynamic_shapes import Dim
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
        output_tensor = torch.zeros(batch_size, self.out_channels, self.out_height, self.out_width, dtype=latent.dtype, device=latent.device)

        for i, latent_slice in enumerate(latent):
            latent_slice_batched = latent_slice.unsqueeze(0)
            decoded_slice = self.vae_decoder(latent_slice_batched)
            
            output_tensor[i:i+1] = decoded_slice

        return output_tensor

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
    scripted_decoder = (simple_vae_decoder_instance)

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
                dynamic_shapes={
                    "latent_sample": {
                        0: Dim("batch_size", min=1, max=4),
                        2: Dim("height", min=64, max=65),
                        3: Dim("width", min=64, max=65),
                    },
                    "sample": {
                        0: Dim("batch_size", min=1, max=4),
                        2: Dim("height_out", min=512, max=520),
                        3: Dim("width_out", min=512, max=520),
                    },
                },
                opset_version=19,
            )
            print(f"✅ Simplified VAE Decoder exported successfully to {onnx_path}")
            return True
    except Exception as e:
        print(f"❌ Simplified VAE Decoder export failed: {e}")
        traceback.print_exc()
        return False

# --- NEW: Optimizer Function ---
def optimize_onnx_model(input_path: str, output_path: str) -> bool:
    """Optimizes an ONNX model using onnx-slim."""
    print(f"--- Optimizing {input_path} -> {output_path} ---")
    try:
        # The command is `onnxslim <input_model> <output_model>`
        result = subprocess.run(
            ["onnxslim", input_path, output_path],
            capture_output=True, text=True, check=True
        )
        print("--- onnx-slim Output ---")
        print(result.stdout)
        print("------------------------")
        if not os.path.exists(output_path):
             raise RuntimeError("onnx-slim ran without error, but the output file was not created.")
        return True
    except FileNotFoundError:
        print("❌ Error: 'onnxslim' command not found. Please ensure onnx-slim is installed (`pip install onnx-slim`).")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during optimization:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during optimization: {e}")
        return False

# --- JSON Export Logic with Deep Dive ---

class OnnxNodeEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle ONNX and NumPy data types that are not
    natively serializable by the standard json library.
    """
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, onnx.TensorProto):
            return {
                "__type__": "TensorProto",
                "name": o.name,
                "dims": list(o.dims),
                "data_type": onnx.TensorProto.DataType.Name(o.data_type),
            }
        if isinstance(o, bytes):
            try:
                return o.decode('utf-8')
            except UnicodeDecodeError:
                return f"<bytes length: {len(o)}>"
        return super().default(o)

def _graph_to_dict(graph: onnx.GraphProto) -> Dict[str, Any]:
    """
    Recursively converts a GraphProto and its nodes into a dictionary,
    diving deep into subgraphs.
    """
    graph_dict = {
        "name": graph.name,
        "inputs": [i.name for i in graph.input],
        "outputs": [o.name for o in graph.output],
        "nodes": []
    }
    for node in graph.node:
        attributes = {}
        for attr in node.attribute:
            value = onnx.helper.get_attribute_value(attr)
            if isinstance(value, onnx.GraphProto):
                attributes[attr.name] = _graph_to_dict(value)
            else:
                attributes[attr.name] = value

        node_info = {
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": attributes
        }
        graph_dict["nodes"].append(node_info)
    return graph_dict

def export_nodes_to_json(onnx_path: str, json_path: str):
    """
    Inspects an ONNX model and exports its entire graph structure, including
    subgraphs, to a JSON file.
    """
    print(f"Recursively exporting ONNX graph from {onnx_path}...")
    try:
        model = onnx.load(onnx_path)
        graph_as_dict = _graph_to_dict(model.graph)
        with open(json_path, 'w') as f:
            json.dump(graph_as_dict, f, indent=4, cls=OnnxNodeEncoder)
        print(f"✅ Deep graph information successfully exported to {json_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to export graph information: {e}")
        traceback.print_exc()
        return False


# --- TensorRT Engine Building Utilities ---

class TQDMProgressMonitor(trt.IProgressMonitor if trt else object):
    def __init__(self):
        if trt: trt.IProgressMonitor.__init__(self)
        self._active_phases, self._step_result, self.max_indent = {}, True, 5
    def phase_start(self, phase_name, parent_phase, num_steps):
        leave = False
        try:
            if parent_phase is not None:
                nbIndents = (self._active_phases.get(parent_phase, {}).get("nbIndents", self.max_indent) + 1)
                if nbIndents >= self.max_indent: return
            else: nbIndents, leave = 0, True
            self._active_phases[phase_name] = {"tq": tqdm(total=num_steps, desc=phase_name, leave=leave, position=nbIndents), "nbIndents": nbIndents, "parent_phase": parent_phase}
        except KeyboardInterrupt: self._step_result = False
    def phase_finish(self, phase_name):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].close()
                del self._active_phases[phase_name]
        except KeyboardInterrupt: self._step_result = False
    def step_complete(self, phase_name, step):
        try:
            if phase_name in self._active_phases.keys(): self._active_phases[phase_name]["tq"].update(step - self._active_phases[phase_name]["tq"].n)
            return self._step_result
        except KeyboardInterrupt: return False

def build_tensorrt_engine(onnx_file: str, engine_file: str, input_profiles: dict, fp16: bool = True, timing_cache_path: str | None = None):
    if not trt: raise ImportError("TensorRT library is not installed.")
    logger = trt.Logger(trt.Logger.VERBOSE)
    print(f"{'='*50}\nExporting ONNX to TensorRT Engine\n  ONNX Path: {onnx_file}\n  Engine Path: {engine_file}\n  FP16: {fp16}\n{'='*50}")
    builder = trt.Builder(logger)
    builder.max_threads = torch.get_num_threads()
    config = builder.create_builder_config()
    if timing_cache_path:
        if os.path.exists(timing_cache_path):
            print(f"Loading timing cache from: {timing_cache_path}")
            with open(timing_cache_path, "rb") as f: cache_data = f.read()
            timing_cache = config.create_timing_cache(cache_data)
        else:
            print("Creating a new timing cache.")
            timing_cache = config.create_timing_cache(b"")
        config.set_timing_cache(timing_cache, ignore_mismatch=False)
    profile = builder.create_optimization_profile()
    for name, dims in input_profiles.items():
        min_shape, opt_shape, max_shape = dims['min'], dims['opt'], dims['max']
        print(f"  Setting profile for input: {name} with min={min_shape}, opt={opt_shape}, max={max_shape}")
        profile.set_shape(name, min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)
    config.progress_monitor = TQDMProgressMonitor()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)
    if not parser.parse_from_file(onnx_file):
        for error in range(parser.num_errors): print(parser.get_error(error))
        raise ValueError(f"Failed to parse ONNX file: {onnx_file}")
    print("Building TensorRT engine. This may take a while...")
    plan = builder.build_serialized_network(network, config)
    if not plan: raise RuntimeError("Failed to build TensorRT engine.")
    if timing_cache_path and config.get_timing_cache():
        new_timing_cache = config.get_timing_cache()
        if new_timing_cache:
            with open(timing_cache_path, "wb") as f: f.write(new_timing_cache.serialize())
            print(f"Timing cache saved to: {timing_cache_path}")
    with open(engine_file, "wb") as f: f.write(plan)
    print(f"✅ TensorRT engine exported successfully to: {engine_file}")
    print("="*50)
    return engine_file

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export, optimize, and inspect VAE models.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--onnx", action='store_true', help="Export the base model to ONNX format.")
    parser.add_argument("--optimize", action='store_true', help="Optimize the ONNX model using onnx-slim.\nRuns after --onnx if present.")
    parser.add_argument("--json", action='store_true', help="Export ONNX graph info to a JSON file.\nRuns after any export or optimization step.")
    parser.add_argument("--tensorrt", action='store_true', help="Build a TensorRT engine from the final ONNX model.")
    parser.add_argument("--onnx_path", type=str, default="onnx/simple_vae_decoder.onnx", help="Path for the base ONNX file.")
    
    args = parser.parse_args()

    # If no specific action is chosen, default to exporting the base ONNX model.
    if not any([args.onnx, args.optimize, args.json, args.tensorrt]):
        args.onnx = True

    os.makedirs("onnx", exist_ok=True)
    
    # This variable will track the most recent version of the ONNX file.
    current_onnx_path = args.onnx_path
    
    # --- Step 1: Export base ONNX model ---
    if args.onnx:
        with torch.no_grad():
            print("Loading original VAE model from HuggingFace...")
            diffusers_vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=torch.float16
            ).to("cuda").eval()
            print("✅ Original VAE model loaded.")
            
            export_onnx_model(diffusers_vae, current_onnx_path)

    # --- Step 2: Optimize the ONNX model if requested ---
    if args.optimize:
        if not os.path.exists(current_onnx_path):
            print(f"❌ Cannot optimize. ONNX file not found at {current_onnx_path}. Please run with --onnx first.")
        else:
            optimized_path = current_onnx_path.replace(".onnx", "_optimized.onnx")
            if optimize_onnx_model(current_onnx_path, optimized_path):
                # IMPORTANT: Update the path to point to the new optimized model for subsequent steps
                current_onnx_path = optimized_path
            else:
                print("⚠️ Optimization failed. Subsequent steps will use the unoptimized model.")

    # --- Step 3: Export JSON graph if requested ---
    if args.json:
        if not os.path.exists(current_onnx_path):
            print(f"❌ Cannot export JSON. ONNX file not found at {current_onnx_path}.")
        else:
            json_path = current_onnx_path.replace(".onnx", "_nodes.json")
            export_nodes_to_json(current_onnx_path, json_path)

    # --- Step 4: Build TensorRT engine if requested ---
    if args.tensorrt:
        if not os.path.exists(current_onnx_path):
            print(f"❌ Cannot build TensorRT engine. ONNX file not found at {current_onnx_path}.")
        else:
            engine_path = current_onnx_path.replace(".onnx", ".trt")
            cache_path = current_onnx_path.replace(".onnx", ".cache")

            input_profiles = OrderedDict([
                ("latent_sample", {
                    "min": (1, 4, 64, 64),
                    "opt": (2, 4, 64, 64),
                    "max": (4, 4, 64, 64),
                }),
            ])

            try:
                build_tensorrt_engine(
                    current_onnx_path,
                    engine_path,
                    input_profiles=input_profiles,
                    timing_cache_path=cache_path
                )
            except Exception as e:
                print(f"❌ TensorRT engine build failed: {e}")
                traceback.print_exc()