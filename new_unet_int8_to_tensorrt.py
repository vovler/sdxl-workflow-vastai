import os
import torch
import tensorrt as trt
from diffusers import UNet2DConditionModel
from huggingface_hub import snapshot_download
import modelopt.torch.opt as mto
from tqdm import tqdm
import argparse
import onnx
from onnxconverter_common import float16

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

class UnetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        output = self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        ).sample
        return output

def analyze_pytorch_model(model):
    """Analyze PyTorch model to understand precision and size."""
    print("\n" + "="*50)
    print("PYTORCH MODEL ANALYSIS")
    print("="*50)
    
    total_params = sum(p.numel() for p in model.parameters())
    total_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_size_gb = total_size_bytes / (1024**3)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Total size: {total_size_gb:.2f} GB ({total_size_bytes:,} bytes)")
    
    # Check the precision of parameters
    precision_counts = {}
    precision_sizes = {}
    
    for name, param in model.named_parameters():
        dtype_str = str(param.dtype)
        param_count = param.numel()
        param_size = param_count * param.element_size()
        
        precision_counts[dtype_str] = precision_counts.get(dtype_str, 0) + param_count
        precision_sizes[dtype_str] = precision_sizes.get(dtype_str, 0) + param_size
    
    print("\nPrecision breakdown:")
    for dtype, count in precision_counts.items():
        size_gb = precision_sizes[dtype] / (1024**3)
        print(f"  {dtype}: {count:,} parameters ({size_gb:.2f} GB)")
    
    print("="*50)

def analyze_onnx_model(onnx_path):
    """Analyze ONNX model to understand weight precision and size."""
    print("\n" + "="*50)
    print(f"ONNX MODEL ANALYSIS: {os.path.basename(onnx_path)}")
    print("="*50)
    
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found at {onnx_path}")
        return
    
    try:
        model = onnx.load(onnx_path, load_external_data=True)
        
        total_params = 0
        precision_counts = {}
        precision_sizes = {}
        
        # Bytes per data type
        bytes_per_type = {
            1: 4,   # FLOAT
            2: 1,   # UINT8
            3: 1,   # INT8
            4: 2,   # UINT16
            5: 2,   # INT16
            6: 4,   # INT32
            7: 4,   # INT64
            8: 8,   # STRING (estimate)
            9: 1,   # BOOL
            10: 2,  # FLOAT16
            11: 8,  # DOUBLE
            12: 4,  # UINT32
            13: 8,  # UINT64
        }
        
        for initializer in model.graph.initializer:
            param_count = 1
            for dim in initializer.dims:
                param_count *= dim
            total_params += param_count
            
            data_type = initializer.data_type
            type_name = onnx.TensorProto.DataType.Name(data_type)
            bytes_per_param = bytes_per_type.get(data_type, 4)
            param_size = param_count * bytes_per_param
            
            precision_counts[type_name] = precision_counts.get(type_name, 0) + param_count
            precision_sizes[type_name] = precision_sizes.get(type_name, 0) + param_size
        
        total_size_bytes = sum(precision_sizes.values())
        total_size_gb = total_size_bytes / (1024**3)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Total size: {total_size_gb:.2f} GB ({total_size_bytes:,} bytes)")
        
        print("\nPrecision breakdown:")
        for precision, count in precision_counts.items():
            size_gb = precision_sizes[precision] / (1024**3)
            percentage = (precision_sizes[precision] / total_size_bytes) * 100
            print(f"  {precision}: {count:,} params ({size_gb:.2f} GB, {percentage:.1f}%)")
            
        # Check file sizes
        onnx_file_size = os.path.getsize(onnx_path) / (1024**3)
        print(f"\nActual ONNX file size: {onnx_file_size:.3f} GB")
        
        # Check for external data file
        directory = os.path.dirname(onnx_path)
        filename = os.path.basename(onnx_path)
        base_name = os.path.splitext(filename)[0]
        data_filename = f"{base_name}.data"
        data_path = os.path.join(directory, data_filename)
        
        if os.path.exists(data_path):
            data_file_size = os.path.getsize(data_path) / (1024**3)
            print(f"External data file size: {data_file_size:.3f} GB")
            print(f"Total files size: {(onnx_file_size + data_file_size):.3f} GB")
        
    except Exception as e:
        print(f"Error analyzing ONNX model: {e}")
    
    print("="*50)

def build_engine(
    engine_path: str,
    onnx_path: str,
    input_profiles: dict,
    fp16: bool = True,
    int8: bool = False,
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
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)

    config.progress_monitor = TQDMProgressMonitor()

    print("Building engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine.")

    print("Engine built successfully.")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"TensorRT engine saved to {engine_path}")

def consolidate_onnx_model(onnx_path, force_fp16=False):
    """Loads an ONNX model and consolidates all external data into a single .data file."""
    print(f"\nConsolidating ONNX model: {onnx_path}")
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found at {onnx_path}")
        return
        
    try:
        # Analyze before consolidation
        print("BEFORE consolidation:")
        analyze_onnx_model(onnx_path)
        
        # Load the model with all its external data files
        print("Loading ONNX model with external data...")
        model = onnx.load(onnx_path, load_external_data=True)
        
        # Get the directory and create the consolidated data filename
        directory = os.path.dirname(onnx_path)
        filename = os.path.basename(onnx_path)
        base_name = os.path.splitext(filename)[0]
        data_filename = f"{base_name}.data"
        
        # Optional FP16 conversion
        if force_fp16:
            print("Converting to FP16...")
            try:
                model = float16.convert_float_to_float16(model, keep_io_types=True)
                print("FP16 conversion successful")
            except Exception as e:
                print(f"FP16 conversion failed: {e}, keeping original precision")
        
        print("Consolidating external data into single file...")
        
        # Save with all tensors in a single external data file
        onnx.save(
            model,
            onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_filename,
            size_threshold=1024,  # Save tensors > 1KB externally
            convert_attribute=False
        )
        
        # Clean up the scattered external data files
        print("Cleaning up scattered external data files...")
        cleaned_files = []
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            # Remove auto-generated external data files (but keep our consolidated one)
            if (file.startswith('onnx__') or 
                (file.endswith('.data') and file != data_filename) or
                (file.endswith('.pb') and file != filename)):
                try:
                    os.remove(file_path)
                    cleaned_files.append(file)
                except Exception as e:
                    print(f"Could not remove {file}: {e}")
        
        if cleaned_files:
            print(f"Removed {len(cleaned_files)} scattered files")
        
        print("AFTER consolidation:")
        analyze_onnx_model(onnx_path)
        
    except Exception as e:
        print(f"Error during ONNX consolidation: {e}")
        return

def main():
    parser = argparse.ArgumentParser(
        description="Export INT8 UNet to ONNX, consolidate it, and/or build TensorRT engine."
    )
    parser.add_argument(
        "--only-onnx",
        action="store_true",
        help="Only export the ONNX model and consolidate it, skip building the TensorRT engine.",
    )
    parser.add_argument(
        "--consolidate-onnx",
        action="store_true",
        help="Only consolidate an existing ONNX model. Skips export and build.",
    )
    parser.add_argument(
        "--force-fp16",
        action="store_true",
        help="Force FP16 conversion during consolidation.",
    )
    args = parser.parse_args()

    if args.only_onnx and args.consolidate_onnx:
        print("Error: --only-onnx and --consolidate-onnx are mutually exclusive.")
        return

    base_model_id = "socks22/sdxl-wai-nsfw-illustriousv14"
    output_dir = "/workflow/wai_dmd2_onnx/unet"
    onnx_output_path = os.path.join(output_dir, "model_int8.onnx")
    engine_output_path = os.path.join(output_dir, "model_int8.plan")

    os.makedirs(output_dir, exist_ok=True)

    if args.consolidate_onnx:
        consolidate_onnx_model(onnx_output_path, force_fp16=args.force_fp16)
        print("Exiting after consolidation.")
        return

    model_dir = snapshot_download(base_model_id)
    int8_checkpoint_path = os.path.join(model_dir, "unet_int8.safetensors")

    print("Loading base UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        base_model_id,
        subfolder="unet",
        torch_dtype=torch.float16,
    ).to("cuda")

    print(f"Restoring INT8 weights from {int8_checkpoint_path}...")
    mto.restore(unet, int8_checkpoint_path)
    unet.eval()
    print("INT8 UNet restored successfully.")

    # Analyze the PyTorch model before export
    analyze_pytorch_model(unet)

    print(f"Exporting INT8 UNet to ONNX: {onnx_output_path}")
    if os.path.exists(onnx_output_path):
        print("ONNX model already exists, skipping export.")
        analyze_onnx_model(onnx_output_path)
    else:
        # Dummy inputs for ONNX export
        batch_size = 1
        latent_height = 120
        latent_width = 120
        
        sample = torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float16).to("cuda")
        timestep = torch.tensor(999, dtype=torch.float16).to("cuda")
        encoder_hidden_states = torch.randn(batch_size, 77, 2048, dtype=torch.float16).to("cuda")
        text_embeds = torch.randn(batch_size, 1280, dtype=torch.float16).to("cuda")
        time_ids = torch.randn(batch_size, 6, dtype=torch.float16).to("cuda")
        
        unet_wrapper = UnetWrapper(unet)
        
        input_names = ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]
        output_names = ["out_sample"]
        dynamic_axes = {
            "sample": {0: "batch_size", 2: "height", 3: "width"},
            "encoder_hidden_states": {0: "batch_size"},
            "text_embeds": {0: "batch_size"},
            "time_ids": {0: "batch_size"},
        }
        
        # Export to a temporary file first
        temp_onnx_path = onnx_output_path + ".temp"
        
        print("Exporting to ONNX...")
        with torch.no_grad():
            torch.onnx.export(
                unet_wrapper,
                (sample, timestep, encoder_hidden_states, text_embeds, time_ids),
                temp_onnx_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=18,
                export_params=True,
                do_constant_folding=False,  # Disable to avoid extra constants
                keep_initializers_as_inputs=False,
                verbose=False
            )
        
        print("ONNX export complete. Analyzing temporary model...")
        analyze_onnx_model(temp_onnx_path)
        
        print("Loading and consolidating...")
        
        # Load the temporary model
        model = onnx.load(temp_onnx_path, load_external_data=True)
        
        # Create data filename
        directory = os.path.dirname(onnx_output_path)
        filename = os.path.basename(onnx_output_path)
        base_name = os.path.splitext(filename)[0]
        data_filename = f"{base_name}.data"
        
        print("Saving consolidated model...")
        onnx.save(
            model,
            onnx_output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_filename,
            size_threshold=1024
        )
        
        # Clean up temporary files
        if os.path.exists(temp_onnx_path):
            os.remove(temp_onnx_path)
        
        # Clean up any scattered external data files from temp export
        temp_dir = os.path.dirname(temp_onnx_path)
        cleaned_count = 0
        for file in os.listdir(temp_dir):
            if file.startswith('onnx__') or (file.endswith('.pb') and file != filename):
                try:
                    os.remove(os.path.join(temp_dir, file))
                    cleaned_count += 1
                except:
                    pass
        
        if cleaned_count > 0:
            print(f"Cleaned up {cleaned_count} temporary files")
        
        print("ONNX model consolidated. Final analysis:")
        analyze_onnx_model(onnx_output_path)

    if args.only_onnx:
        print("Successfully exported ONNX model. Exiting as requested by --only-onnx.")
        return

    print("Building INT8 TensorRT engine...")
    latent_heights = [768 // 8, 1152 // 8, 960 // 8]
    latent_widths = [1152 // 8, 768 // 8, 960 // 8]
    
    min_h, max_h = min(latent_heights), max(latent_heights)
    min_w, max_w = min(latent_widths), max(latent_widths)
    opt_h, opt_w = 960 // 8, 960 // 8
    bs = 1

    unet_input_profiles = {
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
    
    build_engine(
        engine_path=engine_output_path,
        onnx_path=onnx_output_path,
        input_profiles=unet_input_profiles,
        fp16=True,
        int8=True,
    )

    print(f"\nINT8 UNet engine saved successfully to {engine_output_path}")

if __name__ == "__main__":
    main()