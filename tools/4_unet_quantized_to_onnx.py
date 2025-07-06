import os
import torch
from diffusers import UNet2DConditionModel
from huggingface_hub import snapshot_download
import modelopt.torch.opt as mto
from tqdm import tqdm
import argparse
import onnx
from onnxconverter_common import float16
import glob

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

def consolidate_onnx_model(onnx_path):
    """Loads an ONNX model, converts to FP16 if needed, and consolidates external data."""
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
        
        # Automatically convert to FP16 if any FP32 weights are found
        fp32_count = sum(
            1 for initializer in model.graph.initializer 
            if initializer.data_type == onnx.TensorProto.DataType.FLOAT
        )

        if fp32_count > 0:
            print("Model contains FP32 weights. Converting to FP16...")
            try:
                model = float16.convert_float_to_float16(model, keep_io_types=True)
                print("FP16 conversion successful")
            except Exception as e:
                print(f"FP16 conversion failed: {e}, keeping original precision")
        else:
            print("Model is already pure FP16. No conversion needed.")

        # Get the directory and create the consolidated data filename
        directory = os.path.dirname(onnx_path)
        filename = os.path.basename(onnx_path)
        base_name = os.path.splitext(filename)[0]
        data_filename = f"{base_name}.data"
        
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
        cleaned_count = 0
        final_files = {filename, data_filename}
        patterns_to_clean = ['_unet*', 'unet.*', 'onnx__*', '*.data', '*.pb']
        
        for pattern in patterns_to_clean:
            for file_path in glob.glob(os.path.join(directory, pattern)):
                if os.path.basename(file_path) not in final_files:
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                    except OSError as e:
                        print(f"Error removing file {file_path}: {e}")

        if cleaned_count > 0:
            print(f"Removed {cleaned_count} scattered files.")
        
        print("AFTER consolidation:")
        analyze_onnx_model(onnx_path)
        
    except Exception as e:
        print(f"Error during ONNX consolidation: {e}")
        return

def main():
    parser = argparse.ArgumentParser(
        description="Export INT8 UNet to ONNX and consolidate it."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/lab/model",
        help="Path to the downloaded and fused model directory.",
    )
    args = parser.parse_args()

    model_path = args.model_path
    output_dir = os.path.join(model_path, "unet")
    onnx_output_path = os.path.join(output_dir, "model.onnx")

    os.makedirs(output_dir, exist_ok=True)

    int8_checkpoint_path = os.path.join(output_dir, "model_int8.pth")

    if not os.path.exists(int8_checkpoint_path):
        print(f"Error: Quantized UNet checkpoint not found at {int8_checkpoint_path}")
        print("Please run the quantization script (3_unet_quantization_int8.py) first.")
        return

    print("Loading base UNet from local path...")
    unet = UNet2DConditionModel.from_pretrained(
        model_path,
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
        
        # Check the precision of the temporary model
        temp_model = onnx.load(temp_onnx_path, load_external_data=True)
        
        # Check if model is already in FP16
        fp32_count = 0
        fp16_count = 0
        total_count = 0
        
        for initializer in temp_model.graph.initializer:
            param_count = 1
            for dim in initializer.dims:
                param_count *= dim
            total_count += param_count
            
            if initializer.data_type == 1:  # FLOAT (FP32)
                fp32_count += param_count
            elif initializer.data_type == 10:  # FLOAT16
                fp16_count += param_count
        
        fp16_percentage = (fp16_count / total_count * 100) if total_count > 0 else 0
        
        print(f"Model precision: {fp16_percentage:.1f}% FP16, {100 - fp16_percentage:.1f}% FP32")
        
        # Automatically convert to FP16 if there are significant FP32 weights
        should_convert_fp16 = fp32_count > 0
        
        if should_convert_fp16:
            print("Model contains FP32 weights. Converting to FP16...")
            try:
                model = float16.convert_float_to_float16(temp_model, keep_io_types=True)
                print("FP16 conversion successful")
            except Exception as e:
                print(f"FP16 conversion failed: {e}, keeping original precision")
                model = temp_model
        else:
            print("Model is already 100% FP16, skipping conversion.")
            model = temp_model
        
        # Create data filename
        directory = os.path.dirname(onnx_output_path)
        filename = os.path.basename(onnx_output_path)
        base_name = os.path.splitext(filename)[0]
        data_filename = f"{base_name}.data"
        
        print("Saving consolidated model...")
        try:
            onnx.save(
                model,
                onnx_output_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=data_filename,
                size_threshold=1024
            )
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {e}")
            # Fallback: save without external data consolidation
            print("Trying fallback save...")
            onnx.save(model, onnx_output_path)
        
        # Clean up temporary files
        print("Cleaning up temporary and scattered files...")
        cleaned_count = 0
        
        # Remove the main temp file
        if os.path.exists(temp_onnx_path):
            try:
                os.remove(temp_onnx_path)
                cleaned_count += 1
            except Exception as e:
                print(f"Could not remove temp file {temp_onnx_path}: {e}")

        # Clean up other scattered files from export
        final_files = {os.path.basename(onnx_output_path), data_filename}
        patterns_to_clean = ['_unet*', 'unet.*', 'onnx__*', '*.pb']
        
        for pattern in patterns_to_clean:
            for file_path in glob.glob(os.path.join(directory, pattern)):
                if os.path.basename(file_path) not in final_files:
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                    except OSError as e:
                        print(f"Error removing file {file_path}: {e}")

        if cleaned_count > 0:
            print(f"Cleaned up {cleaned_count} temporary files")
        
        print("ONNX model consolidated. Final analysis:")
        analyze_onnx_model(onnx_output_path)

        print(f"\nCleaning up original INT8 safetensors: {os.path.basename(int8_checkpoint_path)}")
        if os.path.exists(int8_checkpoint_path):
            try:
                os.remove(int8_checkpoint_path)
                print("✓ Successfully deleted original INT8 safetensors.")
            except OSError as e:
                print(f"✗ Error deleting original INT8 safetensors: {e}")
        else:
            print("⚠ Original INT8 safetensors not found, skipping cleanup.")

    print("\nSuccessfully exported ONNX model.")

if __name__ == "__main__":
    main()
