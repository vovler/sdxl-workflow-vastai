import torch
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download
import os

# Add necessary imports for quantization
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
from modelopt.torch.quantization.config import (
    QUANT_CONFIG_BY_KEY,
    QuantizationConfig,
    TensorQuantizer,
)
from functools import partial
from tqdm import tqdm


# Define utility functions for quantization
def load_calib_prompts(batch_size, calib_data_path="./calib_prompts.txt"):
    """
    Load calibration prompts from a file.
    """
    with open(calib_data_path, "r", encoding="utf-8") as f:
        lst = [line.rstrip("\n") for line in f]
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def get_percentilequant_config(model, quant_level, percentile, alpha):
    """
    Get the percentile quantization config for a model.
    """
    quant_config_by_module = {
        "default": QuantizationConfig(num_bits=(8, 8), axis=None)
    }
    # Enable INT8 quantizers for all linear layers
    print("Using INT8 quantization")
    quant_config_by_module["aten.linear"] = QuantizationConfig(
        num_bits=(8, 8),
        axis=(0),
        quantizer_type=TensorQuantizer.QuantizerType.INT,
    )

    # Percentile quantization
    if percentile is not None and percentile < 1.0:
        # Use percentile quantization for activations
        print(f"Using percentile quantization with percentile={percentile}")
        quant_config_by_module["default"].update(
            {
                "a_percentile": percentile,
                "w_percentile": percentile,
            }
        )
        quant_config_by_module["aten.linear"].update(
            {
                "a_percentile": percentile,
                "w_percentile": percentile,
            }
        )

    # Alpha-based quantization
    if alpha is not None:
        print(f"Setting alpha to {alpha}")
        quant_config_by_module["default"]["a_method"] = "max"
        quant_config_by_module["default"]["w_method"] = "max"

    def get_module_quant_config(module_name, module_type):
        """
        Get the quantization config for a module.
        """
        quant_config = QUANT_CONFIG_BY_KEY.get(module_name)
        if quant_config is not None:
            return quant_config
        return quant_config_by_module.get(
            module_type, quant_config_by_module.get("default")
        )

    return partial(get_module_quant_config)


def forward_loop(model, prompts, num_inference_steps=20):
    """A basic forward loop for calibration."""
    with torch.no_grad():
        for batch_prompts in tqdm(prompts, desc="Calibrating"):
            _ = model(
                prompt=batch_prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=1.0,
            )


# Base model and LoRA configuration
base_model_id = "socks22/sdxl-wai-nsfw-illustriousv14"

print(f"Loading base model: {base_model_id}")
# Load the main pipeline on CPU to avoid meta device issues
pipeline = DiffusionPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipeline.to("cuda")


model_path = snapshot_download(base_model_id)

# Define the path to save the fused UNet, overwriting the original
unet_path = os.path.join(model_path, "unet")
int8_unet_path = os.path.join(model_path, "unet_int8.safetensors")

print(f"Saving fused UNet to: {unet_path}")
# Save the fused UNet
pipeline.unet.save_pretrained(unet_path)

print("Fused UNet saved successfully.")

# Quantize the UNet
print("Starting UNet quantization...")

# Load calibration prompts
calib_prompts = load_calib_prompts(
    batch_size=1, calib_data_path="./calib_prompts.txt"
)


# Create the int8 quantization recipe
quant_config = get_percentilequant_config(
    pipeline.unet, quant_level=3.0, percentile=1.0, alpha=0.8
)

# Apply the quantization recipe and run calibration
quantized_model = mtq.quantize(
    pipeline.unet, quant_config, lambda: forward_loop(pipeline, calib_prompts)
)

# Save the quantized model
print(f"Saving quantized UNet to: {int8_unet_path}")
mto.save(quantized_model, int8_unet_path)

print("Quantized UNet saved successfully.")
