import torch
from diffusers import StableDiffusionXLPipeline
import os
import sys

# Add necessary imports for quantization
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
from functools import partial
from tqdm import tqdm


# Define utility functions for quantization
def load_calib_prompts(batch_size, calib_data_path):
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
    quant_config = {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 8, "axis": 0},
            "*input_quantizer": {"num_bits": 8, "axis": None},
        },
        "algorithm": {"method": "smoothquant", "alpha": alpha},
    }
    print(f"Using SmoothQuant with alpha={alpha}")
    return quant_config


def forward_loop(model, prompts, num_inference_steps=8):
    """A basic forward loop for calibration."""
    with torch.no_grad():
        for batch_prompts in tqdm(prompts, desc="Calibrating"):
            _ = model(
                prompt=batch_prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=1.0,
            )


def main():
    # Default path
    default_model_path = "/lab/model"

    if len(sys.argv) >= 2:
        model_path = sys.argv[1]
    else:
        model_path = default_model_path
    
    print(f"Loading model from: {model_path}")
    # Load the main pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=False,
    )
    pipeline.to("cuda")

    # Define the path to save the quantized UNet
    unet_dir = os.path.join(model_path, "unet")
    int8_unet_path = os.path.join(unet_dir, "model_int8.pth")

    # Quantize the UNet
    print("Starting UNet quantization...")

    # Load calibration prompts
    calib_prompts = load_calib_prompts(
        batch_size=1, calib_data_path="3_unet_quantization_int8_PROMPTS.txt"
    )

    # Create the int8 quantization recipe
    quant_config = get_percentilequant_config(
        pipeline.unet, quant_level=3.0, percentile=1.0, alpha=0.8, collect_method="min-mean"
    )

    # Apply the quantization recipe and run calibration
    quantized_model = mtq.quantize(
        pipeline.unet, quant_config, lambda: forward_loop(pipeline, calib_prompts)
    )

    # Save the quantized model
    print(f"Saving quantized UNet to: {int8_unet_path}")
    mto.save(quantized_model, int8_unet_path)

    print("Quantized UNet saved successfully.")


if __name__ == "__main__":
    main()
