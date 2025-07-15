#!/usr/bin/env python3
import torch
from diffusers import StableDiffusionXLPipeline
import os
import sys
import re
from typing import Any

# Add necessary imports for quantization
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
from modelopt.torch.quantization import utils as quant_utils
from modelopt.torch.quantization.calib.max import MaxCalibrator
from tqdm import tqdm


# --- Helper functions from NVIDIA's quantization script ---


def filter_func(name):
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|pos_embed|time_text_embed|context_embedder|norm_out|x_embedder).*"
    )
    return pattern.match(name) is not None


def get_int8_config(
    model,
    quant_level=3.0,
    percentile=1.0,
    num_inference_steps=20,
    collect_method="global_min",
):
    quant_config: dict[str, dict[str, Any]] = {
        "quant_cfg": {
            "*output_quantizer": {"enable": False},
            "default": {"enable": False},
        }
    }
    for name, module in model.named_modules():
        w_name = f"{name}*weight_quantizer"
        i_name = f"{name}*input_quantizer"

        if w_name in quant_config["quant_cfg"] or i_name in quant_config["quant_cfg"]:
            continue
        if filter_func(name):
            continue
        if isinstance(module, torch.nn.Linear):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (
                    quant_level >= 2.5
                    and ("to_q" in name or "to_k" in name or "to_v" in name)
                )
                or quant_level == 3
            ):
                quant_config["quant_cfg"][w_name] = {
                    "num_bits": 8,
                    "axis": 0,
                }
                quant_config["quant_cfg"][i_name] = {
                    "num_bits": 8,
                    "axis": -1,
                }
        elif isinstance(module, torch.nn.Conv2d):
            quant_config["quant_cfg"][w_name] = {
                "num_bits": 8,
                "axis": 0,
            }
            quant_config["quant_cfg"][i_name] = {
                "num_bits": 8,
                "axis": None,
                "calibrator": (
                    PercentileCalibrator,
                    (),
                    {
                        "num_bits": 8,
                        "axis": None,
                        "percentile": percentile,
                        "total_step": num_inference_steps,
                        "collect_method": collect_method,
                    },
                ),
            }
    return quant_config


def set_quant_config_attr(
    quant_config, trt_high_precision_dtype, quant_algo, **kwargs
):
    algo_cfg = {"method": quant_algo}

    if quant_algo == "smoothquant" and "alpha" in kwargs:
        algo_cfg["alpha"] = kwargs["alpha"]
    elif quant_algo == "svdquant" and "lowrank" in kwargs:
        algo_cfg["lowrank"] = kwargs["lowrank"]
    quant_config["algorithm"] = algo_cfg

    for p in quant_config["quant_cfg"].values():
        if "num_bits" in p and "trt_high_precision_dtype" not in p:
            p["trt_high_precision_dtype"] = trt_high_precision_dtype


# --- End helper functions ---


def load_calib_prompts(batch_size, calib_data_path):
    """
    Load calibration prompts from a file.
    """
    with open(calib_data_path, "r", encoding="utf-8") as f:
        lst = [line.rstrip("\n") for line in f]
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def do_calibrate(
    pipe: StableDiffusionXLPipeline,
    calibration_prompts: list[str],
    calib_size: int,
    n_steps: int,
) -> None:
    """
    Run calibration steps on the pipeline using the given prompts.
    """
    print(f"Running calibration for {calib_size} batches...")
    with torch.no_grad():
        for i, prompts in enumerate(tqdm(calibration_prompts, desc="Calibrating")):
            if i >= calib_size:
                break
            _ = pipe(
                prompt=prompts,
                num_inference_steps=n_steps,
                guidance_scale=1.0,
            ).images


def main():
    # --- Configuration ---
    # Default path
    default_model_path = "/lab/model"

    if len(sys.argv) >= 2:
        model_path = sys.argv[1]
    else:
        model_path = default_model_path

    # Quantization parameters from user command
    batch_size = 1
    calib_size = 64
    collect_method = "min-mean"
    percentile = 1.0
    alpha = 0.8
    quant_level = 3.0
    n_steps = 8
    quant_algo = "smoothquant"
    model_dtype = torch.float16

    print(f"Loading model from: {model_path}")
    # Load the main pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        use_safetensors=True,
        low_cpu_mem_usage=False,
    )
    pipeline.to("cuda")

    # Choose correct backbone for the loaded pipeline
    backbone = pipeline.unet

    # Define the path to save the quantized UNet
    unet_dir = os.path.join(model_path, "unet")
    int8_unet_path = os.path.join(unet_dir, "model_int8.pth")

    # Quantize the UNet
    print("Starting UNet quantization...")

    # Load calibration prompts
    calib_prompts = load_calib_prompts(
        batch_size=batch_size, calib_data_path="3_unet_quantization_int8_PROMPTS.txt"
    )

    # Adjust calibration steps to be number of batches
    num_calib_batches = calib_size // batch_size

    # Build quant_config based on format
    print(f"Using {quant_algo} with alpha={alpha}")
    quant_config = get_int8_config(
        backbone,
        quant_level=quant_level,
        percentile=percentile,
        num_inference_steps=n_steps,
        collect_method=collect_method,
    )
    # Adjust the quant config
    set_quant_config_attr(
        quant_config, "Half", quant_algo, alpha=alpha  # Assuming FP16
    )

    def forward_loop(mod):
        # Switch the pipeline's backbone, run calibration
        pipeline.unet = mod
        do_calibrate(
            pipe=pipeline,
            calibration_prompts=calib_prompts,
            calib_size=num_calib_batches,
            n_steps=n_steps,
        )

    # Apply the quantization recipe and run calibration
    quantized_model = mtq.quantize(backbone, quant_config, forward_loop)

    # Save the quantized model
    print(f"Saving quantized UNet to: {int8_unet_path}")
    mto.save(quantized_model, int8_unet_path)

    print("Quantized UNet saved successfully.")


class PercentileCalibrator(MaxCalibrator):
    def __init__(self, num_bits=8, axis=None, unsigned=False, track_amax=False, **kwargs):
        super().__init__(num_bits, axis, unsigned, track_amax)
        self.percentile = kwargs["percentile"]
        self.total_step = kwargs["total_step"]
        self.collect_method = kwargs["collect_method"]
        self.data = {}
        self.i = 0

    def collect(self, x):
        """Tracks the absolute max of all tensors.

        Args:
            x: A tensor

        Raises:
            RuntimeError: If amax shape changes
        """
        # Swap axis to reduce.
        reduce_axis = quant_utils.convert_quantization_axis_to_reduce_axis(x, self._axis)
        local_amax = quant_utils.reduce_amax(x, axis=reduce_axis).detach()
        _cur_step = self.i % self.total_step
        if _cur_step not in self.data:
            self.data[_cur_step] = local_amax
        elif self.collect_method == "global_min":
            self.data[_cur_step] = torch.min(self.data[_cur_step], local_amax)
        elif self.collect_method in {"min-max", "mean-max"}:
            self.data[_cur_step] = torch.max(self.data[_cur_step], local_amax)
        else:
            self.data[_cur_step] += local_amax
        if self._track_amax:
            raise NotImplementedError
        self.i += 1

    def compute_amax(self):
        """Return the absolute max of all tensors collected."""
        up_lim = int(self.total_step * self.percentile)
        if self.collect_method == "min-mean":
            amaxs_values = [self.data[i] / self.total_step for i in range(up_lim)]
        else:
            amaxs_values = [self.data[i] for i in range(up_lim)]
        if self.collect_method == "mean-max":
            act_amax = torch.vstack(amaxs_values).mean(axis=0)[0]
        else:
            act_amax = torch.vstack(amaxs_values).min(axis=0)[0]
        self._calib_amax = act_amax
        return self._calib_amax

    def __str__(self):
        s = "PercentileCalibrator"
        return s.format(**self.__dict__)

    def __repr__(self):
        s = "PercentileCalibrator("
        s += super(MaxCalibrator, self).__repr__()
        s += " calib_amax={_calib_amax}"
        if self._track_amax:
            s += " amaxs={_amaxs}"
        s += ")"
        return s.format(**self.__dict__)

if __name__ == "__main__":
    main()
