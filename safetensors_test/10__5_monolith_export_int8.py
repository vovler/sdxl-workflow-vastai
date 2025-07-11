#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from pathlib import Path
import sys
import argparse
import gc
import os
import onnx
import glob
import shutil
import modelopt.torch.opt as mto
from modelopt.torch.quantization.calib.max import MaxCalibrator
from modelopt.torch.quantization import utils as quant_utils

def print_tensor_stats(name, tensor):
    """Prints detailed statistics for a given tensor on a single line."""
    # This function is a no-op during export to avoid side effects.
    return

class ONNXEulerAncestralDiscreteScheduler(nn.Module):
    """
    A custom implementation of the EulerAncestralDiscreteScheduler that is designed
    to be ONNX-exportable. It encapsulates the entire denoising loop.
    """
    def __init__(self, num_inference_steps: int, dtype, num_train_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012, beta_schedule: str = "scaled_linear", timestep_spacing: str = "linspace"):
        super().__init__()
        
        self.num_inference_steps = num_inference_steps
        
        # Beta schedule (using torch)
        if beta_schedule == "scaled_linear":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented.")
            
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Timesteps (using torch)
        if timestep_spacing == "linspace":
            timesteps = torch.linspace(0, num_train_timesteps - 1, num_inference_steps, dtype=torch.float32).flip(0)
        elif timestep_spacing == "trailing":
            step_ratio = num_train_timesteps / self.num_inference_steps
            timesteps = (torch.arange(num_train_timesteps, 0, -step_ratio, dtype=torch.float32) - 1).round()
        else:
            raise NotImplementedError(f"{timestep_spacing} is not supported.")
        
        # Sigmas (using torch)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        
        # Interpolation of sigmas - PyTorch equivalent of np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        from_timesteps = torch.arange(num_train_timesteps, device=sigmas.device)

        # Find the indices for interpolation
        indices = torch.searchsorted(from_timesteps, timesteps, right=True)
        indices = torch.clamp(indices, 1, num_train_timesteps - 1)

        # Get the adjacent timesteps and sigmas for linear interpolation
        t_low = from_timesteps[indices - 1]
        t_high = from_timesteps[indices]
        s_low = sigmas[indices - 1]
        s_high = sigmas[indices]

        # Linearly interpolate
        w_denom = t_high.float() - t_low.float()
        w_denom[w_denom == 0] = 1.0  # Avoid division by zero
        w = (timesteps.float() - t_low.float()) / w_denom
        interpolated_sigmas = s_low.float() + w * (s_high.float() - s_low.float())
        
        # Concatenate with zero sigma for the last step
        sigmas_buffer = torch.cat([interpolated_sigmas, torch.tensor([0.0])])
        
        if timestep_spacing in ["linspace", "trailing"]:
             init_noise_sigma_val = sigmas_buffer.max()
        else: # "leading"
             init_noise_sigma_val = (sigmas_buffer.max() ** 2 + 1) ** 0.5
        
        self.register_buffer('sigmas', sigmas_buffer.to(dtype=dtype))
        self.register_buffer('timesteps', timesteps)
        self.register_buffer('init_noise_sigma', init_noise_sigma_val.clone().detach().to(dtype=dtype))

    def scale_model_input(self, sample: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Scales the model input by 1 / sqrt(sigma^2 + 1)."""
        return sample / torch.sqrt(sigma**2 + 1)

    def forward(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        add_time_ids: torch.Tensor,
        unet: nn.Module,
        all_noises: torch.Tensor
    ) -> torch.Tensor:
        """
        Denoising loop that is compatible with ONNX export.
        """
        latents = latents * self.init_noise_sigma
        
        # The main denoising loop
        for i in range(self.num_inference_steps):
            sigma = self.sigmas[i]
            t = self.timesteps[i].unsqueeze(0) # UNet expects a tensor for timestep
            
            # 1. Scale the latent input
            latent_model_input = self.scale_model_input(latents, sigma)
            
            # 2. Predict noise with the UNet
            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
            noise_pred = unet(
                latent_model_input, t,
                encoder_hidden_states=text_embeddings,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False
            )[0]
            
            # 3. Scheduler step (Euler Ancestral method)
            # 3a. Denoise with a standard Euler step
            pred_original_sample = latents - sigma * noise_pred
            derivative = (latents - pred_original_sample) / sigma
            
            sigma_from = self.sigmas[i]
            sigma_to = self.sigmas[i + 1]
            sigma_up = torch.sqrt((sigma_to**2 * (sigma_from**2 - sigma_to**2)) / sigma_from**2)
            sigma_down = torch.sqrt(sigma_to**2 - sigma_up**2)
            
            dt = sigma_down - sigma
            denoised_latents = latents + derivative * dt
            
            # 3b. Add ancestral noise
            noise = all_noises[i]
            latents = denoised_latents + noise * sigma_up
            
        return latents

# --- The Final, "Ready-to-Save" Monolithic Module ---
class MonolithicSDXL(nn.Module):
    def __init__(self, text_encoder_1, text_encoder_2, unet, vae, scheduler_module):
        super().__init__()
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.unet = unet
        self.vae = vae
        self.denoising_module = scheduler_module
        self.vae_scale_factor = vae.config.scaling_factor
        
    def forward(
        self,
        prompt_ids_1: torch.Tensor,
        prompt_ids_2: torch.Tensor,
        initial_latents: torch.Tensor,
        all_noises: torch.Tensor,
        add_time_ids: torch.Tensor,
    ) -> torch.Tensor:
        
        # --- Encode prompts ---
        prompt_embeds_1_out = self.text_encoder_1(prompt_ids_1, output_hidden_states=True)
        prompt_embeds_1 = prompt_embeds_1_out.hidden_states[-2]

        # Get the output from the second text encoder
        text_encoder_2_out = self.text_encoder_2(prompt_ids_2, output_hidden_states=True)
        prompt_embeds_2 = text_encoder_2_out.hidden_states[-2]
        pooled_prompt_embeds = text_encoder_2_out.text_embeds

        # Concatenate the 3D prompt embeddings
        prompt_embeds = torch.cat((prompt_embeds_1, prompt_embeds_2), dim=-1)
        
        final_latents = self.denoising_module(
            initial_latents, 
            prompt_embeds, 
            pooled_prompt_embeds, 
            add_time_ids, 
            self.unet,
            all_noises
        )

        final_latents = final_latents / self.vae_scale_factor
        image = self.vae.decode(final_latents, return_dict=False)[0]

        return image


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
            10: 2,  # FLOAT16
            7: 8,   # INT64
            3: 1,   # INT8
        }
        
        for initializer in model.graph.initializer:
            param_count = 1
            for dim in initializer.dims:
                param_count *= dim
            total_params += param_count
            
            data_type = initializer.data_type
            type_name = onnx.TensorProto.DataType.Name(data_type)
            bytes_per_param = bytes_per_type.get(data_type, 0) # Default to 0 for unknown types
            param_size = param_count * bytes_per_param
            
            precision_counts[type_name] = precision_counts.get(type_name, 0) + param_count
            precision_sizes[type_name] = precision_sizes.get(type_name, 0) + param_size
        
        total_size_bytes = sum(precision_sizes.values())
        total_size_gb = total_size_bytes / (1024**3)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Total calculated size: {total_size_gb:.3f} GB")
        
        print("\nPrecision breakdown:")
        for precision, count in precision_counts.items():
            size_gb = precision_sizes[precision] / (1024**3)
            percentage = (precision_sizes[precision] / total_size_bytes) * 100 if total_size_bytes > 0 else 0
            print(f"  {precision}: {count:,} params ({size_gb:.3f} GB, {percentage:.1f}%)")
            
        # Check file sizes
        onnx_file_size = os.path.getsize(onnx_path) / (1024**3)
        print(f"\nActual ONNX file size: {onnx_file_size:.3f} GB")
        
        # Check for external data file, using the convention we will establish
        directory = os.path.dirname(onnx_path)
        filename = os.path.basename(onnx_path)
        base_name = os.path.splitext(filename)[0]
        data_filename = f"{base_name}.data"
        data_path = os.path.join(directory, data_filename)
        
        if os.path.exists(data_path):
            data_file_size = os.path.getsize(data_path) / (1024**3)
            print(f"External data file size: {data_file_size:.3f} GB")
            print(f"Total combined size: {(onnx_file_size + data_file_size):.3f} GB")
        
    except Exception as e:
        print(f"Error analyzing ONNX model: {e}")
    
    print("="*50)


def main():
    """
    Exports the MonolithicSDXL model to ONNX format.
    """
    parser = argparse.ArgumentParser(description="Export the Monolithic SDXL model to ONNX.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="monolith.onnx",
        help="The path to save the exported ONNX model."
    )
    args = parser.parse_args()

    with torch.no_grad():
        if not torch.cuda.is_available():
            print("Error: CUDA is not available. This script requires a GPU for model loading.")
            sys.exit(1)

        # --- Configuration ---
        base_dir = Path("/lab/model")
        device = "cuda"
        dtype = torch.float16

        # Pipeline settings
        num_inference_steps = 8
        height = 832
        width = 1216
        batch_size = 1
        
        # --- Load Model Components ---
        print("=== Loading models ===")
        
        vae = AutoencoderKL.from_pretrained(base_dir / "vae", torch_dtype=dtype).to(device)
        print("Enabling VAE tiling for memory-efficient decoding...")
        vae.enable_tiling()
        
        tokenizer_1 = CLIPTokenizer.from_pretrained(str(base_dir), subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(str(base_dir), subfolder="tokenizer_2")
        
        text_encoder_1 = CLIPTextModel.from_pretrained(
            str(base_dir), subfolder="text_encoder", torch_dtype=dtype, use_safetensors=True
        ).to(device)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            str(base_dir), subfolder="text_encoder_2", torch_dtype=dtype, use_safetensors=True
        ).to(device)
        
        print("Loading base FP16 UNet for INT8 restoration...")
        unet = UNet2DConditionModel.from_pretrained(
            str(base_dir / "unet"), torch_dtype=dtype, use_safetensors=True
        ).to(device)

        int8_checkpoint_path = base_dir / "unet" / "model_int8.pth"
        if not os.path.exists(str(int8_checkpoint_path)):
            print(f"Error: Quantized UNet checkpoint not found at {int8_checkpoint_path}")
            print("Please run the quantization script first (e.g., 3_unet_quantization_int8.py).")
            sys.exit(1)
        
        print(f"Restoring INT8 weights from {int8_checkpoint_path}...")
        mto.restore(unet, str(int8_checkpoint_path))
        print("INT8 UNet restored successfully.")
        
        # --- Memory Optimization ---
        # print("Enabling memory-efficient attention...")
        # unet.enable_xformers_memory_efficient_attention()
        
        # --- Instantiate Monolithic Module ---
        print("Instantiating monolithic module...")
        
        onnx_scheduler = ONNXEulerAncestralDiscreteScheduler(
            num_inference_steps=num_inference_steps,
            dtype=dtype,
            timestep_spacing="linspace",
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012
        )
        
        monolith = MonolithicSDXL(
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            unet=unet,
            vae=vae,
            scheduler_module=onnx_scheduler,
        ).to(device).eval()

        # --- Clean up memory ---
        print("Cleaning up memory before export...")
        del text_encoder_1, text_encoder_2, vae
        gc.collect()
        torch.cuda.empty_cache()

        # --- Create Dummy Inputs for ONNX Export ---
        print("\n=== Creating dummy inputs for ONNX export ===")

        max_length_1 = tokenizer_1.model_max_length
        max_length_2 = tokenizer_2.model_max_length
        
        dummy_prompt_ids_1 = torch.randint(0, tokenizer_1.vocab_size, (batch_size, max_length_1), dtype=torch.int64, device=device)
        dummy_prompt_ids_2 = torch.randint(0, tokenizer_2.vocab_size, (batch_size, max_length_2), dtype=torch.int64, device=device)
        
        del tokenizer_1, tokenizer_2
        gc.collect()

        latents_shape = (batch_size, unet.config.in_channels, height // 8, width // 8)
        dummy_initial_latents = torch.randn(latents_shape, device=device, dtype=dtype)
        
        noise_shape = (num_inference_steps, batch_size, unet.config.in_channels, height // 8, width // 8)
        dummy_all_noises = torch.randn(noise_shape, device=device, dtype=dtype)
        
        dummy_add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=device, dtype=dtype).repeat(batch_size, 1)
        
        del unet
        gc.collect()
        torch.cuda.empty_cache()

        dummy_inputs = (
            dummy_prompt_ids_1,
            dummy_prompt_ids_2,
            dummy_initial_latents,
            dummy_all_noises,
            dummy_add_time_ids,
        )
        
        # --- Export to ONNX ---
        onnx_output_path = args.output_path
        # temp_dir = onnx_output_path + "_temp"
        # os.makedirs(temp_dir, exist_ok=True)
        # temp_onnx_path = os.path.join(temp_dir, "model.onnx")

        # print(f"\n=== Exporting model to temporary directory: {temp_dir} ===")
        print(f"\n=== Exporting model to: {onnx_output_path} ===")
        
        input_names = [
            "prompt_ids_1", "prompt_ids_2", "initial_latents",
            "all_noises", "add_time_ids"
        ]
        output_names = ["image"]
        
        dynamic_axes = {
            "prompt_ids_1": {0: "batch_size"},
            "prompt_ids_2": {0: "batch_size"},
            "initial_latents": {0: "batch_size", 2: "height_div_8", 3: "width_div_8"},
            "all_noises": {1: "batch_size", 3: "height_div_8", 4: "width_div_8"},
            "add_time_ids": {0: "batch_size"},
            "image": {0: "batch_size", 2: "height", 3: "width"},
        }

        try:
            torch.onnx.export(
                monolith,
                dummy_inputs,
                onnx_output_path,
                opset_version=18,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False,
                do_constant_folding=False,
                verify=False,
                optimize=False,
                # dynamo=True
            )
            print("✓ ONNX export complete.")
            
            # # --- Consolidate Model Files ---
            # print(f"Consolidating model from {temp_dir} to {onnx_output_path}...")
            # # Move the main ONNX file
            # final_onnx_path_unversioned = os.path.join(os.path.dirname(onnx_output_path), os.path.splitext(os.path.basename(onnx_output_path))[0])
            
            # shutil.move(temp_onnx_path, onnx_output_path)
            
            # # Move any external data files (.data)
            # for data_file in glob.glob(os.path.join(temp_dir, "*.data")):
            #     final_data_path = os.path.join(os.path.dirname(onnx_output_path), os.path.basename(data_file))
            #     print(f"Moving data file: {data_file} to {final_data_path}")
            #     shutil.move(data_file, final_data_path)

            # print("✓ Model consolidation complete.")

        except Exception as e:
            print(f"✗ ONNX export or consolidation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            # --- Clean up temporary directory ---
            # if os.path.exists(temp_dir):
            #     print(f"Cleaning up temporary directory: {temp_dir}")
            #     shutil.rmtree(temp_dir)
            pass

        # --- Final Analysis ---
        print(f"\nFinal analysis of consolidated model: {onnx_output_path}")
        analyze_onnx_model(onnx_output_path)


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
