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
        self.register_buffer('init_noise_sigma', torch.tensor(init_noise_sigma_val, dtype=dtype))

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
    
    tokenizer_1 = CLIPTokenizer.from_pretrained(str(base_dir), subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(str(base_dir), subfolder="tokenizer_2")
    
    text_encoder_1 = CLIPTextModel.from_pretrained(
        str(base_dir), subfolder="text_encoder", torch_dtype=dtype, use_safetensors=True
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        str(base_dir), subfolder="text_encoder_2", torch_dtype=dtype, use_safetensors=True
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        str(base_dir / "unet"), torch_dtype=dtype, use_safetensors=True
    ).to(device)
    
    # --- Memory Optimization ---
    print("Enabling memory-efficient attention...")
    unet.enable_xformers_memory_efficient_attention()
    
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
    ).eval()

    # --- Clean up memory ---
    print("Cleaning up memory before export...")
    del text_encoder_1, text_encoder_2, vae # Keep unet for config
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
    
    del unet # No longer needed
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
    print(f"\n=== Exporting model to {args.output_path} with opset 18 ===")
    
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
            args.output_path,
            opset_version=18,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=True
        )
        print(f"✓ Model exported successfully to {args.output_path}")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
