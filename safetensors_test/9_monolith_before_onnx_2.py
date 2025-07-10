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
from PIL import Image
import time
from tqdm import tqdm
import argparse

def print_tensor_stats(name, tensor):
    """Prints detailed statistics for a given tensor on a single line."""
    if tensor is None:
        print(f"--- {name}: Tensor is None ---")
        return
    
    stats = f"Shape: {str(tuple(tensor.shape)):<20} | Dtype: {str(tensor.dtype):<15}"
    if tensor.numel() > 0:
        tensor_float = tensor.float()
        stats += f" | Mean: {tensor_float.mean().item():<8.4f} | Min: {tensor_float.min().item():<8.4f} | Max: {tensor_float.max().item():<8.4f}"
        stats += f" | Has NaN: {str(torch.isnan(tensor_float).any().item()):<5} | Has Inf: {str(torch.isinf(tensor_float).any().item()):<5}"
    
    print(f"--- {name+':':<30} {stats} ---")

class ONNXEulerAncestralDiscreteScheduler(nn.Module):
    """
    A custom implementation of the EulerAncestralDiscreteScheduler that is designed
    to be ONNX-exportable. It encapsulates the entire denoising loop and handles
    random noise generation from a seed tensor.
    """
    def __init__(self, num_inference_steps: int, device, dtype, num_train_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012, beta_schedule: str = "scaled_linear", timestep_spacing: str = "linspace"):
        super().__init__()
        
        self.num_inference_steps = num_inference_steps
        self.device = device
        self.dtype = dtype
        
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
        self.sigmas = torch.cat([interpolated_sigmas, torch.tensor([0.0])]).to(device=device, dtype=dtype)
        self.timesteps = timesteps.to(device=device)

        # Initial noise sigma
        if timestep_spacing in ["linspace", "trailing"]:
             init_noise_sigma_val = self.sigmas.max()
        else: # "leading"
             init_noise_sigma_val = (self.sigmas.max() ** 2 + 1) ** 0.5
        
        self.init_noise_sigma = torch.tensor(init_noise_sigma_val, device=device, dtype=dtype)

        print("--- Custom ONNX Scheduler Initialized (Standalone) ---")
        print(f"Num inference steps: {self.num_inference_steps}")
        print_tensor_stats("Timesteps", self.timesteps)
        print_tensor_stats("Sigmas", self.sigmas)
        print(f"Initial noise sigma: {self.init_noise_sigma.item():.4f}")
        print("-----------------------------------------")


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
        print_tensor_stats("Initial Latents (before noise sigma scaling)", latents)
        latents = latents * self.init_noise_sigma
        print_tensor_stats("Initial Latents (after noise sigma scaling)", latents)
        
        # The main denoising loop
        for i in range(self.num_inference_steps):
            sigma = self.sigmas[i]
            t = self.timesteps[i].unsqueeze(0) # UNet expects a tensor for timestep
            
            # 1. Scale the latent input
            latent_model_input = self.scale_model_input(latents, sigma)
            
            print(f"\n--- Monolith DenoisingLoop: Step {i} ---")
            print_tensor_stats("Latent Input (scaled)", latent_model_input)
            
            # 2. Predict noise with the UNet
            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
            noise_pred = unet(
                latent_model_input, t,
                encoder_hidden_states=text_embeddings,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False
            )[0]
            print_tensor_stats("Noise Pred", noise_pred)
            
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
            print_tensor_stats("Latents after scheduler step", latents)
            
        return latents

# --- The Final, "Ready-to-Save" Monolithic Module ---
class MonolithicSDXL(nn.Module):
    def __init__(self, text_encoder_1, text_encoder_2, unet, vae, scheduler_module):
        super().__init__()
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.unet = unet
        self.vae_decoder = vae.decode
        self.denoising_module = scheduler_module
        self.vae_scale_factor = vae.config.scaling_factor
        self.latent_channels = unet.config.in_channels
        
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

        print(f"VAE Scale Factor: {self.vae_scale_factor}")
        final_latents = final_latents / self.vae_scale_factor
        image = self.vae_decoder(final_latents, return_dict=False)[0]

        return image


def main():
    """
    Generates an image with SDXL using a monolithic module.
    """
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Generate an image with a custom prompt.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="masterpiece,best quality,amazing quality, general, 1girl, aqua_(konosuba), on a swing, looking at viewer, volumetric_lighting, park, night, shiny clothes, shiny skin, detailed_background",
        help="The prompt to use for image generation."
    )
    parser.add_argument("--random", action="store_true", help="Use a random seed for generation.")
    parser.add_argument("--seed", type=int, default=1020094661, help="The seed to use for generation.")
    args = parser.parse_args()

    with torch.no_grad():
        if not torch.cuda.is_available():
            print("Error: CUDA is not available. This script requires a GPU.")
            sys.exit(1)

        # --- Configuration ---
        base_dir = Path("/lab/model")
        device = "cuda"
        dtype = torch.float16

        prompt = args.prompt
        
        # Pipeline settings
        num_inference_steps = 8
        height = 832
        width = 1216
        
        # --- Load Model Components ---
        print("=== Loading models ===")
        
        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(base_dir / "vae", torch_dtype=dtype)
        vae.to(device)

        print("Loading text encoders and tokenizers...")
        tokenizer_1 = CLIPTokenizer.from_pretrained(str(base_dir), subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(str(base_dir), subfolder="tokenizer_2")
        text_encoder_1 = CLIPTextModel.from_pretrained(
            str(base_dir), subfolder="text_encoder", torch_dtype=dtype, use_safetensors=True
        )
        text_encoder_1.to(device)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            str(base_dir), subfolder="text_encoder_2", torch_dtype=dtype, use_safetensors=True
        )
        text_encoder_2.to(device)

        print("Loading UNet...")
        unet = UNet2DConditionModel.from_pretrained(
            str(base_dir / "unet"), torch_dtype=dtype, use_safetensors=True
        )
        unet.to(device)
        unet.enable_xformers_memory_efficient_attention()

        # --- Manual Inference Process ---
        print("\n=== Starting Manual Inference ===")
        
        # --- Instantiate Monolithic Module ---
        print("Instantiating monolithic module...")
        
        # Instantiate the custom ONNX-exportable scheduler module
        # Using SDXL defaults for beta schedule
        onnx_scheduler = ONNXEulerAncestralDiscreteScheduler(
            num_inference_steps=num_inference_steps,
            device=device,
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
        )

        # Tokenize prompts
        prompt_ids_1 = tokenizer_1(prompt, padding="max_length", max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt").input_ids
        prompt_ids_2 = tokenizer_2(prompt, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids

        if args.random:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            seed = args.seed
        
        print(f"\n--- Generating image with seed: {seed} ---")
        
        # --- Prepare inputs that are now outside the nn.Module ---
        bs = prompt_ids_1.shape[0]
        
        add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=device, dtype=dtype)
        add_time_ids = add_time_ids.repeat(bs, 1)

        generator = torch.Generator(device=device).manual_seed(seed)
        
        latents_shape = (bs, unet.config.in_channels, height // 8, width // 8)
        initial_latents = torch.randn(latents_shape, generator=generator, device=device, dtype=dtype)

        noise_shape = (onnx_scheduler.num_inference_steps, bs, unet.config.in_channels, height // 8, width // 8)
        all_noises = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)

        script_name = Path(__file__).stem
        image_idx = 0
        while True:
            output_path = f"{script_name}__{image_idx:04d}.png"
            if not Path(output_path).exists():
                break
            image_idx += 1

        start_time = time.time()
        
        # --- Call the monolith ---
        raw_image_tensor = monolith(
            prompt_ids_1=prompt_ids_1.to(device),
            prompt_ids_2=prompt_ids_2.to(device),
            initial_latents=initial_latents,
            all_noises=all_noises,
            add_time_ids=add_time_ids,
        )

        end_time = time.time()
        print(f"Monolith execution took: {end_time - start_time:.4f} seconds")
        
        # --- Post-process and save final image ---
        print(f"Saving final image to {output_path}...")
        image = (raw_image_tensor / 2 + 0.5).clamp(0, 1)
        image_uint8 = image.cpu().permute(0, 2, 3, 1).mul(255).round().to(torch.uint8)
        image = Image.fromarray(image_uint8.numpy()[0])
        image.save(output_path)
        
        print("✓ Image generated successfully!")

if __name__ == "__main__":
    main()
