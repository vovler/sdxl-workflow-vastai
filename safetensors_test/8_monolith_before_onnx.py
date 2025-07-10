#!/usr/bin/env python3
import torch
import torch.nn as nn
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    EulerDiscreteScheduler,
)
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from pathlib import Path
import sys
from PIL import Image
import time
from tqdm import tqdm
import numpy as np
import argparse

def print_tensor_stats(name, tensor):
    """Prints detailed statistics for a given tensor."""
    print(f"--- {name} ---")
    if tensor is None:
        print("  Tensor is None")
        return
    print(f"  Shape: {tensor.shape}")
    if tensor.numel() > 0:
        # Cast to float32 for stats to avoid issues with other types
        tensor_float = tensor.float()
        print(f"  Mean:  {tensor_float.mean().item():.4f}")
        print(f"  Min:   {tensor_float.min().item():.4f}")
        print(f"  Max:   {tensor_float.max().item():.4f}")
        print(f"  Has NaN: {torch.isnan(tensor_float).any().item()}")
        print(f"  Has Inf: {torch.isinf(tensor_float).any().item()}")
    print(f"  Dtype: {tensor.dtype}")

# DenoisingLoop class remains unchanged.
class DenoisingLoop(nn.Module):
    def __init__(self, unet: nn.Module, scheduler: EulerDiscreteScheduler):
        super().__init__()
        self.unet = unet
        self.init_sigma = scheduler.init_noise_sigma

    def forward(
        self,
        initial_latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        add_time_ids: torch.Tensor,
        timesteps: torch.Tensor,
        sigmas: torch.Tensor
    ) -> torch.Tensor:
        print(f"Latents before noise sigma scaling: min={initial_latents.min():.4f}, max={initial_latents.max():.4f}, mean={initial_latents.mean():.4f}")
        print(f"Initial noise sigma: {self.init_sigma}")
        latents = initial_latents * self.init_sigma
        print(f"Latents after noise sigma scaling:  min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}")
        
        for i in tqdm(range(timesteps.shape[0])):
            t = timesteps[i]
            sigma_t = sigmas[i]
            
            latent_model_input = latents
            
            # scale the model input by the current sigma
            latent_model_input = latent_model_input / ((sigma_t**2 + 1) ** 0.5)
            print(f"\n--- Step {i} ---")
            print(f"Latent Input (scaled): min={latent_model_input.min():.4f}, max={latent_model_input.max():.4f}, mean={latent_model_input.mean():.4f}")
            # --- Prepare UNet inputs ---
            timestep_input = t.unsqueeze(0)
            added_cond_kwargs_input = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
            
            # --- Debug prints for UNet inputs ---
            print(f"\n--- Monolith UNet Inputs: Step {i} ---")
            print_tensor_stats("latent_model_input", latent_model_input)
            print_tensor_stats("timestep", timestep_input)
            print_tensor_stats("encoder_hidden_states", text_embeddings)
            print_tensor_stats("added_cond_kwargs['text_embeds']", added_cond_kwargs_input["text_embeds"])
            print_tensor_stats("added_cond_kwargs['time_ids']", added_cond_kwargs_input["time_ids"])

            noise_pred = self.unet(latent_model_input, timestep_input,
                                   encoder_hidden_states=text_embeddings,
                                   added_cond_kwargs=added_cond_kwargs_input,
                                   return_dict=False)[0]
            print(f"Noise Pred: min={noise_pred.min():.4f}, max={noise_pred.max():.4f}, mean={noise_pred.mean():.4f}")

            if i < sigmas.shape[0] - 1:
                sigma_next = sigmas[i + 1]
            else:
                sigma_next = torch.tensor(0.0, device=sigmas.device)
            
            # 2. compute previous image: x_t -> x_t-1
            # "Euler" method
            derivative = (latents - noise_pred) / sigma_t
            dt = sigma_next - sigma_t
            latents = latents + derivative * dt
            print(f"Latents after Euler step: min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}")
        return latents

# --- The Final, "Ready-to-Save" Monolithic Module ---
class MonolithicSDXL(nn.Module):
    def __init__(self, text_encoder_1, text_encoder_2, unet, vae, scheduler):
        super().__init__()
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.vae_decoder = vae.decode
        self.denoising_loop = DenoisingLoop(unet, scheduler)
        self.vae_scale_factor = vae.config.scaling_factor
        self.latent_channels = unet.config.in_channels
        self.text_encoder_2_projection_dim = text_encoder_2.config.projection_dim

    def forward(
        self,
        prompt_ids_1: torch.Tensor,
        prompt_ids_2: torch.Tensor,
        timesteps: torch.Tensor,
        sigmas: torch.Tensor,
        height: torch.Tensor,
        width: torch.Tensor,
        generator: torch.Generator
    ) -> torch.Tensor:
        
        # --- (Internal setup, text encoding, and denoising loop) ---
        h = height.item()
        w = width.item()
        bs = prompt_ids_1.shape[0]
        device = prompt_ids_1.device
        
        add_time_ids = torch.tensor([[h, w, 0, 0, h, w]], device=device, dtype=torch.float16)
        add_time_ids = add_time_ids.repeat(bs, 1)

        latents_shape = (bs, self.latent_channels, h // 8, w // 8)
        initial_latents = torch.randn(latents_shape, generator=generator, device=device, dtype=torch.float16)
        
        # --- Encode prompts ---
        # Get the output from the first text encoder
        prompt_embeds_1_out = self.text_encoder_1(prompt_ids_1, output_hidden_states=True)
        # Use the last hidden state as requested
        prompt_embeds_1 = prompt_embeds_1_out.hidden_states[-1]

        # Get the output from the second text encoder
        text_encoder_2_out = self.text_encoder_2(prompt_ids_2, output_hidden_states=True)
        # Use the second-to-last hidden state as requested
        prompt_embeds_2 = text_encoder_2_out.hidden_states[-2]
        # Get the pooled and projected output
        pooled_prompt_embeds = text_encoder_2_out.text_embeds

        # Concatenate the 3D prompt embeddings
        prompt_embeds = torch.cat((prompt_embeds_1, prompt_embeds_2), dim=-1)
        
        final_latents = self.denoising_loop(initial_latents, prompt_embeds, pooled_prompt_embeds, add_time_ids, timesteps, sigmas)
        
        final_latents = final_latents / self.vae_scale_factor
        image = self.vae_decoder(final_latents, return_dict=False)[0]

        # --- The NEW "Ready-to-Save" Post-Processing Block ---
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1)
        image = (image * 255.0).round().to(torch.uint8)
        
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
    parser.add_argument("--batch", type=int, default=1, help="Number of images to generate in a loop.")
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

        scheduler = EulerDiscreteScheduler.from_config(
            str(base_dir / "scheduler"), timestep_spacing="trailing" # linspace or trailing
        )
        print(f"✓ Scheduler set to EulerDiscreteScheduler with 'trailing' spacing.")

        # --- Instantiate Monolithic Module ---
        print("Instantiating monolithic module...")
        monolith = MonolithicSDXL(
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            unet=unet,
            vae=vae,
            scheduler=scheduler,
        )

        # --- Manual Inference Process ---
        print("\n=== Starting Manual Inference ===")
        
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = torch.tensor([999, 749, 499, 249, 187, 125, 63, 1], device=device)
        
        sigmas = np.array(((1 - scheduler.alphas_cumprod.cpu().numpy()) / scheduler.alphas_cumprod.cpu().numpy()) ** 0.5)
        sigmas = np.interp(timesteps.cpu().numpy(), np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        sigmas = torch.from_numpy(sigmas).to(device=device)

        print(f"Using custom timesteps: {timesteps.tolist()}")
        print(f"Recalculated sigmas: {sigmas.tolist()}")

        # Tokenize prompts
        prompt_ids_1 = tokenizer_1(prompt, padding="max_length", max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt").input_ids
        prompt_ids_2 = tokenizer_2(prompt, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids

        for batch_idx in range(args.batch):
            if args.random:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
            else:
                seed = args.seed + batch_idx
            
            print(f"\n--- Generating image {batch_idx+1}/{args.batch} with seed: {seed} ---")
            generator = torch.Generator(device="cuda").manual_seed(seed)
            
            script_name = Path(__file__).stem
            image_idx = 0
            while True:
                output_path = f"{script_name}__{image_idx:04d}.png"
                if not Path(output_path).exists():
                    break
                image_idx += 1

            start_time = time.time()
            
            # --- Call the monolith ---
            image_tensor_uint8 = monolith(
                prompt_ids_1=prompt_ids_1.to(device),
                prompt_ids_2=prompt_ids_2.to(device),
                timesteps=timesteps,
                sigmas=sigmas,
                height=torch.tensor(height),
                width=torch.tensor(width),
                generator=generator,
            )

            end_time = time.time()
            print(f"Monolith execution took: {end_time - start_time:.4f} seconds")
            
            # --- Save final image ---
            print(f"Saving final image to {output_path}...")
            image = Image.fromarray(image_tensor_uint8.cpu().numpy()[0])
            image.save(output_path)
            
            print("✓ Image generated successfully!")
        
        print("\n✓ Batch generation complete!")

if __name__ == "__main__":
    main()
