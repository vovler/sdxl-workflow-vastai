#!/usr/bin/env python3
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
)
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from safetensors.torch import load_file
from pathlib import Path
import sys
from PIL import Image
import shutil
import time
from tqdm import tqdm
import numpy as np

def main():
    """
    Generates an image with SDXL using an unfused UNet, a separate VAE, and a LoRA.
    The process is run manually to decode the UNet output with the VAE explicitly.
    """
    with torch.no_grad():
        if not torch.cuda.is_available():
            print("Error: CUDA is not available. This script requires a GPU.")
            sys.exit(1)

        # --- Configuration ---
        base_dir = Path("/lab/model")
        device = "cuda"
        dtype = torch.float16

        prompt = "masterpiece,best quality,amazing quality, general, 1girl, aqua_(konosuba), on a swing, looking at viewer, volumetric_lighting, park, night, shiny clothes, shiny skin, detailed_background"
        #prompt = "masterpiece,best quality,amazing quality, general, 1girl, aqua_(konosuba), face_focus, looking at viewer, volumetric_lighting"
        #prompt = "masterpiece,best quality,amazing quality, general, 1girl, aqua_(konosuba), dark lolita, running makeup, holding pipe, looking at viewer, volumetric_lighting, street, night, shiny clothes, shiny skin, detailed_background"
        
        # Pipeline settings
        cfg_scale = 1.0
        num_inference_steps = 8
        seed = 1020094661
        generator = torch.Generator(device="cuda").manual_seed(seed)
        height = 832
        width = 1216
        batch_size = 1
        
        # --- Load Model Components ---
        print("=== Loading models ===")
        
        # Load VAE from its own directory
        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            base_dir / "vae",
            torch_dtype=dtype
        )
        vae.to(device)

        # Load text encoders and tokenizers
        print("Loading text encoders and tokenizers...")
        tokenizer = CLIPTokenizer.from_pretrained(str(base_dir), subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(str(base_dir), subfolder="tokenizer_2")
        text_encoder = CLIPTextModel.from_pretrained(
            str(base_dir), subfolder="text_encoder", torch_dtype=dtype, use_safetensors=True
        )
        text_encoder.to(device)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            str(base_dir), subfolder="text_encoder_2", torch_dtype=dtype, use_safetensors=True
        )
        text_encoder_2.to(device)

        # Load the unfused UNet weights
        print("Loading unfused UNet...")
        unet_dir = base_dir / "unet"

        unet = UNet2DConditionModel.from_pretrained(
            str(unet_dir), torch_dtype=dtype, use_safetensors=True
        )
        unet.to(device)
        print("✓ Unfused UNet loaded.")

        # Create the scheduler
        scheduler = EulerAncestralDiscreteScheduler.from_config(
            str(base_dir / "scheduler"), timestep_spacing="linspace"
        )
        #scheduler = scheduler.to(device)
        print(f"✓ Scheduler set to EulerAncestralDiscreteScheduler with 'linspace' spacing.")
        
        # Instantiate pipeline from components
        print("Instantiating pipeline from components...")
        pipe = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_vae_tiling()
        pipe.enable_vae_slicing()


        # --- Manual Inference Process ---
        print("\n=== Starting Manual Inference ===")
        # 1. Encode prompts
        print("Encoding prompts...")
        (
            prompt_embeds,
            _,
            pooled_prompt_embeds,
            _,
        ) = pipe.encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=batch_size,
            do_classifier_free_guidance=1
        )

        # 2. Prepare latents
        print("Preparing latents...")
        latents = torch.randn(
            (batch_size, pipe.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        

        # 3. Prepare timesteps and extra embeds for the denoising loop
        # pipe.scheduler.set_timesteps(num_inference_steps, device=device)

        # Manually create a custom list of step numbers and pass it to the scheduler
        custom_timesteps = [876.0, 751.0, 626.0, 501.0, 376.0, 251.0, 126.0, 1.0]
        print(f"Using custom timesteps: {custom_timesteps}")

        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        
        # Overwrite with custom timesteps and recalculate sigmas
        timesteps_np = np.array(custom_timesteps)
        pipe.scheduler.timesteps = torch.from_numpy(timesteps_np).to(device)

        # Recalculate sigmas for the new timesteps
        scheduler = pipe.scheduler
        sigmas = np.array(((1 - scheduler.alphas_cumprod.cpu().numpy()) / scheduler.alphas_cumprod.cpu().numpy()) ** 0.5)
        sigmas = np.interp(timesteps_np, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        scheduler.sigmas = torch.from_numpy(sigmas).to(device=device)
        print(f"Recalculated sigmas: {sigmas.tolist()}")

        timesteps = pipe.scheduler.timesteps
        
        add_time_ids = pipe._get_add_time_ids((height, width), (0,0), (height, width), dtype, text_encoder_projection_dim=text_encoder_2.config.projection_dim).to(device)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        
        # Scale the initial noise by the scheduler's standard deviation
        latents = latents * pipe.scheduler.init_noise_sigma

        # 4. Denoising loop
        print(f"Running denoising loop for {num_inference_steps} steps...")
        
        script_name = Path(__file__).stem
        image_idx = 0
        while True:
            # Check for the first step's file to determine a unique run index
            if not Path(f"{script_name}__{image_idx:04d}_step0.png").exists():
                break
            image_idx += 1

        start_time = time.time()
        for i, t in enumerate(tqdm(timesteps)):
            # No CFG for cfg_scale=1.0, so we don't duplicate inputs
            latent_model_input = latents
            
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # Prepare added conditioning signals
            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

            # Predict the noise residual
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            
            # No guidance is applied since cfg_scale is 1.0

            # Compute the previous noisy sample x_t -> x_{t-1}
            latents = pipe.scheduler.step(noise_pred, t, latents, generator=generator, return_dict=False)[0]

            # --- Save intermediate image ---
            latents_for_vae = latents / pipe.vae.config.scaling_factor
            image_tensor = pipe.vae.decode(latents_for_vae, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]
            
            output_path = f"{script_name}__{image_idx:04d}_step{i:02d}.png"
            image.save(output_path)
        
        end_time = time.time()
        print(f"Denoising loop took: {end_time - start_time:.4f} seconds")
        print("✓ Denoising loop complete.")
        print("✓ Images generated successfully!")

if __name__ == "__main__":
    main()
