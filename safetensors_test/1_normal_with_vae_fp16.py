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

        prompt = "a cute cat, masterpiece, best quality, ultra-detailed, cinematic lighting"
        
        # Pipeline settings
        cfg_scale = 1.0
        num_inference_steps = 8
        seed = 1020094661
        generator = torch.Generator(device="cuda").manual_seed(seed)
        height = 832
        width = 1216
        
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

        # Load and set LoRA weights
        print("Loading LoRA...")
        lora_path = base_dir / "lora"
        lora_filename = "dmd2_sdxl_4step_lora_fp16.safetensors"
        pipe.load_lora_weights(lora_path, weight_name=lora_filename)
        print("✓ LoRA loaded.")

        print("Enabling LoRA weights...")
        pipe.enable_lora()
        
        #print("Unloading LoRA weights from memory...")
        #pipe.unload_lora_weights()
        #print("✓ LoRA unloaded.")

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
            num_images_per_prompt=1,
            do_classifier_free_guidance=1
        )

        # 2. Prepare latents
        print("Preparing latents...")
        latents = torch.randn(
            (1, pipe.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        # Scale the initial noise by the scheduler's standard deviation
        latents = latents * pipe.scheduler.init_noise_sigma

        # 3. Prepare timesteps and extra embeds for the denoising loop
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps
        
        add_time_ids = pipe._get_add_time_ids((height, width), (0,0), (height, width), dtype, text_encoder_projection_dim=text_encoder_2.config.projection_dim)
        
        # 4. Denoising loop
        print(f"Running denoising loop for {num_inference_steps} steps...")
        for i, t in enumerate(timesteps):
            # No CFG for cfg_scale=1.0, so we don't duplicate inputs
            latent_model_input = latents
            
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
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        print("✓ Denoising loop complete.")
        
        # 5. Manually decode the latents with the VAE
        print("Decoding latents with VAE...")
        # The VAE scales the latents internally
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        
        # 6. Post-process the image
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        
        print("✓ Image generated successfully!")
        
        # 7. Save the image
        output_path = "output_manual_vae.png"
        image.save(output_path)
        print(f"✓ Image saved to: {output_path}")

if __name__ == "__main__":
    main()
