import gc
import numpy as np
import torch
from PIL import Image
import time
import psutil
import os

import loader
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import utils

class SDXLPipeline:
    def __init__(self):
        self.device = "cuda"
        self.components = loader.load_pipeline_components()
        self.tokenizer_l = self.components["tokenizer_1"]
        self.tokenizer_g = self.components["tokenizer_2"]
        self.text_encoder_l = self.components["text_encoder_l"]
        self.text_encoder_g = self.components["text_encoder_g"]
        self.trt_vae = self.components["trt_vae"]
        self.unet = self.components["unet"]
        self.scheduler = self.components["scheduler"]

        self.image_processor = self.components["image_processor"]
        self.vae_scale_factor = self.components["vae_scale_factor"]
        self.process = psutil.Process(os.getpid())

    def __call__(
        self,
        prompt: str,
        height: int = 768,
        width: int = 1152,
        num_inference_steps: int = 8,
        seed: int = 42,
        is_warmup: bool = False,
    ):
        if not is_warmup:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            print("\n" + "="*50)
            print("--- Starting SDXL Pipeline (Monitored Run) ---")
            print(f"Prompt: {prompt}")
            print(f"Height: {height}, Width: {width}, Steps: {num_inference_steps}, Seed: {seed}")
            print("="*50)
        else:
            print("\n" + "="*50)
            print("--- Starting SDXL Pipeline (Warmup Run) ---")
            print("="*50)

        # 1. Get text embeddings
        if not is_warmup:
            print("\n" + "="*40)
            print("--- RUNNING CLIP ---")
            print("="*40)
            torch.cuda.synchronize()
            clip_start_time = time.time()
            clip_start_ram = self.process.memory_info().rss
        
        # Tokenize prompt
        tokenized_l = self.tokenizer_l(prompt, padding="max_length", max_length=self.tokenizer_l.model_max_length, truncation=True, return_tensors="pt")
        tokenized_g = self.tokenizer_g(prompt, padding="max_length", max_length=self.tokenizer_g.model_max_length, truncation=True, return_tensors="pt")

        input_ids_l = tokenized_l.input_ids.to(self.device)
        input_ids_g = tokenized_g.input_ids.to(self.device)

        # Get embeddings
        hidden_states_l = self.text_encoder_l(input_ids=input_ids_l).last_hidden_state
        output_g = self.text_encoder_g(input_ids=input_ids_g)
        hidden_states_g = output_g.last_hidden_state
        pooled_prompt_embeds = output_g.pooler_output

        prompt_embeds = torch.cat([hidden_states_l, hidden_states_g], dim=-1)

        if is_warmup:
            self.text_encoder_l.capture_graph({"input_ids": input_ids_l})
            self.text_encoder_g.capture_graph({"input_ids": input_ids_g})

        if not is_warmup:
            torch.cuda.synchronize()
            clip_end_time = time.time()
            clip_end_ram = self.process.memory_info().rss
            clip_duration = clip_end_time - clip_start_time
            clip_ram_delta = (clip_end_ram - clip_start_ram) / (1024 * 1024)
            print(f"CLIP: took {clip_duration * 1000:.0f}ms - RAM Delta: {clip_ram_delta:.0f}MB")

        if not is_warmup:
            print("--- Final Embeddings ---")
            print(f"prompt_embeds: shape={prompt_embeds.shape}, dtype={prompt_embeds.dtype}, device={prompt_embeds.device}")
            print(f"prompt_embeds | Mean: {prompt_embeds.mean():.6f} | Std: {prompt_embeds.std():.6f} | Sum: {prompt_embeds.sum():.6f}")
            if pooled_prompt_embeds is not None:
                print(f"pooled_prompt_embeds: shape={pooled_prompt_embeds.shape}, dtype={pooled_prompt_embeds.dtype}, device={pooled_prompt_embeds.device}")
                print(f"pooled_prompt_embeds | Mean: {pooled_prompt_embeds.mean():.6f} | Std: {pooled_prompt_embeds.std():.6f} | Sum: {pooled_prompt_embeds.sum():.6f}")
            print("------------------------")

        # 2. Prepare latents
        generator = torch.Generator(device=self.device).manual_seed(seed)
        num_channels_latents = 4 # UNet channel count is fixed

        # 3. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        latents = utils._prepare_latents(
            self.scheduler, 1, num_channels_latents, height, width, pooled_prompt_embeds.dtype, self.device, generator
        )

        latents = latents.to(self.device)
        
        
        timesteps = self.scheduler.timesteps
        if not is_warmup:
            print(f"\n--- Timesteps ({len(timesteps)}) ---")
            print(timesteps)
            print("--------------------")

        # 4. Prepare extra inputs for UNet
        time_ids = utils._get_add_time_ids(
            (height, width), (0, 0), (height, width), dtype=pooled_prompt_embeds.dtype
        )
        time_ids = time_ids.to(self.device)

        # 5. Denoising loop
        if not is_warmup:
            print("\n--- Denoising Loop ---")
            torch.cuda.synchronize()
            unet_loop_start_time = time.time()
            unet_loop_start_ram = self.process.memory_info().rss
            total_scheduler_time = 0

        for i, t in enumerate(timesteps):
            if not is_warmup:
                print(f"\n--- Step {i+1}/{len(timesteps)}, Timestep: {t} ---")
                print(f"latents before scale_mode_input: shape={latents.shape}, dtype={latents.dtype}, device={latents.device}")
                print(f"latents before scale_mode_input | Mean: {latents.mean():.6f} | Std: {latents.std():.6f} | Sum: {latents.sum():.6f}")

            #cast t to int
            latent_model_input = self.scheduler.scale_model_input(latents, t.to(torch.int32))
            if not is_warmup:
                print(f"latent_model_input: shape={latent_model_input.shape}, dtype={latent_model_input.dtype}, device={latent_model_input.device}")
                print(f"latent_model_input | Mean: {latent_model_input.mean():.6f} | Std: {latent_model_input.std():.6f} | Sum: {latent_model_input.sum():.6f}")

            if is_warmup and i == 0:
                self.unet.capture_graph({
                    "sample": latent_model_input,
                    "timestep": t,
                    "encoder_hidden_states": prompt_embeds,
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": time_ids,
                })

            noise_pred = self.unet(
                latent_model_input,
                t,
                prompt_embeds,
                pooled_prompt_embeds,
                time_ids,
            )
            
            if not is_warmup:
                print(f"noise_pred: shape={noise_pred.shape}, dtype={noise_pred.dtype}")
                print(f"noise_pred | Mean: {noise_pred.mean():.6f} | Std: {noise_pred.std():.6f} | Sum: {noise_pred.sum():.6f}")
            
            scheduler_step_start_time = 0
            if not is_warmup:
                torch.cuda.synchronize()
                scheduler_step_start_time = time.time()

            latents = self.scheduler.step(noise_pred, t, latents)[0]

            if not is_warmup:
                torch.cuda.synchronize()
                scheduler_step_end_time = time.time()
                total_scheduler_time += scheduler_step_end_time - scheduler_step_start_time
                print(f"latents after step: shape={latents.shape}, dtype={latents.dtype}")
                print(f"latents after step | Mean: {latents.mean():.6f} | Std: {latents.std():.6f} | Sum: {latents.sum():.6f}")
        
        if not is_warmup:
            torch.cuda.synchronize()
            unet_loop_end_time = time.time()
            unet_loop_end_ram = self.process.memory_info().rss
            unet_loop_duration = unet_loop_end_time - unet_loop_start_time
            unet_time = unet_loop_duration - total_scheduler_time
            unet_loop_ram_delta = (unet_loop_end_ram - unet_loop_start_ram) / (1024 * 1024)
            print("\n--- Denoising Loop End ---")
            print(f"UNET: took {unet_loop_duration:.2f}s - on unet: {unet_time:.2f}s - on scheduler: {total_scheduler_time:.2f}s - RAM Delta: {unet_loop_ram_delta:.0f}MB")

        # 6. Decode latents
        if not is_warmup:
            print("\n--- Decoding Latents ---")
            torch.cuda.synchronize()
            vae_start_time = time.time()
            vae_start_ram = self.process.memory_info().rss
        
        if is_warmup:
            self.trt_vae.capture_graph({"latent_sample": latents})

        image_np = self.trt_vae(latents)

        if not is_warmup:
            torch.cuda.synchronize()
            vae_end_time = time.time()
            vae_end_ram = self.process.memory_info().rss
            vae_duration = vae_end_time - vae_start_time
            vae_ram_delta = (vae_end_ram - vae_start_ram) / (1024 * 1024)
            print(f"VAE: took {vae_duration * 1000:.0f}ms - RAM Delta: {vae_ram_delta:.0f}MB")
            print(f"decoded image (tensor): shape={image_np.shape}, dtype={image_np.dtype}, device={image_np.device}, has_nan={torch.isnan(image_np).any()}, has_inf={torch.isinf(image_np).any()}")

        # 7. Post-process image
        if not is_warmup:
            print("\n--- Post-processing Image ---")
        image = self.image_processor.postprocess(image_np.detach().cpu(), output_type="pil")[0]
        if not is_warmup:
            print(f"Post-processed image: {image}")
            print("--- Post-processing Complete ---")

        # 8. Clear memory
        #utils._clear_memory()
        if not is_warmup:
            torch.cuda.synchronize()

        return image

if __name__ == "__main__":
    #prompt = "masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, night, 1girl, aqua_(konosuba), smiling, looking at viewer, at the park, night"
    
    prompt = (
        "masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, newest,"
        "1girl, aqua_(konosuba), robotic_arms, left_side, facing_right,"
        "1girl, megumin,medieval_armor, blue_sword, right_side, facing_left,"
        "futuristic_city, neon_lights, sunset, background"
    )
    pipeline = SDXLPipeline()
    # Warmup run
    _ = pipeline(prompt, is_warmup=True)

    # Monitored run
    start_time = time.time()
    #prompt = "masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, night, 1girl, aqua_(konosuba), smiling, looking at viewer, at the park, nude"
    image = pipeline(prompt)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.2f} seconds")
    image.save("output.png") 