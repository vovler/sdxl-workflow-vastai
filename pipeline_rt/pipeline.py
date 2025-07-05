import gc
import numpy as np
import torch
from PIL import Image
import time
import psutil
import os

import loader
import models
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import utils
import defaults

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

    def set_unet(self, unet_path: str):
        """
        Load a new UNet model.
        The CUDA graph will be captured on the first inference run with this new UNet.
        """
        print(f"\n--- Swapping UNet to: {unet_path} ---")
        if not os.path.exists(unet_path):
            raise FileNotFoundError(f"UNet model not found at {unet_path}")
        self.unet = models.UNet(unet_path, self.device)
        print("--- UNet swapped successfully ---")

    def __call__(
        self,
        prompt: str,
        height: int = 768,
        width: int = 1152,
        num_inference_steps: int = 12,
        seed: int = 44,
    ):
        print("\n" + "="*50)
        print("--- Starting SDXL Pipeline (Monitored Run) ---")
        print(f"Prompt: {prompt}")
        print(f"Height: {height}, Width: {width}, Steps: {num_inference_steps}, Seed: {seed}")
        print("="*50)

        # 1. Get text embeddings
        print("\n" + "="*40)
        print("--- RUNNING CLIP ---")
        print("="*40)
        clip_start_time = time.time()
        
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

        clip_end_time = time.time()
        clip_duration = clip_end_time - clip_start_time
        print(f"CLIP: took {clip_duration * 1000:.0f}ms")

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
        print(f"\n--- Timesteps ({len(timesteps)}) ---")
        print(timesteps)
        print("--------------------")

        # 4. Prepare extra inputs for UNet
        time_ids = utils._get_add_time_ids(
            (height, width), (0, 0), (height, width), dtype=pooled_prompt_embeds.dtype
        )
        time_ids = time_ids.to(self.device)

        # 5. Denoising loop
        print("\n--- Denoising Loop ---")
        unet_loop_start_time = time.time()
        total_scheduler_time = 0

        for i, t in enumerate(timesteps):
            print(f"\n--- Step {i+1}/{len(timesteps)}, Timestep: {t} ---")
            print(f"latents before scale_mode_input: shape={latents.shape}, dtype={latents.dtype}, device={latents.device}")
            print(f"latents before scale_mode_input | Mean: {latents.mean():.6f} | Std: {latents.std():.6f} | Sum: {latents.sum():.6f}")

            #cast t to int
            latent_model_input = self.scheduler.scale_model_input(latents, t.to(torch.int32))
            print(f"latent_model_input: shape={latent_model_input.shape}, dtype={latent_model_input.dtype}, device={latent_model_input.device}")
            print(f"latent_model_input | Mean: {latent_model_input.mean():.6f} | Std: {latent_model_input.std():.6f} | Sum: {latent_model_input.sum():.6f}")

            noise_pred = self.unet(
                latent_model_input,
                t,
                prompt_embeds,
                pooled_prompt_embeds,
                time_ids,
            )
            
            print(f"noise_pred: shape={noise_pred.shape}, dtype={noise_pred.dtype}")
            print(f"noise_pred | Mean: {noise_pred.mean():.6f} | Std: {noise_pred.std():.6f} | Sum: {noise_pred.sum():.6f}")
            
            scheduler_step_start_time = 0
            scheduler_step_start_time = time.time()

            latents = self.scheduler.step(noise_pred, t, latents)[0]

            scheduler_step_end_time = time.time()
            total_scheduler_time += scheduler_step_end_time - scheduler_step_start_time
            print(f"latents after step: shape={latents.shape}, dtype={latents.dtype}")
            print(f"latents after step | Mean: {latents.mean():.6f} | Std: {latents.std():.6f} | Sum: {latents.sum():.6f}")
        
        unet_loop_end_time = time.time()
        unet_loop_duration = unet_loop_end_time - unet_loop_start_time
        unet_time = unet_loop_duration - total_scheduler_time
        print("\n--- Denoising Loop End ---")
        print(f"UNET: took {unet_loop_duration:.2f}s - on unet: {unet_time:.2f}s - on scheduler: {total_scheduler_time:.2f}s")

        # 6. Decode latents
        print("\n--- Decoding Latents ---")
        vae_start_time = time.time()
        
        image_np = self.trt_vae(latents)

        vae_end_time = time.time()
        vae_duration = vae_end_time - vae_start_time
        print(f"VAE: took {vae_duration * 1000:.0f}ms")
        print(f"decoded image (tensor): shape={image_np.shape}, dtype={image_np.dtype}, device={image_np.device}, has_nan={torch.isnan(image_np).any()}, has_inf={torch.isinf(image_np).any()}")

        # 7. Post-process image
        print("\n--- Post-processing Image ---")
        image = self.image_processor.postprocess(image_np.detach().cpu(), output_type="pil")[0]
        print(f"Post-processed image: {image}")
        print("--- Post-processing Complete ---")

        # 8. Clear memory
        #utils._clear_memory()

        return image

if __name__ == "__main__":
    #prompt = "masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, night, 1girl, aqua_(konosuba), smiling, looking at viewer, at the park, night"
    
    prompt = (
        "masterpiece,best quality,amazing quality, general, 3girls, "
        "aqua_(konosuba), left_side, blue sword, "
        "megumin, at the center, red_sword, "
        "darkness_(konosuba), right_side, green_sword, "
        "shiny skin, shiny clothes, looking at viewer, volumetric_lighting, futuristic_city, day"
    )
    pipeline = SDXLPipeline()

    # Monitored run with default (INT8) UNet
    print("\n--- Running Inference with INT8 UNet ---")
    start_time = time.time()
    image_int8 = pipeline(prompt)
    end_time = time.time()
    print(f"INT8 UNet - Time taken: {end_time - start_time:.2f} seconds")
    image_int8.save("output_int8.png")
    print("--- INT8 UNet Inference Complete ---")

    # Swap to FP16 UNet and run again
    unet_dir = os.path.dirname(defaults.UNET_PATH)
    fp16_unet_path = os.path.join(unet_dir, "model_fp16.plan")
    pipeline.set_unet(fp16_unet_path)

    print("\n--- Running Inference with FP16 UNet ---")
    start_time_fp16 = time.time()
    image_fp16 = pipeline(prompt)
    end_time_fp16 = time.time()
    print(f"FP16 UNet - Time taken: {end_time_fp16 - start_time_fp16:.2f} seconds")
    image_fp16.save("output_fp16.png")
    print("--- FP16 UNet Inference Complete ---") 