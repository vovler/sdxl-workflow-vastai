import gc
import numpy as np
import torch
from PIL import Image

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
        #self.vae_decoder = self.components["vae_decoder"]
        self.vae = self.components["vae"]
        self.unet = self.components["unet"]
        self.scheduler = self.components["scheduler"]

        self.image_processor = self.components["image_processor"]
        self.vae_scale_factor = self.components["vae_scale_factor"]

    def __call__(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 8,
        seed: int = 42,
    ):
        print("\n" + "="*50)
        print("--- Starting SDXL Pipeline ---")
        print(f"Prompt: {prompt}")
        print(f"Height: {height}, Width: {width}, Steps: {num_inference_steps}, Seed: {seed}")
        print("="*50)

        # 1. Get text embeddings
        print("\n" + "="*40)
        print("--- RUNNING ONNX ---")
        print("="*40)
        
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

        print("--- Final Embeddings ---")
        print(f"prompt_embeds: shape={prompt_embeds.shape}, dtype={prompt_embeds.dtype}, device={prompt_embeds.device}")
        print(f"prompt_embeds | Mean: {prompt_embeds.mean():.6f} | Std: {prompt_embeds.std():.6f} | Sum: {prompt_embeds.sum():.6f}")
        if pooled_prompt_embeds is not None:
            print(f"pooled_prompt_embeds: shape={pooled_prompt_embeds.shape}, dtype={pooled_prompt_embeds.dtype}, device={pooled_prompt_embeds.device}")
            print(f"pooled_prompt_embeds | Mean: {pooled_prompt_embeds.mean():.6f} | Std: {pooled_prompt_embeds.std():.6f} | Sum: {pooled_prompt_embeds.sum():.6f}")
        print("------------------------")

        # 2. Prepare latents
        generator = torch.Generator(device=self.device).manual_seed(seed)
        num_channels_latents = self.unet.session.get_inputs()[0].shape[1]

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
        for i, t in enumerate(timesteps):
            print(f"\n--- Step {i+1}/{len(timesteps)}, Timestep: {t} ---")
            print(f"latents before scale_mode_input: shape={latents.shape}, dtype={latents.dtype}, device={latents.device}")
            print(f"latents before scale_mode_input | Mean: {latents.mean():.6f} | Std: {latents.std():.6f} | Sum: {latents.sum():.6f}")

            latent_model_input = self.scheduler.scale_model_input(latents, t)
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
            
            latents = self.scheduler.step(noise_pred, t, latents)[0]
            print(f"latents after step: shape={latents.shape}, dtype={latents.dtype}")
            print(f"latents after step | Mean: {latents.mean():.6f} | Std: {latents.std():.6f} | Sum: {latents.sum():.6f}")
        print("\n--- Denoising Loop End ---")
        
        # 6. Decode latents
        print("\n--- Decoding Latents ---")
        
        # Un-scale and un-shift latents for TinyVAE before decoding
        #latents_to_decode = (latents - self.vae.config.latent_shift) / self.vae.config.latent_magnitude
        #print(f"TinyVAE latent_shift: {self.vae.config.latent_shift}, latent_magnitude: {self.vae.config.latent_magnitude}")
        #latents_to_decode = latents / self.vae_scale_factor
        latents_to_decode = latents / 0.18215
        
        image_np = self.vae.decode(latents_to_decode).sample
        
        print(f"decoded image (tensor): shape={image_np.shape}, dtype={image_np.dtype}, device={image_np.device}, has_nan={torch.isnan(image_np).any()}, has_inf={torch.isinf(image_np).any()}")

        # 7. Post-process image
        print("\n--- Post-processing Image ---")
        image = self.image_processor.postprocess(image_np.detach().cpu(), output_type="pil")[0]
        print(f"Post-processed image: {image}")
        print("--- Post-processing Complete ---")

        # 8. Clear memory
        #utils._clear_memory()
        print("\n--- Memory Cleared, Pipeline Finished ---")

        return image

if __name__ == "__main__":
    prompt = "masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, night, 1girl, aqua_(konosuba), smiling, looking at viewer, at the park, night"
    
    pipeline = SDXLPipeline()
    image = pipeline(prompt)
    image.save("output.png") 