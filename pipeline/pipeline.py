import gc
import numpy as np
import torch
from PIL import Image

import loader
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

class SDXLPipeline:
    def __init__(self):
        self.device = "cuda"
        self.components = loader.load_pipeline_components()
        self.compel_onnx = self.components["compel_onnx"]
        self.tokenizer_l = self.components["tokenizer_1"]
        self.tokenizer_g = self.components["tokenizer_2"]
        self.text_encoder_l = self.components["text_encoder_l"]
        self.text_encoder_g = self.components["text_encoder_g"]
        self.vae = self.components["vae"]
        self.unet = self.components["unet"]
        self.scheduler = self.components["scheduler"]
        self.image_processor = self.components["image_processor"]

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
        print("--- RUNNING ONNX COMPEL ---")
        print("="*40)
        #prompt_embeds, pooled_prompt_embeds = self.compel_onnx(prompt)
        
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
        print(f"prompt_embeds: shape={prompt_embeds.shape}, dtype={prompt_embeds.dtype}, device={prompt_embeds.device}, has_nan={torch.isnan(prompt_embeds).any()}, has_inf={torch.isinf(prompt_embeds).any()}")
        if pooled_prompt_embeds is not None:
            print(f"pooled_prompt_embeds: shape={pooled_prompt_embeds.shape}, dtype={pooled_prompt_embeds.dtype}, device={pooled_prompt_embeds.device}, has_nan={torch.isnan(pooled_prompt_embeds).any()}, has_inf={torch.isinf(pooled_prompt_embeds).any()}")
        print("------------------------")

        # 2. Prepare latents
        latents = self._prepare_latents(height, width, seed)

        # 3. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        print(f"\n--- Timesteps ({len(timesteps)}) ---")
        print(timesteps)
        print("--------------------")

        # 4. Prepare extra inputs for UNet
        time_ids = self._get_time_ids(height, width)

        # 5. Denoising loop
        print("\n--- Denoising Loop ---")
        for i, t in enumerate(timesteps):
            print(f"\n--- Step {i+1}/{len(timesteps)}, Timestep: {t} ---")
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            print(f"latent_model_input: shape={latent_model_input.shape}, dtype={latent_model_input.dtype}, has_nan={torch.isnan(latent_model_input).any()}, has_inf={torch.isinf(latent_model_input).any()}")

            noise_pred = self.unet(
                latent_model_input,
                t,
                prompt_embeds,
                pooled_prompt_embeds,
                time_ids,
            )
            print(f"noise_pred: shape={noise_pred.shape}, dtype={noise_pred.dtype}, has_nan={torch.isnan(noise_pred).any()}, has_inf={torch.isinf(noise_pred).any()}")
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            print(f"latents after step: shape={latents.shape}, dtype={latents.dtype}, has_nan={torch.isnan(latents).any()}, has_inf={torch.isinf(latents).any()}")
        print("\n--- Denoising Loop End ---")
        
        # 6. Decode latents
        print("\n--- Decoding Latents ---")
        latents_to_decode = latents / self.vae.config.scaling_factor
        print(f"VAE scaling factor: {self.vae.config.scaling_factor}")
        # print(f"latents_to_decode: shape={latents_to_decode.shape}, dtype={latents_to_decode.dtype}, has_nan={torch.isnan(latents_to_decode).any()}, has_inf={torch.isinf(latents_to_decode).any()}")
        image_np = self.vae.decode(latents_to_decode, return_dict=False)[0].detach().cpu()
        
        print(f"decoded image (tensor): shape={image_np.shape}, dtype={image_np.dtype}, device={image_np.device}, has_nan={torch.isnan(image_np).any()}, has_inf={torch.isinf(image_np).any()}")

        # 7. Post-process image
        print("\n--- Post-processing Image ---")
        image = self.image_processor.postprocess(image_np, output_type="pil")[0]
        print(f"Post-processed image: {image}")
        print("--- Post-processing Complete ---")

        # 8. Clear memory
        self._clear_memory()
        print("\n--- Memory Cleared, Pipeline Finished ---")

        return image

    def _prepare_latents(self, height, width, seed):
        print("\n--- Preparing Latents ---")
        print(f"height={height}, width={width}, seed={seed}")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        shape = (
            1,
            4,
            height // 8,
            width // 8,
        )
        print(f"latent shape: {shape}")
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=torch.float16)
        print(f"initial random latents: has_nan={torch.isnan(latents).any()}, has_inf={torch.isinf(latents).any()}")
        print(f"scheduler.init_noise_sigma: {self.scheduler.init_noise_sigma}")
        latents = latents * self.scheduler.init_noise_sigma.to(self.device)
        print(f"scaled latents: shape={latents.shape}, dtype={latents.dtype}, has_nan={torch.isnan(latents).any()}, has_inf={torch.isinf(latents).any()}")
        print("-------------------------")
        return latents

    def _get_time_ids(self, height, width):
        print("\n--- Getting Time IDs ---")
        time_ids_list = [
            height,
            width,
            0,
            0,
            height,
            width,
        ]
        time_ids = torch.tensor([time_ids_list], device=self.device, dtype=torch.float16)
        print(f"time_ids: {time_ids}")
        print(f"time_ids shape: {time_ids.shape}, dtype: {time_ids.dtype}")
        print("------------------------")
        return time_ids

    def _postprocess_image(self, image: torch.Tensor) -> Image.Image:
        print("--- In _postprocess_image ---")
        print(f"input image tensor: shape={image.shape}, dtype={image.dtype}, has_nan={torch.isnan(image).any()}, has_inf={torch.isinf(image).any()}")
        #image = torch.nan_to_num(image)
        image = (image / 2 + 0.5).clamp(0, 1)
        print(f"image after scaling and clamping: shape={image.shape}, dtype={image.dtype}, has_nan={torch.isnan(image).any()}, has_inf={torch.isinf(image).any()}")
        image = image.permute(0, 2, 3, 1)
        print(f"image after permute: shape={image.shape}, dtype={image.dtype}")
        image = (image * 255).round().to(torch.uint8)
        print(f"image after converting to uint8: shape={image.shape}, dtype={image.dtype}")
        final_image = Image.fromarray(image.cpu().numpy()[0])
        print("--- Exiting _postprocess_image ---")
        return final_image

    def _clear_memory(self):
        print("--- In _clear_memory ---")
        gc.collect()

if __name__ == "__main__":
    prompt = "masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, night, 1girl, aqua_(konosuba), smiling, looking at viewer, at the park, night"
    
    pipeline = SDXLPipeline()
    image = pipeline(prompt)
    image.save("output.png") 

    #print("\n" + "="*40)
    #print("--- RUNNING NATIVE DIFFUSERS PIPELINE ---")
    #print("="*40)
    
    #scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    #    "socks22/sdxl-wai-nsfw-illustriousv14", subfolder="scheduler", use_karras_sigmas=False
    #)
    
    #native_pipeline = StableDiffusionXLPipeline.from_pretrained(
    #    "socks22/sdxl-wai-nsfw-illustriousv14",
    #    torch_dtype=torch.float16,
    #    scheduler=scheduler,
    #    use_safetensors=True,
    #)
    #native_pipeline.to("cuda:0")
    #native_image = native_pipeline(prompt=prompt, num_inference_steps=8, guidance_scale=1.0).images[0]
    #native_image.save("output_native.png")