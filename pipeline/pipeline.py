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
        self.compel_original = self.components["compel_original"]
        self.tokenizer_l = self.components["tokenizer_1"]
        self.tokenizer_g = self.components["tokenizer_2"]
        self.text_encoder_l = self.components["text_encoder_l"]
        self.text_encoder_g = self.components["text_encoder_g"]
        self.vae_decoder = self.components["vae_decoder"]
        self.unet = self.components["unet"]
        self.scheduler = self.components["scheduler"]
        self.vae_scaling_factor = self.components["vae_scaling_factor"]

    def __call__(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 8,
        seed: int = 42,
    ):
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

        # 4. Prepare extra inputs for UNet
        time_ids = self._get_time_ids(height, width)

        # 5. Denoising loop
        for t in timesteps:
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                prompt_embeds,
                pooled_prompt_embeds,
                time_ids,
            )
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # 6. Decode latents
        image_np = self.vae_decoder(latents / self.vae_scaling_factor)

        # 7. Post-process image
        image = self._postprocess_image(image_np)

        # 8. Clear memory
        self._clear_memory()

        return image

    def _prepare_latents(self, height, width, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        shape = (
            1,
            4,
            height // 8,
            width // 8,
        )
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=torch.float16)
        latents = latents * self.scheduler.init_noise_sigma.to(self.device)
        return latents

    def _get_time_ids(self, height, width):
        time_ids = [
            height,
            width,
            0,
            0,
            height,
            width,
        ]
        return torch.tensor([time_ids], device=self.device, dtype=torch.float16)

    def _postprocess_image(self, image: torch.Tensor) -> Image.Image:
        image = torch.nan_to_num(image)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1)
        image = (image * 255).round().to(torch.uint8)
        return Image.fromarray(image.cpu().numpy()[0])

    def _clear_memory(self):
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
    #    "socks22/sdxl-wai-nsfw-illustriousv14", subfolder="scheduler"
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