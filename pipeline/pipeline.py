import gc
import numpy as np
import torch
from PIL import Image

import loader

class SDXLPipeline:
    def __init__(self):
        self.device = "cuda"
        self.components = loader.load_pipeline_components()
        self.compel = self.components["compel"]
        self.vae_decoder = self.components["vae_decoder"]
        self.unet = self.components["unet"]
        self.scheduler = self.components["scheduler"]
        self.vae_scaling_factor = self.components["vae_scaling_factor"]
        self.scheduler.init_noise_sigma = self.scheduler.init_noise_sigma.to(self.device)

    def __call__(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 8,
        seed: int = 42,
    ):
        # 1. Get text embeddings
        prompt_embeds, pooled_prompt_embeds = self.compel(prompt)
        prompt_embeds = prompt_embeds.to(dtype=torch.float16)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=torch.float16)

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
        latents = latents * self.scheduler.init_noise_sigma
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
    pipeline = SDXLPipeline()
    prompt = "masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, night, 1girl, 1boy, aqua(konosuba)"
    image = pipeline(prompt)
    image.save("output.png") 