import gc
import numpy as np
import torch
from PIL import Image

import loader

class SDXLPipeline:
    def __init__(self):
        self.components = loader.load_pipeline_components()
        self.compel = self.components["compel"]
        self.vae_decoder = self.components["vae_decoder"]
        self.unet = self.components["unet"]
        self.scheduler = self.components["scheduler"]

    def __call__(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        seed: int = 42,
    ):
        # 1. Get text embeddings
        prompt_embeds, pooled_prompt_embeds = self.compel(prompt)

        # 2. Prepare latents
        latents = self._prepare_latents(height, width, seed)

        # 3. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra inputs for UNet
        time_ids = self._get_time_ids(height, width)

        # 5. Denoising loop
        prompt_embeds_np = prompt_embeds.cpu().numpy()
        pooled_prompt_embeds_np = pooled_prompt_embeds.cpu().numpy()

        latents_torch = torch.from_numpy(latents).to("cuda")

        for t in timesteps:
            latent_model_input = latents_torch

            timestep_numpy = np.array([t.item()], dtype=np.float16)

            noise_pred_np = self.unet(
                latent_model_input.cpu().numpy(),
                timestep_numpy,
                prompt_embeds_np,
                pooled_prompt_embeds_np,
                time_ids,
            )
            
            noise_pred_torch = torch.from_numpy(noise_pred_np).to("cuda")
            latents_torch = self.scheduler.step(noise_pred_torch, t, latents_torch).prev_sample
        
        latents = latents_torch.cpu().numpy()

        # 6. Decode latents
        image = self.vae_decoder(latents / self.scheduler.config.scaling_factor)

        # 7. Post-process image
        image = self._postprocess_image(image)

        # 8. Clear memory
        self._clear_memory()

        return image

    def _prepare_latents(self, height, width, seed):
        generator = torch.manual_seed(seed)
        shape = (
            1,
            4,
            height // 8,
            width // 8,
        )
        latents = torch.randn(shape, generator=generator, dtype=torch.float32).numpy()
        return self.scheduler.init_noise_sigma * latents

    def _get_time_ids(self, height, width):
        time_ids = [
            height,
            width,
            0,
            0,
            height,
            width,
        ]
        time_ids = np.array([time_ids], dtype=np.float16)
        return time_ids

    def _postprocess_image(self, image: np.ndarray) -> Image.Image:
        image = (image / 2 + 0.5).clip(0, 1)
        image = image.transpose((0, 2, 3, 1))
        image = (image * 255).round().astype("uint8")
        return Image.fromarray(image[0])

    def _clear_memory(self):
        gc.collect()

if __name__ == "__main__":
    pipeline = SDXLPipeline()
    #prompt = "A cinematic shot of a baby raccoon wearing an intricate italian mafioso suit, saying 'nice one, eh?'"
    #image = pipeline(prompt)
    #image.save("output.png") 