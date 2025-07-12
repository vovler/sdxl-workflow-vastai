import torch
import torch.nn as nn

from scheduler import EulerAncestralDiscreteScheduler

class UNetLoop(nn.Module):
    def __init__(self, unet: nn.Module, scheduler: EulerAncestralDiscreteScheduler):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.num_inference_steps = scheduler.num_inference_steps

    @torch.no_grad()
    def forward(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        add_time_ids: torch.Tensor,
        all_noises: torch.Tensor
    ) -> torch.Tensor:
        """
        Denoising loop that is compatible with ONNX export.
        """
        latents = latents * self.scheduler.init_noise_sigma
        
        # The main denoising loop
        for i in range(self.num_inference_steps):
            sigma = self.scheduler.sigmas[i]
            t = self.scheduler.timesteps[i].unsqueeze(0) # UNet expects a tensor for timestep
            
            # 1. Scale the latent input
            latent_model_input = latents / torch.sqrt(sigma**2 + 1)
            
            # 2. Predict noise with the UNet
            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
            noise_pred = self.unet(
                latent_model_input, t,
                encoder_hidden_states=text_embeddings,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False
            )[0]
            
            # 3. Scheduler step (Euler Ancestral method)
            latents = self.scheduler.step(noise_pred, latents, i, all_noises)
            
        return latents
