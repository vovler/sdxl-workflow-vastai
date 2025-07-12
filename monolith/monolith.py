import torch
import torch.nn as nn

from monolith.unet_loop import UNetLoop

class MonolithicSDXL(nn.Module):
    def __init__(self, text_encoder_1, text_encoder_2, unet, vae, unet_loop):
        super().__init__()
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.unet = unet
        self.vae = vae
        self.unet_loop = unet_loop
        self.vae_scale_factor = vae.config["scaling_factor"]
        
    @torch.no_grad()
    def forward(
        self,
        prompt_ids_1: torch.Tensor,
        prompt_ids_2: torch.Tensor,
        initial_latents: torch.Tensor,
        all_noises: torch.Tensor,
        add_time_ids: torch.Tensor,
    ) -> torch.Tensor:
        
        # --- Encode prompts ---
        prompt_embeds_1_out = self.text_encoder_1(prompt_ids_1, output_hidden_states=True)
        prompt_embeds_1 = prompt_embeds_1_out.hidden_states[-2]

        # Get the output from the second text encoder
        text_encoder_2_out = self.text_encoder_2(prompt_ids_2, output_hidden_states=True)
        prompt_embeds_2 = text_encoder_2_out.hidden_states[-2]
        pooled_prompt_embeds = text_encoder_2_out.text_embeds

        # Concatenate the 3D prompt embeddings
        prompt_embeds = torch.cat((prompt_embeds_1, prompt_embeds_2), dim=-1)
        
        final_latents = self.unet_loop(
            initial_latents, 
            prompt_embeds, 
            pooled_prompt_embeds, 
            add_time_ids, 
            self.unet,
            all_noises
        )

        final_latents = final_latents / self.vae_scale_factor
        image = self.vae.decode(final_latents)

        return image