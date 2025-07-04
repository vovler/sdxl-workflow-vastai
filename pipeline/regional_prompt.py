import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
import math

# --- 1. Helper Functions ---
def find_token_indices(tokenizer, prompt: str, sub_prompt: str) -> List[int]:
    tokens_prompt = tokenizer.tokenize(prompt)
    tokens_sub_prompt = tokenizer.tokenize(sub_prompt)
    for i in range(len(tokens_prompt) - len(tokens_sub_prompt) + 1):
        if tokens_prompt[i:i + len(tokens_sub_prompt)] == tokens_sub_prompt:
            # +1 to account for the starting BOS token
            return list(range(i + 1, i + len(tokens_sub_prompt) + 1))
    raise ValueError(f"Sub-prompt '{sub_prompt}' not found in prompt '{prompt}'.")

# This is a helper function from sd-forge-couple's attention_masks.py
# It is needed to correctly downsample the mask to the latent's resolution.
def repeat_div(value: int, iterations: int) -> int:
    for _ in range(iterations):
        value = math.ceil(value / 2)
    return value
    
# --- 2. The Core Logic: Corrected Custom Attention Processor ---
class RegionalAttnProcessorV2:
    def __init__(self, regions_config: Dict, full_prompt_embeds: torch.Tensor, tokenizer):
        self.regions_config = regions_config
        self.num_regions = len(regions_config)
        self.device = full_prompt_embeds.device
        self.dtype = full_prompt_embeds.dtype
        self.tokenizer = tokenizer
        
        self.uncond_embeds, self.cond_embeds = full_prompt_embeds.chunk(2)
        
        self.regional_cond_embeds = []
        
        max_len = 0
        for region_data in self.regions_config.values():
            max_len = max(max_len, len(region_data["indices"]))

        for region_data in self.regions_config.values():
            indices = region_data["indices"]
            
            padding_len = max_len - len(indices)
            if padding_len > 0:
                # Pad with EOS token ID
                padding_index = self.cond_embeds.shape[1] - 1
                indices = indices + [padding_index] * padding_len

            indices_tensor = torch.tensor(indices).to(self.device)
            region_embed = torch.index_select(self.cond_embeds, 1, indices_tensor)
            self.regional_cond_embeds.append(region_embed)
            
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        hidden_states_uncond, hidden_states_cond = hidden_states.chunk(2)
        encoder_hidden_states_uncond, encoder_hidden_states_cond = encoder_hidden_states.chunk(2)

        # 1. Process Unconditional Pass (standard attention)
        output_uncond = F.scaled_dot_product_attention(
            attn.to_q(hidden_states_uncond), attn.to_k(encoder_hidden_states_uncond), attn.to_v(encoder_hidden_states_uncond),
        )

        # 2. Process Conditional Pass (regional attention)
        hidden_states_cond_repeated = hidden_states_cond.repeat(self.num_regions, 1, 1)
        regional_prompts_stacked = torch.cat(self.regional_cond_embeds, dim=0)

        q = attn.to_q(hidden_states_cond_repeated)
        k = attn.to_k(regional_prompts_stacked)
        v = attn.to_v(regional_prompts_stacked)
        regional_outputs = F.scaled_dot_product_attention(q, k, v)

        latent_h = int(np.sqrt(sequence_length))
        latent_w = latent_h
        
        final_cond_output = torch.zeros_like(hidden_states_cond)
        for i, region_data in enumerate(self.regions_config.values()):
            mask = region_data["mask"].to(self.device, self.dtype)
            mask_downsampled = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(latent_h, latent_w), mode='bilinear', align_corners=False)
            mask_downsampled = mask_downsampled.squeeze().view(sequence_length, 1)
            
            region_output = regional_outputs[i].unsqueeze(0)
            final_cond_output += region_output * mask_downsampled

        # 3. Combine Unconditional and Conditional Outputs
        output = torch.cat([output_uncond, final_cond_output], dim=0)
        output = attn.batch_to_head_dim(output)
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        
        return output

# --- 3. Main Script Setup and Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "socks22/sdxl-wai-nsfw-illustriousv14",
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    prompt_left = "1girl, aqua (konosuba), blue hair, smiling, masterpiece, best quality"
    prompt_right = "1girl, megumin (konosuba), brown hair, eyepatch, sad, masterpiece, best quality"
    
    # The full prompt MUST contain the regional prompts verbatim.
    full_prompt = f"{prompt_left} AND {prompt_right}"
    negative_prompt = "blurry, ugly, deformed, text, watermark, worst quality, low quality"
    
    width, height = 1024, 768
    
    # --- Encode prompts to get embeddings (needed for the processor) ---
    print("Encoding prompts...")
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True
    )
    # The processor needs both cond and uncond embeds
    full_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # --- Find Token Indices for each region ---
    print("Finding token indices...")
    # SDXL uses two tokenizers, we care about the first one for cross-attention
    tokenizer1 = pipe.tokenizer
    
    indices_left = find_token_indices(tokenizer1, full_prompt, prompt_left)
    indices_right = find_token_indices(tokenizer1, full_prompt, prompt_right)
    print(f"Left prompt indices: {indices_left}")
    print(f"Right prompt indices: {indices_right}")

    # --- Create Spatial Masks ---
    print("Creating spatial masks...")
    mask_left = torch.zeros((height, width))
    mask_left[:, :width // 2] = 1.0

    mask_right = torch.zeros((height, width))
    mask_right[:, width // 2:] = 1.0
    
    # --- Prepare for Injection ---
    regions_config = {
        "left": {"mask": mask_left, "indices": indices_left},
        "right": {"mask": mask_right, "indices": indices_right}
    }

    # --- Inject the Custom Attention Processor ---
    print("Injecting custom attention processor...")
    attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        if "attn2" in name: # Target cross-attention blocks
            attn_procs[name] = RegionalAttnProcessorV2(regions_config, full_prompt_embeds, tokenizer1)
        else: # Use default for self-attention
            attn_procs[name] = pipe.unet.attn_processors[name]
    pipe.unet.set_attn_processor(attn_procs)

    # --- Generate the Image ---
    print("Generating image with attention coupling...")
    generator = torch.Generator("cuda").manual_seed(12345)
    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        width=width,
        height=height,
        guidance_scale=7.0,
        num_inference_steps=28,
        generator=generator,
    ).images[0]
    
    print("Image generation complete.")
    image.save("multi_character_composition_sdxl_fixed.png")
    print("Saved image to 'multi_character_composition_sdxl_fixed.png'")