import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
import math

# --- 1. Helper Functions (Unchanged) ---
def find_token_indices(tokenizer, prompt: str, sub_prompt: str) -> List[int]:
    tokens_prompt = tokenizer.tokenize(prompt)
    tokens_sub_prompt = tokenizer.tokenize(sub_prompt)
    for i in range(len(tokens_prompt) - len(tokens_sub_prompt) + 1):
        if tokens_prompt[i:i + len(tokens_sub_prompt)] == tokens_sub_prompt:
            return list(range(i + 1, i + len(tokens_sub_prompt) + 1))
    raise ValueError(f"Sub-prompt '{sub_prompt}' not found in prompt '{prompt}'.")

def lcm(a, b):
    return abs(a * b) // math.gcd(a, b) if a != 0 and b != 0 else 0

def lcm_for_list(numbers):
    if not numbers: return 0
    result = numbers[0]
    for i in range(1, len(numbers)):
        result = lcm(result, numbers[i])
    return result

# --- 2. The Core Logic: Architecturally Correct Attention Processor ---
class RegionalAttnProcessorV3:
    def __init__(self, regions_config: Dict, full_prompt_embeds: torch.Tensor, width: int, height: int):
        self.regions_config = regions_config
        self.num_regions = len(regions_config)
        self.device = full_prompt_embeds.device
        self.dtype = full_prompt_embedsF.dtype
        self.width = width
        self.height = height
        
        self.uncond_embeds, self.cond_embeds = full_prompt_embeds.chunk(2)
        
        regional_cond_embeds = []
        for region_data in self.regions_config.values():
            indices = torch.tensor(region_data["indices"]).to(self.device)
            region_embed = torch.index_select(self.cond_embeds, 1, indices)
            regional_cond_embeds.append(region_embed)
            
        regional_token_counts = [embed.shape[1] for embed in regional_cond_embeds]
        self.common_len = lcm_for_list(regional_token_counts) if regional_token_counts else 0
        
        self.padded_regional_cond_embeds = []
        for embed in regional_cond_embeds:
            if embed.shape[1] == self.common_len:
                self.padded_regional_cond_embeds.append(embed)
            elif self.common_len != 0:
                repeat_factor = self.common_len // embed.shape[1]
                padded_embed = embed.repeat(1, repeat_factor, 1)
                self.padded_regional_cond_embeds.append(padded_embed)
            
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        residual = hidden_states
        
        # --- Start of diffusers AttentionProcessor flow ---
        # 1. Reshape inputs to `(batch_size * num_heads, ...)`
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        
        # Split into uncond and cond parts *after* reshaping
        query_uncond, query_cond = query.chunk(2)

        # --- Process Unconditional Pass ---
        key_uncond = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states.chunk(2)[0]))
        value_uncond = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states.chunk(2)[0]))
        output_uncond = F.scaled_dot_product_attention(query_uncond, key_uncond, value_uncond)

        # --- Process Conditional Pass (Regional Logic) ---
        # Project and reshape the stacked regional prompts
        regional_prompts_stacked = torch.cat(self.padded_regional_cond_embeds, dim=0)
        key_regional = attn.head_to_batch_dim(attn.to_k(regional_prompts_stacked))
        value_regional = attn.head_to_batch_dim(attn.to_v(regional_prompts_stacked))

        # The query_cond already has shape `(1*heads, seq_len, inner_dim)`.
        # We need to repeat it for each region.
        query_cond_repeated = query_cond.repeat(self.num_regions, 1, 1)

        # Perform attention for all regions at once
        regional_outputs = F.scaled_dot_product_attention(query_cond_repeated, key_regional, value_regional)

        # Correctly calculate latent shape
        sequence_length = hidden_states.shape[1]
        image_ratio = self.height / self.width
        latent_w = int(math.sqrt(sequence_length / image_ratio))
        latent_h = sequence_length // latent_w
        
        # Recombine regional outputs using masks
        # The output must have the same shape as query_cond: `(1*heads, seq_len, inner_dim)`
        final_cond_output = torch.zeros_like(query_cond)
        
        # Each chunk of regional_outputs has shape `(1*heads, seq_len, inner_dim)`
        regional_outputs_chunked = regional_outputs.chunk(self.num_regions, dim=0)

        for i, region_data in enumerate(self.regions_config.values()):
            mask = region_data["mask"].to(self.device, self.dtype)
            mask_downsampled = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(latent_h, latent_w), mode='bilinear', align_corners=False)
            mask_downsampled = mask_downsampled.view(1, sequence_length, 1) # Reshape for broadcasting
            
            final_cond_output += regional_outputs_chunked[i] * mask_downsampled

        # --- Combine and Reshape Back ---
        # Concatenate along the batch*heads dimension
        output = torch.cat([output_uncond, final_cond_output], dim=0)
        
        # 5. Reshape back to `(batch_size, ...)`
        output = attn.batch_to_head_dim(output)
        
        # 6. Final projection and residual
        output = attn.to_out[0](output)
        output = attn.to_out[1](output) # Dropout
        output = output + residual

        return output

# --- 3. Main Script Setup and Execution (Unchanged from last version) ---
if __name__ == "__main__":
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "socks22/sdxl-wai-nsfw-illustriousv14",
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    prompt_left = "1girl, aqua (konosuba), blue hair, smiling, masterpiece, best quality"
    prompt_right = "1girl, megumin (konosuba), brown hair, eyepatch, sad, masterpiece, best quality"
    full_prompt = f"{prompt_left} AND {prompt_right}"
    negative_prompt = "blurry, ugly, deformed, text, watermark, worst quality, low quality, (bad anatomy)"
    width, height = 1024, 768
    
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(prompt=full_prompt, negative_prompt=negative_prompt, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=True)
    full_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    tokenizer1 = pipe.tokenizer
    indices_left = find_token_indices(tokenizer1, full_prompt, prompt_left)
    indices_right = find_token_indices(tokenizer1, full_prompt, prompt_right)
    print(f"Left prompt indices: {indices_left} (Length: {len(indices_left)})")
    print(f"Right prompt indices: {indices_right} (Length: {len(indices_right)})")

    mask_left = torch.zeros((height, width))
    mask_left[:, :width // 2] = 1.0
    mask_right = torch.zeros((height, width))
    mask_right[:, width // 2:] = 1.0
    regions_config = {"left": {"mask": mask_left, "indices": indices_left}, "right": {"mask": mask_right, "indices": indices_right}}

    print("Injecting custom attention processor...")
    attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        if "attn2" in name:
            attn_procs[name] = RegionalAttnProcessorV3(regions_config, full_prompt_embeds, width, height)
        else:
            attn_procs[name] = pipe.unet.attn_processors[name]
    pipe.unet.set_attn_processor(attn_procs)

    print("Generating image with attention coupling...")
    generator = torch.Generator("cuda").manual_seed(12345)
    image = pipe(
        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        width=width, height=height, guidance_scale=1, num_inference_steps=8, generator=generator
    ).images[0]
    
    print("Image generation complete.")
    image.save("multi_character_composition_sdxl_final_final_final_fix.png")
    print("Saved image to 'multi_character_composition_sdxl_final_final_final_fix.png'")