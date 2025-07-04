import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import numpy as np
from typing import List
import math

# --- 1. Helper Functions (Unchanged) ---
def lcm(a, b):
    return abs(a * b) // math.gcd(a, b) if a != 0 and b != 0 else 0

def lcm_for_list(numbers):
    if not numbers: return 0
    result = numbers[0]
    for i in range(1, len(numbers)):
        result = lcm(result, numbers[i])
    return result

# --- 2. The Final, Architecturally Correct Processor ---
class RegionalAttnProcessorV8:
    def __init__(self, region_masks_and_weights: List[tuple[torch.Tensor, float]], region_token_counts: List[int], width: int, height: int):
        self.region_masks_and_weights = region_masks_and_weights
        self.num_regions = len(region_masks_and_weights)
        self.region_token_counts = region_token_counts
        self.width = width
        self.height = height

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        residual = hidden_states
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        
        query_uncond, query_cond = query.chunk(2)
        encoder_states_uncond, encoder_states_cond = encoder_hidden_states.chunk(2)
        
        # Unconditional Pass
        key_uncond = attn.head_to_batch_dim(attn.to_k(encoder_states_uncond))
        value_uncond = attn.head_to_batch_dim(attn.to_v(encoder_states_uncond))
        output_uncond = F.scaled_dot_product_attention(query_uncond, key_uncond, value_uncond)

        # Conditional Pass
        final_cond_output = torch.zeros_like(query_cond)
        key_cond_all = attn.head_to_batch_dim(attn.to_k(encoder_states_cond))
        value_cond_all = attn.head_to_batch_dim(attn.to_v(encoder_states_cond))

        token_start_index = 0
        for i in range(self.num_regions):
            token_len = self.region_token_counts[i]
            token_end_index = token_start_index + token_len
            
            key_regional = key_cond_all[:, token_start_index:token_end_index, :]
            value_regional = value_cond_all[:, token_start_index:token_end_index, :]
            
            regional_output = F.scaled_dot_product_attention(query_cond, key_regional, value_regional)

            mask, weight = self.region_masks_and_weights[i]
            
            sequence_length = hidden_states.shape[1]
            image_ratio = self.height / self.width
            latent_w = int(math.sqrt(sequence_length / image_ratio))
            latent_h = sequence_length // latent_w
            
            # Apply the weight to the mask
            weighted_mask = mask * weight
            mask_downsampled = F.interpolate(weighted_mask.unsqueeze(0).unsqueeze(0), size=(latent_h, latent_w), mode='bilinear', align_corners=False)
            mask_downsampled = mask_downsampled.view(1, sequence_length, 1)
            
            final_cond_output += regional_output * mask_downsampled
            token_start_index = token_end_index

        output = torch.cat([output_uncond, final_cond_output], dim=0)
        output = attn.batch_to_head_dim(output)
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        output = output + residual
        return output

# --- 3. Main Script Setup and Execution ---
if __name__ == "__main__":
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "John6666/wai-nsfw-illustrious-v130-sdxl",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    # --- DEFINE PROMPTS AND WEIGHTS ---
    prompt_background = "masterpiece, best quality, sharp focus, intricate details, cinematic lighting, indoors, tavern"
    prompt_region_1 = "1girl, aqua (konosuba), smiling, upper body"
    prompt_region_2 = "1girl, megumin (konosuba), sad, eyepatch, upper body"
    
    # These weights are crucial and mimic the defaults in sd-forge-couple
    bg_weight = 0.5
    tile_weight = 1.0

    # Combine all prompts into a single string for the one-time, holistic pooled embedding generation
    full_prompt_text = f"{prompt_background}\n{prompt_region_1}\n{prompt_region_2}"
    
    negative_prompt = "blurry, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name"
    width, height = 1024, 768
    
    print("Encoding prompts...")
    # 1. Encode the FULL prompt text to get the ONE TRUE global pooled embedding
    _, _, global_pooled_embeds, _ = pipe.encode_prompt(prompt=full_prompt_text, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)

    # 2. Encode each part separately for its cross-attention context
    cond_embeds_bg, _, _, _ = pipe.encode_prompt(prompt=prompt_background, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)
    cond_embeds_r1, _, _, _ = pipe.encode_prompt(prompt=prompt_region_1, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)
    cond_embeds_r2, _, _, _ = pipe.encode_prompt(prompt=prompt_region_2, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)
    
    # 3. Encode negative prompt
    _, uncond_embeds, _, uncond_pooled_embeds = pipe.encode_prompt(prompt="", negative_prompt=negative_prompt, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=True)
    
    # --- PREPARE EMBEDDINGS FOR PIPELINE ---
    regional_cond_embeds_list = [cond_embeds_bg, cond_embeds_r1, cond_embeds_r2]
    regional_token_counts = [embed.shape[1] for embed in regional_cond_embeds_list]
    common_len = lcm_for_list(regional_token_counts)
    
    padded_regional_conds = []
    for embed in regional_cond_embeds_list:
        if embed.shape[1] < common_len:
            padded_regional_conds.append(embed.repeat(1, common_len // embed.shape[1], 1))
        else:
            padded_regional_conds.append(embed)

    final_cond_embeds = torch.cat(padded_regional_conds, dim=1)
    padded_token_counts = [common_len] * len(regional_cond_embeds_list)
    
    # Pad the negative prompt to match the concatenated positive prompt's length
    if uncond_embeds.shape[1] < final_cond_embeds.shape[1]:
        padding_shape = (1, final_cond_embeds.shape[1] - uncond_embeds.shape[1], uncond_embeds.shape[2])
        padding = torch.zeros(padding_shape, dtype=uncond_embeds.dtype, device=uncond_embeds.device)
        padded_uncond_embeds = torch.cat([uncond_embeds, padding], dim=1)
    else:
        padded_uncond_embeds = uncond_embeds

    # --- CREATE MASKS AND PAIR WITH WEIGHTS ---
    device, dtype = "cuda", torch.float16
    mask_background = torch.ones((height, width), device=device, dtype=dtype)
    mask_region_1 = torch.zeros((height, width), device=device, dtype=dtype)
    mask_region_1[:, :width // 2] = 1.0
    mask_region_2 = torch.zeros((height, width), device=device, dtype=dtype)
    mask_region_2[:, width // 2:] = 1.0
    
    # Pair each mask with its corresponding weight
    region_masks_and_weights = [
        (mask_background, bg_weight),
        (mask_region_1, tile_weight),
        (mask_region_2, tile_weight)
    ]
    
    # --- INJECT PROCESSOR ---
    print("Injecting custom attention processor...")
    attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        if "attn2" in name:
            attn_procs[name] = RegionalAttnProcessorV8(region_masks_and_weights, padded_token_counts, width, height)
        else:
            attn_procs[name] = pipe.unet.attn_processors[name]
    pipe.unet.set_attn_processor(attn_procs)

    # --- GENERATE IMAGE ---
    print("Generating image...")
    generator = torch.Generator("cuda").manual_seed(12345)
    
    image = pipe(
        prompt_embeds=final_cond_embeds,
        negative_prompt_embeds=padded_uncond_embeds,
        pooled_prompt_embeds=global_pooled_embeds, # Use the true global pooled embeds
        negative_pooled_prompt_embeds=uncond_pooled_embeds,
        width=width, height=height, guidance_scale=7.0, num_inference_steps=28, generator=generator
    ).images[0]
    
    print("Image generation complete.")
    image.save("final_image.png")
    print("Saved image to final_image.png")