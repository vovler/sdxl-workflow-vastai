import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import numpy as np
from typing import List
import math

# --- Helper Functions from sd-forge-couple ---
def repeat_div(value: int, iterations: int) -> int:
    """Repeatedly divide value by 2, ceiling the result"""
    for _ in range(iterations):
        value = math.ceil(value / 2)
    return value

def get_mask(mask, batch_size, num_tokens, original_shape):
    """
    Credit: hako-mikan, arcusmaximus & www from sd-forge-couple
    Properly calculate mask for attention layers
    """
    image_width: int = original_shape[3]
    image_height: int = original_shape[2]

    scale = math.ceil(math.log2(math.sqrt(image_height * image_width / num_tokens)))
    size = (repeat_div(image_height, scale), repeat_div(image_width, scale))

    num_conds = mask.shape[0]
    mask_downsample = F.interpolate(mask.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    mask_downsample = mask_downsample.view(num_conds, num_tokens, 1).repeat_interleave(
        batch_size, dim=0
    )

    return mask_downsample

def lcm(a, b):
    return abs(a * b) // math.gcd(a, b) if a != 0 and b != 0 else 0

def lcm_for_list(numbers):
    if not numbers: return 0
    result = numbers[0]
    for i in range(1, len(numbers)):
        result = lcm(result, numbers[i])
    return result

# --- Fixed Regional Attention Processor ---
class RegionalAttnProcessorV8:
    def __init__(self, region_masks_and_weights: List[tuple[torch.Tensor, float]], region_token_counts: List[int], width: int, height: int):
        self.region_masks_and_weights = region_masks_and_weights
        self.num_regions = len(region_masks_and_weights)
        self.region_token_counts = region_token_counts
        self.width = width
        self.height = height

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        residual = hidden_states
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        
        batch_size = query.shape[0] // 2  # Account for CFG (unconditional + conditional)
        sequence_length = hidden_states.shape[1]
        
        # Split unconditional and conditional
        query_uncond, query_cond = query.chunk(2)
        encoder_states_uncond, encoder_states_cond = encoder_hidden_states.chunk(2)
        
        # Unconditional Pass (unchanged)
        key_uncond = attn.head_to_batch_dim(attn.to_k(encoder_states_uncond))
        value_uncond = attn.head_to_batch_dim(attn.to_v(encoder_states_uncond))
        output_uncond = F.scaled_dot_product_attention(query_uncond, key_uncond, value_uncond)

        # Conditional Pass with Regional Attention
        final_cond_output = torch.zeros_like(query_cond)
        
        # Prepare masks using forge-couple method
        original_shape = (batch_size, 3, self.height, self.width)
        all_masks = torch.stack([mask for mask, _ in self.region_masks_and_weights])
        
        # Get properly downsampled masks
        mask_downsample = get_mask(all_masks, batch_size, sequence_length, original_shape)
        
        token_start_index = 0
        for i in range(self.num_regions):
            token_len = self.region_token_counts[i]
            token_end_index = token_start_index + token_len
            
            # Get regional keys and values
            key_regional = attn.head_to_batch_dim(attn.to_k(encoder_states_cond[:, token_start_index:token_end_index, :]))
            value_regional = attn.head_to_batch_dim(attn.to_v(encoder_states_cond[:, token_start_index:token_end_index, :]))
            
            # Compute regional attention
            regional_output = F.scaled_dot_product_attention(query_cond, key_regional, value_regional)

            # Get mask for this region (accounting for batch_size repetition)
            mask, weight = self.region_masks_and_weights[i]
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            regional_mask = mask_downsample[start_idx:end_idx]  # Shape: [batch_size, sequence_length, 1]
            
            # Apply weight to the mask and accumulate
            weighted_mask = regional_mask * weight
            final_cond_output += regional_output * weighted_mask
            
            token_start_index = token_end_index

        # Combine unconditional and conditional outputs
        output = torch.cat([output_uncond, final_cond_output], dim=0)
        output = attn.batch_to_head_dim(output)
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        output = output + residual
        return output

# --- Main Script Setup and Execution ---
if __name__ == "__main__":
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "John6666/wai-nsfw-illustrious-v130-sdxl",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    # --- DEFINE PROMPTS AND WEIGHTS ---
    prompt_background = "masterpiece, best quality, sharp focus, intricate details, cinematic lighting, indoors, tavern"
    prompt_region_1 = "1girl, aqua_(konosuba), smiling, upper body"
    prompt_region_2 = "1girl, megumin (konosuba), sad, eyepatch, upper body"
    
    # These weights follow sd-forge-couple defaults
    bg_weight = 0.5
    tile_weight = 1.0

    negative_prompt = "blurry, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name"
    width, height = 1024, 768
    
    print("Encoding prompts...")
    
    # Encode each prompt separately to get proper embeddings
    cond_embeds_bg, _, _, _ = pipe.encode_prompt(prompt=prompt_background, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)
    cond_embeds_r1, _, _, _ = pipe.encode_prompt(prompt=prompt_region_1, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)
    cond_embeds_r2, _, _, _ = pipe.encode_prompt(prompt=prompt_region_2, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)
    
    # For pooled embeddings, use the background prompt as the global context
    _, _, global_pooled_embeds, _ = pipe.encode_prompt(prompt=prompt_background, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)
    
    # Encode negative prompt
    _, uncond_embeds, _, uncond_pooled_embeds = pipe.encode_prompt(prompt="", negative_prompt=negative_prompt, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=True)
    
    # --- PREPARE EMBEDDINGS FOR PIPELINE ---
    regional_cond_embeds_list = [cond_embeds_bg, cond_embeds_r1, cond_embeds_r2]
    regional_token_counts = [embed.shape[1] for embed in regional_cond_embeds_list]
    
    # Ensure all embeddings have the same length by padding to LCM
    common_len = lcm_for_list(regional_token_counts)
    
    padded_regional_conds = []
    for embed in regional_cond_embeds_list:
        if embed.shape[1] < common_len:
            # Repeat the embedding to reach common length
            repeat_factor = common_len // embed.shape[1]
            padded_embed = embed.repeat(1, repeat_factor, 1)
            if padded_embed.shape[1] < common_len:
                # Handle any remaining tokens by repeating the last token
                remaining = common_len - padded_embed.shape[1]
                last_token = embed[:, -1:, :].repeat(1, remaining, 1)
                padded_embed = torch.cat([padded_embed, last_token], dim=1)
            padded_regional_conds.append(padded_embed)
        else:
            padded_regional_conds.append(embed[:, :common_len, :])

    final_cond_embeds = torch.cat(padded_regional_conds, dim=1)
    padded_token_counts = [common_len] * len(regional_cond_embeds_list)
    
    # Pad the negative prompt to match the concatenated positive prompt's length
    if uncond_embeds.shape[1] < final_cond_embeds.shape[1]:
        padding_length = final_cond_embeds.shape[1] - uncond_embeds.shape[1]
        # Repeat the last token of uncond_embeds to pad
        last_token = uncond_embeds[:, -1:, :].repeat(1, padding_length, 1)
        padded_uncond_embeds = torch.cat([uncond_embeds, last_token], dim=1)
    else:
        padded_uncond_embeds = uncond_embeds[:, :final_cond_embeds.shape[1], :]

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
        if "attn2" in name:  # Only replace cross-attention processors
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
        pooled_prompt_embeds=global_pooled_embeds,
        negative_pooled_prompt_embeds=uncond_pooled_embeds,
        width=width, height=height, guidance_scale=7.0, num_inference_steps=28, generator=generator
    ).images[0]
    
    print("Image generation complete.")
    image.save("final_image.png")
    print("Saved image to final_image.png")