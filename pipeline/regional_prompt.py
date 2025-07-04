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

# --- 2. The Final, Correct Attention Processor ---
# This version replicates the sd-forge-couple logic faithfully.
class RegionalAttnProcessorV7:
    def __init__(self, name: str, region_masks: List[torch.Tensor], region_token_counts: List[int], base_context_token_count: int, width: int, height: int):
        self.name = name
        self.region_masks = region_masks
        self.num_regions = len(region_masks)
        self.region_token_counts = region_token_counts
        self.base_context_token_count = base_context_token_count
        self.width = width
        self.height = height
        
        # This mask is for the 'base' context derived from the full prompt.
        # In sd-forge-couple, this is effectively zero, replacing the base context.
        self.base_mask = torch.zeros((height, width), device="cuda", dtype=torch.float16)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        if not hasattr(self, 'call_count'):
            self.call_count = 0
        
        is_debug_call = self.call_count < 2

        if is_debug_call:
            print(f"\n--- AttnProc Call Start: {self.name}, Call Count: {self.call_count} ---")
            print(f"  hidden_states shape: {hidden_states.shape}, stats: min={hidden_states.min():.4f}, max={hidden_states.max():.4f}, mean={hidden_states.mean():.4f}")
            if encoder_hidden_states is not None:
                print(f"  encoder_hidden_states shape: {encoder_hidden_states.shape}, stats: min={encoder_hidden_states.min():.4f}, max={encoder_hidden_states.max():.4f}, mean={encoder_hidden_states.mean():.4f}")

        residual = hidden_states
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        
        if is_debug_call:
            print(f"  query shape: {query.shape}, stats: min={query.min():.4f}, max={query.max():.4f}, mean={query.mean():.4f}")
        
        query_uncond, query_cond = query.chunk(2)
        encoder_states_uncond, encoder_states_cond = encoder_hidden_states.chunk(2)
        
        if is_debug_call:
            print(f"  query_uncond shape: {query_uncond.shape}, stats: min={query_uncond.min():.4f}, max={query_uncond.max():.4f}, mean={query_uncond.mean():.4f}")
            print(f"  query_cond shape: {query_cond.shape}, stats: min={query_cond.min():.4f}, max={query_cond.max():.4f}, mean={query_cond.mean():.4f}")
            print(f"  encoder_states_uncond shape: {encoder_states_uncond.shape}")
            print(f"  encoder_states_cond shape: {encoder_states_cond.shape}")

        # --- Unconditional Pass (Standard) ---
        key_uncond = attn.head_to_batch_dim(attn.to_k(encoder_states_uncond))
        value_uncond = attn.head_to_batch_dim(attn.to_v(encoder_states_uncond))
        output_uncond = F.scaled_dot_product_attention(query_uncond, key_uncond, value_uncond)

        if is_debug_call:
            print(f"  key_uncond shape: {key_uncond.shape}, stats: min={key_uncond.min():.4f}, max={key_uncond.max():.4f}, mean={key_uncond.mean():.4f}")
            print(f"  value_uncond shape: {value_uncond.shape}, stats: min={value_uncond.min():.4f}, max={value_uncond.max():.4f}, mean={value_uncond.mean():.4f}")
            print(f"  output_uncond shape: {output_uncond.shape}, stats: min={output_uncond.min():.4f}, max={output_uncond.max():.4f}, mean={output_uncond.mean():.4f}")

        # --- Conditional Pass (Regional) ---
        final_cond_output = torch.zeros_like(query_cond)
        
        # The encoder_states_cond now contains the base context AND all regional contexts concatenated.
        key_cond_all = attn.head_to_batch_dim(attn.to_k(encoder_states_cond))
        value_cond_all = attn.head_to_batch_dim(attn.to_v(encoder_states_cond))
        
        if is_debug_call:
            print(f"  key_cond_all shape: {key_cond_all.shape}, stats: min={key_cond_all.min():.4f}, max={key_cond_all.max():.4f}, mean={key_cond_all.mean():.4f}")
            print(f"  value_cond_all shape: {value_cond_all.shape}, stats: min={value_cond_all.min():.4f}, max={value_cond_all.max():.4f}, mean={value_cond_all.mean():.4f}")
        
        # --- Attention for Base Context (from full prompt) ---
        key_base = key_cond_all[:, :self.base_context_token_count, :]
        value_base = value_cond_all[:, :self.base_context_token_count, :]
        base_output = F.scaled_dot_product_attention(query_cond, key_base, value_base)
        
        if is_debug_call:
            print(f"  key_base shape: {key_base.shape}")
            print(f"  value_base shape: {value_base.shape}")
            print(f"  base_output shape: {base_output.shape}, stats: min={base_output.min():.4f}, max={base_output.max():.4f}, mean={base_output.mean():.4f}")

        # Downsample the base mask
        sequence_length = hidden_states.shape[1]
        image_ratio = self.height / self.width
        latent_w = int(math.sqrt(sequence_length / image_ratio))
        latent_h = sequence_length // latent_w
        base_mask_downsampled = F.interpolate(self.base_mask.unsqueeze(0).unsqueeze(0), size=(latent_h, latent_w), mode='bilinear', align_corners=False)
        base_mask_downsampled = base_mask_downsampled.view(1, sequence_length, 1)

        if is_debug_call:
            print(f"  latent_h={latent_h}, latent_w={latent_w}")
            print(f"  base_mask_downsampled shape: {base_mask_downsampled.shape}, sum: {base_mask_downsampled.sum():.4f}")

        # Add the (zeroed out) base output
        final_cond_output += base_output * base_mask_downsampled

        if is_debug_call:
            print(f"  final_cond_output after base (should be 0): stats: min={final_cond_output.min():.4f}, max={final_cond_output.max():.4f}, mean={final_cond_output.mean():.4f}")
        
        # --- Attention for Regional Contexts ---
        token_start_index = self.base_context_token_count
        for i in range(self.num_regions):
            token_len = self.region_token_counts[i]
            token_end_index = token_start_index + token_len
            
            key_regional = key_cond_all[:, token_start_index:token_end_index, :]
            value_regional = value_cond_all[:, token_start_index:token_end_index, :]
            regional_output = F.scaled_dot_product_attention(query_cond, key_regional, value_regional)

            mask = self.region_masks[i]
            mask_downsampled = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(latent_h, latent_w), mode='bilinear', align_corners=False)
            mask_downsampled = mask_downsampled.view(1, sequence_length, 1)
            
            if is_debug_call:
                print(f"  --- Region {i} ---")
                print(f"    token_len: {token_len}, start: {token_start_index}, end: {token_end_index}")
                print(f"    key_regional shape: {key_regional.shape}")
                print(f"    value_regional shape: {value_regional.shape}")
                print(f"    regional_output stats: min={regional_output.min():.4f}, max={regional_output.max():.4f}, mean={regional_output.mean():.4f}")
                print(f"    mask shape: {mask.shape}, sum: {mask.sum():.4f}")
                print(f"    mask_downsampled shape: {mask_downsampled.shape}, sum: {mask_downsampled.sum():.4f}")
            
            final_cond_output += regional_output * mask_downsampled
            if is_debug_call:
                print(f"    final_cond_output after region {i}: stats: min={final_cond_output.min():.4f}, max={final_cond_output.max():.4f}, mean={final_cond_output.mean():.4f}")

            token_start_index = token_end_index

        # --- Combine and Reshape Back ---
        output = torch.cat([output_uncond, final_cond_output], dim=0)
        output = attn.batch_to_head_dim(output)
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        output = output + residual

        if is_debug_call:
            print(f"  Final output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
            print(f"--- AttnProc Call End: {self.name} ---")
        
        self.call_count += 1
        return output

# --- 3. Main Script Setup and Execution ---
if __name__ == "__main__":
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "John6666/wai-nsfw-illustrious-v130-sdxl",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    # --- DEFINE PROMPTS ---
    prompt_background = "masterpiece, best quality, sharp focus, intricate details, cinematic lighting, indoors, tavern"
    prompt_region_1 = "1girl, aqua (konosuba), smiling, upper body"
    prompt_region_2 = "1girl, megumin (konosuba), sad, eyepatch, upper body"
    
    # Combine all prompts into a single string for holistic pooled embedding generation
    full_prompt_text = f"{prompt_background}\n{prompt_region_1}\n{prompt_region_2}"
    
    negative_prompt = "blurry, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name"
    width, height = 1024, 768
    
    # --- ENCODE PROMPTS ---
    print("Encoding prompts...")
    # 1. Encode the FULL prompt to get the global pooled embedding and a base context
    base_cond_embeds, _, global_pooled_embeds, _ = pipe.encode_prompt(prompt=full_prompt_text, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)

    # 2. Encode each regional part separately for cross-attention
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

    # Concatenate the base context AND all padded regional contexts
    final_cond_embeds = torch.cat([base_cond_embeds] + padded_regional_conds, dim=1)
    
    # The processor needs to know the lengths of the padded regional parts
    padded_token_counts = [common_len] * len(regional_cond_embeds_list)

    # Pad the negative prompt to match the final positive prompt's massive length
    if uncond_embeds.shape[1] < final_cond_embeds.shape[1]:
        padding_shape = (1, final_cond_embeds.shape[1] - uncond_embeds.shape[1], uncond_embeds.shape[2])
        padding = torch.zeros(padding_shape, dtype=uncond_embeds.dtype, device=uncond_embeds.device)
        padded_uncond_embeds = torch.cat([uncond_embeds, padding], dim=1)
    else:
        padded_uncond_embeds = uncond_embeds

    # --- CREATE MASKS ---
    device, dtype = "cuda", torch.float16
    mask_background = torch.ones((height, width), device=device, dtype=dtype)
    mask_region_1 = torch.zeros((height, width), device=device, dtype=dtype)
    mask_region_1[:, :width // 2] = 1.0
    mask_region_2 = torch.zeros((height, width), device=device, dtype=dtype)
    mask_region_2[:, width // 2:] = 1.0
    
    region_masks = [mask_background, mask_region_1, mask_region_2]
    
    # --- INJECT PROCESSOR ---
    print("Injecting custom attention processor...")
    attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        if "attn2" in name:
            attn_procs[name] = RegionalAttnProcessorV7(
                name,
                region_masks, 
                padded_token_counts, 
                base_cond_embeds.shape[1], # Pass the length of the base context
                width, 
                height
            )
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