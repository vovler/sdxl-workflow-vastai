import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
import math

# --- 1. Helper Functions (LCM only) ---
def lcm(a, b):
    return abs(a * b) // math.gcd(a, b) if a != 0 and b != 0 else 0

def lcm_for_list(numbers):
    if not numbers: return 0
    result = numbers[0]
    for i in range(1, len(numbers)):
        result = lcm(result, numbers[i])
    return result

# --- 2. The Core Logic: Final Attention Processor ---
# This version takes individually encoded regional prompts, the most robust method.
class RegionalAttnProcessorV4:
    def __init__(self, regional_cond_embeds: List[torch.Tensor], uncond_embeds: torch.Tensor, width: int, height: int):
        self.num_regions = len(regional_cond_embeds)
        self.uncond_embeds = uncond_embeds
        self.device = uncond_embeds.device
        self.dtype = uncond_embeds.dtype
        self.width = width
        self.height = height
        
        # --- Padding Logic ---
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
        
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        
        # --- Unconditional Pass ---
        # The uncond key/value are taken from the single uncond_embeds passed during init
        key_uncond = attn.head_to_batch_dim(attn.to_k(self.uncond_embeds))
        value_uncond = attn.head_to_batch_dim(attn.to_v(self.uncond_embeds))
        output_uncond = F.scaled_dot_product_attention(query.chunk(2)[0], key_uncond, value_uncond)

        # --- Conditional Pass (Regional Logic) ---
        regional_prompts_stacked = torch.cat(self.padded_regional_cond_embeds, dim=0)
        key_regional = attn.head_to_batch_dim(attn.to_k(regional_prompts_stacked))
        value_regional = attn.head_to_batch_dim(attn.to_v(regional_prompts_stacked))

        query_cond_repeated = query.chunk(2)[1].repeat(self.num_regions, 1, 1)
        regional_outputs = F.scaled_dot_product_attention(query_cond_repeated, key_regional, value_regional)

        sequence_length = hidden_states.shape[1]
        image_ratio = self.height / self.width
        latent_w = int(math.sqrt(sequence_length / image_ratio))
        latent_h = sequence_length // latent_w
        
        final_cond_output = torch.zeros_like(query.chunk(2)[1])
        regional_outputs_chunked = regional_outputs.chunk(self.num_regions, dim=0)

        for i, mask_tensor in enumerate(self.region_masks):
            mask_downsampled = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0), size=(latent_h, latent_w), mode='bilinear', align_corners=False)
            mask_downsampled = mask_downsampled.view(1, sequence_length, 1)
            final_cond_output += regional_outputs_chunked[i] * mask_downsampled

        # --- Combine and Reshape Back ---
        output = torch.cat([output_uncond, final_cond_output], dim=0)
        output = attn.batch_to_head_dim(output)
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        output = output + residual
        return output

# --- 3. Main Script Setup and Execution ---
if __name__ == "__main__":
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "John6666/wai-nsfw-illustrious-v130-sdxl", vae=vae, torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    # Define regional prompts and negative prompt
    prompt_left = "masterpiece,best quality,amazing quality, 1girl, aqua_(konosuba)"
    prompt_right = "masterpiece,best quality,amazing quality, 1girl, megumin"
    negative_prompt = "bad quality,worst quality,worst detail,sketch,censor,"
    width, height = 1024, 768
    
    # --- Encode each prompt SEPARATELY ---
    print("Encoding regional prompts separately...")
    cond_embeds_left, _, _, _ = pipe.encode_prompt(prompt=prompt_left, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)
    cond_embeds_right, _, _, _ = pipe.encode_prompt(prompt=prompt_right, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)
    
    # Encode negative prompt once
    _, uncond_embeds, _, uncond_pooled_embeds = pipe.encode_prompt(prompt="", negative_prompt=negative_prompt, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=True)
    
    # We need pooled embeds for the pipeline call. We'll just use the ones from the first prompt.
    # The processor doesn't use them, so this is just to satisfy the pipeline's signature.
    _, _, pooled_embeds, _ = pipe.encode_prompt(prompt=prompt_left, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)

    regional_cond_embeds = [cond_embeds_left, cond_embeds_right]

    # Create masks
    mask_left = torch.zeros((height, width))
    mask_left[:, :width // 2] = 1.0
    mask_right = torch.zeros((height, width))
    mask_right[:, width // 2:] = 1.0
    
    print("Injecting custom attention processor...")
    attn_procs = {}
    # The processor needs the masks to be available
    RegionalAttnProcessorV4.region_masks = [mask_left.to("cuda", torch.float16), mask_right.to("cuda", torch.float16)]
    for name in pipe.unet.attn_processors.keys():
        if "attn2" in name: # Target cross-attention blocks
            attn_procs[name] = RegionalAttnProcessorV4(regional_cond_embeds, uncond_embeds, width, height)
        else: # Use default for self-attention
            attn_procs[name] = pipe.unet.attn_processors[name]
    pipe.unet.set_attn_processor(attn_procs)

    print("Generating image with attention coupling...")
    generator = torch.Generator("cuda").manual_seed(12345)
    
    # The main prompt_embeds passed to the pipe are now just placeholders.
    # Our processor will ignore them and use its own internal uncond/regional embeds.
    image = pipe(
        prompt_embeds=cond_embeds_left, # Placeholder
        negative_prompt_embeds=uncond_embeds,
        pooled_prompt_embeds=pooled_embeds, # Placeholder
        negative_pooled_prompt_embeds=uncond_pooled_embeds,
        width=width, height=height, guidance_scale=5, num_inference_steps=28, generator=generator
    ).images[0]
    
    print("Image generation complete.")
    image.save("multi_character_composition_sdxl_final.png")
    print("Saved image to 'multi_character_composition_sdxl_final.png'")