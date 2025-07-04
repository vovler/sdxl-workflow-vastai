import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
import math

# --- 1. Helper Functions ---
def find_token_indices(tokenizer, prompt: str, sub_prompt: str) -> List[int]:
    # (Same as before, no changes needed)
    tokens_prompt = tokenizer.tokenize(prompt)
    tokens_sub_prompt = tokenizer.tokenize(sub_prompt)
    for i in range(len(tokens_prompt) - len(tokens_sub_prompt) + 1):
        if tokens_prompt[i:i + len(tokens_sub_prompt)] == tokens_sub_prompt:
            return list(range(i + 1, i + len(tokens_sub_prompt) + 1))
    raise ValueError(f"Sub-prompt '{sub_prompt}' not found in prompt '{prompt}'.")

def lcm(a, b):
    return abs(a * b) // math.gcd(a, b) if a != 0 and b != 0 else 0

def lcm_for_list(numbers):
    # (Same as before, no changes needed)
    if not numbers:
        return 0
    result = numbers[0]
    for i in range(1, len(numbers)):
        result = lcm(result, numbers[i])
    return result

# --- 2. The Core Logic: Corrected Custom Attention Processor ---
class RegionalAttnProcessorV2:
    # MODIFICATION: Added width and height to __init__
    def __init__(self, regions_config: Dict, full_prompt_embeds: torch.Tensor, width: int, height: int):
        self.regions_config = regions_config
        self.num_regions = len(regions_config)
        self.device = full_prompt_embeds.device
        self.dtype = full_prompt_embeds.dtype
        self.width = width
        self.height = height
        
        self.uncond_embeds, self.cond_embeds = full_prompt_embeds.chunk(2)
        
        regional_cond_embeds = []
        for region_data in self.regions_config.values():
            indices = torch.tensor(region_data["indices"]).to(self.device)
            region_embed = torch.index_select(self.cond_embeds, 1, indices)
            regional_cond_embeds.append(region_embed)
            
        regional_token_counts = [embed.shape[1] for embed in regional_cond_embeds]

        if len(regional_token_counts) > 1:
            self.common_len = lcm_for_list(regional_token_counts)
        elif regional_token_counts:
            self.common_len = regional_token_counts[0]
        else:
            self.common_len = 0 # Handle case with no regions
        
        self.padded_regional_cond_embeds = []
        for embed in regional_cond_embeds:
            if embed.shape[1] == self.common_len:
                self.padded_regional_cond_embeds.append(embed)
            else:
                repeat_factor = self.common_len // embed.shape[1]
                padded_embed = embed.repeat(1, repeat_factor, 1)
                self.padded_regional_cond_embeds.append(padded_embed)
            
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        hidden_states_uncond, hidden_states_cond = hidden_states.chunk(2)
        encoder_hidden_states_uncond, encoder_hidden_states_cond = encoder_hidden_states.chunk(2)

        output_uncond = F.scaled_dot_product_attention(
            attn.to_q(hidden_states_uncond), attn.to_k(encoder_hidden_states_uncond), attn.to_v(encoder_hidden_states_uncond),
        )

        hidden_states_cond_repeated = hidden_states_cond.repeat(self.num_regions, 1, 1)
        regional_prompts_stacked = torch.cat(self.padded_regional_cond_embeds, dim=0)

        q = attn.to_q(hidden_states_cond_repeated)
        k = attn.to_k(regional_prompts_stacked)
        v = attn.to_v(regional_prompts_stacked)
        regional_outputs = F.scaled_dot_product_attention(q, k, v)

        # --- START: ASPECT-RATIO-CORRECT SHAPE CALCULATION ---
        # This is the key fix for non-square images.
        image_ratio = self.height / self.width
        # We solve a system of equations:
        # 1) latent_h * latent_w = sequence_length
        # 2) latent_h / latent_w = image_ratio
        # From (2), latent_h = latent_w * image_ratio. Substitute into (1):
        # (latent_w * image_ratio) * latent_w = sequence_length
        # latent_w^2 = sequence_length / image_ratio
        latent_w = int(math.sqrt(sequence_length / image_ratio))
        latent_h = sequence_length // latent_w
        # --- END: ASPECT-RATIO-CORRECT SHAPE CALCULATION ---
        
        final_cond_output = torch.zeros_like(hidden_states_cond)
        for i, region_data in enumerate(self.regions_config.values()):
            mask = region_data["mask"].to(self.device, self.dtype)
            mask_downsampled = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(latent_h, latent_w), mode='bilinear', align_corners=False)
            mask_downsampled = mask_downsampled.squeeze().view(sequence_length, 1)
            
            region_output = regional_outputs[i].unsqueeze(0)
            final_cond_output += region_output * mask_downsampled

        output = torch.cat([output_uncond, final_cond_output], dim=0)
        output = attn.batch_to_head_dim(output)
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        
        return output

# --- 3. Main Script Setup and Execution ---
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
    
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True
    )
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
    
    regions_config = {
        "left": {"mask": mask_left, "indices": indices_left},
        "right": {"mask": mask_right, "indices": indices_right}
    }

    print("Injecting custom attention processor...")
    attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        if "attn2" in name:
            # MODIFICATION: Pass width and height to the processor
            attn_procs[name] = RegionalAttnProcessorV2(regions_config, full_prompt_embeds, width, height)
        else:
            attn_procs[name] = pipe.unet.attn_processors[name]
    pipe.unet.set_attn_processor(attn_procs)

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
    image.save("multi_character_composition_sdxl_final_final_fix.png")
    print("Saved image to 'multi_character_composition_sdxl_final_final_fix.png'")