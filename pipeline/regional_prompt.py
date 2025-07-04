# regional_prompting_final_professional.py
import torch
import time
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, AutoencoderTiny
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# --- 1. SETTINGS ---
prompt_background = "masterpiece, best quality, absurdres, cinematic lighting, ultra-detailed, dark city street at night, heavy rain, puddles on the ground with glowing reflections of neon signs, wet asphalt"
prompt_left = "1girl, megumin, witch_hat, eyepatch, short_brown_hair, red_dress, black_cape, smug, hands_behind_back"
prompt_center = "1girl, aqua_(konosuba), long_blue_hair, blue_dress, bare_shoulders, hands_up, arms_up, happy, joyful"
prompt_right = "1boy, satou_kazuma, green_tracksuit, short_brown_hair, annoyed, holding_plushie, frog_plushie"

height = 832
width = 1216
num_inference_steps = 15 # A few more steps helps with the complex blending
seed = 45
guidance_scale = 1.0

# --- TUNING KNOBS from the GUIDE ---
# Equivalent to "Base Ratio" in the guide
background_blend_ratio = 0.35
# Equivalent to "Overlay Ratio". Controls the blur/softness of mask edges. Higher sigma = softer edge.
mask_blur_sigma = 4.0

device = "cuda:0"
base_model_id = "socks22/sdxl-wai-nsfw-illustriousv14"


# --- 2. PIPELINE SETUP (Identical) ---
print("Loading pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, use_safetensors=True)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
pipe.enable_xformers_memory_efficient_attention()


# --- 3. ENCODE & BLEND PROMPTS (Identical) ---
print("Encoding and blending prompts...")
def encode_prompt_cfg_less(prompt):
    prompt_embeds, pooled_embeds = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)
    return prompt_embeds, pooled_embeds

prompt_embeds_bg, pooled_embeds_bg = encode_prompt_cfg_less(prompt_background)
prompt_embeds_left, pooled_embeds_left = encode_prompt_cfg_less(prompt_left)
prompt_embeds_center, pooled_embeds_center = encode_prompt_cfg_less(prompt_center)
prompt_embeds_right, pooled_embeds_right = encode_prompt_cfg_less(prompt_right)

def blend_conditionings(cond1, pooled1, cond2, pooled2, ratio):
    blended_cond = (cond1 * ratio) + (cond2 * (1.0 - ratio))
    blended_pooled = (pooled1 * ratio) + (pooled2 * (1.0 - ratio))
    return blended_cond, blended_pooled

prompt_embeds_blended_left, pooled_blended_left = blend_conditionings(prompt_embeds_bg, pooled_embeds_bg, prompt_embeds_left, pooled_embeds_left, background_blend_ratio)
prompt_embeds_blended_center, pooled_blended_center = blend_conditionings(prompt_embeds_bg, pooled_embeds_bg, prompt_embeds_center, pooled_embeds_center, background_blend_ratio)
prompt_embeds_blended_right, pooled_blended_right = blend_conditionings(prompt_embeds_bg, pooled_embeds_bg, prompt_embeds_right, pooled_embeds_right, background_blend_ratio)
print("Conditionings blended.")


# --- 4. PREPARE LATENTS AND SOFT MASKS ---
generator = torch.Generator(device).manual_seed(seed)
latents = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8), generator=generator, device=device, dtype=torch.float16)
latent_height, latent_width = height // 8, width // 8

# Create the initial hard-edged masks
masks_np = np.zeros((3, latent_height, latent_width), dtype=np.float32)
third_width = latent_width // 3
masks_np[0, :, :third_width] = 1.0
masks_np[1, :, third_width:third_width*2] = 1.0
masks_np[2, :, third_width*2:] = 1.0

# --- THE NEW LOGIC: Blur the masks to create soft overlaps ---
print(f"Blurring masks with sigma={mask_blur_sigma}...")
blurred_masks_np = np.zeros_like(masks_np)
for i in range(masks_np.shape[0]):
    blurred_masks_np[i] = gaussian_filter(masks_np[i], sigma=mask_blur_sigma)

# Normalize the blurred masks so that at any point, the sum of all mask weights is 1.0
# This ensures a smooth, weighted average blend.
mask_sum = np.sum(blurred_masks_np, axis=0, keepdims=True)
normalized_masks_np = blurred_masks_np / (mask_sum + 1e-6) # add epsilon to avoid division by zero

# Convert to PyTorch tensors
masks = torch.from_numpy(normalized_masks_np).to(device, dtype=torch.float16)


# --- 5. THE CUSTOM DENOISING LOOP ---
print("Starting custom denoising loop...")
pipe.scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = pipe.scheduler.timesteps
latents = latents * pipe.scheduler.init_noise_sigma

start_time = time.time()
with torch.no_grad():
    for i, t in enumerate(pipe.progress_bar(timesteps)):
        latent_model_input = pipe.scheduler.scale_model_input(latents, t)
        
        # We no longer need the cropping technique because the soft masks handle blending.
        add_kwargs = {"time_ids": pipe._get_add_time_ids(
            (height, width), (0,0), (height, width), dtype=torch.float16,
            text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim
        ).to(device)}

        # Predict noise for each blended region on the FULL canvas
        add_kwargs["text_embeds"] = pooled_blended_left
        pred_left = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds_blended_left, added_cond_kwargs=add_kwargs).sample
        
        add_kwargs["text_embeds"] = pooled_blended_center
        pred_center = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds_blended_center, added_cond_kwargs=add_kwargs).sample
        
        add_kwargs["text_embeds"] = pooled_blended_right
        pred_right = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds_blended_right, added_cond_kwargs=add_kwargs).sample
        
        # --- The Final Blend using Soft Masks ---
        # This is a weighted average of the noise predictions.
        noise_pred = (pred_left * masks[0] + 
                      pred_center * masks[1] + 
                      pred_right * masks[2])
        
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

end_time = time.time()
print(f"Custom pipeline execution time: {end_time - start_time:.2f} seconds")


# --- 6. DECODE AND SAVE ---
latents = 1 / pipe.vae.config.scaling_factor * latents
image = pipe.vae.decode(latents).sample
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
image = (image * 255).round().astype("uint8")
pil_image = Image.fromarray(image[0])

pil_image.save("regional_prompting_final_output.png")
print("Saved final regional prompting image to regional_prompting_final_output.png")