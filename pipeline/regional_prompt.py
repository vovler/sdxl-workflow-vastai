# regional_prompting_blended_conditioning.py
import torch
import time
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, AutoencoderTiny
import numpy as np
from PIL import Image

# --- 1. SETTINGS ---
# Prompts are now clearly separated for their roles.
prompt_background = "masterpiece, best quality, absurdres, cinematic lighting, ultra-detailed, dark city street at night, heavy rain, puddles on the ground with glowing reflections of neon signs, wet asphalt"
prompt_left = "1girl, megumin, witch_hat, eyepatch, short_brown_hair, red_dress, black_cape, smug, hands_behind_back"
prompt_center = "1girl, aqua_(konosuba), long_blue_hair, blue_dress, bare_shoulders, hands_up, arms_up, happy, joyful"
prompt_right = "1boy, satou_kazuma, green_tracksuit, short_brown_hair, annoyed, holding_plushie, frog_plushie"

height = 832
width = 1216
num_inference_steps = 12 # A few more steps helps with complex blending
seed = 45
guidance_scale = 1.0

# This is the new, critical tuning knob!
# It controls how much influence the background has on the characters.
# 0.3 means the final prompt is 30% background and 70% character.
background_blend_ratio = 0.3

device = "cuda:0"
base_model_id = "socks22/sdxl-wai-nsfw-illustriousv14"


# --- 2. PIPELINE SETUP (Identical) ---
print("Loading pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, use_safetensors=True)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
pipe.enable_xformers_memory_efficient_attention()


# --- 3. ENCODE & BLEND PROMPTS ---
print("Encoding and blending prompts...")
def encode_prompt_cfg_less(prompt):
    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False
    )
    return prompt_embeds, pooled_prompt_embeds

# Encode all base prompts
prompt_embeds_bg, pooled_embeds_bg = encode_prompt_cfg_less(prompt_background)
prompt_embeds_left, pooled_embeds_left = encode_prompt_cfg_less(prompt_left)
prompt_embeds_center, pooled_embeds_center = encode_prompt_cfg_less(prompt_center)
prompt_embeds_right, pooled_embeds_right = encode_prompt_cfg_less(prompt_right)

# --- THE NEW LOGIC: Create blended conditionings ---
def blend_conditionings(cond1, pooled1, cond2, pooled2, ratio):
    """Performs a weighted average (lerp) on the embeddings."""
    blended_cond = (cond1 * ratio) + (cond2 * (1.0 - ratio))
    blended_pooled = (pooled1 * ratio) + (pooled2 * (1.0 - ratio))
    return blended_cond, blended_pooled

# Create a unique, blended prompt for each region
prompt_embeds_blended_left, pooled_blended_left = blend_conditionings(
    prompt_embeds_bg, pooled_embeds_bg, prompt_embeds_left, pooled_embeds_left, background_blend_ratio
)
prompt_embeds_blended_center, pooled_blended_center = blend_conditionings(
    prompt_embeds_bg, pooled_embeds_bg, prompt_embeds_center, pooled_embeds_center, background_blend_ratio
)
prompt_embeds_blended_right, pooled_blended_right = blend_conditionings(
    prompt_embeds_bg, pooled_embeds_bg, prompt_embeds_right, pooled_embeds_right, background_blend_ratio
)
print("Conditionings blended successfully.")


# --- 4. PREPARE LATENTS AND REGIONS (Identical) ---
generator = torch.Generator(device).manual_seed(seed)
latents = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8), generator=generator, device=device, dtype=torch.float16)
latent_height, latent_width = height // 8, width // 8
third_width = latent_width // 3
bbox_left = (0, third_width)
bbox_center = (third_width, third_width * 2)
bbox_right = (third_width * 2, latent_width)


# --- 5. THE CUSTOM DENOISING LOOP (Using Blended Prompts) ---
print("Starting custom denoising loop with blended conditionings...")
pipe.scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = pipe.scheduler.timesteps
latents = latents * pipe.scheduler.init_noise_sigma

start_time = time.time()
with torch.no_grad():
    # Initialize a blank canvas for our noise predictions
    noise_pred = torch.zeros_like(latents)

    for i, t in enumerate(pipe.progress_bar(timesteps)):
        latent_model_input = pipe.scheduler.scale_model_input(latents, t)

        def process_region(bbox, regional_latents, blended_prompt_embeds, blended_pooled_embeds):
            x_start, x_end = bbox
            cropped_latents = regional_latents[:, :, :, x_start:x_end]
            
            add_kwargs = {"time_ids": pipe._get_add_time_ids(
                (height, width), (0,0), (height, width), dtype=torch.float16,
                text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim
            ).to(device)}
            add_kwargs["text_embeds"] = blended_pooled_embeds
            
            # Predict noise on the cropped latent using the BLENDED prompt
            regional_pred = pipe.unet(
                cropped_latents, t,
                encoder_hidden_states=blended_prompt_embeds,
                added_cond_kwargs=add_kwargs
            ).sample
            
            # Paste the regional prediction into its correct place on the canvas
            noise_pred[:, :, :, x_start:x_end] = regional_pred

        # Process each region using its unique blended prompt
        process_region(bbox_left, latent_model_input, prompt_embeds_blended_left, pooled_blended_left)
        process_region(bbox_center, latent_model_input, prompt_embeds_blended_center, pooled_blended_center)
        process_region(bbox_right, latent_model_input, prompt_embeds_blended_right, pooled_blended_right)
        
        # Step with the fully composited noise prediction
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

end_time = time.time()
print(f"Custom pipeline execution time: {end_time - start_time:.2f} seconds")


# --- 6. DECODE AND SAVE (Identical) ---
latents = 1 / pipe.vae.config.scaling_factor * latents
image = pipe.vae.decode(latents).sample
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
image = (image * 255).round().astype("uint8")
pil_image = Image.fromarray(image[0])

pil_image.save("regional_prompting_blended_output.png")
print("Saved blended regional prompting image to regional_prompting_blended_output.png")