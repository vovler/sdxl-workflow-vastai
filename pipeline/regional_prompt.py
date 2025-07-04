# regional_prompting_distilled.py
import torch
import time
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, AutoencoderTiny
import numpy as np
from PIL import Image

# --- 1. SETTINGS FOR A DISTILLED MODEL ---
# The logic remains the same: a global prompt for style and local prompts for characters.

global_prompt = "masterpiece, best quality, absurdres, cinematic lighting, ultra-detailed, dancing in the heavy rain at night on a city street, puddles on the ground with reflections, wet clothes, (group of friends:1.1)"

prompt_left = "1girl, megumin, witch_hat, eyepatch, short_brown_hair, red_dress, black_cape, smug, hands_behind_back"
prompt_center = "1girl, aqua_(konosuba), long_blue_hair, blue_dress, bare_shoulders, hands_up, arms_up, happy, joyful"
prompt_right = "1boy, satou_kazuma, green_tracksuit, short_brown_hair, annoyed, holding_plushie, frog_plushie"

# NOTE: Negative prompt is no longer used.
# NOTE: Guidance scale must be 1.0 (or even 0.0 for some Turbo models) for CFG-less generation.
# NOTE: Inference steps are very low, as required by distilled models.
height = 832
width = 1216
num_inference_steps = 8 # Drastically reduced for a distilled model
seed = 45
guidance_scale = 1.0 # Disables CFG

device = "cuda:0"
# IMPORTANT: You would point this to your actual distilled model checkpoint.
# Using the original as a placeholder.
base_model_id = "socks22/sdxl-wai-nsfw-illustriousv14"


# --- 2. PIPELINE SETUP ---
print("Loading pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, use_safetensors=True)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
# The scheduler choice is still important. Euler A is a good default.
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)


# --- 3. ENCODE ALL PROMPTS (SIMPLIFIED) ---
print("Encoding all prompts (CFG-less)...")

# This encoding function is now much simpler as it doesn't need to handle CFG.
def encode_prompt_cfg_less(prompt):
    # We call the internal method but tell it not to handle CFG.
    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False # The key change is here
    )
    return prompt_embeds, pooled_prompt_embeds

prompt_embeds_global, pooled_prompt_embeds_global = encode_prompt_cfg_less(global_prompt)
prompt_embeds_left, pooled_prompt_embeds_left = encode_prompt_cfg_less(prompt_left)
prompt_embeds_center, pooled_prompt_embeds_center = encode_prompt_cfg_less(prompt_center)
prompt_embeds_right, pooled_prompt_embeds_right = encode_prompt_cfg_less(prompt_right)

print("Prompts encoded.")


# --- 4. PREPARE LATENTS AND MASKS ---
# This part remains identical to the previous script.
generator = torch.Generator(device).manual_seed(seed)
latents = torch.randn(
    (1, pipe.unet.config.in_channels, height // 8, width // 8),
    generator=generator,
    device=device,
    dtype=torch.float16
)

latent_height, latent_width = height // 8, width // 8
mask = torch.zeros((1, 1, latent_height, latent_width), device=device, dtype=torch.float16)

third_width = latent_width // 3
mask_left = mask.clone()
mask_left[:, :, :, :third_width] = 1.0
mask_center = mask.clone()
mask_center[:, :, :, third_width : third_width*2] = 1.0
mask_right = mask.clone()
mask_right[:, :, :, third_width*2:] = 1.0

print(f"Created 3 masks for latent space of size {latent_height}x{latent_width}")


# --- 5. THE CUSTOM DENOISING LOOP (CFG-LESS LATENT COUPLING) ---
print("Starting custom CFG-less denoising loop...")
# For distilled models, it's crucial to scale the timesteps correctly.
# SD-Turbo/DMD2 models often need a specific timestep range. Here we assume the scheduler handles it.
pipe.scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = pipe.scheduler.timesteps

# Scale the initial noise by the scheduler's initial sigma. This is critical for Turbo models.
latents = latents * pipe.scheduler.init_noise_sigma

start_time = time.time()

# Main loop is now simpler
for i, t in enumerate(pipe.progress_bar(timesteps)):
    # --- A. CONDITIONAL PASSES (Global + Each Region) ---
    # We no longer need to duplicate the latent or run an unconditional pass.
    
    # Global prompt prediction
    pred_global = pipe.unet(
        latents, t,
        encoder_hidden_states=prompt_embeds_global,
        added_cond_kwargs={"text_embeds": pooled_prompt_embeds_global, "time_ids": pipe._get_add_time_ids(
            (height, width), (0,0), (height, width), dtype=torch.float16, device=device
        )}
    ).sample
    
    # Left region prediction
    pred_left = pipe.unet(
        latents, t,
        encoder_hidden_states=prompt_embeds_left,
        added_cond_kwargs={"text_embeds": pooled_prompt_embeds_left, "time_ids": pipe._get_add_time_ids(
            (height, width), (0,0), (height, width), dtype=torch.float16, device=device
        )}
    ).sample

    # Center region prediction
    pred_center = pipe.unet(
        latents, t,
        encoder_hidden_states=prompt_embeds_center,
        added_cond_kwargs={"text_embeds": pooled_prompt_embeds_center, "time_ids": pipe._get_add_time_ids(
            (height, width), (0,0), (height, width), dtype=torch.float16, device=device
        )}
    ).sample

    # Right region prediction
    pred_right = pipe.unet(
        latents, t,
        encoder_hidden_states=prompt_embeds_right,
        added_cond_kwargs={"text_embeds": pooled_prompt_embeds_right, "time_ids": pipe._get_add_time_ids(
            (height, width), (0,0), (height, width), dtype=torch.float16, device=device
        )}
    ).sample
    
    # --- B. BLENDING THE PREDICTIONS (LATENT COUPLING) ---
    # This logic remains the same.
    noise_pred = pred_global
    noise_pred = (noise_pred * (1.0 - mask_left)) + (pred_left * mask_left)
    noise_pred = (noise_pred * (1.0 - mask_center)) + (pred_center * mask_center)
    noise_pred = (noise_pred * (1.0 - mask_right)) + (pred_right * mask_right)
    
    # --- C. STEP ---
    # We directly step with the blended prediction. No CFG calculation.
    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample


end_time = time.time()
print(f"Custom pipeline execution time: {end_time - start_time:.2f} seconds")

# --- 6. DECODE AND SAVE ---
# This part is identical.
latents = 1 / pipe.vae.config.scaling_factor * latents
image = pipe.vae.decode(latents).sample
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
image = (image * 255).round().astype("uint8")
pil_image = Image.fromarray(image[0])

pil_image.save("regional_prompting_distilled_output.png")
print("Saved distilled regional prompting image to regional_prompting_distilled_output.png")