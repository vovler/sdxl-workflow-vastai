# regional_prompting_final_fixed_api_v2.py
import torch
import time
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, AutoencoderTiny
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# --- 1. SETTINGS ---
prompt_left = "masterpiece, best quality, absurdres, cinematic lighting, ultra-detailed, dark city street at night, 1girl, megumin, witch_hat, eyepatch, short_brown_hair, red_dress, black_cape, smug, hands_behind_back"
prompt_center = "masterpiece, best quality, absurdres, cinematic lighting, ultra-detailed, dark city street at night, 1girl, aqua_(konosuba), long_blue_hair, blue_dress, bare_shoulders, hands_up, arms_up, happy, joyful"
prompt_right = "masterpiece, best quality, absurdres, cinematic lighting, ultra-detailed, dark city street at night, 1boy, satou_kazuma, green_tracksuit, short_brown_hair, annoyed, holding_plushie, frog_plushie"

height = 832
width = 1216
num_inference_steps = 8
seed = 45
guidance_scale = 1.0
background_blend_ratio = 0.35
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
    prompt_embeds, _, pooled_prompt_embeds, _  = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False
    )
    return prompt_embeds, pooled_prompt_embeds

prompt_embeds_left, pooled_embeds_left = encode_prompt_cfg_less(prompt_left)
prompt_embeds_center, pooled_embeds_center = encode_prompt_cfg_less(prompt_center)
prompt_embeds_right, pooled_embeds_right = encode_prompt_cfg_less(prompt_right)


# --- 4. PREPARE LATENTS AND SOFT MASKS (Identical) ---
generator = torch.Generator(device).manual_seed(seed)
latents = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8), generator=generator, device=device, dtype=torch.float16)
latent_height, latent_width = height // 8, width // 8

masks_np = np.zeros((3, latent_height, latent_width), dtype=np.float32)
third_width = latent_width // 3
masks_np[0, :, :third_width] = 1.0
masks_np[1, :, third_width:third_width*2] = 1.0
masks_np[2, :, third_width*2:] = 1.0

print(f"Blurring masks with sigma={mask_blur_sigma}...")
blurred_masks_np = np.zeros_like(masks_np)
for i in range(masks_np.shape[0]):
    blurred_masks_np[i] = gaussian_filter(masks_np[i], sigma=mask_blur_sigma)
mask_sum = np.sum(blurred_masks_np, axis=0, keepdims=True)
normalized_masks_np = blurred_masks_np / (mask_sum + 1e-6)
masks = torch.from_numpy(normalized_masks_np).to(device, dtype=torch.float16)


# --- Calculate shifts for regional prompts ---
latent_center_col = latent_width // 2

# Center of left mask
left_mask_center_col = (latent_width // 3) // 2
shift_left = left_mask_center_col - latent_center_col

# Center of right mask
right_mask_start_col = (latent_width // 3) * 2
right_mask_width = latent_width - right_mask_start_col
right_mask_center_col = right_mask_start_col + (right_mask_width // 2)
shift_right = right_mask_center_col - latent_center_col

print(f"Shifting left by {shift_left} and right by {shift_right}")


# --- 5. THE CUSTOM DENOISING LOOP (Identical) ---
print("Starting custom denoising loop...")
pipe.scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = pipe.scheduler.timesteps
latents = latents * pipe.scheduler.init_noise_sigma

start_time = time.time()
with torch.no_grad():
    for i, t in enumerate(pipe.progress_bar(timesteps)):
        latent_model_input = pipe.scheduler.scale_model_input(latents, t)
        
        add_kwargs = {"time_ids": pipe._get_add_time_ids(
            (height, width), (0,0), (height, width), dtype=torch.float16,
            text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim
        ).to(device)}

        add_kwargs["text_embeds"] = pooled_embeds_left
        pred_left = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds_left, added_cond_kwargs=add_kwargs).sample
        
        add_kwargs["text_embeds"] = pooled_embeds_center
        pred_center = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds_center, added_cond_kwargs=add_kwargs).sample
        
        add_kwargs["text_embeds"] = pooled_embeds_right
        pred_right = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds_right, added_cond_kwargs=add_kwargs).sample
        
        pred_left_shifted = torch.roll(pred_left, shifts=shift_left, dims=3)
        pred_right_shifted = torch.roll(pred_right, shifts=shift_right, dims=3)

        noise_pred = (pred_left_shifted * masks[0] +
                      pred_center * masks[1] +
                      pred_right_shifted * masks[2])
        
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

pil_image.save("regional_prompting_final_output.png")
print("Saved final regional prompting image to regional_prompting_final_output.png")