# debug_native.py
import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

# Ensure you use the exact same settings as your pipeline
prompt = "masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, night, 1girl, aqua_(konosuba), smiling, looking at viewer, at the park, night"
height = 1024
width = 1024
num_inference_steps = 8
seed = 42
guidance_scale = 1.0 # This is the key to matching your CFG-less setup

device = "cuda"

# Load native pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "socks22/sdxl-wai-nsfw-illustriousv14", # Your base model
    torch_dtype=torch.float16,
    use_safetensors=True,
).to(device)

# Set the exact same scheduler
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=False
)

# Prepare latents
generator = torch.Generator(device=device).manual_seed(seed)
latents = pipe.prepare_latents(1, pipe.unet.config.in_channels, height, width, torch.float16, device, generator)

# Run the pipeline, but intercept the call to the UNet
# We will monkey-patch the unet forward pass to capture the inputs
original_unet_forward = pipe.unet.forward

def new_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    added_cond_kwargs = kwargs.get("added_cond_kwargs", {})
    text_embeds = added_cond_kwargs.get("text_embeds")
    time_ids = added_cond_kwargs.get("time_ids")

    print("\n" + "="*20 + f" NATIVE DIFFUSERS - CAPTURING TENSORS (Timestep: {timestep.item()}) " + "="*20)
    # 1. Latent Model Input
    print(f"LATENT_MODEL_INPUT | Shape: {sample.shape} | Dtype: {sample.dtype}")
    print(f"LATENT_MODEL_INPUT | Mean: {sample.mean():.6f} | Std: {sample.std():.6f} | Sum: {sample.sum():.6f}")

    # 2. Prompt Embeds (needs to be split for comparison)
    uncond_embeds, cond_embeds = encoder_hidden_states.chunk(2)
    print(f"PROMPT_EMBEDS (cond) | Shape: {cond_embeds.shape} | Dtype: {cond_embeds.dtype}")
    print(f"PROMPT_EMBEDS (cond) | Mean: {cond_embeds.mean():.6f} | Std: {cond_embeds.std():.6f} | Sum: {cond_embeds.sum():.6f}")

    # 3. Pooled Prompt Embeds (needs to be split)
    uncond_pooled, cond_pooled = text_embeds.chunk(2)
    print(f"POOLED_EMBEDS (cond) | Shape: {cond_pooled.shape} | Dtype: {cond_pooled.dtype}")
    print(f"POOLED_EMBEDS (cond) | Mean: {cond_pooled.mean():.6f} | Std: {cond_pooled.std():.6f} | Sum: {cond_pooled.sum():.6f}")
    
    # 4. Time IDs (needs to be split)
    uncond_time, cond_time = time_ids.chunk(2)
    print(f"TIME_IDS (cond) | Shape: {cond_time.shape} | Dtype: {cond_time.dtype}")
    print(f"TIME_IDS (cond) | Values: {cond_time.flatten().tolist()}")

    # 5. The Timestep
    print(f"TIMESTEP | Value: {timestep.item()} | Dtype of tensor: {timestep.dtype}")
    print("="*78 + "\n")
    
    # Call the original UNet
    noise_pred = original_unet_forward(sample, timestep, encoder_hidden_states, **kwargs)
    
    print("\n" + "="*20 + f" NATIVE DIFFUSERS - CAPTURING UNET OUTPUT " + "="*20)
    # Perform the guidance manually to get the final noise_pred that the scheduler sees
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    final_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    print(f"NOISE_PRED (final) | Shape: {final_noise_pred.shape} | Dtype: {final_noise_pred.dtype}")
    print(f"NOISE_PRED (final) | Mean: {final_noise_pred.mean():.6f} | Std: {final_noise_pred.std():.6f} | Sum: {final_noise_pred.sum():.6f}")
    print("="*68 + "\n")
    
    # We only need to run one step, so we can raise an exception to stop the pipeline
    raise ValueError("Stopping after one step for debugging.")

# Patch the UNet
pipe.unet.forward = new_unet_forward

try:
    # Run the pipeline. It will stop after the first UNet call.
    _ = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale, # Set to 1.0 to mimic your pipeline
        generator=generator,
        latents=latents,
    )
except ValueError as e:
    print(str(e))

print("Native debug script finished.") 