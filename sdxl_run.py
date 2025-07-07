import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import os
from tqdm import tqdm

# Set up the pipeline
pipe = StableDiffusionXLPipeline.from_single_file(
    "/lab/waiNSFWIllustrious_v140.safetensors",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.load_lora_weights("/lab/dmd2_sdxl_4step_lora_fp16.safetensors")
pipe.fuse_lora(lora_scale=1.0)
pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="linspace", num_train_timesteps=1000
)

print(pipe)
print(pipe.scheduler)
# Define the prompt and parameters for the pipeline
prompt = "masterpiece,best quality,amazing quality, general, 1girl, aqua_(konosuba), on a swing, looking at viewer, volumetric_lighting, park, night, shiny clothes, shiny skin, detailed_background"
guidance_scale = 0
num_inference_steps = 4
seed = 1020094661
generator = torch.Generator(device="cuda").manual_seed(seed)
height = 832
width = 1216

# Manually unpack the pipeline to allow for custom scheduler logic
with torch.no_grad():
    # 1. Encode input prompt
    prompt_embeds, _, _, _ = pipe.encode_prompt(prompt)

    # 2. Prepare timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=pipe.device)
    timesteps = pipe.scheduler.timesteps
    print(f"Generated timesteps: {timesteps}")

    # 3. Prepare latent variables
    num_channels_latents = pipe.unet.config.in_channels
    latents = pipe.prepare_latents(
        1,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        pipe.device,
        generator,
    )

    # 4. Denoising loop
    for i, t in enumerate(tqdm(timesteps)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = latents

        # predict the noise residual
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # 5. Post-processing
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]


# Save the generated image
base_name = "sdxl_output"
extension = ".png"
output_filename = f"{base_name}{extension}"
i = 1
while os.path.exists(output_filename):
    output_filename = f"{base_name}_{i}{extension}"
    i += 1
image.save(output_filename)
print(f"Image saved as {output_filename}")
