import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import os

# Set up the pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "/lab/model",
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing", num_train_timesteps=1000
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

pipe.scheduler.set_timesteps(num_inference_steps, device=pipe.device)
timesteps = pipe.scheduler.timesteps
print(f"Generated timesteps: {timesteps}")

# Generate the image
with torch.no_grad():
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    ).images[0]

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
