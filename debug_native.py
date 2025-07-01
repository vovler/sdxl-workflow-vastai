pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config
)

print("\n" + "="*20 + " NATIVE SCHEDULER CONFIG " + "="*20)
for key, value in pipe.scheduler.config.items():
    print(f"{key}: {value}")
print("="*67 + "\n")

# Prepare latents
generator=torch.Generator("cpu").manual_seed(0x7A35D)
#latents = pipe.prepare_latents(1, pipe.unet.config.in_channels, height, width, torch.float16, device, generator) 