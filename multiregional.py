from pipeline_seg import StableDiffusionXLSEGPipeline
import torch


pipe = StableDiffusionXLSEGPipeline.from_pretrained(
    "John6666/wai-nsfw-illustrious-v130-sdxl",
    torch_dtype=torch.float16
)

device = "cuda"
pipe = pipe.to(device)
prompts = ["masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, 2girls, ON THE LEFT: aqua_(konosuba), smiling, ON THE RIGHT: megumin, sad, upper_body"]
seed = 1

generator = torch.Generator(device="cuda").manual_seed(seed)
output = pipe(
    prompts,
    num_inference_steps=25,
    guidance_scale=1.0,
    seg_scale=2.0,
    seg_blur_sigma=25.0,
    seg_applied_layers=['mid'],
    generator=generator,
)

output.images[0].save("output.png")