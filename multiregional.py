from pipeline_seg import StableDiffusionXLSEGPipeline
import torch


pipe = StableDiffusionXLSEGPipeline.from_pretrained(
    "John6666/wai-nsfw-illustrious-v130-sdxl",
    torch_dtype=torch.float16
)

device = "cuda"
pipe = pipe.to(device)
prompts = ["masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, 2girls, aqua_(konosuba) on the left, smiling, 1girl, megumin, sad"]
seed = 10

generator = torch.Generator(device="cuda").manual_seed(seed)
output = pipe(
    prompts,
    num_inference_steps=25,
    guidance_scale=1.0,
    seg_scale=3.0,
    seg_blur_sigma=100.0,
    seg_applied_layers=['mid'],
    generator=generator,
)

output.save_image("output.png")