import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import io
from typing import Optional, Dict, Any, Tuple

# Global pipeline variable
pipe = None

def initialize_pipeline():
    """Initialize the SDXL pipeline with DMD2 LoRA"""
    global pipe
    if pipe is not None:
        return pipe
        
    print("Loading SDXL pipeline...", flush=True)
   
        
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "socks22/sdxl-wai-nsfw-illustriousv14",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    print("SDXL pipeline loaded from pretrained", flush=True)
    
    # Load and apply DMD2 LoRA
    print("Loading DMD2 LoRA...", flush=True)
    pipe.load_lora_weights("tianweiy/DMD2", weight_name="dmd2_sdxl_4step_lora_fp16.safetensors", torch_dtype=torch.float16, variant="fp16")
    pipe.fuse_lora(lora_scale=0.8)
    
    # Set up Euler Ancestral scheduler
    print("Setting up Euler Ancestral scheduler...", flush=True)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    print("Moving pipeline to CUDA...", flush=True)
    pipe.to("cuda")

    # Optional memory optimizations
    print("Enabling memory optimizations...", flush=True)
    #pipe.enable_pytorch_attention()
    pipe.enable_xformers_memory_efficient_attention()  # efficient attention
    #pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)  # speed-up (requires torch>=2.0)
    
    print("SDXL pipeline loaded successfully with Euler Ancestral scheduler and DMD2 LoRA!", flush=True)
    return pipe

def generate_image(
    prompt: str
) -> Tuple[bytes, Any]:
    """
    Generate an image using SDXL pipeline with DMD2 LoRA
    
    Returns: 
        Tuple of (image_bytes, pil_image) 
    """
    global pipe
    
    # Initialize pipeline if not already done
    if pipe is None:
        initialize_pipeline()
    
    print(f"Generating image with prompt: {prompt}", flush=True)
    
    prompt = "masterpiece,best quality,amazing quality, scenery, 1girl, " + prompt + ", looking_at_viewer, Thigh Up, background visible, high detail"
    negative_prompt = "worst quality, bad quality, very displeasing, displeasing, bad anatomy, artistic error, anatomical nonsense, lowres, bad hands, watermark,signature, artist name, variations, old, oldest, extra hands, multiple_penises, deformed, mutated, ugly, disfigured, missing fingers, cropped, very displeasing, bad anatomy, conjoined, bad ai-generated, multiple_girls, multiple_boys, multiple_views"
    # Define seed
    seed = 12345
    generator = torch.Generator(device="cuda").manual_seed(seed)

    width = 1152
    height = 768
    num_steps = 8 # DMD2 is optimized for 4 steps
    guidance = 1.2

    out = pipe(
        prompt=prompt,
        prompt_2=prompt,  # same prompt for both encoders; you can customize prompt_2 differently
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        output_type="pil",
        generator=generator
    )
    
    img = out.images[0]
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    image_bytes = img_buffer.getvalue()
    
    # Debug: Check the generated image bytes
    print(f"Generated image bytes type: {type(image_bytes)}", flush=True)
    print(f"Generated image bytes size: {len(image_bytes)}", flush=True)
    
    return image_bytes, img
