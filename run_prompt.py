import torch
from diffusers import StableDiffusionXLPipeline
import io
from typing import Optional, Dict, Any, Tuple

# Global pipeline variable
pipe = None

def initialize_pipeline():
    """Initialize the SDXL pipeline"""
    global pipe
    if pipe is not None:
        return pipe
        
    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "John6666/ntr-mix-illustrious-xl-noob-xl-xiii-sdxl",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe.to("cuda")

    # Optional memory optimizations
    pipe.enable_xformers_memory_efficient_attention()  # efficient attention
    #pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)  # speed-up (requires torch>=2.0)
    
    print("SDXL pipeline loaded successfully!")
    return pipe

def generate_image(
    prompt: str = "A majestic anime knight standing on a cliff, cinematic lighting, ultra-detailed",
    negative_prompt: str = "lowres, bad anatomy, watermark",
    width: int = 1024,
    height: int = 1024,
    num_steps: int = 30,
    guidance: float = 5.0,
    save_path: Optional[str] = None
) -> Tuple[bytes, Any]:
    """
    Generate an image using SDXL pipeline
    
    Returns:
        Tuple of (image_bytes, pil_image)
    """
    global pipe
    
    # Initialize pipeline if not already done
    if pipe is None:
        initialize_pipeline()
    
    print(f"Generating image with prompt: {prompt}")
    
    # Generate image
    out = pipe(
        prompt=prompt,
        prompt_2=prompt,  # same prompt for both encoders; you can customize prompt_2 differently
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        output_type="pil"
    )
    
    img = out.images[0]
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    image_bytes = img_buffer.getvalue()
    
    # Save to file if path provided
    if save_path:
        img.save(save_path)
        print(f"Image saved as {save_path}")
    
    return image_bytes, img
