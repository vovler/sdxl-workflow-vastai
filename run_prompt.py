import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import io
from typing import Optional, Dict, Any, Tuple

# Global pipeline variable
pipe = None

def initialize_pipeline():
    """Initialize the SDXL pipeline with DMD2 LoRA"""
    global pipe
    if pipe is not None:
        return pipe
        
    print("Loading SDXL pipeline...")
   
        
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "John6666/ntr-mix-illustrious-xl-noob-xl-xiii-sdxl",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    # Load and apply DMD2 LoRA
    print("Loading DMD2 LoRA...")
    try:
        pipe.load_lora_weights("tianweiy/DMD2", weight_name="dmd2_sdxl_4step_lora_fp16.safetensors")
        pipe.fuse_lora(lora_scale=0.8)
        print("DMD2 LoRA loaded and applied successfully!")
    except Exception as e:
        print(f"Warning: Failed to load DMD2 LoRA: {e}")
        print("Continuing without LoRA...")
    
    # Set up DPM++ 2M SGM Uniform scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=False,
        timestep_spacing="trailing"
    )
    
    pipe.to("cuda")

    # Optional memory optimizations
    pipe.enable_xformers_memory_efficient_attention()  # efficient attention
    #pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)  # speed-up (requires torch>=2.0)
    
    print("SDXL pipeline loaded successfully with DPM++ 2M SGM Uniform scheduler and DMD2 LoRA!")
    return pipe

def generate_image(
    prompt: str = "A majestic anime knight standing on a cliff, cinematic lighting, ultra-detailed",
    negative_prompt: str = "lowres, bad anatomy, watermark",
    width: int = 1024,
    height: int = 1024,
    num_steps: int = 4,  # DMD2 is optimized for 4 steps
    guidance: float = 5.0,
    save_path: Optional[str] = None
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
    
    print(f"Generating image with prompt: {prompt}")
    
    prompt = "masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, 1girl, " + prompt + ", looking_at_viewer, Thigh Up, background visible, high detail, volumetric lighting, highly detailed, high quality"
    negative_prompt = "worst quality, bad quality, very displeasing, displeasing, bad anatomy, \
	artistic error, anatomical nonsense, lowres, bad hands, signature, artist name, variations, \
	old, oldest, extra hands, multiple_penises, deformed, mutated, ugly, disfigured, long body, \
	missing fingers, cropped, very displeasing, bad anatomy, conjoined, bad ai-generated, \
	multiple_girls, multiple_boys, multiple_views"
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
