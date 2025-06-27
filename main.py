import torch
from diffusers import DDIMScheduler
from diffusers.pipelines import StableDiffusionXLPipeline
from pipeline import TensorRTStableDiffusionXLPipeline

if True:

    print("Step 1: Loading PyTorch models from Hugging Face Hub...")
    
    # Load the base SDXL pipeline
    base_pipeline = StableDiffusionXLPipeline.from_pretrained(
        "socks22/sdxl-wai-nsfw-illustriousv14",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    print("Step 2: Creating and building the TensorRT pipeline...")
    
    # Create the TensorRT pipeline
    trt_pipeline = TensorRTStableDiffusionXLPipeline(
        vae=base_pipeline.vae,
        text_encoder=base_pipeline.text_encoder,
        tokenizer=base_pipeline.tokenizer,
        text_encoder_2=base_pipeline.text_encoder_2,
        tokenizer_2=base_pipeline.tokenizer_2,
        unet=base_pipeline.unet,
        scheduler=base_pipeline.scheduler,
        force_engine_rebuild=False,
    )
    
    # This is where the magic happens. The first time this is called, it will build the engines.
    trt_pipeline.to("cuda")
    
    # Free up memory from the original PyTorch pipelines
    del base_pipeline
    #del refiner_pipeline
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Step 3: Running inference with the TensorRT pipeline...")
    
    prompt = "A majestic lion jumping from a big stone at night"
    generator = torch.Generator(device="cuda").manual_seed(42)
    
    # Run inference
    # denoising_end=0.8 means the base model runs for 80% of the steps, refiner for the last 20%.
    image = trt_pipeline(
        prompt=prompt,
        num_inference_steps=30,
        generator=generator
    ).images[0]
    
    # Save the image
    image.save("sdxl_tensorrt_output.png")