from diffusers import EulerAncestralDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTokenizer
import torch

from . import models
from . import defaults

def load_pipeline_components():
    """
    Load all the components of the SDXL TensorRT pipeline.
    """
    device = torch.device("cuda:0")
    
    # Tokenizers
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        defaults.DEFAULT_BASE_MODEL, subfolder="tokenizer"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        defaults.DEFAULT_BASE_MODEL, subfolder="tokenizer_2"
    )

    # TensorRT Engines
    text_encoder_l = models.CLIPTextEncoder(defaults.CLIP_TEXT_ENCODER_1_PATH, device, name="CLIP-L")
    text_encoder_g = models.CLIPTextEncoder(defaults.CLIP_TEXT_ENCODER_2_PATH, device, name="CLIP-G")
    
    trt_vae = models.VAEDecoder(defaults.VAE_DECODER_PATH, device)
    
    unet = models.UNet(defaults.UNET_PATH, device)
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        defaults.DEFAULT_BASE_MODEL, subfolder="scheduler"
    )
    
    vae_scale_factor = 8
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    return {
        "tokenizer_1": tokenizer_1,
        "tokenizer_2": tokenizer_2,
        "text_encoder_l": text_encoder_l,
        "text_encoder_g": text_encoder_g,
        "scheduler": scheduler,
        "unet": unet,
        "trt_vae": trt_vae,
        "vae_scale_factor": vae_scale_factor,
        "image_processor": image_processor,
    } 