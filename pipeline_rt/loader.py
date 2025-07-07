from diffusers import EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTokenizer
import torch
import os
import json

import models
import defaults

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
    
    vae = models.VAEDecoder(defaults.VAE_DECODER_PATH, device)
    #vae = None
    #vae_alt_config_path = os.path.join(os.path.dirname(defaults.VAE_ALT_PATH), "config.json")

    #vae_alt = AutoencoderKL.from_single_file(
    #    defaults.VAE_ALT_PATH,
    #    dtype=torch.float16,
    #    config=vae_alt_config_path
    #).to(device)
    #vae_alt.enable_xformers_memory_efficient_attention()
    #vae_alt.enable_tiling()
    
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
        "vae": vae,
        #"vae_alt": vae_alt,
        "vae_scale_factor": vae_scale_factor,
        "image_processor": image_processor,
    } 