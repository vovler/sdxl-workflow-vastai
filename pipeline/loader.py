from diffusers import EulerAncestralDiscreteScheduler, AutoencoderTiny
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import torch

import models
import defaults
from compel import Compel, ReturnedEmbeddingsType
import json
import os


def load_pipeline_components():
    """
    Load all the components of the SDXL pipeline.
    """
    device = torch.device("cpu")
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        defaults.DEFAULT_BASE_MODEL, subfolder="tokenizer"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        defaults.DEFAULT_BASE_MODEL, subfolder="tokenizer_2"
    )

    # ONNX Models
    onnx_text_encoder_1 = models.CLIPTextEncoder(defaults.CLIP_TEXT_ENCODER_1_PATH, device, name="CLIP-L")
    onnx_text_encoder_2 = models.CLIPTextEncoder(defaults.CLIP_TEXT_ENCODER_2_PATH, device, name="CLIP-G")
    
    onnx_vae = models.VAEDecoder(defaults.VAE_DECODER_PATH, device)
    vae = AutoencoderTiny.from_pretrained("cqyan/hybrid-sd-tinyvae-xl", torch_dtype=torch.float16).to(device)
    
    vae_scale_factor = 2 ** (len(vae.config.decoder_block_out_channels) - 1)
    image_processor = VaeImageProcessor(1)

    unet = models.UNet(defaults.UNET_PATH, device)
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        defaults.DEFAULT_BASE_MODEL, subfolder="scheduler"
    )

    return {
        "tokenizer_1": tokenizer_1,
        "tokenizer_2": tokenizer_2,
        "text_encoder_l": onnx_text_encoder_1,
        "text_encoder_g": onnx_text_encoder_2,
        "scheduler": scheduler,
        "unet": unet,
        "onnx_vae": onnx_vae,
        "vae": vae,
        "vae_scale_factor": vae_scale_factor,
        "image_processor": image_processor,
    } 