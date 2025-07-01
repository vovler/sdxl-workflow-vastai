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
    device = torch.device("cuda:0")
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        defaults.DEFAULT_BASE_MODEL, subfolder="tokenizer"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        defaults.DEFAULT_BASE_MODEL, subfolder="tokenizer_2"
    )

    # ONNX Models
    onnx_text_encoder_1 = models.CLIPTextEncoder(defaults.CLIP_TEXT_ENCODER_1_PATH, device, name="CLIP-L")
    onnx_text_encoder_2 = models.CLIPTextEncoder(defaults.CLIP_TEXT_ENCODER_2_PATH, device, name="CLIP-G")

    compel_onnx = Compel(
        tokenizer=[tokenizer_1, tokenizer_2],
        text_encoder=[onnx_text_encoder_1, onnx_text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        device=device
    )

    vae_decoder = models.VAEDecoder(defaults.VAE_DECODER_PATH, device)
    
    vae_config_path = os.path.join(os.path.dirname(defaults.VAE_DECODER_PATH), "config.json")
    with open(vae_config_path, "r") as f:
        vae_config = json.load(f)
    vae_scaling_factor = vae_config["scaling_factor"]
    
    #vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16).to(device)
    
    #vae_scale_factor = 2 ** (len(vae.config.decoder_block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scaling_factor)

    unet = models.UNet(defaults.UNET_PATH, device)
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        defaults.DEFAULT_BASE_MODEL, subfolder="scheduler"
    )

    return {
        "compel_onnx": compel_onnx,
        "tokenizer_1": tokenizer_1,
        "tokenizer_2": tokenizer_2,
        "text_encoder_l": onnx_text_encoder_1,
        "text_encoder_g": onnx_text_encoder_2,
        "vae_decoder": vae_decoder,
        "vae": vae,
        "image_processor": image_processor,
        "unet": unet,
        "scheduler": scheduler,
        "vae_scale_factor": vae_scale_factor,
        # "vae_scaling_factor": vae_scaling_factor,
    } 