from diffusers import EulerAncestralDiscreteScheduler
from transformers import CLIPTokenizer

import models
import defaults
from pipeline.compel.src.compel import Compel


def load_pipeline_components():
    """
    Load all the components of the SDXL pipeline.
    """
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        defaults.DEFAULT_BASE_MODEL, subfolder="tokenizer"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        defaults.DEFAULT_BASE_MODEL, subfolder="tokenizer_2"
    )

    text_encoder_1 = models.CLIPTextEncoder(defaults.CLIP_TEXT_ENCODER_1_PATH)
    text_encoder_2 = models.CLIPTextEncoder(defaults.CLIP_TEXT_ENCODER_2_PATH)

    compel_instance = Compel(
        tokenizer=[tokenizer_1, tokenizer_2],
        text_encoder=[text_encoder_1, text_encoder_2],
        returned_embeddings_type=Compel.ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True]
    )

    vae_decoder = models.VAEDecoder(defaults.VAE_DECODER_PATH)
    unet = models.UNet(defaults.UNET_PATH)
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        defaults.DEFAULT_BASE_MODEL, subfolder="scheduler"
    )

    return {
        "compel": compel_instance,
        "vae_decoder": vae_decoder,
        "unet": unet,
        "scheduler": scheduler,
    } 