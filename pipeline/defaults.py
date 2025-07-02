import os

# Get the absolute path of the directory where this file is located
_this_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the project root
_project_root = os.path.realpath(os.path.join(_this_dir, ".."))

# ONNX Models
DEFAULT_BASE_MODEL = "socks22/sdxl-wai-nsfw-illustriousv14"
ONNX_MODELS_DIR = "/workflow/wai_dmd2_onnx"

# VAE
VAE_DECODER_PATH = os.path.join(ONNX_MODELS_DIR, "tiny_vae_decoder", "model_nonopt.onnx")

# UNet
UNET_PATH = os.path.join(ONNX_MODELS_DIR, "unet", "model_opt.onnx")

# Text Encoders
CLIP_TEXT_ENCODER_1_PATH = os.path.join(ONNX_MODELS_DIR, "text_encoder", "model.onnx")
CLIP_TEXT_ENCODER_2_PATH = os.path.join(ONNX_MODELS_DIR, "text_encoder_2", "model.onnx") 