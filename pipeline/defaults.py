import os

# Get the absolute path of the directory where this file is located
_this_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the project root
_project_root = os.path.realpath(os.path.join(_this_dir, ".."))

# ONNX Models
DEFAULT_BASE_MODEL = "socks22/sdxl-wai-nsfw-illustriousv14"

# VAE
VAE_DECODER_PATH = os.path.join(_project_root, "vae_decoder.onnx")

# UNet
UNET_PATH = os.path.join(_project_root, "unet.patched.onnx")

# Text Encoders
CLIP_TEXT_ENCODER_1_PATH = os.path.join(_project_root, "clip_l.onnx")
CLIP_TEXT_ENCODER_2_PATH = os.path.join(_project_root, "clip_g.onnx") 