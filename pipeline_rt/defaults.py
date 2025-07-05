import os

# Get the absolute path of the directory where this file is located
_this_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the project root
_project_root = os.path.realpath(os.path.join(_this_dir, ".."))

# Model Paths
DEFAULT_BASE_MODEL = "socks22/sdxl-wai-nsfw-illustriousv14"
TENSORRT_ENGINES_DIR = "/workflow/wai_dmd2_onnx"

# VAE
VAE_DECODER_PATH = os.path.join(TENSORRT_ENGINES_DIR, "vae_decoder", "model_opt.plan")

# UNet
UNET_PATH = os.path.join(TENSORRT_ENGINES_DIR, "unet", "model_int8_optimized.plan")

# Text Encoders
CLIP_TEXT_ENCODER_1_PATH = os.path.join(TENSORRT_ENGINES_DIR, "text_encoder", "model_opt.plan")
CLIP_TEXT_ENCODER_2_PATH = os.path.join(TENSORRT_ENGINES_DIR, "text_encoder_2", "model_opt.plan") 