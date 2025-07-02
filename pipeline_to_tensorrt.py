import os
import torch
import tensorrt as trt
from polygraphy.backend.trt import CreateConfig, Profile, TrtRunner, engine_from_network, network_from_onnx_path, save_engine
from collections import OrderedDict
from pipeline import defaults

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_abs_path(path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

def build_engine(
    engine_path: str,
    onnx_path: str,
    input_profiles: dict,
    fp16: bool = True,
):
    print(f"Building TensorRT engine for {onnx_path}: {engine_path}")
    
    if os.path.exists(engine_path):
        print("Engine already exists, skipping build.")
        return

    profile = Profile()
    for name, (min_shape, opt_shape, max_shape) in input_profiles.items():
        profile.add(name, min=min_shape, opt=opt_shape, max=max_shape)

    config = CreateConfig(
        trt_logger=TRT_LOGGER,
        fp16=fp16,
        memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 2 * 1024 * 1024 * 1024},  # 2GB
        profiles=[profile]
    )

    network = network_from_onnx_path(onnx_path)
    engine = engine_from_network(network, config=config)
    save_engine(engine, path=engine_path)
    print(f"TensorRT engine saved to {engine_path}")

def get_engine_path(onnx_path: str):
    """
    Returns the TensorRT engine path from an ONNX model path.
    e.g. /path/to/model.onnx -> /path/to/model.plan
    """
    return os.path.splitext(onnx_path)[0] + ".plan"

def main():
    """
    Builds TensorRT engines for all SDXL pipeline models.
    """
    print("Building TensorRT engines for SDXL pipeline...")
    device = torch.device("cuda:0")

    # UNet
    print("\n--- Building UNet Engine ---")
    latent_heights = [768 // 8, 1152 // 8, 944 // 8] # 96, 144, 118
    latent_widths = [1152 // 8, 768 // 8, 944 // 8]  # 144, 96, 118
    
    min_h, max_h = min(latent_heights), max(latent_heights)
    min_w, max_w = min(latent_widths), max(latent_widths)
    opt_h, opt_w = 944 // 8, 944 // 8

    unet_input_profiles = {
        "sample": (
            (1, 4, min_h, min_w),
            (1, 4, opt_h, opt_w),
            (1, 4, max_h, max_w),
        ),
        "timestep": ((1,), (1,), (1,)),
        "encoder_hidden_states": ((1, 77, 2048), (1, 77, 2048), (1, 77, 2048)),
        "text_embeds": ((1, 1280), (1, 1280), (1, 1280)),
        "time_ids": ((1, 6), (1, 6), (1, 6)),
    }
    build_engine(
        engine_path=get_engine_path(defaults.UNET_PATH),
        onnx_path=defaults.UNET_PATH,
        input_profiles=unet_input_profiles,
    )

    # VAE Decoder
    print("\n--- Building VAE Decoder Engine ---")
    vae_input_profiles = {
        "latent_sample": (
            (1, 4, min_h, min_w),
            (1, 4, opt_h, opt_w),
            (1, 4, max_h, max_w),
        ),
    }
    build_engine(
        engine_path=get_engine_path(defaults.VAE_DECODER_PATH),
        onnx_path=defaults.VAE_DECODER_PATH,
        input_profiles=vae_input_profiles,
    )

    # CLIP Text Encoder 1 (CLIP-L)
    print("\n--- Building CLIP-L Text Encoder Engine ---")
    clip_l_input_profiles = {
        "input_ids": ((1, 77), (1, 77), (1, 77)),
    }
    build_engine(
        engine_path=get_engine_path(defaults.CLIP_TEXT_ENCODER_1_PATH),
        onnx_path=defaults.CLIP_TEXT_ENCODER_1_PATH,
        input_profiles=clip_l_input_profiles,
    )

    # CLIP Text Encoder 2 (CLIP-G)
    print("\n--- Building CLIP-G Text Encoder Engine ---")
    clip_g_input_profiles = {
        "input_ids": ((1, 77), (1, 77), (1, 77)),
    }
    build_engine(
        engine_path=get_engine_path(defaults.CLIP_TEXT_ENCODER_2_PATH),
        onnx_path=defaults.CLIP_TEXT_ENCODER_2_PATH,
        input_profiles=clip_g_input_profiles,
    )

    print("\nAll TensorRT engines built successfully!")

if __name__ == "__main__":
    main() 