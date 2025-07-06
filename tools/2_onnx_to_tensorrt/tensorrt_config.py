import tensorrt as trt
import torch

# General Builder Configuration
# These settings are applied to all models unless overridden.
BUILDER_CONFIG = {
    
    # Maximum number of auxiliary streams TensorRT is allowed to use.
    # Setting this to 1 can sometimes help with performance or stability.
    "MAX_AUX_STREAMS": 1,

    # Fraction of total GPU VRAM to be used for the workspace.
    # The workspace is a memory pool used by TensorRT for temporary storage
    # during engine building. 0.5 means 50% of VRAM.
    "MEMORY_WORKSPACE_FRACTION": 0.5,
    
    # Builder optimization level. This controls the aggressiveness of optimization algorithms.
    # The levels range from 0 to 5. Higher levels enable more optimizations and may
    # result in better performance but increase build time.
    # - Level 0: Disables all optimizations.
    # - Level 1-2: Enables basic optimizations.
    # - Level 3 (DEFAULT): Default level, provides a good balance between build time and performance.
    # - Level 4: Enables more aggressive optimizations.
    # - Level 5: Maximum optimization level. Enables all available optimizations, 
    #            including those with high build time cost.
    "BUILDER_OPTIMIZATION_LEVEL": 0,

    # Hardware compatibility level. This ensures the engine is portable across different
    # GPU architectures.
    # - NONE (DEFAULT): No compatibility. The engine is tuned for the specific GPU it was built on.
    # - SAME_COMPUTE_CAPABILITY: The engine is portable to other GPUs with the same
    #   major compute capability (e.g., across different Ampere GPUs).
    # - AMPERE_PLUS: Guarantees portability across Ampere and newer architectures (Hopper, Ada).
    "HARDWARE_COMPATIBILITY_LEVEL": trt.HardwareCompatibilityLevel.NONE,
    
    # Tiling optimization level. Controls how aggressively TensorRT explores different
    # tiling strategies for network layers. A higher level allows TensorRT to spend
    # more time searching for a better tiling strategy.
    # NONE (DEFAULT): Do not apply any tiling strategy.
    # FAST: Use a fast algorithm and heuristic based strategy. Slightly increases engine build time.
    # MODERATE: Increase search space and use a mixed heuristic/profiling strategy. Moderately increases engine build time.
    # FULL: Increase search space even wider. Significantly increases engine build time.
    "TILING_OPTIMIZATION_LEVEL": trt.TilingOptimizationLevel.NONE,
    
    # Preview features to enable. These are experimental features that may change
    # or be removed in future TensorRT versions.
    "PREVIEW_FEATURES": {
        "RUNTIME_ACTIVATION_RESIZE_10_10": True,
    },

    # Builder flags.
    "BUILDER_FLAGS": {
        "FP16": True,
        "INT8": False,  # Default to False, overridden for UNet
    }
}

# --- Model Specific Configurations ---

# UNet Configuration
UNET_PROFILES = {
    "min": {"bs": 1, "height": 768, "width": 768, "seq_len": 77},
    "opt": {"bs": 1, "height": 768, "width": 1152, "seq_len": 77},
    "max": {"bs": 4, "height": 1024, "width": 1152, "seq_len": 77},
}
UNET_INT8_CONFIG = {
    "flags": {
        "INT8": True, # UNet is quantized
        "FP16": True,
    }
}

# VAE Decoder Configuration
VAE_DECODER_PROFILES = {
    "min": {"bs": 1, "height": 768, "width": 768},
    "opt": {"bs": 1, "height": 768, "width": 1152},
    "max": {"bs": 4, "height": 1024, "width": 1152},
}

# CLIP Text Encoders Configuration
CLIP_PROFILES = {
    "min": {"bs": 1, "seq_len": 77},
    "opt": {"bs": 1, "seq_len": 77},
    "max": {"bs": 1, "seq_len": 144},
}

# WD-Tagger Configuration
WD_TAGGER_PROFILES = {
    "min": {"bs": 1, "height": 448, "width": 448},
    "opt": {"bs": 1, "height": 448, "width": 448},
    "max": {"bs": 4, "height": 448, "width": 448},
}

# YOLO Configuration
YOLO_PROFILES = {
    "min": {"bs": 1, "height": 768, "width": 768},
    "opt": {"bs": 1, "height": 768, "width": 1152},
    "max": {"bs": 4, "height": 1024, "width": 1152},
}

# SAM Configuration
SAM_PROFILES = {
    "min": {"bs": 1, "height": 30, "width": 30},
    "opt": {"bs": 1, "height": 100, "width": 100},
    "max": {"bs": 4, "height": 800, "width": 800},
}

# Upscaler Configuration
UPSCALER_PROFILES = {
    "min": {"bs": 1, "height": 768, "width": 768},
    "opt": {"bs": 1, "height": 768, "width": 1152},
    "max": {"bs": 4, "height": 1024, "width": 1152},
}


def apply_builder_config(config: trt.IBuilderConfig, model_flags: dict = {}):
    """Applies the global builder settings and any model-specific overrides."""
    
    config.builder_optimization_level = BUILDER_CONFIG["BUILDER_OPTIMIZATION_LEVEL"]
    
    # Set Max Aux Streams
    if "MAX_AUX_STREAMS" in BUILDER_CONFIG:
        config.max_aux_streams = BUILDER_CONFIG["MAX_AUX_STREAMS"]

    # Set Workspace Size
    if "MEMORY_WORKSPACE_FRACTION" in BUILDER_CONFIG and torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory
        workspace_size = int(total_vram * BUILDER_CONFIG["MEMORY_WORKSPACE_FRACTION"])
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
        print(f"Setting TensorRT workspace size to {workspace_size / (1024**3):.2f} GB")

    for feature_name, enabled in BUILDER_CONFIG["PREVIEW_FEATURES"].items():
        if enabled:
            feature = getattr(trt.PreviewFeature, feature_name)
            config.set_preview_feature(feature, True)
            
    config.hardware_compatibility_level = BUILDER_CONFIG["HARDWARE_COMPATIBILITY_LEVEL"]

    config.tiling_optimization_level = BUILDER_CONFIG["TILING_OPTIMIZATION_LEVEL"]

    # Combine global and model-specific flags
    final_flags = BUILDER_CONFIG["BUILDER_FLAGS"].copy()
    final_flags.update(model_flags)

    if final_flags.get("FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
    if final_flags.get("INT8"):
        config.set_flag(trt.BuilderFlag.INT8) 