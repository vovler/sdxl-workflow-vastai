import tensorrt as trt
import os

def build_engine(onnx_path, engine_path, use_fp16=True):
    """
    Builds a TensorRT engine from an ONNX file.
    """
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    
    # Check if the ONNX file exists
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found at {onnx_path}")
        print("Please run unet_to_onnx.py first to generate the ONNX file.")
        return

    # Initialize TensorRT builder, network, and parser
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # Set memory constraints
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 * (1024 ** 3)) # 1 GiB

    # Enable FP16 if supported and requested
    if use_fp16 and builder.platform_has_fast_fp16:
        print("Enabling FP16 mode.")
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print("FP16 mode not enabled.")

    # Parse the ONNX model
    print(f"Parsing ONNX file: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("Completed parsing ONNX file.")

    # --- Define Optimization Profile for Dynamic Shapes ---
    profile = builder.create_optimization_profile()
    
    # Define min, optimal, and max shapes for each input
    # These should match the names printed by the previous script.
    # The ONNX graph has a static shape, so min/opt/max for H/W must match it.
    # Shape: (batch_size * 2, channels, height/8, width/8)
    profile.set_shape("sample", (1*2, 4, 128, 128), (1*2, 4, 128, 128), (2*2, 4, 128, 128))
    # Shape: (batch_size * 2,)
    profile.set_shape("timestep", (1*2,), (1*2,), (2*2,))
    # Shape: (batch_size * 2, sequence_length, hidden_size)
    profile.set_shape("encoder_hidden_states", (1*2, 77, 2048), (1*2, 77, 2048), (2*2, 77, 2048))
    # Shape: (batch_size * 2, pooled_projection_dim)
    profile.set_shape("text_embeds", (1*2, 1280), (1*2, 1280), (2*2, 1280))
    # Shape: (batch_size * 2, 6)
    profile.set_shape("time_ids", (1*2, 6), (1*2, 6), (2*2, 6))
    
    config.add_optimization_profile(profile)

    # Build the engine
    print("Building TensorRT engine... (This may take a few minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("ERROR: Failed to build the engine.")
        return None
    print("Successfully built TensorRT engine.")

    # Save the engine
    print(f"Saving engine to file: {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print("Engine saved.")

def main():
    onnx_file_path = "unet.onnx"
    engine_file_path = "unet.engine"
    build_engine(onnx_file_path, engine_file_path, use_fp16=True)

if __name__ == "__main__":
    main() 