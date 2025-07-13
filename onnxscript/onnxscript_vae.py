import spox
import spox.opset.ai.onnx.v21 as op
import numpy as np
from safetensors.numpy import load_file
from typing import Dict, Any
import onnx  # Required for the onnx.ModelProto type hint
import json
import sys

import onnxruntime

from typing import Dict, Any, Tuple

from PIL import Image
# --- Parameter Loading Utilities ---

def load_config_from_json(filepath: str) -> Dict[str, Any]:
    """Loads the model's configuration from a JSON file."""
    print(f"Loading configuration from: {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)

def load_state_dict_from_safetensors(filepath: str) -> Dict[str, np.ndarray]:
    """Loads a state_dict from a .safetensors file."""
    print(f"Loading state dictionary from: {filepath}")
    return load_file(filepath)

def load_and_create_spox_params(state_dict: Dict[str, np.ndarray], target_dtype: np.dtype) -> Dict[str, Any]:
    """
    Converts a flat state_dict into a nested dictionary of Spox constant Vars,
    ensuring all float tensors are cast to the target data type.
    """
    spox_params = {}
    for key, value in state_dict.items():
        parts = key.split('.')
        current_level = spox_params
        for part in parts[:-1]:
            part_key = str(part) if part.isdigit() else part
            if part_key not in current_level:
                current_level[part_key] = {}
            current_level = current_level[part_key]
        
        last_part = parts[-1]
        
        # FIX: Ensure all floating point weights are cast to the target dtype
        if np.issubdtype(value.dtype, np.floating):
            value = value.astype(target_dtype)
        
        current_level[last_part] = op.const(value)
    return spox_params

# --- Spox Implementations of PyTorch Modules ---

def to_const(arr: np.ndarray) -> spox.Var:
    return op.const(arr)

def spox_silu(x: spox.Var) -> spox.Var:
    return op.mul(x, op.sigmoid(x))

def spox_group_norm(
    x: spox.Var, weight: spox.Var, bias: spox.Var, num_groups: int, epsilon: float = 1e-6
) -> spox.Var:
    # Get the shape of the input tensor before the operation
    original_shape = op.shape(x)

    # Perform the group normalization
    normalized_x = op.group_normalization(x, weight, bias, num_groups=num_groups, epsilon=epsilon)

    # Explicitly reshape the output to match the original input shape.
    # This is a no-op at runtime but provides the necessary shape info to the builder.
    return op.reshape(normalized_x, original_shape)



def spox_conv_2d(
    x: spox.Var, weight: spox.Var, bias: spox.Var, stride: int = 1, padding: int = 0
) -> spox.Var:
    pads = [padding, padding, padding, padding]
    strides = [stride, stride]
    return op.conv(x, weight, bias, strides=strides, pads=pads)

def spox_linear(x: spox.Var, weight: spox.Var, bias: spox.Var) -> spox.Var:
    weight_t = op.transpose(weight, perm=[1, 0])
    return op.add(op.matmul(x, weight_t), bias)

def spox_resnet_block_2d(
    input_tensor: spox.Var,
    in_channels: int,
    out_channels: int,
    params: Dict[str, Any],
    norm_num_groups: int,
    param_path: str 
) -> spox.Var:
    try:
        hidden_states = input_tensor
        
        norm1_out = spox_group_norm(hidden_states, params["norm1"]["weight"], params["norm1"]["bias"], norm_num_groups)
        act1_out = spox_silu(norm1_out)
        conv1_out = spox_conv_2d(act1_out, params["conv1"]["weight"], params["conv1"]["bias"], padding=1)
        
        norm2_out = spox_group_norm(conv1_out, params["norm2"]["weight"], params["norm2"]["bias"], norm_num_groups)
        act2_out = spox_silu(norm2_out)
        conv2_out = spox_conv_2d(act2_out, params["conv2"]["weight"], params["conv2"]["bias"], padding=1)

        shortcut = input_tensor
        if "conv_shortcut" in params:
            shortcut = spox_conv_2d(input_tensor, params["conv_shortcut"]["weight"], params["conv_shortcut"]["bias"], padding=0, stride=1)
        
        return op.add(shortcut, conv2_out)
    except KeyError as e:
        raise KeyError(f"Missing parameter in ResNet Block at '{param_path}'. Required key: {e}") from e

def spox_attention_block(
    hidden_states: spox.Var,
    channels: int,
    params: Dict[str, Any],
    norm_num_groups: int,
    param_path: str,
    target_dtype: np.dtype
) -> spox.Var:
    try:
        residual = hidden_states
        shape_of_hs = op.shape(hidden_states)
        batch = op.gather(shape_of_hs, to_const(np.array([0], dtype=np.int64)))
        height = op.gather(shape_of_hs, to_const(np.array([2], dtype=np.int64)))
        width = op.gather(shape_of_hs, to_const(np.array([3], dtype=np.int64)))

        norm_out = spox_group_norm(hidden_states, params["group_norm"]["weight"], params["group_norm"]["bias"], norm_num_groups)

        hw = op.mul(height, width)
        reshaped_norm = op.reshape(norm_out, op.concat([batch, to_const(np.array([channels], dtype=np.int64)), hw], axis=0))
        transposed_norm = op.transpose(reshaped_norm, perm=[0, 2, 1])

        q = spox_linear(transposed_norm, params["to_q"]["weight"], params["to_q"]["bias"])
        k = spox_linear(transposed_norm, params["to_k"]["weight"], params["to_k"]["bias"])
        v = spox_linear(transposed_norm, params["to_v"]["weight"], params["to_v"]["bias"])
        
        scale = to_const(np.array(channels**-0.5, dtype=target_dtype))
        
        q = op.reshape(q, op.concat([batch, hw, to_const(np.array([1, channels], dtype=np.int64))], axis=0))
        k = op.reshape(k, op.concat([batch, hw, to_const(np.array([1, channels], dtype=np.int64))], axis=0))
        v = op.reshape(v, op.concat([batch, hw, to_const(np.array([1, channels], dtype=np.int64))], axis=0))

        q, k, v = [op.transpose(t, perm=[0, 2, 1, 3]) for t in (q, k, v)]
        
        scaled_q = op.mul(q, scale)
        attention_scores = op.matmul(scaled_q, op.transpose(k, perm=[0, 1, 3, 2]))
        attention_probs = op.softmax(attention_scores, axis=-1)
        
        hidden_states_attn = op.matmul(attention_probs, v)
        hidden_states_attn = op.transpose(hidden_states_attn, perm=[0, 2, 1, 3])
        hidden_states_attn = op.reshape(hidden_states_attn, op.concat([batch, hw, to_const(np.array([channels], dtype=np.int64))], axis=0))
        
        hidden_states_out_proj = spox_linear(hidden_states_attn, params["to_out"]["0"]["weight"], params["to_out"]["0"]["bias"])

        transposed_out = op.transpose(hidden_states_out_proj, perm=[0, 2, 1])
        reshaped_out = op.reshape(transposed_out, op.concat([batch, to_const(np.array([channels], dtype=np.int64)), height, width], axis=0))

        return op.add(reshaped_out, residual)
    except KeyError as e:
        raise KeyError(f"Missing parameter in Attention Block at '{param_path}'. Required key: {e}") from e

def spox_vae_mid_block(
    hidden_states: spox.Var,
    in_channels: int,
    params: Dict[str, Any],
    norm_num_groups: int,
    param_path: str,
    target_dtype: np.dtype
) -> spox.Var:
    hidden_states = spox_resnet_block_2d(hidden_states, in_channels, in_channels, params["resnets"]["0"], norm_num_groups, f"{param_path}.resnets.0")
    hidden_states = spox_attention_block(hidden_states, in_channels, params["attentions"]["0"], norm_num_groups, f"{param_path}.attentions.0", target_dtype)
    hidden_states = spox_resnet_block_2d(hidden_states, in_channels, in_channels, params["resnets"]["1"], norm_num_groups, f"{param_path}.resnets.1")
    return hidden_states

def spox_downsample(
    hidden_states: spox.Var,
    params: Dict[str, spox.Var],
) -> spox.Var:
    return spox_conv_2d(hidden_states, params["conv"]["weight"], params["conv"]["bias"], stride=2, padding=1)

def spox_upsample(
    hidden_states: spox.Var,
    params: Dict[str, spox.Var],
) -> spox.Var:
    scales = to_const(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)) # Resize scales are float
    hidden_states = op.resize(hidden_states, scales=scales, mode='nearest')
    hidden_states = spox_conv_2d(hidden_states, params["conv"]["weight"], params["conv"]["bias"], padding=1)
    return hidden_states

def spox_down_encoder_block_2d(
    hidden_states: spox.Var,
    in_channels: int,
    out_channels: int,
    num_layers: int,
    params: Dict[str, Any],
    add_downsample: bool,
    norm_num_groups: int,
    param_path: str,
) -> spox.Var:
    current_in_channels = in_channels
    for i in range(num_layers):
        hidden_states = spox_resnet_block_2d(
            hidden_states, current_in_channels, out_channels, params["resnets"][str(i)], norm_num_groups, f"{param_path}.resnets.{i}"
        )
        current_in_channels = out_channels

    if add_downsample:
        hidden_states = spox_downsample(hidden_states, params["downsamplers"]["0"])
    
    return hidden_states

def spox_up_decoder_block_2d(
    hidden_states: spox.Var,
    in_channels: int,
    out_channels: int,
    num_layers: int,
    params: Dict[str, Any],
    add_upsample: bool,
    norm_num_groups: int,
    param_path: str,
) -> spox.Var:
    current_in_channels = in_channels
    for i in range(num_layers):
        hidden_states = spox_resnet_block_2d(
            hidden_states, current_in_channels, out_channels, params["resnets"][str(i)], norm_num_groups, f"{param_path}.resnets.{i}"
        )
        current_in_channels = out_channels

    if add_upsample:
        hidden_states = spox_upsample(hidden_states, params["upsamplers"]["0"])
    
    return hidden_states

def spox_encoder(
    x: spox.Var, params: Dict[str, Any], config: Dict, target_dtype: np.dtype
) -> spox.Var:
    try:
        x = spox_conv_2d(x, params["conv_in"]["weight"], params["conv_in"]["bias"], padding=1)
        
        in_channel = config["block_out_channels"][0]
        for i, _ in enumerate(config["down_block_types"]):
            out_channel = config["block_out_channels"][i]
            is_final_block = i == len(config["block_out_channels"]) - 1
            
            x = spox_down_encoder_block_2d(
                hidden_states=x,
                in_channels=in_channel,
                out_channels=out_channel,
                num_layers=config["layers_per_block"],
                params=params["down_blocks"][str(i)],
                add_downsample=not is_final_block,
                norm_num_groups=config["norm_num_groups"],
                param_path=f"encoder.down_blocks.{i}"
            )
            in_channel = out_channel

        if config.get("mid_block_add_attention", True):
            x = spox_vae_mid_block(x, in_channel, params["mid_block"], config["norm_num_groups"], "encoder.mid_block", target_dtype)
        
        x = spox_group_norm(x, params["conv_norm_out"]["weight"], params["conv_norm_out"]["bias"], config["norm_num_groups"])
        x = spox_silu(x)
        x = spox_conv_2d(x, params["conv_out"]["weight"], params["conv_out"]["bias"], padding=1)
        return x
    except KeyError as e:
        raise KeyError(f"Missing parameter in Encoder. Required key: {e}") from e

def spox_decoder(
    z: spox.Var, params: Dict[str, Any], config: Dict, target_dtype: np.dtype
) -> spox.Var:
    try:
        z = spox_conv_2d(z, params["conv_in"]["weight"], params["conv_in"]["bias"], padding=1)

        if config.get("mid_block_add_attention", True):
            z = spox_vae_mid_block(z, config["block_out_channels"][-1], params["mid_block"], config["norm_num_groups"], "decoder.mid_block", target_dtype)

        reversed_block_out_channels = list(reversed(config["block_out_channels"]))
        in_channel = reversed_block_out_channels[0]
        for i, _ in enumerate(config["up_block_types"]):
            out_channel = reversed_block_out_channels[i]
            is_final_block = i == len(config["block_out_channels"]) - 1
            
            z = spox_up_decoder_block_2d(
                hidden_states=z,
                in_channels=in_channel,
                out_channels=out_channel,
                num_layers=config["layers_per_block"] + 1,
                params=params["up_blocks"][str(i)],
                add_upsample=not is_final_block,
                norm_num_groups=config["norm_num_groups"],
                param_path=f"decoder.up_blocks.{i}"
            )
            in_channel = out_channel

        z = spox_group_norm(z, params["conv_norm_out"]["weight"], params["conv_norm_out"]["bias"], config["norm_num_groups"])
        z = spox_silu(z)
        z = spox_conv_2d(z, params["conv_out"]["weight"], params["conv_out"]["bias"], padding=1)
        return z
    except KeyError as e:
        raise KeyError(f"Missing parameter in Decoder. Required key: {e}") from e

def spox_diagonal_gaussian_distribution_sample(parameters: spox.Var, target_dtype: np.dtype) -> spox.Var:
    mean, logvar = op.split(parameters, num_outputs=2, axis=1)
    logvar = op.clip(logvar, min=to_const(np.array(-30.0, dtype=target_dtype)), max=to_const(np.array(20.0, dtype=target_dtype)))
    std = op.exp(op.mul(logvar, to_const(np.array(0.5, dtype=target_dtype))))
    shape = op.shape(std)
    epsilon = op.random_normal_like(std, dtype=target_dtype, mean=0.0, scale=1.0)
    return op.add(mean, op.mul(std, epsilon))

def spox_diagonal_gaussian_distribution_mode(parameters: spox.Var) -> spox.Var:
    mean, _ = op.split(parameters, num_outputs=2, axis=1)
    return mean

def spox_autoencoder_kl_forward(
    sample: spox.Var,
    sample_posterior: spox.Var,
    params: Dict[str, Any],
    config: Dict,
    target_dtype: np.dtype
) -> spox.Var:
    try:
        h = spox_encoder(sample, params["encoder"], config, target_dtype)
        
        if config.get("use_quant_conv", True):
            moments = spox_conv_2d(h, params["quant_conv"]["weight"], params["quant_conv"]["bias"], padding=0)
        else:
            moments = h
        
        (z,) = op.if_(
            sample_posterior,
            then_branch=lambda: [spox_diagonal_gaussian_distribution_sample(moments, target_dtype)],
            else_branch=lambda: [spox_diagonal_gaussian_distribution_mode(moments)]
        )
        
        if config.get("use_post_quant_conv", True):
            z = spox_conv_2d(z, params["post_quant_conv"]["weight"], params["post_quant_conv"]["bias"], padding=0)
        
        dec = spox_decoder(z, params["decoder"], config, target_dtype)
        return dec
    except KeyError as e:
        raise KeyError(f"Missing parameter at top level. Required key: {e}") from e

# --- Main Build Function ---

def build_encoder_onnx_model(state_dict: Dict[str, np.ndarray], config: Dict) -> onnx.ModelProto:
    """Builds and returns the ONNX model for the VAE Encoder."""
    force_upcast = config.get("force_upcast", False)
    target_dtype = np.float32 if force_upcast else np.float16
    print(f"Building ENCODER with target data type: {target_dtype.__name__}")

    spox_params = load_and_create_spox_params(state_dict, target_dtype)

    sample_type = spox.Tensor(target_dtype, ('batch_size', config["in_channels"], 'height', 'width'))
    sample_arg = spox.argument(sample_type)

    # Encoder forward pass
    h = spox_encoder(sample_arg, spox_params["encoder"], config, target_dtype)
    moments = spox_conv_2d(h, spox_params["quant_conv"]["weight"], spox_params["quant_conv"]["bias"], padding=0)

    # Build the encoder model
    encoder_model = spox.build(
        inputs={"sample": sample_arg},
        outputs={"latent_dist": moments}
    )
    print("Successfully built Encoder ONNX ModelProto.")
    return encoder_model

def build_decoder_onnx_model(state_dict: Dict[str, np.ndarray], config: Dict) -> onnx.ModelProto:
    """Builds and returns the ONNX model for the VAE Decoder."""
    force_upcast = config.get("force_upcast", False)
    target_dtype = np.float32 if force_upcast else np.float16
    print(f"Building DECODER with target data type: {target_dtype.__name__}")

    spox_params = load_and_create_spox_params(state_dict, target_dtype)

    # Define the input for the decoder, which is the latent sample 'z'
    latent_channels = config["latent_channels"]
    latent_type = spox.Tensor(target_dtype, ('batch_size', latent_channels, 'latent_height', 'latent_width'))
    latent_sample_arg = spox.argument(latent_type)

    # Decoder forward pass
    z = spox_conv_2d(latent_sample_arg, spox_params["post_quant_conv"]["weight"], spox_params["post_quant_conv"]["bias"], padding=0)
    dec = spox_decoder(z, spox_params["decoder"], config, target_dtype)

    # Build the decoder model
    decoder_model = spox.build(
        inputs={"latent_sample": latent_sample_arg},
        outputs={"sample": dec}
    )
    print("Successfully built Decoder ONNX ModelProto.")
    return decoder_model

# --- New Inference Helper Functions ---

def preprocess_image(image_path: str, target_dtype: np.dtype) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Loads an image, pads it to be divisible by 8, normalizes, and transposes it.
    Returns the processed numpy array and the original image dimensions.
    """
    img = Image.open(image_path).convert("RGB")
    original_width, original_height = img.size

    # Calculate the new size, padding up to the nearest multiple of 8
    target_width = math.ceil(original_width / 8) * 8
    target_height = math.ceil(original_height / 8) * 8

    # Create a new black image canvas and paste the original image onto it
    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    canvas.paste(img, (0, 0))

    # Continue with standard preprocessing on the padded image (canvas)
    img_array = np.array(canvas).astype(target_dtype)
    img_array = (img_array / 127.5) - 1.0  # Normalize from [0, 255] to [-1, 1]
    img_array = img_array.transpose(2, 0, 1) # HWC to CHW
    img_array = np.expand_dims(img_array, 0) # Add batch dimension

    return img_array, (original_width, original_height)

def postprocess_image(image_tensor: np.ndarray, original_size: Tuple[int, int]) -> Image.Image:
    """Denormalize, transpose, convert to PIL Image, and crop to original size."""
    img = image_tensor[0]
    img = (img + 1.0) * 127.5 # Denormalize
    img = np.clip(img, 0, 255)
    img = img.transpose(1, 2, 0) # CHW to HWC
    reconstructed_image = Image.fromarray(img.astype(np.uint8))

    # Crop the image back to its original size to remove the padding
    original_width, original_height = original_size
    cropped_image = reconstructed_image.crop((0, 0, original_width, original_height))

    return cropped_image


# --- Main Execution ---

if __name__ == '__main__':
    SAFETENSORS_FILE_PATH = "/lab/model/vae/diffusion_pytorch_model.safetensors"
    CONFIG_FILE_PATH = "/lab/model/vae/config.json"
    TEST_IMAGE_PATH = "/lab/test.png"

    try:
        # --- Build and Save Models ---
        config = load_config_from_json(CONFIG_FILE_PATH)
        state_dict = load_state_dict_from_safetensors(SAFETENSORS_FILE_PATH)

        encoder_proto = build_encoder_onnx_model(state_dict, config)
        with open("encoder.onnx", "wb") as f:
            f.write(encoder_proto.SerializeToString())
        print("Saved encoder model to encoder.onnx")

        decoder_proto = build_decoder_onnx_model(state_dict, config)
        with open("decoder.onnx", "wb") as f:
            f.write(decoder_proto.SerializeToString())
        print("Saved decoder model to decoder.onnx")

        # --- Run Inference ---
        print("\n--- Running Inference ---")
        target_dtype = np.float32 if config.get("force_upcast", False) else np.float16

        print("Loading ONNX models into inference sessions...")
        encoder_sess = onnxruntime.InferenceSession("encoder.onnx")
        decoder_sess = onnxruntime.InferenceSession("decoder.onnx")

        print(f"Loading and preprocessing image: {TEST_IMAGE_PATH}")
        # Capture both the processed image and its original size
        image_np, original_size = preprocess_image(TEST_IMAGE_PATH, target_dtype)

        print("Encoding image into latent space...")
        encoder_inputs = {encoder_sess.get_inputs()[0].name: image_np}
        latent_dist_result = encoder_sess.run(None, encoder_inputs)[0]

        print("Sampling from the latent distribution...")
        mean, logvar = np.split(latent_dist_result, 2, axis=1)
        std = np.exp(0.5 * logvar)
        epsilon = np.random.randn(*mean.shape).astype(mean.dtype)
        latents = mean + std * epsilon
        latents = latents * config["scaling_factor"]

        print("Decoding latents back into an image...")
        decoder_inputs = {decoder_sess.get_inputs()[0].name: latents}
        reconstructed_image_np = decoder_sess.run(None, decoder_inputs)[0]

        print("Postprocessing and saving the output image...")
        # Pass the original size to the postprocessing function
        final_image = postprocess_image(reconstructed_image_np, original_size)
        final_image.save("test_out.png")
        print("\nSuccessfully saved reconstructed image to test_out.png")

    except FileNotFoundError as e:
        print(f"\nERROR: Could not find a required file: {e.filename}", file=sys.stderr)
        print("Please check the file paths at the top of the main execution block.", file=sys.stderr)
    except KeyError as e:
        print(f"\n--- MODEL BUILDING FAILED ---", file=sys.stderr)
        print(f"A required weight/bias was not found in the safetensors file. This usually", file=sys.stderr)
        print(f"means there is a mismatch between the model architecture defined in the script", file=sys.stderr)
        print(f"and the weights in the .safetensors file.", file=sys.stderr)
        print(f"\nDETAILS: Missing Key -> {e}\n", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)