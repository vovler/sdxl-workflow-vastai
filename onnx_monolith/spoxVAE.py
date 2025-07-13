import spox
import spox.opset.ai.onnx.v21 as op
import numpy as np
from safetensors.numpy import load_file
from typing import Dict, Any
import onnx  # Required for the onnx.ModelProto type hint
import json
from typing import Dict, Any, Tuple

# --- Parameter Loading Utilities ---

# It's good practice to wrap complex graph-building logic to provide context on errors.
def with_error_context(name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Re-raise the exception with more context about where it happened.
                raise type(e)(f"Error in ONNX graph construction at '{name}': {e}") from e
        return wrapper
    return decorator


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

# --- Main Build Function ---

def build_encoder_onnx_model(state_dict: Dict[str, np.ndarray], config: Dict) -> onnx.ModelProto:
    """
    Builds the VAE Encoder ONNX model, self-containing the reparameterization trick.
    """
    target_dtype = np.float16
    print(f"Building ENCODER with target data type: {target_dtype.__name__}")

    spox_params = load_and_create_spox_params(state_dict, target_dtype)

    # Define the symbolic input shape
    sample_type = spox.Tensor(target_dtype, ('batch_size', config["in_channels"], 'height', 'width'))
    sample_arg = spox.argument(sample_type)

    # --- Encoder and Sampling Graph Logic ---
    # 1. Get the 8-channel latent distribution (moments) from the encoder backbone
    h = spox_encoder(sample_arg, spox_params["encoder"], config, target_dtype)
    moments = spox_conv_2d(h, spox_params["quant_conv"]["weight"], spox_params["quant_conv"]["bias"], padding=0)

    # 2. Split the moments into mean and log-variance
    mean, logvar = op.split(moments, num_outputs=2, axis=1)

    # 3. Calculate standard deviation: std = exp(0.5 * logvar)
    half_const = op.const(np.array(0.5, dtype=target_dtype))
    std = op.exp(op.mul(logvar, half_const))

    # 4. Generate noise of the same shape as mean/std
    epsilon = op.random_normal_like(mean, dtype=target_dtype, mean=0.0, scale=1.0)

    # 5. Compute the final latent sample: z = mean + std * epsilon
    latent_sample = op.add(mean, op.mul(std, epsilon))

    # --- Build the model ---
    # The output is the self-contained latent sample (z)
    encoder_model = spox.build(
        inputs={"sample": sample_arg},
        outputs={"latent_sample": latent_sample}
    )
    print("Successfully built Encoder ONNX ModelProto (with self-contained sampling).")
    return encoder_model


def build_decoder_onnx_model(state_dict: Dict[str, np.ndarray], config: Dict) -> onnx.ModelProto:
    """
    Builds the VAE Decoder ONNX model, which takes a 4-channel latent sample as input.
    """
    target_dtype = np.float16
    print(f"Building DECODER with target data type: {target_dtype.__name__}")

    spox_params = load_and_create_spox_params(state_dict, target_dtype)

    # The input is now the 4-channel latent sample
    latent_channels = config["latent_channels"]
    latent_type = spox.Tensor(target_dtype, ('batch_size', latent_channels, 'latent_height', 'latent_width'))
    latent_sample_arg = spox.argument(latent_type)

    # The first step in the decoder is the post-quantization conv
    z = spox_conv_2d(latent_sample_arg, spox_params["post_quant_conv"]["weight"], spox_params["post_quant_conv"]["bias"], padding=0)
    
    # Run the main decoder backbone
    reconstructed_sample = spox_decoder(z, spox_params["decoder"], config, target_dtype)

    decoder_model = spox.build(
        inputs={"latent_sample": latent_sample_arg},
        outputs={"reconstructed_sample": reconstructed_sample}
    )
    print("Successfully built Decoder ONNX ModelProto.")
    return decoder_model







def spox_blend_v(
    top_tile: spox.Var, 
    bottom_tile: spox.Var, 
    blend_extent: int, 
    tile_size: int, 
    target_dtype: np.dtype
) -> spox.Var:
    """
    Blends the bottom rows of top_tile with the top rows of bottom_tile.
    """
    # Create a vertical blending ramp (1.0 -> 0.0)
    ramp_np = np.linspace(1.0, 0.0, blend_extent, dtype=target_dtype).astype(target_dtype)
    ramp = to_const(ramp_np.reshape(1, 1, blend_extent, 1))

    # Slice the regions to be blended
    # Slice the bottom `blend_extent` rows from the top tile
    top_blend_region = op.slice(
        top_tile, 
        starts=to_const(np.array([tile_size - blend_extent])), 
        ends=to_const(np.array([tile_size])), 
        axes=to_const(np.array([2]))
    )
    
    # Slice the top `blend_extent` rows from the bottom tile
    bottom_blend_region = op.slice(
        bottom_tile, 
        starts=to_const(np.array([0])), 
        ends=to_const(np.array([blend_extent])), 
        axes=to_const(np.array([2]))
    )

    # Perform the blend: (bottom * ramp) + (top * (1 - ramp))
    one_const = to_const(np.array(1.0, dtype=target_dtype))
    blended_region = op.add(
        op.mul(bottom_blend_region, ramp),
        op.mul(top_blend_region, op.sub(one_const, ramp))
    )

    # Get the remaining part of the bottom tile that is not blended
    bottom_remaining = op.slice(
        bottom_tile, 
        starts=to_const(np.array([blend_extent])), 
        ends=to_const(np.array([tile_size])), 
        axes=to_const(np.array([2]))
    )
    
    # Concatenate the blended region with the rest of the tile
    return op.concat([blended_region, bottom_remaining], axis=2)

def spox_blend_h(
    left_tile: spox.Var, 
    right_tile: spox.Var, 
    blend_extent: int, 
    tile_size: int, 
    target_dtype: np.dtype
) -> spox.Var:
    """
    Blends the right columns of left_tile with the left columns of right_tile.
    """
    # Create a horizontal blending ramp (1.0 -> 0.0)
    ramp_np = np.linspace(1.0, 0.0, blend_extent, dtype=target_dtype).astype(target_dtype)
    ramp = to_const(ramp_np.reshape(1, 1, 1, blend_extent))
    
    # Slice the regions to be blended
    # Slice the right `blend_extent` columns from the left tile
    left_blend_region = op.slice(
        left_tile, 
        starts=to_const(np.array([tile_size - blend_extent])), 
        ends=to_const(np.array([tile_size])), 
        axes=to_const(np.array([3]))
    )
    
    # Slice the left `blend_extent` columns from the right tile
    right_blend_region = op.slice(
        right_tile, 
        starts=to_const(np.array([0])), 
        ends=to_const(np.array([blend_extent])), 
        axes=to_const(np.array([3]))
    )

    # Perform the blend: (right * ramp) + (left * (1 - ramp))
    one_const = to_const(np.array(1.0, dtype=target_dtype))
    blended_region = op.add(
        op.mul(right_blend_region, ramp),
        op.mul(left_blend_region, op.sub(one_const, ramp))
    )

    # Get the remaining part of the right tile that is not blended
    right_remaining = op.slice(
        right_tile, 
        starts=to_const(np.array([blend_extent])), 
        ends=to_const(np.array([tile_size])), 
        axes=to_const(np.array([3]))
    )
    
    # Concatenate the blended region with the rest of the tile
    return op.concat([blended_region, right_remaining], axis=3)




@with_error_context("Tiled Decoder Model")
def build_tiled_decoder_onnx_model_with_loop(
    state_dict: Dict[str, np.ndarray],
    config: Dict
) -> onnx.ModelProto:
    """
    Builds the VAE Tiled Decoder ONNX model using the `Loop` operator for dynamic tiling.
    """
    target_dtype = np.float16
    print(f"Building TILED DECODER with target data type: {target_dtype.__name__}")

    # --- 1. Calculate Tiling Parameters ---
    tile_sample_min_size = config["sample_size"]
    tile_latent_min_size = int(tile_sample_min_size / (2 ** (len(config["block_out_channels"]) - 1)))
    tile_overlap_factor = 0.25
    downsample_factor = 2**(len(config["block_out_channels"]) - 1)

    overlap_size = int(tile_latent_min_size * (1 - tile_overlap_factor))
    blend_extent = int(tile_sample_min_size * tile_overlap_factor)
    
    print(f"Tiling config: latent_tile={tile_latent_min_size}, sample_tile={tile_sample_min_size}, overlap={overlap_size}, blend={blend_extent}")

    # --- 2. Load Parameters & Define Inputs ---
    spox_params = load_and_create_spox_params(state_dict, target_dtype)

    latent_channels = config["latent_channels"]
    latent_type = spox.Tensor(target_dtype, ('batch_size', latent_channels, 'latent_height', 'latent_width'))
    latent_z_arg = spox.argument(latent_type)

    # --- 3. Setup Loop for Tile Decoding and Blending ---
    latent_shape = op.shape(latent_z_arg)
    batch_size = op.gather(latent_shape, to_const(np.array([0], dtype=np.int64)))
    latent_height = op.gather(latent_shape, to_const(np.array([2], dtype=np.int64)))
    latent_width = op.gather(latent_shape, to_const(np.array([3], dtype=np.int64)))

    overlap_size_const = to_const(np.array(overlap_size, dtype=np.int64))
    num_rows = op.div(op.add(latent_height, op.sub(overlap_size_const, to_const(np.array(1, dtype=np.int64)))), overlap_size_const)
    num_cols = op.div(op.add(latent_width, op.sub(overlap_size_const, to_const(np.array(1, dtype=np.int64)))), overlap_size_const)

    trip_count = op.mul(num_rows, num_cols)
    
    # We will build up the final image inside a single loop, row by row.
    # The initial state is an empty canvas to which we will add rows.
    initial_canvas_shape = op.concat([
        batch_size,
        to_const(np.array([config['out_channels'], 0], dtype=np.int64)),
        op.mul(latent_width, to_const(np.array(downsample_factor, dtype=np.int64)))
    ], axis=0)
    initial_canvas = op.constant_of_shape(initial_canvas_shape, value=to_const(np.array(0, dtype=target_dtype)))
    
    # We also need to carry the previously decoded row for vertical blending.
    prev_row_shape = op.concat([
        batch_size, 
        to_const(np.array([config['out_channels'], tile_sample_min_size], dtype=np.int64)), 
        op.mul(num_cols, to_const(np.array(overlap_size, dtype=np.int64)))
    ], axis=0)
    initial_prev_row = op.constant_of_shape(prev_row_shape, value=to_const(np.array(0, dtype=target_dtype)))

    @with_error_context("Main Loop Body")
    def main_loop_body(row_idx, _, current_canvas, previous_row):
        # This loop iterates over rows.
        
        # Inner loop to build one full row of tiles.
        @with_error_context("Inner Loop Body (Columns)")
        def col_loop_body(col_idx, _, prev_tile_in_row):
            start_h = op.mul(row_idx, overlap_size_const)
            start_w = op.mul(col_idx, overlap_size_const)
            
            latent_tile = op.slice(
                latent_z_arg,
                starts=op.concat([start_h, start_w], axis=0),
                ends=op.concat([op.add(start_h, to_const(np.array(tile_latent_min_size, dtype=np.int64))),
                                 op.add(start_w, to_const(np.array(tile_latent_min_size, dtype=np.int64)))], axis=0),
                axes=to_const(np.array([2, 3], dtype=np.int64))
            )
            
            post_quant_tile = spox_conv_2d(latent_tile, spox_params["post_quant_conv"]["weight"], spox_params["post_quant_conv"]["bias"], padding=0)
            decoded_tile = spox_decoder(post_quant_tile, spox_params["decoder"], config, target_dtype)

            # Blend Horizontally
            def blend_left():
                return [spox_blend_h(prev_tile_in_row, decoded_tile, blend_extent, tile_sample_min_size, target_dtype)]
            def no_blend_left():
                return [decoded_tile]
            is_not_first_col = op.greater(col_idx, to_const(np.array(0, dtype=np.int64)))
            (blended_tile,) = op.if_(is_not_first_col, then_branch=blend_left, else_branch=no_blend_left)

            # Return the condition, and the blended tile which becomes the `prev_tile_in_row` for the next column.
            return op.const(True), blended_tile
        
        # Run the inner loop to get the last blended tile of the row, which contains the full row.
        initial_inner_tile_shape = op.concat([
            batch_size, to_const(np.array([config['out_channels'], tile_sample_min_size, 0], dtype=np.int64))
        ], axis=0)
        initial_inner_tile = op.constant_of_shape(initial_inner_tile_shape, value=to_const(np.array(0, dtype=target_dtype)))
        
        (final_row_tile,) = op.loop(num_cols, v_initial=[initial_inner_tile], body=col_loop_body)

        # Blend Vertically with the previous row
        def blend_up():
            return [spox_blend_v(previous_row, final_row_tile, blend_extent, tile_sample_min_size, target_dtype)]
        def no_blend_up():
             # For the first row, we don't blend, but we still need to crop the top blend_extent off
            return [op.slice(final_row_tile, starts=to_const(np.array([blend_extent])), ends=to_const(np.array([tile_sample_min_size])), axes=to_const(np.array([2])))]
            
        is_not_first_row = op.greater(row_idx, to_const(np.array(0, dtype=np.int64)))
        (final_row_for_canvas,) = op.if_(is_not_first_row, then_branch=blend_up, else_branch=no_blend_up)

        # Concatenate the new row to the canvas
        updated_canvas = op.concat([current_canvas, final_row_for_canvas], axis=2)
        
        # Return loop condition, updated canvas, and the full row to be used in the next iteration.
        return op.const(True), updated_canvas, final_row_tile

    # --- 5. Execute the Main Loop ---
    (final_canvas, _,) = op.loop(
        num_rows,
        v_initial=[initial_canvas, initial_prev_row],
        body=main_loop_body
    )

    # --- 6. Crop Final Image to Dynamic Size ---
    final_out_height = op.mul(latent_height, to_const(np.array(downsample_factor, dtype=np.int64)))
    final_out_width = op.mul(latent_width, to_const(np.array(downsample_factor, dtype=np.int64)))

    final_image_cropped = op.slice(
        final_canvas,
        starts=to_const(np.array([0, 0, 0, 0], dtype=np.int64)),
        ends=op.concat([
            batch_size, 
            to_const(np.array([config['out_channels']], dtype=np.int64)), 
            final_out_height, 
            final_out_width
        ], axis=0),
        axes=to_const(np.array([0, 1, 2, 3], dtype=np.int64))
    )

    # --- 7. Build the Model ---
    decoder_model = spox.build(
        inputs={"latent_sample": latent_z_arg},
        outputs={"reconstructed_sample": final_image_cropped}
    )
    print("Successfully built Tiled Decoder ONNX ModelProto using op.loop.")
    return decoder_model