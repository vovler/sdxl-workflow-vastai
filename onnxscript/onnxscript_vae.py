import spox
import spox.opset.ai.onnx.v21 as op
import numpy as np
from safetensors.numpy import load_file
from typing import Dict, Any
import onnx # Required for the onnx.ModelProto type hint

# Model configuration
config = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
    "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
    "block_out_channels": [128, 256, 512, 512],
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "latent_channels": 4,
    "mid_block_add_attention": True,
    "use_quant_conv": True,
    "use_post_quant_conv": True,
}

# --- Parameter Loading Utilities ---

def load_state_dict_from_safetensors(filepath: str) -> Dict[str, np.ndarray]:
    """Loads a state_dict from a .safetensors file."""
    print(f"Loading state dictionary from: {filepath}")
    return load_file(filepath)

def load_and_create_spox_params(state_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Converts a flat state_dict into a nested dictionary of Spox constant Vars,
    ensuring all float tensors are cast to float16.
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
        
        # --- FIX: Ensure all floating point weights are float16 ---
        if np.issubdtype(value.dtype, np.floating):
            value = value.astype(np.float16)
        
        current_level[last_part] = op.const(value)
    return spox_params


# --- Spox Implementations of PyTorch Modules (Unchanged) ---

def to_const(arr: np.ndarray) -> spox.Var:
    return op.const(arr)

def spox_silu(x: spox.Var) -> spox.Var:
    return op.mul(x, op.sigmoid(x))

def spox_group_norm(
    x: spox.Var, weight: spox.Var, bias: spox.Var, num_groups: int, epsilon: float = 1e-6
) -> spox.Var:
    return op.group_normalization(x, weight, bias, num_groups=num_groups, epsilon=epsilon)

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

        if "conv_shortcut" in params:
            input_tensor = spox_conv_2d(input_tensor, params["conv_shortcut"]["weight"], params["conv_shortcut"]["bias"], padding=0, stride=1)
        
        return op.add(input_tensor, conv2_out)
    except KeyError as e:
        raise KeyError(f"Missing parameter in ResNet Block at '{param_path}'. Required key: {e}") from e

def spox_attention_block(
    hidden_states: spox.Var,
    channels: int,
    params: Dict[str, Any],
    norm_num_groups: int,
    param_path: str
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
        
        scale = to_const(np.array(channels**-0.5, dtype=np.float16))
        
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
    param_path: str
) -> spox.Var:
    hidden_states = spox_resnet_block_2d(hidden_states, params["resnets"]["0"], norm_num_groups, f"{param_path}.resnets.0")
    hidden_states = spox_attention_block(hidden_states, in_channels, params["attentions"]["0"], norm_num_groups, f"{param_path}.attentions.0")
    hidden_states = spox_resnet_block_2d(hidden_states, params["resnets"]["1"], norm_num_groups, f"{param_path}.resnets.1")
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
    scales = to_const(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32))
    hidden_states = op.resize(hidden_states, scales=scales, mode='nearest')
    hidden_states = spox_conv_2d(hidden_states, params["conv"]["weight"], params["conv"]["bias"], padding=1)
    return hidden_states

def spox_down_encoder_block_2d(
    hidden_states: spox.Var,
    params: Dict[str, Any],
    add_downsample: bool,
    norm_num_groups: int,
    param_path: str,
) -> spox.Var:
    num_layers = len(params["resnets"])
    for i in range(num_layers):
        hidden_states = spox_resnet_block_2d(hidden_states, params["resnets"][str(i)], norm_num_groups, f"{param_path}.resnets.{i}")

    if add_downsample:
        hidden_states = spox_downsample(hidden_states, params["downsamplers"]["0"])
    
    return hidden_states

def spox_up_decoder_block_2d(
    hidden_states: spox.Var,
    params: Dict[str, Any],
    add_upsample: bool,
    norm_num_groups: int,
    param_path: str,
) -> spox.Var:
    num_layers = len(params["resnets"])
    for i in range(num_layers):
        hidden_states = spox_resnet_block_2d(hidden_states, params["resnets"][str(i)], norm_num_groups, f"{param_path}.resnets.{i}")

    if add_upsample:
        hidden_states = spox_upsample(hidden_states, params["upsamplers"]["0"])
    
    return hidden_states

def spox_encoder(
    x: spox.Var, params: Dict[str, Any], config: Dict
) -> spox.Var:
    try:
        x = spox_conv_2d(x, params["conv_in"]["weight"], params["conv_in"]["bias"], padding=1)
        
        for i, down_block_type in enumerate(config["down_block_types"]):
            is_final_block = i == len(config["block_out_channels"]) - 1
            x = spox_down_encoder_block_2d(
                hidden_states=x,
                params=params["down_blocks"][str(i)],
                add_downsample=not is_final_block,
                norm_num_groups=config["norm_num_groups"],
                param_path=f"encoder.down_blocks.{i}"
            )

        if config.get("mid_block_add_attention", True):
            x = spox_vae_mid_block(x, config["block_out_channels"][-1], params["mid_block"], config["norm_num_groups"], "encoder.mid_block")
        
        x = spox_group_norm(x, params["conv_norm_out"]["weight"], params["conv_norm_out"]["bias"], config["norm_num_groups"])
        x = spox_silu(x)
        x = spox_conv_2d(x, params["conv_out"]["weight"], params["conv_out"]["bias"], padding=1)
        return x
    except KeyError as e:
        raise KeyError(f"Missing parameter in Encoder. Required key: {e}") from e

def spox_decoder(
    z: spox.Var, params: Dict[str, Any], config: Dict
) -> spox.Var:
    try:
        z = spox_conv_2d(z, params["conv_in"]["weight"], params["conv_in"]["bias"], padding=1)

        if config.get("mid_block_add_attention", True):
            z = spox_vae_mid_block(z, config["block_out_channels"][-1], params["mid_block"], config["norm_num_groups"], "decoder.mid_block")

        for i, up_block_type in enumerate(config["up_block_types"]):
            is_final_block = i == len(config["block_out_channels"]) - 1
            z = spox_up_decoder_block_2d(
                hidden_states=z,
                params=params["up_blocks"][str(i)],
                add_upsample=not is_final_block,
                norm_num_groups=config["norm_num_groups"],
                param_path=f"decoder.up_blocks.{i}"
            )

        z = spox_group_norm(z, params["conv_norm_out"]["weight"], params["conv_norm_out"]["bias"], config["norm_num_groups"])
        z = spox_silu(z)
        z = spox_conv_2d(z, params["conv_out"]["weight"], params["conv_out"]["bias"], padding=1)
        return z
    except KeyError as e:
        raise KeyError(f"Missing parameter in Decoder. Required key: {e}") from e

def spox_diagonal_gaussian_distribution_sample(parameters: spox.Var) -> spox.Var:
    mean, logvar = op.split(parameters, num_outputs=2, axis=1)
    logvar = op.clip(logvar, min=to_const(np.array(-30.0, dtype=np.float16)), max=to_const(np.array(20.0, dtype=np.float16)))
    std = op.exp(op.mul(logvar, to_const(np.array(0.5, dtype=np.float16))))
    shape = op.shape(std)
    epsilon = op.random_normal(shape=shape, dtype=np.float16, mean=0.0, scale=1.0)
    return op.add(mean, op.mul(std, epsilon))

def spox_diagonal_gaussian_distribution_mode(parameters: spox.Var) -> spox.Var:
    mean, _ = op.split(parameters, num_outputs=2, axis=1)
    return mean

def spox_autoencoder_kl_forward(
    sample: spox.Var,
    sample_posterior: spox.Var,
    params: Dict[str, Any],
    config: Dict
) -> spox.Var:
    try:
        h = spox_encoder(sample, params["encoder"], config)
        
        if config.get("use_quant_conv", True):
            moments = spox_conv_2d(h, params["quant_conv"]["weight"], params["quant_conv"]["bias"], padding=0)
        else:
            moments = h
        
        (z,) = op.if_(
            sample_posterior,
            then_branch=lambda: [spox_diagonal_gaussian_distribution_sample(moments)],
            else_branch=lambda: [spox_diagonal_gaussian_distribution_mode(moments)]
        )
        
        if config.get("use_post_quant_conv", True):
            z = spox_conv_2d(z, params["post_quant_conv"]["weight"], params["post_quant_conv"]["bias"], padding=0)
        
        dec = spox_decoder(z, params["decoder"], config)
        return dec
    except KeyError as e:
        raise KeyError(f"Missing parameter at top level. Required key: {e}") from e


# --- Main Build Function ---

def build_autoencoder_kl_onnx_model(state_dict: Dict[str, np.ndarray], config: Dict) -> onnx.ModelProto:
    print("Loading parameters into Spox constants...")
    spox_params = load_and_create_spox_params(state_dict)
    
    print("Defining model inputs and building graph...")
    sample_type = spox.Tensor(np.float16, ('batch_size', config["in_channels"], 'height', 'width'))
    sample_arg = spox.argument(sample_type)
    
    sample_posterior_type = spox.Tensor(np.bool_, ())
    sample_posterior_arg = spox.argument(sample_posterior_type)
    
    final_output = spox_autoencoder_kl_forward(
        sample=sample_arg,
        sample_posterior=sample_posterior_arg,
        params=spox_params,
        config=config,
    )
    
    onnx_model = spox.build(
        inputs={"sample": sample_arg, "sample_posterior": sample_posterior_arg},
        outputs={"output": final_output}
    )
    
    print("Successfully built ONNX ModelProto.")
    return onnx_model

# --- Main Execution ---

if __name__ == '__main__':
    SAFETENSORS_FILE_PATH = "/lab/model/vae/diffusion_pytorch_model.safetensors"

    try:
        state_dict = load_state_dict_from_safetensors(SAFETENSORS_FILE_PATH)
        model_proto = build_autoencoder_kl_onnx_model(state_dict, config)
        
        output_filename = "autoencoder_kl_spox_from_safetensors.onnx"
        with open(output_filename, "wb") as f:
            f.write(model_proto.SerializeToString())
        print(f"\nSaved complete model to {output_filename}")

    except FileNotFoundError:
        print(f"\nERROR: Could not find the safetensors file at '{SAFETENSORS_FILE_PATH}'.")
        print("Please update the SAFETENSORS_FILE_PATH variable in the script.")
    except KeyError as e:
        print(f"\n--- MODEL BUILDING FAILED ---")
        print(f"A required weight/bias was not found. This usually means there is a mismatch")
        print(f"between the model architecture defined in the script and the weights in")
        print(f"the .safetensors file.")
        print(f"\nDETAILS: {e}\n")
    except Exception as e:
        print(f"An unexpected error occurred during model building: {e}")