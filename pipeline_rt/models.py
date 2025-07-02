import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import os
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")
runtime = trt.Runtime(TRT_LOGGER)

def trt_dtype_to_torch(dtype: trt.DataType) -> torch.dtype:
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.bfloat16:
        return torch.bfloat16
    else:
        raise TypeError(f"Unsupported TensorRT data type: {dtype}")

@dataclass
class CLIPTextOutput:
    last_hidden_state: torch.Tensor
    pooler_output: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    text_embeds: Optional[torch.Tensor] = None

class TensorRTModel:
    def __init__(self, engine_path: str, device: torch.device):
        self.device = device
        subfolder = os.path.basename(os.path.dirname(engine_path))
        filename = os.path.basename(engine_path)
        print(f"\n--- Loading TensorRT engine for: {subfolder}/{filename} ---")

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        self.input_map = {}
        self.output_map = {}
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if is_input:
                self.input_map[name] = {'dtype': dtype}
            else:
                self.output_map[name] = {'dtype': dtype}

    def __call__(self, feed_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bindings = {}
        
        for name, tensor in feed_dict.items():
            if name not in self.input_map:
                continue
            self.context.set_input_shape(name, tensor.shape)
            bindings[name] = tensor.contiguous().to(device=self.device, dtype=self.input_map[name]['dtype'])

        outputs = {}
        for name, properties in self.output_map.items():
            shape = self.context.get_tensor_shape(name)
            outputs[name] = torch.empty(tuple(shape), dtype=properties['dtype'], device=self.device)
            bindings[name] = outputs[name]
        
        stream = torch.cuda.current_stream().cuda_stream

        for name, tensor in bindings.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        self.context.execute_async_v3(stream_handle=stream)
        torch.cuda.current_stream().synchronize()
        
        return outputs


class VAEDecoder(TensorRTModel):
    def __init__(self, model_path: str, device: torch.device):
        engine_path = os.path.splitext(model_path)[0] + ".plan"
        super().__init__(engine_path, device)

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        print("--- VAEDecoder Input ---")
        print(f"latent: shape={latent.shape}, dtype={latent.dtype}, device={latent.device}")
        print(f"latent | Mean: {latent.mean():.6f} | Std: {latent.std():.6f} | Sum: {latent.sum():.6f}")
        print("------------------------")

        feed_dict = {"latent_sample": latent}
        outputs = super().__call__(feed_dict)
        return outputs["sample"]


class UNet(TensorRTModel):
    def __init__(self, model_path: str, device: torch.device):
        engine_path = os.path.splitext(model_path)[0] + ".plan"
        super().__init__(engine_path, device)

    def __call__(self, latent: torch.Tensor, timestep: torch.Tensor, text_embedding: torch.Tensor, text_embeds: torch.Tensor, time_ids: torch.Tensor) -> torch.Tensor:
        timestep = timestep.unsqueeze(0) # Add batch dimension for TRT

        print("--- UNet Inputs ---")
        print(f"latent: shape={latent.shape}, dtype={latent.dtype}, device={latent.device}")
        print(f"latent | Mean: {latent.mean():.6f} | Std: {latent.std():.6f} | Sum: {latent.sum():.6f}")
        print(f"timestep: shape={timestep.shape}, dtype={timestep.dtype}, device={timestep.device}, value: {timestep.item()}")
        print(f"text_embedding: shape={text_embedding.shape}, dtype={text_embedding.dtype}, device={text_embedding.device}")
        print(f"text_embeds: shape={text_embeds.shape}, dtype={text_embeds.dtype}, device={text_embeds.device}")
        print(f"time_ids: shape={time_ids.shape}, dtype={time_ids.dtype}, device={time_ids.device}")
        print(f"time_ids | Mean: {time_ids.mean():.6f} | Std: {time_ids.std():.6f} | Sum: {time_ids.sum():.6f}")
        print("--------------------")

        feed_dict = {
            "sample": latent,
            "timestep": timestep,
            "encoder_hidden_states": text_embedding,
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }
        
        outputs = super().__call__(feed_dict)
        return outputs["out_sample"]


class CLIPTextEncoder(TensorRTModel):
    def __init__(self, model_path: str, device: torch.device, name: str = "CLIPTextEncoder"):
        engine_path = os.path.splitext(model_path)[0] + ".plan"
        super().__init__(engine_path, device)
        self.name = name

        if self.name == "CLIP-L":
            self.last_hidden_state_name = "hidden_states.11"
            self.pooler_output_name = None
        elif self.name == "CLIP-G":
            self.last_hidden_state_name = "hidden_states.31"
            self.pooler_output_name = "text_embeds"
        else:
            raise ValueError(f"Unknown CLIP model name: {self.name}")

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        input_ids = input_ids.to(torch.int32) # TRT requires int32

        print(f"--- {self.name} ONNX Input ---")
        print(f"input_ids: shape={input_ids.shape}, dtype={input_ids.dtype}, device={input_ids.device}")
        print(f"tokens: {input_ids.flatten().tolist()}")
        if attention_mask is not None:
            print(f"attention_mask: shape={attention_mask.shape}, dtype={attention_mask.dtype}, device={attention_mask.device}")
        if output_hidden_states is not None:
            print(f"output_hidden_states: {output_hidden_states}")
        print("---------------------------")

        feed_dict = {"input_ids": input_ids}
        outputs = super().__call__(feed_dict)
        
        last_hidden_state = outputs[self.last_hidden_state_name]
        pooler_output = None
        if self.name == "CLIP-G":
            pooler_output = outputs[self.pooler_output_name]
        
        print(f"--- {self.name} ONNX Output ---")
        
        last_hidden_state_nan_count = torch.isnan(last_hidden_state).sum()
        pooler_output_nan_count = torch.isnan(pooler_output).sum() if pooler_output is not None else 0
        
        print(f"{self.last_hidden_state_name}: shape={last_hidden_state.shape}, dtype={last_hidden_state.dtype}, device={last_hidden_state.device}, nans={last_hidden_state_nan_count}/{last_hidden_state.numel()}")
        print(f"{self.last_hidden_state_name} | Mean: {last_hidden_state.mean():.6f} | Std: {last_hidden_state.std():.6f} | Sum: {last_hidden_state.sum():.6f}")
        if pooler_output is not None:
            print(f"{self.pooler_output_name}: shape={pooler_output.shape}, dtype={pooler_output.dtype}, device={pooler_output.device}, nans={pooler_output_nan_count}/{pooler_output.numel()}")
            print(f"{self.pooler_output_name} | Mean: {pooler_output.mean():.6f} | Std: {pooler_output.std():.6f} | Sum: {pooler_output.sum():.6f}")
        print("----------------------------")

        hidden_states = None
        if output_hidden_states:
            hidden_states = (last_hidden_state,)

        return CLIPTextOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=hidden_states,
            text_embeds=pooler_output,
        ) 