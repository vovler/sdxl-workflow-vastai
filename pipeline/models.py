import onnxruntime as ort
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Tuple
import os
from transformers import CLIPTextModel, CLIPTextModelWithProjection


@dataclass
class ONNXCLIPTextOutput:
    last_hidden_state: torch.Tensor
    pooler_output: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    text_embeds: Optional[torch.Tensor] = None

class ONNXModel:
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        subfolder = os.path.basename(os.path.dirname(model_path))
        filename = os.path.basename(model_path)
        print(f"\\n--- Creating InferenceSession for: {subfolder}/{filename} ---")
        #so = ort.SessionOptions()
        #so.log_severity_level = 1
        provider_options = [{"device_id": self.device.index}]
        self.session = ort.InferenceSession(
            model_path, providers=[("CUDAExecutionProvider")]
        )
        self.io_binding = self.session.io_binding()
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def bind_input(self, name: str, tensor: torch.Tensor):
        tensor = tensor.contiguous()
        self.io_binding.bind_input(
            name=name,
            device_type='cuda',
            device_id=self.device.index,
            element_type=np.float16 if tensor.dtype == torch.float16 else np.float32 if tensor.dtype == torch.float32 else np.int64,
            shape=tensor.shape,
            buffer_ptr=tensor.data_ptr(),
        )

    def bind_output(self, name: str, tensor: torch.Tensor):
        tensor = tensor.contiguous()
        self.io_binding.bind_output(
            name=name,
            device_type='cuda',
            device_id=self.device.index,
            element_type=np.float16 if tensor.dtype == torch.float16 else np.float32 if tensor.dtype == torch.float32 else np.int64,
            shape=tensor.shape,
            buffer_ptr=tensor.data_ptr(),
        )

class VAEDecoder(ONNXModel):
    def __init__(self, model_path: str, device: torch.device):
        super().__init__(model_path, device)

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        self.io_binding.clear_binding_inputs()
        self.io_binding.clear_binding_outputs()
        
        print("--- VAEDecoder Input ---")
        print(f"latent: shape={latent.shape}, dtype={latent.dtype}, device={latent.device}, has_nan={torch.isnan(latent).any()}, has_inf={torch.isinf(latent).any()}")
        print("------------------------")

        self.bind_input("latent_sample", latent)
        
        output_shape = (latent.shape[0], 3, latent.shape[2] * 8, latent.shape[3] * 8)
        output_tensor = torch.empty(output_shape, dtype=latent.dtype, device=self.device)
        self.bind_output("sample", output_tensor)
        
        self.session.run_with_iobinding(self.io_binding)
        return output_tensor


class UNet(ONNXModel):
    def __init__(self, model_path: str, device: torch.device):
        super().__init__(model_path, device)

    def __call__(self, latent: torch.Tensor, timestep: torch.Tensor, text_embedding: torch.Tensor, text_embeds: torch.Tensor, time_ids: torch.Tensor) -> torch.Tensor:
        self.io_binding.clear_binding_inputs()
        self.io_binding.clear_binding_outputs()

        print("--- UNet Inputs ---")
        print(f"latent: shape={latent.shape}, dtype={latent.dtype}, device={latent.device}, has_nan={torch.isnan(latent).any()}, has_inf={torch.isinf(latent).any()}")
        print(f"timestep: shape={timestep.shape}, dtype={timestep.dtype}, device={timestep.device}, has_nan={torch.isnan(timestep).any()}, has_inf={torch.isinf(timestep).any()}")
        print(f"text_embedding: shape={text_embedding.shape}, dtype={text_embedding.dtype}, device={text_embedding.device}, has_nan={torch.isnan(text_embedding).any()}, has_inf={torch.isinf(text_embedding).any()}")
        print(f"text_embeds: shape={text_embeds.shape}, dtype={text_embeds.dtype}, device={text_embeds.device}, has_nan={torch.isnan(text_embeds).any()}, has_inf={torch.isinf(text_embeds).any()}")
        print(f"time_ids: shape={time_ids.shape}, dtype={time_ids.dtype}, device={time_ids.device}, has_nan={torch.isnan(time_ids).any()}, has_inf={torch.isinf(time_ids).any()}")
        print("--------------------")

        self.bind_input("sample", latent)
        self.bind_input("timestep", timestep.to(torch.float16))
        self.bind_input("encoder_hidden_states", text_embedding.to(torch.float16))
        self.bind_input("text_embeds", text_embeds.to(torch.float16))
        self.bind_input("time_ids", time_ids)

        output_tensor = torch.empty(latent.shape, dtype=latent.dtype, device=self.device)
        self.bind_output("out_sample", output_tensor)
        
        self.session.run_with_iobinding(self.io_binding)
        return output_tensor


class CLIPTextEncoder(ONNXModel):
    def __init__(self, model_path: str, device: torch.device, name: str = "CLIPTextEncoder"):
        super().__init__(model_path, device)
        self.name = name
        self.hidden_size = None
        self.pooler_dim = None

        self.last_hidden_state_name = "last_hidden_state"
        if "text_encoder_2" in model_path: # CLIP-G
            self.pooler_output_name = "text_embeds"
        else: # CLIP-L
            self.last_hidden_state_name = "hidden_states.11"
            self.pooler_output_name = "pooler_output"

        for output in self.session.get_outputs():
            if output.name == self.last_hidden_state_name:
                self.hidden_size = output.shape[2]
            elif output.name == self.pooler_output_name:
                self.pooler_dim = output.shape[1]

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        self.io_binding.clear_binding_inputs()
        self.io_binding.clear_binding_outputs()

        input_ids = input_ids.to(torch.int64)

        print(f"--- {self.name} ONNX Input ---")
        print(f"input_ids: shape={input_ids.shape}, dtype={input_ids.dtype}, device={input_ids.device}")
        print(f"tokens: {input_ids.flatten().tolist()}")
        if attention_mask is not None:
            print(f"attention_mask: shape={attention_mask.shape}, dtype={attention_mask.dtype}, device={attention_mask.device}")
        if output_hidden_states is not None:
            print(f"output_hidden_states: {output_hidden_states}")
        print("---------------------------")

        self.bind_input("input_ids", input_ids)

        # Prepare output tensors
        batch_size, seq_len = input_ids.shape
        
        print(f"--- {self.name} Prepared Output Shapes ---")
        last_hidden_state_shape = (batch_size, seq_len, self.hidden_size)
        print(f"last_hidden_state_shape: {last_hidden_state_shape}")
        last_hidden_state = torch.empty(last_hidden_state_shape, dtype=torch.float16, device=self.device)
        self.bind_output(self.last_hidden_state_name, last_hidden_state)
        
        pooler_output = None
        # For CLIP-L, the pooler output is not used, so we don't need to bind it.
        if self.pooler_dim is not None and "text_encoder_2" in self.session.get_outputs()[0].name:
            pooler_output_shape = (batch_size, self.pooler_dim)
            print(f"pooler_output_shape: {pooler_output_shape}")
            pooler_output = torch.empty(pooler_output_shape, dtype=torch.float16, device=self.device)
            self.bind_output(self.pooler_output_name, pooler_output)
        print("------------------------------------")

        self.session.run_with_iobinding(self.io_binding)

        print(f"--- {self.name} ONNX Output ---")
        
        last_hidden_state_nan_count = torch.isnan(last_hidden_state).sum()
        pooler_output_nan_count = torch.isnan(pooler_output).sum() if pooler_output is not None else 0
        
        print(f"{self.last_hidden_state_name}: shape={last_hidden_state.shape}, dtype={last_hidden_state.dtype}, device={last_hidden_state.device}, nans={last_hidden_state_nan_count}/{last_hidden_state.numel()}")
        if pooler_output is not None:
            print(f"{self.pooler_output_name}: shape={pooler_output.shape}, dtype={pooler_output.dtype}, device={pooler_output.device}, nans={pooler_output_nan_count}/{pooler_output.numel()}")
        print("----------------------------")

        hidden_states = None
        if output_hidden_states:
            hidden_states = (last_hidden_state, last_hidden_state)

        return ONNXCLIPTextOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=hidden_states,
            text_embeds=pooler_output,
        )

class DebugCLIPTextModel(CLIPTextModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "CLIP-L Original"

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        print(f"--- {self.name} Input ---")
        if input_ids is not None:
            print(f"input_ids: shape={input_ids.shape}, dtype={input_ids.dtype}, device={input_ids.device}")
            print(f"tokens: {input_ids.flatten().tolist()}")
        if attention_mask is not None:
            print(f"attention_mask: shape={attention_mask.shape}, dtype={attention_mask.dtype}, device={attention_mask.device}")
        if position_ids is not None:
            print(f"position_ids: shape={position_ids.shape}, dtype={position_ids.dtype}, device={position_ids.device}")
        if output_attentions is not None:
            print(f"output_attentions: {output_attentions}")
        if output_hidden_states is not None:
            print(f"output_hidden_states: {output_hidden_states}")
        if return_dict is not None:
            print(f"return_dict: {return_dict}")
        print("---------------------------")

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        print(f"--- {self.name} Output ---")
        last_hidden_state = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        last_hidden_state_nan_count = torch.isnan(last_hidden_state).sum()
        pooler_output_nan_count = torch.isnan(pooler_output).sum()
        print(f"last_hidden_state: shape={last_hidden_state.shape}, dtype={last_hidden_state.dtype}, device={last_hidden_state.device}, nans={last_hidden_state_nan_count}/{last_hidden_state.numel()}")
        print(f"pooler_output: shape={pooler_output.shape}, dtype={pooler_output.dtype}, device={pooler_output.device}, nans={pooler_output_nan_count}/{pooler_output.numel()}")
        print("----------------------------")

        return outputs

class DebugCLIPTextModelWithProjection(CLIPTextModelWithProjection):
    def __init__(self, config):
        super().__init__(config)
        self.name = "CLIP-G Original"

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        print(f"--- {self.name} Input ---")
        if input_ids is not None:
            print(f"input_ids: shape={input_ids.shape}, dtype={input_ids.dtype}, device={input_ids.device}")
            print(f"tokens: {input_ids.flatten().tolist()}")
        if attention_mask is not None:
            print(f"attention_mask: shape={attention_mask.shape}, dtype={attention_mask.dtype}, device={attention_mask.device}")
        if position_ids is not None:
            print(f"position_ids: shape={position_ids.shape}, dtype={position_ids.dtype}, device={position_ids.device}")
        if output_attentions is not None:
            print(f"output_attentions: {output_attentions}")
        if output_hidden_states is not None:
            print(f"output_hidden_states: {output_hidden_states}")
        if return_dict is not None:
            print(f"return_dict: {return_dict}")
        print("---------------------------")

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        print(f"--- {self.name} Output ---")
        last_hidden_state = outputs.last_hidden_state
        text_embeds = outputs.text_embeds
        last_hidden_state_nan_count = torch.isnan(last_hidden_state).sum()
        text_embeds_nan_count = torch.isnan(text_embeds).sum()
        print(f"last_hidden_state: shape={last_hidden_state.shape}, dtype={last_hidden_state.dtype}, device={last_hidden_state.device}, nans={last_hidden_state_nan_count}/{last_hidden_state.numel()}")
        print(f"text_embeds: shape={text_embeds.shape}, dtype={text_embeds.dtype}, device={text_embeds.device}, nans={text_embeds_nan_count}/{text_embeds.numel()}")
        print("----------------------------")
        
        return outputs 