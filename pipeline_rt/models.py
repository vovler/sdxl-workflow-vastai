import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import os
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from polygraphy.backend.trt import TrtRunner


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
        print(f"\n--- Creating InferenceSession for: {subfolder}/{filename} ---")
        
        self.runner = TrtRunner(engine_path)
        self.runner.activate()

    def __call__(self, feed_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Convert input tensors to contiguous and ensure they are on the correct device
        for name, tensor in feed_dict.items():
            feed_dict[name] = tensor.contiguous().to(self.device)

        # Run inference
        outputs = self.runner.infer(feed_dict)
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

        feed_dict = {"latent_sample": latent.to(torch.float16)}
        outputs = super().__call__(feed_dict)
        return outputs["sample"]


class UNet(TensorRTModel):
    def __init__(self, model_path: str, device: torch.device):
        engine_path = os.path.splitext(model_path)[0] + ".plan"
        super().__init__(engine_path, device)

    def __call__(self, latent: torch.Tensor, timestep: torch.Tensor, text_embedding: torch.Tensor, text_embeds: torch.Tensor, time_ids: torch.Tensor) -> torch.Tensor:
        latent = latent.to(torch.float16)
        timestep = timestep.unsqueeze(0).to(torch.float16) # Add batch dimension for TRT
        text_embedding = text_embedding.to(torch.float16)
        text_embeds = text_embeds.to(torch.float16)
        time_ids = time_ids.to(torch.float16)

        print("--- UNet Inputs ---")
        print(f"latent: shape={latent.shape}, dtype={latent.dtype}, device={latent.device}")
        print(f"latent | Mean: {latent.mean():.6f} | Std: {latent.std():.6f} | Sum: {latent.sum():.6f}")
        print(f"timestep: shape={timestep.shape}, dtype={timestep.dtype}, device={timestep.device}, value: {timestep.item()}")
        print(f"text_embedding: shape={text_embedding.shape}, dtype={text_embedding.dtype}, device={text_embedding.device}")
        print(f"text_embedding | Mean: {text_embedding.mean():.6f} | Std: {text_embedding.std():.6f} | Sum: {text_embedding.sum():.6f}")
        print(f"text_embeds: shape={text_embeds.shape}, dtype={text_embeds.dtype}, device={text_embeds.device}")
        print(f"text_embeds | Mean: {text_embeds.mean():.6f} | Std: {text_embeds.std():.6f} | Sum: {text_embeds.sum():.6f}")
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