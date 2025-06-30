import onnxruntime as ort
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ONNXCLIPTextOutput:
    last_hidden_state: torch.Tensor
    pooler_output: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    text_embeds: Optional[torch.Tensor] = None

class ONNXModel:
    def __init__(self, model_path: str):
        #so = ort.SessionOptions()
        #so.log_severity_level = 0
        self.session = ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider"]
        )

    def __call__(self, **kwargs):
        inputs = {
            input.name: np.ascontiguousarray(value)
            for input, value in zip(self.session.get_inputs(), kwargs.values())
        }
        return self.session.run(None, inputs)


class VAEDecoder(ONNXModel):
    def __init__(self, model_path: str):
        super().__init__(model_path)

    def __call__(self, latent: np.ndarray) -> np.ndarray:
        return super().__call__(latent_sample=latent)[0]


class UNet(ONNXModel):
    def __init__(self, model_path: str):
        super().__init__(model_path)

    def __call__(self, latent: np.ndarray, timestep: np.ndarray, text_embedding: np.ndarray, text_embeds: np.ndarray, time_ids: np.ndarray) -> np.ndarray:
        return super().__call__(
            sample=latent,
            timestep=timestep,
            encoder_hidden_states=text_embedding,
            text_embeds=text_embeds,
            time_ids=time_ids,
        )[0]


class CLIPTextEncoder:
    def __init__(self, model_path: str, device: str = "cuda"):
        #so = ort.SessionOptions()
        #so.log_severity_level = 0
        self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
        self.device = torch.device(device)
        self.output_names = [o.name for o in self.session.get_outputs()]

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        input_feed = {"input_ids": input_ids.cpu().numpy()}
        
        outputs = self.session.run(self.output_names, input_feed)
        outputs_map = dict(zip(self.output_names, outputs))

        last_hidden_state = torch.from_numpy(outputs_map['last_hidden_state']).to(self.device)
        
        pooler_output = None
        if "pooler_output" in outputs_map:
            pooler_output = torch.from_numpy(outputs_map['pooler_output']).to(self.device)

        hidden_states = None
        if output_hidden_states:
            # Creating a mock hidden_states tuple for clip-skip compatibility.
            # It contains the last_hidden_state twice to simulate penultimate and last layers.
            hidden_states = (last_hidden_state, last_hidden_state)

        return ONNXCLIPTextOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=hidden_states,
            text_embeds=pooler_output, # For SDXL, compel looks for text_embeds
        ) 