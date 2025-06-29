import torch
from diffusers import StableDiffusionXLPipeline
from torch.export import Dim
import os


class CLIPTextModelWrapper(torch.nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        # We only need the last_hidden_state.
        # The pipeline's CLIPTextEncoder class will handle the output.
        return self.text_encoder(input_ids, output_hidden_states=False).last_hidden_state


def main():
    """
    Exports the CLIP-L (text_encoder) of an SDXL model to ONNX.
    """
    # Using a standard SDXL model. The LoRA from unet_to_onnx.py is for the UNet.
    model_id = "socks22/sdxl-wai-nsfw-illustriousv14"
    output_path = "clip_l.onnx"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading SDXL model: {model_id}")
    # Load model and force it into FP16, then use that for all tensors.
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True
    )

    text_encoder = pipe.text_encoder
    text_encoder.to(device)
    text_encoder.eval()
    text_encoder_dtype = next(text_encoder.parameters()).dtype
    print(f"Text encoder dtype: {text_encoder_dtype}")

    print("Preparing dummy inputs for text encoder export...")
    batch_size = 1
    # SDXL uses a fixed sequence length of 77 for its text encoders.
    seq_len = 77

    input_ids_shape = (batch_size, seq_len)

    # The input_ids are integer token IDs.
    input_ids = torch.randint(
        0, text_encoder.config.vocab_size, input_ids_shape, dtype=torch.int32
    ).to(device)

    model_args = (input_ids,)

    print("Wrapping text encoder for ONNX export.")
    text_encoder_wrapper = CLIPTextModelWrapper(text_encoder)

    print("Exporting text encoder to ONNX with TorchDynamo...")

    # Define dynamic axes for the model inputs.
    batch_dim = Dim("batch_size")
    seq_dim = Dim("sequence_length")
    dynamic_shapes = {
        "input_ids": {
            0: batch_dim,
            1: seq_dim,
        },
    }

    onnx_program = torch.onnx.export(
        text_encoder_wrapper,
        model_args,
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
        opset_version=18,
    )

    # The pipeline's CLIPTextEncoder expects the output name to be "last_hidden_state".
    print("Renaming model output to 'last_hidden_state' for compatibility.")
    onnx_program.model_proto.graph.output[0].name = "last_hidden_state"

    print("\n--- ONNX Model Inputs ---")
    for i, input_proto in enumerate(onnx_program.model_proto.graph.input):
        print(f"{i}: {input_proto.name}")

    print("\n--- ONNX Model Outputs ---")
    for i, output_proto in enumerate(onnx_program.model_proto.graph.output):
        print(f"{i}: {output_proto.name}\n")

    print(f"Saving ONNX model to {output_path}...")
    onnx_program.save(output_path)

    print(f"Text encoder successfully exported to {output_path}")


if __name__ == "__main__":
    main() 