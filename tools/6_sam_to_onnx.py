import os
import argparse
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoderOnnxModel(nn.Module):
    """
    This model wraps the SAM image encoder for ONNX export.
    It handles image preprocessing, including resizing and normalization.
    """

    def __init__(
        self,
        model,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    ):
        super().__init__()
        self.model = model
        self.img_size = model.image_encoder.img_size
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False
        )

    def forward(self, input_image: torch.Tensor):
        # Resize
        input_image = F.interpolate(
            input_image,
            (self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        # Normalize
        input_image = (input_image - self.pixel_mean) / self.pixel_std
        # Forward
        image_embeddings = self.model.image_encoder(input_image)
        return image_embeddings


def export_encoder(sam_model, onnx_path: str, opset: int):
    """Export the SAM image encoder to an ONNX model."""
    if os.path.exists(onnx_path):
        print(f"Encoder ONNX file already exists at {onnx_path}, skipping export.")
        return

    print(f"Exporting encoder to {onnx_path}...")
    onnx_model = ImageEncoderOnnxModel(model=sam_model)
    
    # input_image shape: (N, 3, H, W)
    dummy_input = {"input_image": torch.randn(1, 3, 1024, 1024, dtype=torch.float)}
    dynamic_axes = {
        "input_image": {0: "batch_size", 2: "height", 3: "width"},
    }
    output_names = ["image_embeddings"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_input.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_input.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
    print(f"✓ Encoder exported to {onnx_path}")


def export_decoder(sam_model, onnx_path: str, opset: int):
    """Export the SAM prompt encoder and mask decoder to an ONNX model."""
    if os.path.exists(onnx_path):
        print(f"Decoder ONNX file already exists at {onnx_path}, skipping export.")
        return

    print(f"Exporting decoder to {onnx_path}...")
    from segment_anything.utils.onnx import SamOnnxModel

    onnx_model = SamOnnxModel(model=sam_model, return_single_mask=True)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam_model.prompt_encoder.embed_dim
    embed_size = sam_model.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }

    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
    print(f"✓ Decoder exported to {onnx_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Exports a SAM .pth model to ONNX format, creating separate encoder and decoder models."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/lab/model",
        help="Path to the model directory containing the SAM sam_b.pt file.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vit_b",
        help="The type of SAM model to export (e.g., 'vit_b', 'vit_l', 'vit_h').",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="The ONNX opset version to use.",
    )
    args = parser.parse_args()

    print("--- Exporting SAM (sam_b.pt) to ONNX ---")
    
    subfolder = "sam"
    model_dir = Path(args.model_path) / subfolder
    model_dir.mkdir(parents=True, exist_ok=True)
    
    pt_path = model_dir / "sam_b.pt"
    encoder_onnx_path = model_dir / "encoder.onnx"
    decoder_onnx_path = model_dir / "decoder.onnx"

    if not pt_path.exists():
        print(f"Error: SAM checkpoint not found at {pt_path}")
        # Add download logic here if needed, e.g.,
        # print("Downloading SAM checkpoint...")
        # torch.hub.download_url_to_file("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", pt_path)
        return 1

    try:
        from segment_anything import sam_model_registry
        
        sam = sam_model_registry[args.model_type](checkpoint=str(pt_path))
        
        export_encoder(sam, str(encoder_onnx_path), args.opset)
        export_decoder(sam, str(decoder_onnx_path), args.opset)

        # Cleanup original .pth file
        print(f"\nCleaning up original PyTorch model: {pt_path.name}")
        try:
            os.remove(pt_path)
            print(f"✓ Removed {pt_path.name}")
        except OSError as e:
            print(f"✗ Error deleting .pth file: {e}")

        print("\nSAM ONNX export successful.")
        return 0
    except ImportError:
        print("✗ Error: `segment-anything` package not found.")
        print("Please install it with: pip install segment-anything")
        return 1
    except Exception as e:
        print(f"✗ An error occurred during SAM ONNX export: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
