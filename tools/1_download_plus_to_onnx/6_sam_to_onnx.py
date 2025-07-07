import os
import argparse
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import onnx


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
        # Normalize
        input_image = (input_image - self.pixel_mean) / self.pixel_std
        # Forward
        image_embeddings = self.model.image_encoder(input_image)
        return image_embeddings


class CombinedSamModel(nn.Module):
    """
    This model wraps the SAM image encoder and decoder for ONNX export.
    It handles image preprocessing and combines both parts of the model.
    """
    def __init__(self, model):
        super().__init__()
        self.image_encoder = ImageEncoderOnnxModel(model)
        from segment_anything.utils.onnx import SamOnnxModel
        self.mask_decoder = SamOnnxModel(model=model, return_single_mask=True)

    def forward(
        self,
        input_image: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
        orig_im_size: torch.Tensor,
    ):
        image_embeddings = self.image_encoder(input_image)
        
        masks, iou_predictions, low_res_masks = self.mask_decoder(
            image_embeddings=image_embeddings,
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            has_mask_input=has_mask_input,
            orig_im_size=orig_im_size,
        )
        return masks, iou_predictions, low_res_masks


def export_combined_sam(sam_model, onnx_path: str, opset: int):
    """Export the combined SAM model to a single ONNX file."""
    if os.path.exists(onnx_path):
        print(f"Combined SAM ONNX file already exists at {onnx_path}, skipping export.")
        return

    print(f"Exporting combined SAM model to {onnx_path}...")
    onnx_model = CombinedSamModel(model=sam_model)
    
    # Dummy inputs
    embed_dim = sam_model.prompt_encoder.embed_dim
    embed_size = sam_model.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]

    dummy_inputs = {
        "input_image": torch.randn(1, 3, 1024, 1024, dtype=torch.float),
        "point_coords": torch.tensor([[[512, 512]]], dtype=torch.float),
        "point_labels": torch.ones((1, 1), dtype=torch.float),
        "mask_input": torch.zeros(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([0], dtype=torch.float),
        "orig_im_size": torch.tensor([1024, 1024], dtype=torch.int64),
    }

    dynamic_axes = {
        "input_image": {0: "batch_size"},
        "point_coords": {0: "batch_size"},
        "point_labels": {0: "batch_size"},
        "mask_input": {0: "batch_size"},
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
    print(f"✓ Combined SAM exported to {onnx_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Exports a SAM .pth model to a single ONNX model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/lab/model",
        help="Path to the model directory containing the SAM model.pth file.",
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

    print("--- Exporting SAM (model.pth) to ONNX ---")
    
    subfolder = "sam"
    model_dir = Path(args.model_path) / subfolder
    model_dir.mkdir(parents=True, exist_ok=True)
    
    pt_path = model_dir / "model.pth"
    onnx_path = model_dir / "sam.onnx"

    if not pt_path.exists():
        print(f"Error: SAM checkpoint not found at {pt_path}")
        # Add download logic here if needed, e.g.,
        # print("Downloading SAM checkpoint...")
        # torch.hub.download_url_to_file("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", pt_path)
        return 1

    try:
        from segment_anything import sam_model_registry
        
        sam = sam_model_registry[args.model_type](checkpoint=str(pt_path))
        
        export_combined_sam(sam, str(onnx_path), args.opset)

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
