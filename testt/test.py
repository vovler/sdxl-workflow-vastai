import torch
import torch.nn as nn
import onnx
import os
from typing import List
from diffusers import AutoencoderKL
import traceback

@torch.jit.script
def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[2], b.shape[2], blend_extent)
    for y in range(blend_extent):
        b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
    return b

@torch.jit.script
def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[3], b.shape[3], blend_extent)
    for x in range(blend_extent):
        b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
    return b

# VAE Decoder Wrapper for ONNX export
class VaeDecoder(nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae

    @torch.jit.ignore
    def _ignored_decode(self, latent_tile: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent_tile).sample

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        r"""
        Decode a batch of images using a tiled decoder.
        """
        tile_latent_min_size = 64
        tile_sample_min_size = 64
        tile_overlap_factor = 0.25

        overlap_size = int(tile_latent_min_size * (1.0 - tile_overlap_factor))
        blend_extent = int(tile_sample_min_size * tile_overlap_factor)
        row_limit = tile_sample_min_size - blend_extent

        h_steps = list(range(0, latent.shape[2], overlap_size))
        w_steps = list(range(0, latent.shape[3], overlap_size))

        output_rows: List[torch.Tensor] = []
        prev_row_tiles: List[torch.Tensor] = []

        for i in h_steps:
            decoded_row_tiles: List[torch.Tensor] = []
            for j in w_steps:
                tile_latent = latent[:, :, i : i + tile_latent_min_size, j : j + tile_latent_min_size]
                decoded_tile = self._ignored_decode(tile_latent)
                decoded_row_tiles.append(decoded_tile)
            
            if len(prev_row_tiles) > 0:
                for j in range(len(w_steps)):
                    decoded_row_tiles[j] = blend_v(prev_row_tiles[j], decoded_row_tiles[j], blend_extent)

            stitched_row_tiles: List[torch.Tensor] = []
            for j in range(len(w_steps)):
                tile = decoded_row_tiles[j]
                if j > 0:
                    tile = blend_h(decoded_row_tiles[j - 1], tile, blend_extent)
                
                is_last_col = (j == len(w_steps) - 1)
                slice_width = tile.shape[-1] if is_last_col else row_limit
                stitched_row_tiles.append(tile[..., :slice_width])
            
            output_rows.append(torch.cat(stitched_row_tiles, dim=-1))
            prev_row_tiles = decoded_row_tiles

        final_image_cat: List[torch.Tensor] = []
        for i in range(len(h_steps)):
            row = output_rows[i]
            is_last_row = (i == len(h_steps) - 1)
            slice_height = row.shape[-2] if is_last_row else row_limit
            final_image_cat.append(row[..., :slice_height, :])

        return torch.cat(final_image_cat, dim=-2)

# Test export
def test_export(vae: AutoencoderKL):
    # VAE wrapper
    vae_decoder = torch.jit.script(VaeDecoder(vae))

    # Sample input
    latent_sample = torch.randn(1, 4, 128, 128, device="cuda", dtype=torch.float16)

    print("Testing ONNX export:")
    try:
        with torch.no_grad():
            torch.onnx.export(
                vae_decoder,
                (latent_sample,),
                "onnx/vae_decoder.onnx",
                input_names=['latent_sample'],
                output_names=['sample'],
                dynamic_axes={
                    'latent_sample': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'sample': {0: 'batch_size', 2: 'height_out', 3: 'width_out'}
                },
                opset_version=17
            )
            print("✅ VAE Decoder exported successfully to onnx/vae_decoder.onnx")
    except Exception as e:
        print(f"❌ VAE Decoder export failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    os.makedirs("onnx", exist_ok=True)
    
    with torch.no_grad():
        print("Loading original VAE model from HuggingFace...")
        # Use diffusers to load pretrained weights
        diffusers_vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16
        )
        diffusers_vae.to("cuda")
        diffusers_vae.eval()
        print("✅ Original VAE model loaded.")

        test_export(diffusers_vae)