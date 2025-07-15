import torch
import torch.nn as nn
import onnx
import os
from typing import List
from diffusers import AutoencoderKL
import traceback

@torch.jit.script
def _onnx_friendly_min(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Calculates min(a, b) using a mathematical formula that is more stable for
    ONNX export than torch.min or Python's min() when dealing with dynamic shapes.
    """
    return (0.5 * (a + b - torch.abs(a - b))).to(torch.long)

@torch.jit.script
def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent_in: int) -> torch.Tensor:
    # Explicitly convert all Python numbers and shape values to 0-dim tensors
    shape_a_dim = torch.tensor(a.shape[2], device=a.device, dtype=torch.long)
    shape_b_dim = torch.tensor(b.shape[2], device=a.device, dtype=torch.long)
    blend_extent_in_tensor = torch.tensor(blend_extent_in, device=a.device, dtype=torch.long)

    # --- ONNX EXPORT FIX: Replace torch.min with a math-based equivalent ---
    min_of_shapes = _onnx_friendly_min(shape_a_dim, shape_b_dim)
    blend_extent = _onnx_friendly_min(min_of_shapes, blend_extent_in_tensor)

    if blend_extent == 0:
        return b

    # Create a blending weight tensor
    y = torch.arange(blend_extent, device=a.device, dtype=a.dtype).view(1, 1, -1, 1)
    weight = y / blend_extent.to(a.dtype)

    # Use the "Flip Trick" to avoid dynamic start_index calculation
    a_flipped = torch.flip(a, [2])
    a_sliced = a_flipped[:, :, :blend_extent, :]
    a_restored = torch.flip(a_sliced, [2])

    blended_slice = a_restored * (1 - weight) + b[:, :, :blend_extent, :] * weight

    # Create a new tensor for the result to avoid in-place operations
    result = b.clone()
    result[:, :, :blend_extent, :] = blended_slice
    return result

@torch.jit.script
def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent_in: int) -> torch.Tensor:
    # Explicitly convert all Python numbers and shape values to 0-dim tensors
    shape_a_dim = torch.tensor(a.shape[3], device=a.device, dtype=torch.long)
    shape_b_dim = torch.tensor(b.shape[3], device=a.device, dtype=torch.long)
    blend_extent_in_tensor = torch.tensor(blend_extent_in, device=a.device, dtype=torch.long)

    # --- ONNX EXPORT FIX: Replace torch.min with a math-based equivalent ---
    min_of_shapes = _onnx_friendly_min(shape_a_dim, shape_b_dim)
    blend_extent = _onnx_friendly_min(min_of_shapes, blend_extent_in_tensor)

    if blend_extent == 0:
        return b

    # Create a blending weight tensor
    x = torch.arange(blend_extent, device=a.device, dtype=a.dtype).view(1, 1, 1, -1)
    weight = x / blend_extent.to(a.dtype)

    # Use the "Flip Trick" to avoid dynamic start_index calculation
    a_flipped = torch.flip(a, [3])
    a_sliced = a_flipped[:, :, :, :blend_extent]
    a_restored = torch.flip(a_sliced, [3])

    blended_slice = a_restored * (1 - weight) + b[:, :, :, :blend_extent] * weight
    
    # Create a new tensor for the result to avoid in-place operations
    result = b.clone()
    result[:, :, :, :blend_extent] = blended_slice
    return result

# VAE Decoder Wrapper for ONNX export
class VaeDecoder(nn.Module):
    def __init__(self, traced_vae_decoder):
        super().__init__()
        self.vae_decoder = traced_vae_decoder

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
                decoded_tile = self.vae_decoder(tile_latent)
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
    # Wrapper for tracing the VAE decode method
    class VaeDecodeWrapper(nn.Module):
        def __init__(self, vae_model):
            super().__init__()
            self.vae = vae_model
        def forward(self, latents):
            return self.vae.decode(latents).sample

    # Trace the VAE decoder part to get a scriptable graph
    tile_latent_min_size = 64
    dummy_latent_tile = torch.randn(1, 4, tile_latent_min_size, tile_latent_min_size, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        traced_vae_decoder = torch.jit.trace(VaeDecodeWrapper(vae), dummy_latent_tile)

    # Script the VAE tiling wrapper with the traced decoder
    vae_decoder = torch.jit.script(VaeDecoder(traced_vae_decoder))

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
                opset_version=20
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