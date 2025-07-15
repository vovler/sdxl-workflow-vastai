import torch
import torch.nn as nn
import onnx
import os
from typing import List
from diffusers import AutoencoderKL
import traceback
import torch.nn.functional as F

# The blend functions no longer need dynamic min calculations, as all tiles will be the same size.
# They can assume the blend_extent is always valid.
@torch.jit.script
def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """Blends the bottom of tensor 'a' with the top of tensor 'b'."""
    blend_extent_tensor = torch.tensor(blend_extent, device=a.device, dtype=torch.long)

    # Create a blending weight tensor
    y = torch.arange(blend_extent_tensor, device=a.device, dtype=a.dtype).view(1, 1, -1, 1)
    weight = y / blend_extent_tensor.to(a.dtype)

    # Slice the tensors to the blend region
    a_slice = a[:, :, a.shape[2] - blend_extent :, :]
    b_slice = b[:, :, :blend_extent, :]

    blended_slice = a_slice * (1 - weight) + b_slice * weight

    # Create a new tensor for the result to avoid in-place operations
    result = b.clone()
    result[:, :, :blend_extent, :] = blended_slice
    return result

@torch.jit.script
def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """Blends the right side of tensor 'a' with the left side of tensor 'b'."""
    blend_extent_tensor = torch.tensor(blend_extent, device=a.device, dtype=torch.long)
    
    # Create a blending weight tensor
    x = torch.arange(blend_extent_tensor, device=a.device, dtype=a.dtype).view(1, 1, 1, -1)
    weight = x / blend_extent_tensor.to(a.dtype)

    # Slice the tensors to the blend region
    a_slice = a[:, :, :, a.shape[3] - blend_extent :]
    b_slice = b[:, :, :, :blend_extent]

    blended_slice = a_slice * (1 - weight) + b_slice * weight
    
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
        Decode a batch of images using a tiled decoder with pre-padding
        to ensure all tiles have a static shape for ONNX compatibility.
        """
        tile_latent_min_size = 64
        tile_sample_min_size = 64 # This should be tile_latent_min_size * vae_scale_factor
        tile_overlap_factor = 0.25

        # --- FIX: Calculate padding needed for the latent tensor ---
        # This ensures all tiles are of size `tile_latent_min_size`
        latent_height, latent_width = latent.shape[2], latent.shape[3]
        overlap_size = int(tile_latent_min_size * (1.0 - tile_overlap_factor))
        blend_extent = int(tile_sample_min_size * tile_overlap_factor)
        row_limit = tile_sample_min_size - blend_extent

        # Calculate padding values
        pad_h = (overlap_size - (latent_height - tile_latent_min_size) % overlap_size) % overlap_size
        pad_w = (overlap_size - (latent_width - tile_latent_min_size) % overlap_size) % overlap_size

        # Pad the latent tensor. `F.pad` is more flexible for scripting.
        # The padding format is (pad_left, pad_right, pad_top, pad_bottom)
        padded_latent = F.pad(latent, (0, pad_w, 0, pad_h), mode='replicate')
        
        padded_h, padded_w = padded_latent.shape[2], padded_latent.shape[3]

        h_steps = list(range(0, padded_h - tile_latent_min_size + 1, overlap_size))
        w_steps = list(range(0, padded_w - tile_latent_min_size + 1, overlap_size))

        output_rows: List[torch.Tensor] = []
        prev_row_tiles: List[torch.Tensor] = []

        for i in h_steps:
            decoded_row_tiles: List[torch.Tensor] = []
            for j in w_steps:
                # All tiles are now guaranteed to be the same size
                tile_latent = padded_latent[:, :, i : i + tile_latent_min_size, j : j + tile_latent_min_size]
                decoded_tile = self.vae_decoder(tile_latent)
                decoded_row_tiles.append(decoded_tile)
            
            if len(prev_row_tiles) > 0:
                for j in range(len(w_steps)):
                    # The blend functions are now much simpler
                    decoded_row_tiles[j] = blend_v(prev_row_tiles[j], decoded_row_tiles[j], blend_extent)

            stitched_row_tiles: List[torch.Tensor] = []
            for j in range(len(w_steps)):
                tile = decoded_row_tiles[j]
                if j > 0:
                    tile = blend_h(decoded_row_tiles[j - 1], tile, blend_extent)
                
                # Slicing is now done with a constant `row_limit`
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

        stitched_image = torch.cat(final_image_cat, dim=-2)

        # --- FIX: Crop the final image back to the original desired size ---
        # The output dimensions must be scaled by the VAE scale factor (usually 8)
        vae_scale_factor = stitched_image.shape[2] // padded_h
        original_height_out = latent_height * vae_scale_factor
        original_width_out = latent_width * vae_scale_factor
        
        final_image = stitched_image[:, :, :original_height_out, :original_width_out]

        return final_image

# Test export
def test_export(vae: AutoencoderKL):
    # Wrapper for tracing the VAE decode method
    class VaeDecodeWrapper(nn.Module):
        def __init__(self, vae_model):
            super().__init__()
            self.vae = vae_model
        def forward(self, latents):
            # This wrapper now receives statically-sized tiles
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