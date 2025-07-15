import torch
import torch.nn as nn
import onnx
import os
from typing import List
from diffusers import AutoencoderKL
import traceback
import torch.nn.functional as F

@torch.jit.script
def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """
    Blends the bottom of tensor 'a' with the top of tensor 'b' using a
    static slicing method ("Flip Trick") for ONNX compatibility.
    """
    blend_extent_tensor = torch.tensor(blend_extent, device=a.device, dtype=torch.long)

    # Create a blending weight tensor (this is fine as blend_extent is a constant)
    y = torch.arange(blend_extent_tensor, device=a.device, dtype=a.dtype).view(1, 1, -1, 1)
    weight = y / blend_extent_tensor.to(a.dtype)

    # --- FIX: Use the "Flip Trick" to avoid dynamic start_index calculation ---
    a_flipped = torch.flip(a, [2])
    a_sliced_flipped = a_flipped[:, :, :blend_extent, :]
    a_slice = torch.flip(a_sliced_flipped, [2])

    b_slice = b[:, :, :blend_extent, :] # This slice is already static

    blended_slice = a_slice * (1 - weight) + b_slice * weight

    result = b.clone()
    result[:, :, :blend_extent, :] = blended_slice
    return result

@torch.jit.script
def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """
    Blends the right side of tensor 'a' with the left side of tensor 'b' using a
    static slicing method ("Flip Trick") for ONNX compatibility.
    """
    blend_extent_tensor = torch.tensor(blend_extent, device=a.device, dtype=torch.long)
    
    x = torch.arange(blend_extent_tensor, device=a.device, dtype=a.dtype).view(1, 1, 1, -1)
    weight = x / blend_extent_tensor.to(a.dtype)

    # --- FIX: Use the "Flip Trick" to avoid dynamic start_index calculation ---
    a_flipped = torch.flip(a, [3])
    a_sliced_flipped = a_flipped[:, :, :, :blend_extent]
    a_slice = torch.flip(a_sliced_flipped, [3])

    b_slice = b[:, :, :, :blend_extent] # This slice is already static

    blended_slice = a_slice * (1 - weight) + b_slice * weight
    
    result = b.clone()
    result[:, :, :, :blend_extent] = blended_slice
    return result

# VAE Decoder Wrapper for ONNX export
class VaeDecoder(nn.Module):
    def __init__(self, traced_vae_decoder):
        super().__init__()
        self.vae_decoder = traced_vae_decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        tile_latent_min_size = 64
        # Assuming VAE scale factor is 8 (standard for Stable Diffusion models)
        vae_scale_factor = 8 
        tile_sample_min_size = tile_latent_min_size * vae_scale_factor
        tile_overlap_factor = 0.25

        # These values are now constants, which is critical for ONNX export
        overlap_size = int(tile_latent_min_size * (1.0 - tile_overlap_factor))
        blend_extent = int(tile_sample_min_size * tile_overlap_factor)
        
        # --- Padding Strategy ---
        latent_height, latent_width = latent.shape[2], latent.shape[3]
        
        # Calculate padding needed so that the padded dimensions are a multiple of overlap_size
        pad_h = (overlap_size - (latent_height - tile_latent_min_size) % overlap_size) % overlap_size
        pad_w = (overlap_size - (latent_width - tile_latent_min_size) % overlap_size) % overlap_size

        # Use F.pad which is ONNX-friendly
        padded_latent = F.pad(latent, (0, pad_w, 0, pad_h), mode='replicate')
        
        padded_h, padded_w = padded_latent.shape[2], padded_latent.shape[3]

        # All loops and slicing logic from here now operate on static shapes and constants
        h_steps = range(0, padded_h - tile_latent_min_size + 1, overlap_size)
        w_steps = range(0, padded_w - tile_latent_min_size + 1, overlap_size)

        output_rows = []
        # Initializing prev_row_tiles for the first iteration
        prev_row_tiles = [torch.zeros(1, 3, tile_sample_min_size, tile_sample_min_size, dtype=latent.dtype, device=latent.device) for _ in w_steps]

        for i, h_step in enumerate(h_steps):
            decoded_row_tiles = []
            for w_step in w_steps:
                tile_latent = padded_latent[:, :, h_step:h_step + tile_latent_min_size, w_step:w_step + tile_latent_min_size]
                decoded_tile = self.vae_decoder(tile_latent)
                decoded_row_tiles.append(decoded_tile)
            
            if i > 0:
                for j in range(len(w_steps)):
                    decoded_row_tiles[j] = blend_v(prev_row_tiles[j], decoded_row_tiles[j], blend_extent)

            row_cat_list = []
            if len(decoded_row_tiles) > 1:
                # Handle all but the last tile in a separate loop for TorchScript compatibility
                for j in range(len(decoded_row_tiles) - 1):
                    tile = decoded_row_tiles[j]
                    if j > 0:
                        tile = blend_h(decoded_row_tiles[j-1], tile, blend_extent)
                    row_cat_list.append(tile[..., :-blend_extent]) # Use negative index for static slicing
                
                # Handle the last tile
                last_tile = blend_h(decoded_row_tiles[-2], decoded_row_tiles[-1], blend_extent)
                row_cat_list.append(last_tile)
            else:
                 row_cat_list.append(decoded_row_tiles[0])
            
            output_rows.append(torch.cat(row_cat_list, dim=-1))
            prev_row_tiles = decoded_row_tiles

        # Stitch the rows together
        if len(output_rows) > 1:
            stitched_image = torch.cat(output_rows, dim=-2)
        else:
            stitched_image = output_rows[0]

        # Crop the padded final image back to the original desired output size
        original_height_out = latent_height * vae_scale_factor
        original_width_out = latent_width * vae_scale_factor
        
        final_image = stitched_image[:, :, :original_height_out, :original_width_out]

        return final_image

# Test export
def test_export(vae: AutoencoderKL):
    class VaeDecodeWrapper(nn.Module):
        def __init__(self, vae_model):
            super().__init__()
            self.vae = vae_model
        def forward(self, latents):
            return self.vae.decode(latents).sample

    tile_latent_min_size = 64
    dummy_latent_tile = torch.randn(1, 4, tile_latent_min_size, tile_latent_min_size, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        traced_vae_decoder = torch.jit.trace(VaeDecodeWrapper(vae), dummy_latent_tile)

    vae_decoder = torch.jit.script(VaeDecoder(traced_vae_decoder))
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
        diffusers_vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16
        )
        diffusers_vae.to("cuda")
        diffusers_vae.eval()
        print("✅ Original VAE model loaded.")

        test_export(diffusers_vae)