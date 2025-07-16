import torch
import torch.nn as nn
import onnx
import os
from typing import List
from diffusers import AutoencoderKL
import traceback
import torch.nn.functional as F

@torch.jit.script
def blend_v(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Blends the bottom of tensor 'a' with the top of tensor 'b' using a
    pre-computed weight tensor. This is ONNX-safe.
    """
    blend_extent = weight.shape[2]

    # Use the "Flip Trick" to get the bottom slice of 'a'
    a_flipped = torch.flip(a, [2])
    a_sliced_flipped = a_flipped[:, :, :blend_extent, :]
    a_slice = torch.flip(a_sliced_flipped, [2])

    b_slice = b[:, :, :blend_extent, :]

    blended_slice = a_slice * (1 - weight) + b_slice * weight

    result = b.clone()
    result[:, :, :blend_extent, :] = blended_slice
    return result

@torch.jit.script
def blend_h(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Blends the right side of tensor 'a' with the left side of tensor 'b' using a
    pre-computed weight tensor. This is ONNX-safe.
    """
    blend_extent = weight.shape[3]

    # Use the "Flip Trick" to get the right slice of 'a'
    a_flipped = torch.flip(a, [3])
    a_sliced_flipped = a_flipped[:, :, :, :blend_extent]
    a_slice = torch.flip(a_sliced_flipped, [3])

    b_slice = b[:, :, :, :blend_extent]

    blended_slice = a_slice * (1 - weight) + b_slice * weight
    
    result = b.clone()
    result[:, :, :, :blend_extent] = blended_slice
    return result

class VaeDecoder(nn.Module):
    def __init__(self, traced_vae_decoder):
        super().__init__()
        self.vae_decoder = traced_vae_decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        tile_latent_size = 64
        vae_scale_factor = 8 
        tile_sample_size = tile_latent_size * vae_scale_factor
        tile_overlap_factor = 0.25

        overlap_size = int(tile_latent_size * (1.0 - tile_overlap_factor))
        blend_extent = int(tile_sample_size * tile_overlap_factor)
        
        # --- FIX: Create arange with float32 and then cast to the target dtype ---
        blend_extent_tensor_long = torch.tensor(blend_extent, device=latent.device, dtype=torch.long)
        
        # Create weight for vertical blending
        y = torch.arange(blend_extent_tensor_long, device=latent.device, dtype=torch.float32).view(1, 1, -1, 1)
        weight_v = y.to(latent.dtype) / blend_extent_tensor_long.to(latent.dtype)

        # Create weight for horizontal blending
        x = torch.arange(blend_extent_tensor_long, device=latent.device, dtype=torch.float32).view(1, 1, 1, -1)
        weight_h = x.to(latent.dtype) / blend_extent_tensor_long.to(latent.dtype)
        
        # --- Padding and Tiling Logic ---
        latent_height, latent_width = latent.shape[2], latent.shape[3]
        pad_h = (overlap_size - (latent_height - tile_latent_size) % overlap_size) % overlap_size
        pad_w = (overlap_size - (latent_width - tile_latent_size) % overlap_size) % overlap_size
        padded_latent = F.pad(latent, (0, pad_w, 0, pad_h), mode='replicate')
        padded_h, padded_w = padded_latent.shape[2], padded_latent.shape[3]

        h_steps = list(range(0, padded_h - tile_latent_size + 1, overlap_size))
        w_steps = list(range(0, padded_w - tile_latent_size + 1, overlap_size))

        output_rows: List[torch.Tensor] = []
        prev_row_tiles: List[torch.Tensor] = []

        for h_step in h_steps:
            decoded_row_tiles: List[torch.Tensor] = []
            for w_step in w_steps:
                tile_latent = padded_latent[:, :, h_step:h_step + tile_latent_size, w_step:w_step + tile_latent_size]
                decoded_tile = self.vae_decoder(tile_latent)
                decoded_row_tiles.append(decoded_tile)
            
            if len(prev_row_tiles) > 0:
                v_blended_tiles: List[torch.Tensor] = []
                for i in range(len(w_steps)):
                    blended = blend_v(prev_row_tiles[i], decoded_row_tiles[i], weight_v)
                    v_blended_tiles.append(blended)
                decoded_row_tiles = v_blended_tiles

            # Stitch the current row horizontally
            if len(decoded_row_tiles) > 1:
                row_parts: List[torch.Tensor] = []
                # First tile is blended with the second, but we only keep the left part
                first_tile = blend_h(decoded_row_tiles[0], decoded_row_tiles[1], weight_h)
                row_parts.append(first_tile[..., :-blend_extent])

                # Middle tiles are blended on both sides
                for i in range(1, len(decoded_row_tiles) - 1):
                    left_blend = blend_h(decoded_row_tiles[i-1], decoded_row_tiles[i], weight_h)
                    right_blend = blend_h(decoded_row_tiles[i], decoded_row_tiles[i+1], weight_h)
                    # We only need the middle part of the current tile after blending
                    row_parts.append(left_blend[..., blend_extent:-blend_extent])
                
                # Last tile is only blended on the left
                last_tile = blend_h(decoded_row_tiles[-2], decoded_row_tiles[-1], weight_h)
                row_parts.append(last_tile[..., blend_extent:])

                stitched_row = torch.cat(row_parts, dim=-1)
            else:
                stitched_row = decoded_row_tiles[0]
            
            output_rows.append(stitched_row)
            prev_row_tiles = decoded_row_tiles

        # Stitch all rows vertically
        if len(output_rows) > 1:
            final_parts: List[torch.Tensor] = []
            for i in range(len(output_rows) - 1):
                final_parts.append(output_rows[i][..., :-blend_extent, :])
            final_parts.append(output_rows[-1])
            stitched_image = torch.cat(final_parts, dim=-2)
        else:
            stitched_image = output_rows[0]

        # Crop to original size
        original_height_out = latent_height * vae_scale_factor
        original_width_out = latent_width * vae_scale_factor
        
        final_image = stitched_image[:, :, :original_height_out, :original_width_out]
        return final_image

def test_export(vae: AutoencoderKL):
    class VaeDecodeWrapper(nn.Module):
        def __init__(self, vae_model):
            super().__init__()
            self.vae = vae_model
        def forward(self, latents):
            return self.vae.decode(latents).sample

    tile_latent_size = 64
    dummy_latent_tile = torch.randn(1, 4, tile_latent_size, tile_latent_size, device="cuda", dtype=torch.float16)
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