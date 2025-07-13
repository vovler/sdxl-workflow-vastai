# Obviously modified from the original source code
# https://github.com/huggingface/diffusers
# So has APACHE 2.0 license

# Author : Simo Ryu

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class DiagonalGaussianDistribution:
    def __init__(self, parameters):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.std)

    def mode(self):
        return self.mean

class AttentionBlock(nn.Module):
    def __init__(self, channels, norm_num_groups=32):
        super().__init__()
        self.group_norm = nn.GroupNorm(norm_num_groups, channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Sequential(nn.Linear(channels, channels))

    def forward(self, hidden_states):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape
        
        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        q = q.view(batch, -1, 1, channel)
        k = k.view(batch, -1, 1, channel)
        v = v.view(batch, -1, 1, channel)
        
        q,k,v = [t.transpose(1, 2) for t in (q,k,v)]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (channel ** 0.5)
        attn_probs = F.softmax(scores, dim=-1)
        hidden_states = torch.matmul(attn_probs, v)
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch, -1, channel)
        hidden_states = self.to_out(hidden_states)
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch, channel, height, width)

        return hidden_states + residual

class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, norm_num_groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(norm_num_groups, in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(norm_num_groups, out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor):
        hidden_states = input_tensor
        
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        return input_tensor + hidden_states

class VAEMidBlock(nn.Module):
    def __init__(self, in_channels, norm_num_groups=32):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels, in_channels, norm_num_groups=norm_num_groups),
            ResnetBlock2D(in_channels, in_channels, norm_num_groups=norm_num_groups),
        ])
        self.attentions = nn.ModuleList([
            AttentionBlock(in_channels, norm_num_groups=norm_num_groups),
        ])

    def forward(self, hidden_states):
        hidden_states = self.resnets[0](hidden_states)
        hidden_states = self.attentions[0](hidden_states)
        hidden_states = self.resnets[1](hidden_states)
        return hidden_states

class Downsample(nn.Module):
    def __init__(self, channels, use_conv=True, padding=1):
        super().__init__()
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=padding)
        else:
            self.conv = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, hidden_states):
        assert hidden_states.shape[2] % 2 == 0 and hidden_states.shape[3] % 2 == 0
        if isinstance(self.conv, nn.Conv2d):
            hidden_states = self.conv(hidden_states)
        else:
            hidden_states = self.conv(hidden_states)
        return hidden_states

class Upsample(nn.Module):
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[2] % 2 == 0 and hidden_states.shape[3] % 2 == 0
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")
        
        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states

class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, add_downsample=True, norm_num_groups=32):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_channels_ = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels_, out_channels, norm_num_groups=norm_num_groups))
        self.resnets = nn.ModuleList(resnets)

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                Downsample(out_channels, use_conv=True)
            ])

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states

class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, add_upsample=True, norm_num_groups=32):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_channels_ = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels_, out_channels, norm_num_groups=norm_num_groups))
        self.resnets = nn.ModuleList(resnets)

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                Upsample(out_channels, use_conv=True)
            ])

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        
        return hidden_states

class Encoder(nn.Module):
    def __init__(self, in_channels, down_block_types, block_out_channels, layers_per_block, norm_num_groups, latent_channels, mid_block_add_attention):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            
            if down_block_type == "DownEncoderBlock2D":
                down_block = DownEncoderBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block,
                    add_downsample=not is_final_block,
                    norm_num_groups=norm_num_groups,
                )
                self.down_blocks.append(down_block)
        
        self.mid_block = None
        if mid_block_add_attention:
            self.mid_block = VAEMidBlock(block_out_channels[-1], norm_num_groups=norm_num_groups)
            
        self.conv_norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[-1], eps=1e-6, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[-1], 2 * latent_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)

        if self.mid_block is not None:
            x = self.mid_block(x)
        
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels, up_block_types, block_out_channels, layers_per_block, norm_num_groups, latent_channels, mid_block_add_attention):
        super().__init__()
        
        self.conv_in = nn.Conv2d(latent_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        if mid_block_add_attention:
            self.mid_block = VAEMidBlock(block_out_channels[-1], norm_num_groups=norm_num_groups)

        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            
            is_final_block = i == len(block_out_channels) - 1
            
            if up_block_type == "UpDecoderBlock2D":
                up_block = UpDecoderBlock2D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block + 1,
                    add_upsample=not is_final_block,
                    norm_num_groups=norm_num_groups,
                )
                self.up_blocks.append(up_block)
        
        self.conv_norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[0], eps=1e-6, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        z = self.conv_in(z)

        if self.mid_block is not None:
            z = self.mid_block(z)
        
        for block in self.up_blocks:
            z = block(z)

        z = self.conv_norm_out(z)
        z = self.conv_act(z)
        z = self.conv_out(z)
        
        return z

class AutoEncoderKL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
            in_channels=config["in_channels"],
            down_block_types=config["down_block_types"],
            block_out_channels=config["block_out_channels"],
            layers_per_block=config["layers_per_block"],
            norm_num_groups=config["norm_num_groups"],
            latent_channels=config["latent_channels"],
            mid_block_add_attention=config.get("mid_block_add_attention", True)
        )
        self.decoder = Decoder(
            out_channels=config["out_channels"],
            up_block_types=config["up_block_types"],
            block_out_channels=config["block_out_channels"],
            layers_per_block=config["layers_per_block"],
            norm_num_groups=config["norm_num_groups"],
            latent_channels=config["latent_channels"],
            mid_block_add_attention=config.get("mid_block_add_attention", True),
        )
        
        if config.get("use_quant_conv", True):
            self.quant_conv = nn.Conv2d(2 * config["latent_channels"], 2 * config["latent_channels"], 1)
        else:
            self.quant_conv = nn.Identity()

        if config.get("use_post_quant_conv", True):
            self.post_quant_conv = nn.Conv2d(config["latent_channels"], config["latent_channels"], 1)
        else:
            self.post_quant_conv = nn.Identity()

        self.tile_decode = False
        self.tile_sample_min_size = config.get("sample_size", 512)
        self.tile_overlap_factor = config.get("tile_overlap_factor", 0.25)
        self.scale_factor = 2 ** (len(config["block_out_channels"]) - 1)
        self.tile_latent_min_size = self.tile_sample_min_size // self.scale_factor


    def enable_tiling(self, use_tiling=True):
        self.tile_decode = use_tiling

    def disable_tiling(self):
        self.enable_tiling(False)

    def blend_h(self, a, b, blend_extent):
        b_shape = torch.tensor(b.shape, device=b.device)
        a_shape = torch.tensor(a.shape, device=a.device)

        blend_amount = torch.minimum(b_shape[3], a_shape[3])
        blend_amount = torch.minimum(blend_amount, torch.tensor(blend_extent, device=b.device))

        ramp = torch.linspace(0.0, 1.0, blend_extent, device=a.device, dtype=torch.float32).view(1, 1, 1, blend_extent)
        
        a_fp32 = a.to(torch.float32)
        b_fp32 = b.to(torch.float32)

        slice_a = a_fp32.narrow(3, a_shape[3] - blend_amount, blend_amount)
        slice_b = b_fp32.narrow(3, 0, blend_amount)
        
        sliced_ramp = ramp.narrow(3, blend_extent - blend_amount, blend_amount)
        
        blended_slice = slice_a * (1.0 - sliced_ramp) + slice_b * sliced_ramp
        
        b_fp32.narrow(3, 0, blend_amount).copy_(blended_slice)
        
        return b_fp32.to(a.dtype)

    def blend_v(self, a, b, blend_extent):
        b_shape = torch.tensor(b.shape, device=b.device)
        a_shape = torch.tensor(a.shape, device=a.device)

        blend_amount = torch.minimum(b_shape[2], a_shape[2])
        blend_amount = torch.minimum(blend_amount, torch.tensor(blend_extent, device=b.device))
        
        ramp = torch.linspace(0.0, 1.0, blend_extent, device=a.device, dtype=torch.float32).view(1, 1, blend_extent, 1)
        
        a_fp32 = a.to(torch.float32)
        b_fp32 = b.to(torch.float32)

        slice_a = a_fp32.narrow(2, a_shape[2] - blend_amount, blend_amount)
        slice_b = b_fp32.narrow(2, 0, blend_amount)
        
        sliced_ramp = ramp.narrow(2, blend_extent - blend_amount, blend_amount)
        
        blended_slice = slice_a * (1.0 - sliced_ramp) + slice_b * sliced_ramp
        
        b_fp32.narrow(2, 0, blend_amount).copy_(blended_slice)
        
        return b_fp32.to(a.dtype)

    def tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        Decode a batch of images using a tiled decoder.
        """
        overlap_size = self.tile_latent_min_size * (100 - int(self.tile_overlap_factor * 100)) // 100
        blend_extent = self.tile_sample_min_size * int(self.tile_overlap_factor * 100) // 100
        row_limit = self.tile_sample_min_size - blend_extent

        output_rows: List[torch.Tensor] = []
        prev_row_tiles: Optional[List[torch.Tensor]] = None

        i = 0
        while i < z.shape[2]:
            decoded_row_tiles: List[torch.Tensor] = []
            j = 0
            while j < z.shape[3]:
                tile_z = z[:, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                decoded_tile = self.decoder(self.post_quant_conv(tile_z))
                decoded_row_tiles.append(decoded_tile)
                j += overlap_size

            if prev_row_tiles is not None:
                for k in range(len(decoded_row_tiles)):
                    decoded_row_tiles[k] = self.blend_v(prev_row_tiles[k], decoded_row_tiles[k], blend_extent)

            stitched_row_tiles: List[torch.Tensor] = []
            for k in range(len(decoded_row_tiles)):
                tile = decoded_row_tiles[k]
                if k > 0:
                    tile = self.blend_h(decoded_row_tiles[k - 1], tile, blend_extent)
                
                is_last_col = k == len(decoded_row_tiles) - 1
                
                slice_width = tile.shape[3]
                if not is_last_col:
                    slice_width = row_limit

                stitched_row_tiles.append(tile[:, :, :, :slice_width])
            
            output_rows.append(torch.cat(stitched_row_tiles, dim=-1))
            prev_row_tiles = decoded_row_tiles
            i += overlap_size

        final_image_cat: List[torch.Tensor] = []
        for i in range(len(output_rows)):
            row = output_rows[i]
            is_last_row = i == len(output_rows) - 1
            
            slice_height = row.shape[2]
            if not is_last_row:
                slice_height = row_limit

            final_image_cat.append(row[:, :, :slice_height, :])

        if len(final_image_cat) == 0:
            return torch.zeros(z.shape[0], self.config["out_channels"], 0, 0, device=z.device, dtype=z.dtype)

        return torch.cat(final_image_cat, dim=-2)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution(moments)

    
    def decode(self, z):
        if self.tile_decode:
            return self.tiled_decode(z)

        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, sample, sample_posterior=False):
        posterior = self.encode(sample)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec
