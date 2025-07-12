# Obviously modified from the original source code
# https://github.com/huggingface/diffusers
# So has APACHE 2.0 license

# Author : Simo Ryu

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.to_out = nn.Linear(channels, channels)

    @torch.no_grad()
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
        
        q,k,v = map(lambda t: t.transpose(1, 2), (q,k,v))
        
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

    @torch.no_grad()
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
            self.mid_block = nn.ModuleList([
                ResnetBlock2D(block_out_channels[-1], block_out_channels[-1], norm_num_groups=norm_num_groups),
                AttentionBlock(block_out_channels[-1], norm_num_groups=norm_num_groups),
                ResnetBlock2D(block_out_channels[-1], block_out_channels[-1], norm_num_groups=norm_num_groups),
            ])
            
        self.norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[-1], eps=1e-6, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[-1], 2 * latent_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)

        if self.mid_block is not None:
            x = self.mid_block[0](x)
            x = self.mid_block[1](x)
            x = self.mid_block[2](x)
        
        x = self.norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels, up_block_types, block_out_channels, layers_per_block, norm_num_groups, latent_channels, mid_block_add_attention):
        super().__init__()
        
        self.conv_in = nn.Conv2d(latent_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        if mid_block_add_attention:
            self.mid_block = nn.ModuleList([
                ResnetBlock2D(block_out_channels[-1], block_out_channels[-1], norm_num_groups=norm_num_groups),
                AttentionBlock(block_out_channels[-1], norm_num_groups=norm_num_groups),
                ResnetBlock2D(block_out_channels[-1], block_out_channels[-1], norm_num_groups=norm_num_groups),
            ])

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
        
        self.norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[0], eps=1e-6, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        z = self.conv_in(z)

        if self.mid_block is not None:
            z = self.mid_block[0](z)
            z = self.mid_block[1](z)
            z = self.mid_block[2](z)
        
        for block in self.up_blocks:
            z = block(z)

        z = self.norm_out(z)
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
        self.tile_size = config.get("tile_size", 512)
        self.tile_stride = config.get("tile_stride", 256)

    def enable_tiling(self, use_tiling=True):
        self.tile_decode = use_tiling

    def disable_tiling(self):
        self.enable_tiling(False)

    @torch.no_grad()
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution(moments)

    @torch.no_grad()
    def decode(self, z):
        if self.tile_decode:
            return self.tiled_decode(z)

        z = self.post_quant_conv(z)
        return self.decoder(z)
    
    def tiled_decode(self, z):
        r"""
        Decode a batch of images using a tiled decoder.
        """
        
        # ... implementation of tiled decoding ...
        # This is a simplified version of the logic in diffusers
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py#L422
        
        batch_size, channels, height, width = z.shape
        
        # calculate the number of tiles
        num_tiles_h = (height - self.tile_size) // self.tile_stride + 1
        num_tiles_w = (width - self.tile_size) // self.tile_stride + 1
        
        # output and blending mask
        output = torch.zeros(batch_size, self.config["out_channels"], height * 8, width * 8, device=z.device)
        blend_mask = torch.zeros(batch_size, 1, height * 8, width * 8, device=z.device)
        
        for i in range(num_tiles_h):
            for j in range(num_tiles_w):
                # get the current tile
                h_start = i * self.tile_stride
                h_end = h_start + self.tile_size
                w_start = j * self.tile_stride
                w_end = w_start + self.tile_size
                
                tile_z = z[:, :, h_start:h_end, w_start:w_end]
                
                # decode the tile
                decoded_tile = self.decoder(self.post_quant_conv(tile_z))
                
                # blend the tile into the output
                output_h_start = h_start * 8
                output_h_end = h_end * 8
                output_w_start = w_start * 8
                output_w_end = w_end * 8
                
                output[:, :, output_h_start:output_h_end, output_w_start:output_w_end] += decoded_tile
                blend_mask[:, :, output_h_start:output_h_end, output_w_start:output_w_end] += 1
        
        return output / blend_mask

    def forward(self, sample, sample_posterior=False):
        posterior = self.encode(sample)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec
