import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import EqualizedConv2d, EqualizedLinear, AdaptiveInstanceNorm, NoiseInjection
from .mapping import MappingNetwork


class StyleConv(nn.Module):
    """
    Standard Style Convolution block:
    EqualizedConv2d -> NoiseInjection -> LeakyReLU -> AdaIN
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, style_dim: int):
        super().__init__()
        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.noise = NoiseInjection(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.adain = AdaptiveInstanceNorm(out_channels, style_dim)

    def forward(self, x: torch.Tensor, style_w: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        out = self.conv(x)
        out = self.noise(out, noise)
        out = self.act(out)
        out = self.adain(out, style_w)
        return out


class ToRGB(nn.Module):
    """
    Projects feature maps to 3-channel RGB image.
    """
    def __init__(self, in_channels: int, style_dim: int):
        super().__init__()
        self.conv = EqualizedConv2d(in_channels, 3, kernel_size=1, padding=0, gain=1.0)
        self.adain = AdaptiveInstanceNorm(3, style_dim)

    def forward(self, x: torch.Tensor, style_w: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.adain(out, style_w)
        return torch.tanh(out)


class SynthesisBlock(nn.Module):
    """
    Synthesis Block that doubles spatial resolution via bilinear upsampling
    and applies two StyleConvs.
    """
    def __init__(self, in_channels: int, out_channels: int, style_dim: int):
        super().__init__()
        self.conv1 = StyleConv(in_channels, out_channels, kernel_size=3, style_dim=style_dim)
        self.conv2 = StyleConv(out_channels, out_channels, kernel_size=3, style_dim=style_dim)

    def forward(self, x: torch.Tensor, style_w: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv1(x, style_w, noise)
        x = self.conv2(x, style_w, noise)
        return x


class Generator(nn.Module):
    """
    StyleGAN Generator.
    Combines Mapping Network f: Z -> W and Synthesis Network g: W -> Image.
    Supports:
    - Learned constant input (4x4)
    - AdaIN conditioning from intermediate style space W
    - Per-layer noise injection
    - Style mixing regularization
    - Truncation trick in W space
    """
    def __init__(
        self,
        image_size: int = 64,
        latent_dim: int = 512,
        style_dim: int = 512,
        max_channels: int = 512,
        mapping_layers: int = 8,
    ):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.style_dim = style_dim

        # Mapping Network
        self.mapping = MappingNetwork(latent_dim=latent_dim, style_dim=style_dim, num_layers=mapping_layers)

        # Base 4x4 resolution setup
        self.initial_constant = nn.Parameter(torch.randn(1, max_channels, 4, 4))
        self.init_conv = StyleConv(max_channels, max_channels, kernel_size=3, style_dim=style_dim)

        # Resolution channel schedule (powers of 2 from 4 to image_size)
        # e.g., 4->512, 8->512, 16->256, 32->128, 64->64
        channel_dict = {
            4: max_channels,
            8: max_channels,
            16: min(max_channels, 256),
            32: min(max_channels, 128),
            64: min(max_channels, 64),
            128: min(max_channels, 32),
            256: min(max_channels, 16),
        }

        self.num_blocks = int(math.log2(image_size)) - 2  # e.g., 64 -> log2(64) - 2 = 6 - 2 = 4 blocks
        self.blocks = nn.ModuleList()

        current_res = 4
        in_c = channel_dict[current_res]
        for _ in range(self.num_blocks):
            next_res = current_res * 2
            out_c = channel_dict.get(next_res, 32)
            self.blocks.append(SynthesisBlock(in_c, out_c, style_dim=style_dim))
            in_c = out_c
            current_res = next_res

        self.to_rgb = ToRGB(in_c, style_dim=style_dim)

        # Register running average of style vector for truncation trick
        self.register_buffer("w_avg", torch.zeros(style_dim))

    def update_w_avg(self, w: torch.Tensor, momentum: float = 0.99):
        """Updates exponential moving average of w for truncation trick."""
        with torch.no_grad():
            batch_avg = w.mean(dim=0)
            self.w_avg.copy_(self.w_avg * momentum + batch_avg * (1.0 - momentum))

    def forward(
        self,
        z: torch.Tensor,
        style_mixing_prob: float = 0.9,
        truncation_psi: float = 1.0,
        return_styles: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with optional style mixing and truncation trick.
        z: latent tensor of shape (batch_size, latent_dim) or tuple of two latents for mixing.
        """
        batch_size = z.shape[0] if isinstance(z, torch.Tensor) else z[0].shape[0]

        # 1. Map latent codes to W space
        if self.training and random.random() < style_mixing_prob:
            # Style mixing: draw a second latent vector and switch styles at random layer
            z2 = torch.randn_like(z)
            w1 = self.mapping(z)
            w2 = self.mapping(z2)
            mix_cutoff = random.randint(1, len(self.blocks))
        else:
            w1 = self.mapping(z)
            w2 = w1
            mix_cutoff = len(self.blocks) + 1

        if self.training:
            self.update_w_avg(w1)

        # Truncation trick: w = w_avg + psi * (w - w_avg)
        if truncation_psi < 1.0:
            w1 = self.w_avg + truncation_psi * (w1 - self.w_avg)
            w2 = self.w_avg + truncation_psi * (w2 - self.w_avg)

        # 2. Synthesis Network
        x = self.initial_constant.repeat(batch_size, 1, 1, 1)
        x = self.init_conv(x, w1)

        for i, block in enumerate(self.blocks):
            current_w = w1 if i < mix_cutoff else w2
            x = block(x, current_w)

        rgb = self.to_rgb(x, w2)

        if return_styles:
            return rgb, (w1, w2)
        return rgb

    @torch.no_grad()
    def generate(self, z: torch.Tensor, truncation_psi: float = 0.7) -> torch.Tensor:
        """Helper for non-training sample inference."""
        self.eval()
        return self.forward(z, style_mixing_prob=0.0, truncation_psi=truncation_psi)
