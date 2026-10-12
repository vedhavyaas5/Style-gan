import torch
import torch.nn as nn
from .blocks import PixelNorm, EqualizedLinear


class MappingNetwork(nn.Module):
    """
    Mapping Network f: Z -> W
    Consists of 8 fully-connected layers mapping a normalized latent code z in Z
    into an intermediate disentangled style space W.
    """
    def __init__(self, latent_dim: int = 512, style_dim: int = 512, num_layers: int = 8, lr_mlp: float = 0.01):
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.pixel_norm = PixelNorm()

        layers = []
        for i in range(num_layers):
            in_dim = latent_dim if i == 0 else style_dim
            layers.append(EqualizedLinear(in_dim, style_dim, lrmul=lr_mlp))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.mapping = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.pixel_norm(z)
        w = self.mapping(z)
        return w
