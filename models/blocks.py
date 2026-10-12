import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNorm(nn.Module):
    """
    Pixel Normalization applied to latent vectors z in the mapping network.
    Normalizes feature vector along the channel dimension.
    """
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class EqualizedWeight(nn.Module):
    """
    Equalized learning rate weight wrapper (He initialization scaled dynamically).
    Scales weights dynamically at runtime by He constant: sqrt(2 / fan_in).
    """
    def __init__(self, shape, gain: float = 2 ** 0.5, lrmul: float = 1.0):
        super().__init__()
        self.lrmul = lrmul
        fan_in = shape[1] * (shape[2] * shape[3] if len(shape) > 2 else 1)
        self.scale = (gain / math.sqrt(fan_in)) * lrmul
        self.weight = nn.Parameter(torch.randn(shape) / lrmul)

    def forward(self) -> torch.Tensor:
        return self.weight * self.scale


class EqualizedLinear(nn.Module):
    """
    Linear layer with equalized learning rate.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, gain: float = 2 ** 0.5, lrmul: float = 1.0):
        super().__init__()
        self.eq_weight = EqualizedWeight((out_features, in_features), gain=gain, lrmul=lrmul)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        self.lrmul = lrmul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.eq_weight()
        b = self.bias * self.lrmul if self.bias is not None else None
        return F.linear(x, w, b)


class EqualizedConv2d(nn.Module):
    """
    Conv2d layer with equalized learning rate.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        gain: float = 2 ** 0.5,
        lrmul: float = 1.0,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.eq_weight = EqualizedWeight(
            (out_channels, in_channels, kernel_size, kernel_size),
            gain=gain,
            lrmul=lrmul,
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.lrmul = lrmul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.eq_weight()
        b = self.bias * self.lrmul if self.bias is not None else None
        return F.conv2d(x, w, b, stride=self.stride, padding=self.padding)


class NoiseInjection(nn.Module):
    """
    Adds scaled Gaussian noise to feature map channels.
    Each channel has a single learned scaling factor.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            batch_size, _, height, width = x.shape
            noise = torch.randn(batch_size, 1, height, width, device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class AdaptiveInstanceNorm(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN).
    Modulates normalized feature channels using style scale (y_s) and bias (y_b) vectors:
    AdaIN(x, y) = y_s * ((x - mu(x)) / sigma(x)) + y_b
    """
    def __init__(self, in_channels: int, style_dim: int, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.style_transform = EqualizedLinear(style_dim, in_channels * 2, gain=1.0)

    def forward(self, x: torch.Tensor, style_w: torch.Tensor) -> torch.Tensor:
        # Style affine transformation: produces scale and bias
        style = self.style_transform(style_w).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = style.chunk(2, dim=1)

        # Instance normalization (per instance, per channel)
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.epsilon)

        return (1.0 + gamma) * norm_x + beta


class MinibatchStdDev(nn.Module):
    """
    Minibatch standard deviation layer for the Discriminator.
    Computes average standard deviation across spatial positions and minibatch,
    and appends it as an extra channel to promote sample diversity.
    """
    def __init__(self, group_size: int = 4, epsilon: float = 1e-8):
        super().__init__()
        self.group_size = group_size
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        # Compute std across minibatch
        std = torch.sqrt(x.var(dim=0, keepdim=True, unbiased=False) + self.epsilon)
        # Average over all channels and spatial pixels
        mean_std = std.mean().expand(batch_size, 1, height, width)
        return torch.cat([x, mean_std], dim=1)
