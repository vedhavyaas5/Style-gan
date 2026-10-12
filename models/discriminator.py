import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import EqualizedConv2d, EqualizedLinear, MinibatchStdDev


class FromRGB(nn.Module):
    """
    Maps 3-channel RGB image to internal feature space.
    """
    def __init__(self, out_channels: int):
        super().__init__()
        self.conv = EqualizedConv2d(3, out_channels, kernel_size=1, padding=0)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class DiscriminatorBlock(nn.Module):
    """
    Discriminator residual block with downsampling.
    Conv -> LeakyReLU -> Conv -> Downsample (AvgPool) + Residual skip
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.skip = EqualizedConv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = F.avg_pool2d(self.skip(x), kernel_size=2, stride=2)

        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out = F.avg_pool2d(out, kernel_size=2, stride=2)

        return (out + res) / math.sqrt(2.0)


class Discriminator(nn.Module):
    """
    StyleGAN Discriminator with residual downsampling blocks and Minibatch StdDev.
    """
    def __init__(self, image_size: int = 64, max_channels: int = 512):
        super().__init__()
        self.image_size = image_size

        channel_dict = {
            4: max_channels,
            8: max_channels,
            16: min(max_channels, 256),
            32: min(max_channels, 128),
            64: min(max_channels, 64),
            128: min(max_channels, 32),
            256: min(max_channels, 16),
        }

        num_blocks = int(math.log2(image_size)) - 2
        self.from_rgb = FromRGB(channel_dict[image_size])

        blocks = []
        cur_res = image_size
        while cur_res > 4:
            next_res = cur_res // 2
            blocks.append(DiscriminatorBlock(channel_dict[cur_res], channel_dict[next_res]))
            cur_res = next_res
        self.blocks = nn.ModuleList(blocks)

        # At 4x4 resolution: add MinibatchStdDev (+1 channel)
        final_in = channel_dict[4] + 1
        self.minibatch_std = MinibatchStdDev()
        self.final_conv = EqualizedConv2d(final_in, channel_dict[4], kernel_size=3, padding=1)
        self.final_linear = EqualizedLinear(channel_dict[4] * 4 * 4, channel_dict[4])
        self.out_linear = EqualizedLinear(channel_dict[4], 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.from_rgb(x)
        for block in self.blocks:
            out = block(out)

        out = self.minibatch_std(out)
        out = self.act(self.final_conv(out))
        out = out.flatten(1)
        out = self.act(self.final_linear(out))
        score = self.out_linear(out)
        return score
