"""
Convolutional Block Attention Module (CBAM)

标准 CBAM 实现，用于与 DWGSA 进行对比实验。
Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """CBAM 通道注意力子模块。"""

    def __init__(self, c1, reduction=16):
        super().__init__()
        mid = max(c1 // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(c1, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, c1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        return self.sigmoid(avg_out + max_out).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM 空间注意力子模块。"""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        return self.conv(combined)


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    串行结构：先通道注意力，再空间注意力。

    Args:
        c1 (int): 输入/输出通道数
        kernel_size (int): 空间注意力卷积核大小，默认 7
        reduction (int): 通道缩减比，默认 16
    """

    def __init__(self, c1, c2=None, kernel_size=7, reduction=16):
        super().__init__()
        c2 = c2 or c1
        self.channel_attn = ChannelAttention(c1, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)
        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return self.proj(x)
