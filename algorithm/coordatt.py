"""
CoordAtt: Coordinate Attention (CVPR 2021)

用于对比实验。通过坐标编码将位置信息嵌入通道注意力。
"""

import torch
import torch.nn as nn


class CoordAtt(nn.Module):
    """Coordinate Attention Module (CVPR 2021).

    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        reduction (int): 中间层通道缩减比
    """

    def __init__(self, c1, c2=None, reduction=32):
        super().__init__()
        c2 = c2 or c1
        mid = max(8, c1 // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(c1, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.SiLU(inplace=True)

        self.conv_h = nn.Conv2d(mid, c1, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, c1, 1, bias=False)

        self.conv_out = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1) -> (B, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = x * a_h * a_w
        return self.conv_out(out)
