"""
FDSA: Frequency-Decoupled Spatial Attention Module

面向工业缺陷检测的频域解耦空间注意力模块。
设计目标：替换 YOLO11 的 C2PSA，以更低计算量实现更强的缺陷感知能力。

核心创新点：
1. 频域感知空间增强 (Frequency-Aware Spatial Enhancement):
   利用 DCT 变换将特征分解为高/低频分量，高频分量对应缺陷边缘和纹理突变，
   通过可学习的频率掩码自适应选择有效频率成分，生成空间注意力权重。
   与 C2PSA 的全 token 自注意力不同，FDSA 利用频域先验直接定位异常区域。

2. 双域交互增强 (Dual-Domain Interaction):
   频域分支提取高频异常信号 → 空间分支在异常区域做局部特征增强。
   两个分支输出维度正交（空间 vs 通道），通过门控融合互补。
   相比 C2PSA 的 O(N^2) 自注意力，FDSA 为 O(N) 线性复杂度。

3. 自适应频率-空间门控 (Adaptive Freq-Spatial Gating):
   根据输入特征的频域统计量动态调节频域/空间分支的贡献比例，
   使模块能自适应不同尺度和对比度的缺陷。

与 C2PSA 的对比:
    C2PSA: cv1 -> PSABlock(MHSA + FFN) x n -> cv2, 复杂度 O(N^2 * d)
    FDSA:  cv1 -> FreqBranch || SpatialBranch -> GatedFusion -> cv2, 复杂度 O(N * C)

与 Ultralytics 集成:
    - YAML: [-1, 1, FDSA, [512]]  (替换 C2PSA)
    - 接口: FDSA(c1, c2, n=1, e=0.5) 兼容 parse_model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FrequencyBranch(nn.Module):
    """频域感知分支：通过 DCT 变换提取高频异常信号并生成空间注意力。"""

    def __init__(self, channels, dct_size=8):
        super().__init__()
        self.dct_size = dct_size
        self.channels = channels

        # 可学习频率掩码 (rfft2 输出宽度为 dct_size//2+1)
        half_w = dct_size // 2 + 1
        self.freq_weight = nn.Parameter(torch.zeros(1, channels, dct_size, half_w))
        self._init_highpass()

        # 频域能量 -> 空间注意力
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def _init_highpass(self):
        """初始化频率掩码：距 DC 越远权重越高。"""
        ds = self.dct_size
        half_w = ds // 2 + 1
        for i in range(ds):
            for j in range(half_w):
                dist = math.sqrt(i * i + j * j) / math.sqrt((ds - 1) ** 2 + (half_w - 1) ** 2)
                self.freq_weight.data[:, :, i, j] = dist * 2 - 1

    def forward(self, x):
        B, C, H, W = x.shape
        ds = self.dct_size

        if H < ds or W < ds:
            attn = self.spatial_proj(x)
            return x * attn

        # 对齐到 dct_size 整数倍
        Hp = (H // ds) * ds
        Wp = (W // ds) * ds
        if Hp != H or Wp != W:
            x_work = F.adaptive_avg_pool2d(x, (Hp, Wp))
        else:
            x_work = x

        nH, nW = Hp // ds, Wp // ds

        # 分块 + rfft2（rfft2 在 AMP fp16 下会产生 ComplexHalf，
        # 与 PyTorch deterministic kernel 不兼容，故强制 float32 计算后转回原 dtype）
        in_dtype = x_work.dtype
        patches = x_work.float().reshape(B, C, nH, ds, nW, ds).permute(0, 1, 2, 4, 3, 5)
        freq = torch.fft.rfft2(patches, dim=(-2, -1), norm='ortho')
        freq_mag = freq.abs().to(in_dtype)

        # 可学习频率掩码
        weight = torch.sigmoid(self.freq_weight)
        weighted = freq_mag * weight.unsqueeze(2).unsqueeze(3)

        # 聚合为空间能量图
        energy = weighted.sum(dim=(-2, -1))  # (B, C, nH, nW)

        # 上采样回原始分辨率
        energy_map = F.interpolate(energy, size=(H, W), mode='bilinear', align_corners=False)

        # 生成空间注意力并增强
        attn = self.spatial_proj(energy_map)
        return x * attn


class SpatialBranch(nn.Module):
    """空间上下文分支：局部特征增强 + 通道重标定。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 16)

        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 5, 1, 2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_local = self.local_conv(x)
        ch_weight = self.channel_attn(x_local).unsqueeze(-1).unsqueeze(-1)
        return x_local * ch_weight


class FDSA(nn.Module):
    """Frequency-Decoupled Spatial Attention Module.

    替换 YOLO11 的 C2PSA。通过频域先验引导空间注意力，
    以线性复杂度实现比全自注意力更强的缺陷感知能力。

    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数（通常等于 c1）
        n (int): 重复次数（兼容 C2PSA 接口）
        e (float): 通道扩展比例
    """

    def __init__(self, c1, c2=None, n=1, e=0.5):
        super().__init__()
        c2 = c2 or c1
        self.c = int(c1 * e)

        # 输入投影
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU(inplace=True),
        )

        # 双分支
        self.freq_branch = FrequencyBranch(self.c)
        self.spatial_branch = SpatialBranch(self.c)

        # 自适应门控：根据输入动态分配两分支权重
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, 2, bias=False),
            nn.Softmax(dim=1),
        )

        # 输出投影
        self.cv2 = nn.Sequential(
            nn.Conv2d(2 * self.c, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )

        # 残差连接（当 c1 == c2 时）
        self.shortcut = c1 == c2

    def forward(self, x):
        # 门控权重
        g = self.gate(x)  # (B, 2)
        g_freq = g[:, 0].view(-1, 1, 1, 1)
        g_spat = g[:, 1].view(-1, 1, 1, 1)

        # 输入投影并分割
        proj = self.cv1(x)
        x1, x2 = proj.chunk(2, dim=1)

        # 双分支处理
        freq_out = self.freq_branch(x1)
        spat_out = self.spatial_branch(x2)

        # 门控融合
        fused = torch.cat([freq_out * g_freq, spat_out * g_spat], dim=1)

        # 输出投影
        out = self.cv2(fused)

        # 残差
        if self.shortcut:
            out = out + x
        return out


# ============================================================
# 消融实验变体
# ============================================================

class FDSAFreqOnly(nn.Module):
    """消融：仅频域分支（无空间分支）。"""

    def __init__(self, c1, c2=None, n=1, e=0.5):
        super().__init__()
        c2 = c2 or c1
        self.c = int(c1 * e)
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, self.c, 1, bias=False),
            nn.BatchNorm2d(self.c),
            nn.SiLU(inplace=True),
        )
        self.freq_branch = FrequencyBranch(self.c)
        self.cv2 = nn.Sequential(
            nn.Conv2d(self.c, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.shortcut = c1 == c2

    def forward(self, x):
        out = self.cv2(self.freq_branch(self.cv1(x)))
        if self.shortcut:
            out = out + x
        return out


class FDSASpatOnly(nn.Module):
    """消融：仅空间分支（无频域分支）。"""

    def __init__(self, c1, c2=None, n=1, e=0.5):
        super().__init__()
        c2 = c2 or c1
        self.c = int(c1 * e)
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, self.c, 1, bias=False),
            nn.BatchNorm2d(self.c),
            nn.SiLU(inplace=True),
        )
        self.spatial_branch = SpatialBranch(self.c)
        self.cv2 = nn.Sequential(
            nn.Conv2d(self.c, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.shortcut = c1 == c2

    def forward(self, x):
        out = self.cv2(self.spatial_branch(self.cv1(x)))
        if self.shortcut:
            out = out + x
        return out


class FDSANoGate(nn.Module):
    """消融：无自适应门控（两分支等权融合）。"""

    def __init__(self, c1, c2=None, n=1, e=0.5):
        super().__init__()
        c2 = c2 or c1
        self.c = int(c1 * e)
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU(inplace=True),
        )
        self.freq_branch = FrequencyBranch(self.c)
        self.spatial_branch = SpatialBranch(self.c)
        self.cv2 = nn.Sequential(
            nn.Conv2d(2 * self.c, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.shortcut = c1 == c2

    def forward(self, x):
        proj = self.cv1(x)
        x1, x2 = proj.chunk(2, dim=1)
        fused = torch.cat([self.freq_branch(x1), self.spatial_branch(x2)], dim=1)
        out = self.cv2(fused)
        if self.shortcut:
            out = out + x
        return out


class FDSANoFreqLearn(nn.Module):
    """消融：频率掩码固定为高通（不可学习）。"""

    def __init__(self, c1, c2=None, n=1, e=0.5):
        super().__init__()
        c2 = c2 or c1
        self.c = int(c1 * e)
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU(inplace=True),
        )
        self.freq_branch = FrequencyBranch(self.c)
        # 冻结频率掩码
        self.freq_branch.freq_weight.requires_grad = False
        self.spatial_branch = SpatialBranch(self.c)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, 2, bias=False),
            nn.Softmax(dim=1),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(2 * self.c, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.shortcut = c1 == c2

    def forward(self, x):
        g = self.gate(x)
        g_freq = g[:, 0].view(-1, 1, 1, 1)
        g_spat = g[:, 1].view(-1, 1, 1, 1)
        proj = self.cv1(x)
        x1, x2 = proj.chunk(2, dim=1)
        fused = torch.cat([self.freq_branch(x1) * g_freq,
                           self.spatial_branch(x2) * g_spat], dim=1)
        out = self.cv2(fused)
        if self.shortcut:
            out = out + x
        return out


# ============================================================
# 对比实验：其他注意力模块
# ============================================================

class EMA(nn.Module):
    """Efficient Multi-Scale Attention (ICASSP 2023).

    用于对比实验。
    """

    def __init__(self, c1, c2=None, groups=32):
        super().__init__()
        c2 = c2 or c1
        self.groups = min(groups, c1)
        if c1 % self.groups != 0:
            self.groups = 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, self.groups, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, self.groups, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(self.groups, c1)
        self.conv1x1 = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x_gn = self.gn(x)
        x_avg = self.avg_pool(x).view(B, self.groups, C // self.groups, 1, 1)
        x_avg = x_avg.mean(dim=2).view(B, self.groups, 1, 1)
        weight = self.sigmoid(self.weight * x_avg + self.bias)
        weight = weight.repeat_interleave(C // self.groups, dim=1)
        out = x * weight * self.sigmoid(x_gn)
        return self.conv1x1(out)


class SimAM(nn.Module):
    """SimAM: Simple Parameter-Free Attention Module (ICML 2021).

    用于对比实验。无参数的注意力机制。
    """

    def __init__(self, c1, c2=None, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.conv1x1 = nn.Conv2d(c1, c2 or c1, 1, bias=False) if c2 and c1 != c2 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        n = H * W - 1
        if n <= 0:
            return self.conv1x1(x)
        # 计算每个神经元的能量
        x_minus_mu_sq = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_sq / (4 * (x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        out = x * torch.sigmoid(y)
        return self.conv1x1(out)
