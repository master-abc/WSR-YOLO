"""
DWGSA: Discrete Wavelet Geometry-prior Sparse Attention

核心创新：
1. 基于 Haar 小波的多级频域-空域联合分解（替代 FFT patch-wise 处理）
2. 利用 LL 低频子带估计 PCB 几何先验，驱动稀疏注意力计算
3. 基于物理信号（HF 能量比 + 空间稀疏度）的自适应双分支融合

接口兼容 Ultralytics YOLO：DWGSA(c1, c2, n=1, e=0.5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HaarDWT2D(nn.Module):
    """2D Haar 离散小波变换，基于 grouped Conv2d stride=2 实现。

    将输入分解为 LL（低频近似）、LH（水平细节）、HL（垂直细节）、HH（对角细节）四个子带，
    每个子带空间分辨率为输入的 1/2。
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # Haar 滤波器: low=[1/√2, 1/√2], high=[1/√2, -1/√2]
        inv_sqrt2 = 1.0 / math.sqrt(2.0)

        # 构造 4 个 2D 滤波器 (LL, LH, HL, HH)
        # LL = low_row * low_col
        # LH = high_row * low_col (水平边缘)
        # HL = low_row * high_col (垂直边缘)
        # HH = high_row * high_col (对角边缘)
        ll = torch.tensor([[inv_sqrt2, inv_sqrt2],
                           [inv_sqrt2, inv_sqrt2]]) * inv_sqrt2
        lh = torch.tensor([[-inv_sqrt2, -inv_sqrt2],
                           [inv_sqrt2, inv_sqrt2]]) * inv_sqrt2
        hl = torch.tensor([[-inv_sqrt2, inv_sqrt2],
                           [-inv_sqrt2, inv_sqrt2]]) * inv_sqrt2
        hh = torch.tensor([[inv_sqrt2, -inv_sqrt2],
                           [-inv_sqrt2, inv_sqrt2]]) * inv_sqrt2

        # 组合为 (4*C, 1, 2, 2) 的 grouped conv 权重
        filters = torch.stack([ll, lh, hl, hh], dim=0)  # (4, 2, 2)
        filters = filters.unsqueeze(1)  # (4, 1, 2, 2)
        # 扩展到所有通道: (4*C, 1, 2, 2) for groups=C
        weight = filters.repeat(channels, 1, 1, 1)  # (4*C, 1, 2, 2)
        self.register_buffer("weight", weight)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) — H, W 需为偶数
        Returns:
            (LL, LH, HL, HH): 每个 shape 为 (B, C, H/2, W/2)
        """
        B, C, H, W = x.shape
        # 确保偶数尺寸
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode="reflect")
            H += 1
        if W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0), mode="reflect")
            W += 1

        # Grouped conv: 每个通道独立应用 4 个滤波器
        # weight shape: (4*C, 1, 2, 2), groups=C
        out = F.conv2d(x, self.weight, stride=2, groups=C)  # (B, 4*C, H/2, W/2)

        # 拆分为 4 个子带
        # 输出排列: [ch0_LL, ch0_LH, ch0_HL, ch0_HH, ch1_LL, ...]
        out = out.view(B, C, 4, H // 2, W // 2)
        ll = out[:, :, 0]  # (B, C, H/2, W/2)
        lh = out[:, :, 1]
        hl = out[:, :, 2]
        hh = out[:, :, 3]
        return ll, lh, hl, hh


class WaveletBranch(nn.Module):
    """基于 DWT 的多分辨率高频特征提取分支。

    2 级 Haar 小波分解后，对 HH/HL/LH 高频子带分别处理，
    融合后生成空间注意力图，突出微小缺陷的边缘响应。
    """

    def __init__(self, channels, levels=2):
        super().__init__()
        self.levels = levels
        self.dwt = HaarDWT2D(channels)

        # 各高频子带独立处理 (depthwise conv 保持轻量)
        self.hh_proc = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.hl_proc = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.lh_proc = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        # 融合 3 个高频子带 → 单通道空间注意力
        self.subband_fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W) — 高频注意力加权后的特征
            ll_final: (B, C, H/4, W/4) — 最终 LL 子带（供几何先验使用）
            hf_energy: (B, C, H/2, W/2) — 融合后的高频能量图
        """
        # Level 1 DWT
        ll1, lh1, hl1, hh1 = self.dwt(x)

        # Level 2 DWT (对 LL 子带递归分解, 需要 LL1 至少 4x4 才能产生有效的 2x2 子带)
        if self.levels >= 2 and ll1.shape[2] >= 4 and ll1.shape[3] >= 4:
            ll2, lh2, hl2, hh2 = self.dwt(ll1)
            ll_final = ll2
        else:
            ll_final = ll1
            lh2, hl2, hh2 = None, None, None

        # 处理 Level 1 高频子带
        hh_feat = self.hh_proc(hh1)
        hl_feat = self.hl_proc(hl1)
        lh_feat = self.lh_proc(lh1)

        # 如果有 Level 2，处理后叠加 Level 2 高频信息（上采样到 Level 1 尺寸）
        # 注意：Level 1 和 Level 2 共享 hh_proc/hl_proc/lh_proc 权重（有意设计，减少参数量）
        if lh2 is not None and hh2.shape[2] > 1 and hh2.shape[3] > 1:
            h1, w1 = hh1.shape[2:]
            hh_feat = hh_feat + F.interpolate(self.hh_proc(hh2), size=(h1, w1), mode="bilinear", align_corners=False)
            hl_feat = hl_feat + F.interpolate(self.hl_proc(hl2), size=(h1, w1), mode="bilinear", align_corners=False)
            lh_feat = lh_feat + F.interpolate(self.lh_proc(lh2), size=(h1, w1), mode="bilinear", align_corners=False)

        # 融合高频子带
        hf_cat = torch.cat([hh_feat, hl_feat, lh_feat], dim=1)  # (B, 3C, H/2, W/2)
        hf_fused = self.subband_fusion(hf_cat)  # (B, C, H/2, W/2)

        # 生成空间注意力并上采样到原始分辨率
        attn = self.spatial_gate(hf_fused)  # (B, 1, H/2, W/2)
        attn = F.interpolate(attn, size=x.shape[2:], mode="bilinear", align_corners=False)

        out = x * attn
        return out, ll_final, hf_fused


class GeometryPriorEstimator(nn.Module):
    """从 LL 低频子带估计 PCB 几何先验 mask。

    LL 子带保留了 PCB 的走线、焊盘等低频结构信息。
    通过轻量卷积网络将其转化为 soft mask，标识缺陷高发区域。
    训练时使用 soft mask（可微），推理时可选硬阈值实现真正稀疏。
    """

    def __init__(self, channels):
        super().__init__()
        mid = max(channels // 4, 16)
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, 1, 3, 1, 1, bias=True),
        )
        self.tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, ll_subband):
        """
        Args:
            ll_subband: (B, C, H/4, W/4)
        Returns:
            mask: (B, 1, H/4, W/4) — soft geometry prior mask [0, 1]
        """
        logits = self.net(ll_subband)
        mask = torch.sigmoid(logits - self.tau)
        return mask


class GeometrySparseAttn(nn.Module):
    """几何先验驱动的稀疏注意力分支。

    利用 LL 子带生成的几何 mask，在感兴趣区域（走线/焊盘）执行精细注意力，
    背景区域保持原始特征，实现计算的选择性分配。
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.geo_estimator = GeometryPriorEstimator(channels)

        # 精细注意力：局部空间增强 + 通道重标定
        self.local_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 5, 1, 2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        mid = max(channels // reduction, 16)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, ll_subband):
        """
        Args:
            x: (B, C, H, W) — 待处理特征
            ll_subband: (B, C, H/4, W/4) — 来自 WaveletBranch 的 LL 子带
        Returns:
            out: (B, C, H, W) — 稀疏注意力增强后的特征
            mask: (B, 1, H/4, W/4) — 几何先验 mask（用于可视化和融合）
        """
        # 生成几何先验 mask
        mask = self.geo_estimator(ll_subband)  # (B, 1, H/4, W/4)
        mask_full = F.interpolate(mask, size=x.shape[2:], mode="bilinear", align_corners=False)

        # 精细空间注意力
        refined = self.local_refine(x)
        ch_w = self.channel_attn(refined).unsqueeze(-1).unsqueeze(-1)
        refined = refined * ch_w

        # 稀疏组合：mask 区域用精细特征，非 mask 区域保留原始
        out = refined * mask_full + x * (1.0 - mask_full)
        return out, mask


class AdaptiveFusion(nn.Module):
    """基于物理信号的自适应双分支融合门控。

    融合决策基于两个可解释的物理量：
    1. HF 能量比：高频子带能量占总能量的比例（缺陷丰富度指标）
    2. 空间稀疏度：几何 mask 的覆盖率（结构复杂度指标）

    当 HF 能量高时倾向 Wavelet 分支；当结构复杂时倾向 Sparse 分支。
    """

    def __init__(self, c1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(c1 // 16, 8)
        self.mlp = nn.Sequential(
            nn.Linear(c1 + 2, hidden, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 2, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x, hf_fused, geometry_mask):
        """
        Args:
            x: (B, C, H, W) — 原始输入（提供全局上下文）
            hf_fused: (B, C, H/2, W/2) — 融合后的高频能量图
            geometry_mask: (B, 1, H/4, W/4) — 几何先验 mask
        Returns:
            g_wave: (B, 1, 1, 1) — Wavelet 分支权重
            g_sparse: (B, 1, 1, 1) — Sparse 分支权重
        """
        # 全局特征
        global_feat = self.pool(x).flatten(1)  # (B, C)

        # 物理信号 1: HF 能量比
        hf_energy = hf_fused.pow(2).mean(dim=[1, 2, 3])  # (B,)
        total_energy = x.pow(2).mean(dim=[1, 2, 3]) + 1e-8  # (B,)
        energy_ratio = (hf_energy / total_energy).unsqueeze(1)  # (B, 1)

        # 物理信号 2: 空间稀疏度
        sparsity = geometry_mask.mean(dim=[1, 2, 3]).unsqueeze(1)  # (B, 1)

        # 融合决策
        context = torch.cat([global_feat, energy_ratio, sparsity], dim=1)  # (B, C+2)
        weights = self.mlp(context)  # (B, 2)

        g_wave = weights[:, 0:1].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        g_sparse = weights[:, 1:2].unsqueeze(-1).unsqueeze(-1)
        return g_wave, g_sparse


class DWGSA(nn.Module):
    """Discrete Wavelet Geometry-prior Sparse Attention.

    双分支架构：
    - WaveletBranch: DWT 多级分解 → 高频子带处理 → 空间注意力
    - GeometrySparseAttn: LL 子带 → 几何先验 mask → 稀疏精细注意力
    - AdaptiveFusion: 基于 HF 能量比和空间稀疏度的自适应门控

    接口: DWGSA(c1, c2=None, n=1, e=0.5)
    """

    def __init__(self, c1, c2=None, n=1, e=0.5):
        super().__init__()
        c2 = c2 or c1
        self.c = int(c1 * e)

        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU(inplace=True),
        )
        self.wavelet_branch = WaveletBranch(self.c, levels=2)
        self.geometry_sparse_attn = GeometrySparseAttn(self.c)
        self.adaptive_fusion = AdaptiveFusion(c1)
        self.cv2 = nn.Sequential(
            nn.Conv2d(2 * self.c, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.shortcut = (c1 == c2)

    def forward(self, x):
        # 特征图过小时跳过复杂处理（DWT stride=2 需要至少 4x4 输入）
        if x.shape[2] < 4 or x.shape[3] < 4:
            out = self.cv2(self.cv1(x))
            if self.shortcut:
                out = out + x
            return out

        # 输入投影 + 分支分割
        proj = self.cv1(x)
        x_wave, x_sparse = proj.chunk(2, dim=1)

        # Wavelet 分支: DWT 分解 → 高频注意力
        wave_out, ll_subband, hf_fused = self.wavelet_branch(x_wave)

        # Geometry Sparse 分支: LL → mask → 稀疏精细注意力
        sparse_out, geo_mask = self.geometry_sparse_attn(x_sparse, ll_subband)

        # 自适应融合
        g_wave, g_sparse = self.adaptive_fusion(x, hf_fused, geo_mask)

        # 门控拼接 + 输出投影
        fused = torch.cat([wave_out * g_wave, sparse_out * g_sparse], dim=1)
        out = self.cv2(fused)

        if self.shortcut:
            out = out + x
        return out


# ============================================================
# 消融变体
# ============================================================

class DWGSAWaveOnly(nn.Module):
    """消融：仅 Wavelet 分支，移除 GeometrySparseAttn。"""

    def __init__(self, c1, c2=None, n=1, e=0.5):
        super().__init__()
        c2 = c2 or c1
        self.c = int(c1 * e)

        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, self.c, 1, bias=False),
            nn.BatchNorm2d(self.c),
            nn.SiLU(inplace=True),
        )
        self.wavelet_branch = WaveletBranch(self.c, levels=2)
        self.cv2 = nn.Sequential(
            nn.Conv2d(self.c, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.shortcut = (c1 == c2)

    def forward(self, x):
        if x.shape[2] < 4 or x.shape[3] < 4:
            out = self.cv2(self.cv1(x))
            if self.shortcut:
                out = out + x
            return out
        proj = self.cv1(x)
        wave_out, _, _ = self.wavelet_branch(proj)
        out = self.cv2(wave_out)
        if self.shortcut:
            out = out + x
        return out


class DWGSASparseOnly(nn.Module):
    """消融：仅 Sparse Attention 分支，移除 WaveletBranch。

    几何先验从原始特征的 AvgPool 降采样中估计（替代 LL 子带）。
    """

    def __init__(self, c1, c2=None, n=1, e=0.5):
        super().__init__()
        c2 = c2 or c1
        self.c = int(c1 * e)

        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, self.c, 1, bias=False),
            nn.BatchNorm2d(self.c),
            nn.SiLU(inplace=True),
        )
        self.geometry_sparse_attn = GeometrySparseAttn(self.c)
        self.cv2 = nn.Sequential(
            nn.Conv2d(self.c, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.shortcut = (c1 == c2)

    def forward(self, x):
        if x.shape[2] < 4 or x.shape[3] < 4:
            out = self.cv2(self.cv1(x))
            if self.shortcut:
                out = out + x
            return out
        proj = self.cv1(x)
        # 用 AvgPool 模拟 LL 子带（2 级下采样 = 1/4 分辨率）
        H, W = proj.shape[2:]
        ll_proxy = F.adaptive_avg_pool2d(proj, (max(H // 4, 1), max(W // 4, 1)))
        sparse_out, _ = self.geometry_sparse_attn(proj, ll_proxy)
        out = self.cv2(sparse_out)
        if self.shortcut:
            out = out + x
        return out


class DWGSANoGeoPrior(nn.Module):
    """消融：移除几何先验（mask 恒为 1），验证 geometry prior 的贡献。"""

    def __init__(self, c1, c2=None, n=1, e=0.5):
        super().__init__()
        c2 = c2 or c1
        self.c = int(c1 * e)

        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU(inplace=True),
        )
        self.wavelet_branch = WaveletBranch(self.c, levels=2)
        # 无几何先验的全注意力精细处理
        self.full_refine = nn.Sequential(
            nn.Conv2d(self.c, self.c, 5, 1, 2, groups=self.c, bias=False),
            nn.BatchNorm2d(self.c),
            nn.SiLU(inplace=True),
        )
        mid = max(self.c // 16, 16)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.c, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, self.c, bias=False),
            nn.Sigmoid(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(2 * self.c, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.shortcut = (c1 == c2)

    def forward(self, x):
        if x.shape[2] < 4 or x.shape[3] < 4:
            out = self.cv2(self.cv1(x))
            if self.shortcut:
                out = out + x
            return out
        proj = self.cv1(x)
        x_wave, x_full = proj.chunk(2, dim=1)

        wave_out, _, _ = self.wavelet_branch(x_wave)

        # 全区域精细注意力（无 mask）
        refined = self.full_refine(x_full)
        ch_w = self.channel_attn(refined).unsqueeze(-1).unsqueeze(-1)
        full_out = refined * ch_w

        fused = torch.cat([wave_out, full_out], dim=1)
        out = self.cv2(fused)
        if self.shortcut:
            out = out + x
        return out


class DWGSANoAdaptive(nn.Module):
    """消融：移除自适应融合，使用等权(0.5, 0.5)融合。"""

    def __init__(self, c1, c2=None, n=1, e=0.5):
        super().__init__()
        c2 = c2 or c1
        self.c = int(c1 * e)

        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU(inplace=True),
        )
        self.wavelet_branch = WaveletBranch(self.c, levels=2)
        self.geometry_sparse_attn = GeometrySparseAttn(self.c)
        self.cv2 = nn.Sequential(
            nn.Conv2d(2 * self.c, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.shortcut = (c1 == c2)

    def forward(self, x):
        if x.shape[2] < 4 or x.shape[3] < 4:
            out = self.cv2(self.cv1(x))
            if self.shortcut:
                out = out + x
            return out
        proj = self.cv1(x)
        x_wave, x_sparse = proj.chunk(2, dim=1)

        wave_out, ll_subband, _ = self.wavelet_branch(x_wave)
        sparse_out, _ = self.geometry_sparse_attn(x_sparse, ll_subband)

        # 等权融合
        fused = torch.cat([wave_out * 0.5, sparse_out * 0.5], dim=1)
        out = self.cv2(fused)
        if self.shortcut:
            out = out + x
        return out


class DWGSASingleLevel(nn.Module):
    """消融：仅 1 级 DWT（验证多级分解的价值）。"""

    def __init__(self, c1, c2=None, n=1, e=0.5):
        super().__init__()
        c2 = c2 or c1
        self.c = int(c1 * e)

        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU(inplace=True),
        )
        self.wavelet_branch = WaveletBranch(self.c, levels=1)  # 仅 1 级
        self.geometry_sparse_attn = GeometrySparseAttn(self.c)
        self.adaptive_fusion = AdaptiveFusion(c1)
        self.cv2 = nn.Sequential(
            nn.Conv2d(2 * self.c, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.shortcut = (c1 == c2)

    def forward(self, x):
        if x.shape[2] < 4 or x.shape[3] < 4:
            out = self.cv2(self.cv1(x))
            if self.shortcut:
                out = out + x
            return out
        proj = self.cv1(x)
        x_wave, x_sparse = proj.chunk(2, dim=1)

        wave_out, ll_subband, hf_fused = self.wavelet_branch(x_wave)
        # levels=1 时 ll_subband 为 H/2 x W/2，需下采样到 H/4 x W/4 匹配 GeometrySparseAttn
        H, W = ll_subband.shape[2:]
        ll_for_sparse = F.adaptive_avg_pool2d(ll_subband, (max(H // 2, 1), max(W // 2, 1)))
        sparse_out, geo_mask = self.geometry_sparse_attn(x_sparse, ll_for_sparse)
        g_wave, g_sparse = self.adaptive_fusion(x, hf_fused, geo_mask)

        fused = torch.cat([wave_out * g_wave, sparse_out * g_sparse], dim=1)
        out = self.cv2(fused)
        if self.shortcut:
            out = out + x
        return out
