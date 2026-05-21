"""
小波子带可视化：展示 DWGSA 模块的 Haar DWT 分解效果和几何先验 Mask。

Usage:
    python experiment/scripts/visualize_wavelets.py \
        --image datasets/DeepPCB/images/test/sample.jpg \
        --weights runs/sota/dwgsa_yolo11m/weights/best.pt \
        --output results/wavelet_vis.png

输出:
    6 格子图：原图 | LL (低频) | LH (水平) | HL (垂直) | HH (对角) | 几何先验 Mask
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.dwgsa import HaarDWT2D, DWGSA, GeometryPriorEstimator
from algorithm.register import register_custom_modules


def preprocess_image(image_path, imgsz=640):
    """加载并预处理图片。"""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (imgsz, imgsz))
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    return img_resized, img_tensor


def normalize_for_display(tensor):
    """将张量归一化到 [0, 1] 用于显示。"""
    t = tensor.float()
    t = t - t.min()
    t = t / (t.max() + 1e-8)
    return t.cpu().numpy()


@torch.no_grad()
def extract_wavelet_features(img_tensor, channels=64):
    """使用 Haar DWT 分解图像特征并提取几何先验。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    conv_proj = torch.nn.Conv2d(3, channels, 3, 1, 1, bias=False).to(device)
    torch.nn.init.kaiming_normal_(conv_proj.weight)

    x = conv_proj(img_tensor.to(device))

    dwt = HaarDWT2D(channels).to(device)
    ll1, lh1, hl1, hh1 = dwt(x)

    ll2, lh2, hl2, hh2 = dwt(ll1)

    geo_est = GeometryPriorEstimator(channels).to(device)
    geo_mask = geo_est(ll2)

    return {
        "ll": ll1.mean(dim=1, keepdim=True).squeeze(),
        "lh": lh1.mean(dim=1, keepdim=True).squeeze().abs(),
        "hl": hl1.mean(dim=1, keepdim=True).squeeze().abs(),
        "hh": hh1.mean(dim=1, keepdim=True).squeeze().abs(),
        "geo_mask": geo_mask.squeeze(),
    }


@torch.no_grad()
def extract_from_trained_model(img_tensor, model_path):
    """从训练好的 DWGSA 模型中提取中间特征。

    实现方式：通过 forward hook 捕获 DWGSA 模块的真实输入（layer 10 接收的特征图），
    然后复现 cv1 投影 → chunk → DWT 流程得到子带。
    这些子带就是模型在推理 img_tensor 时实际计算的高/低频特征。
    """
    from ultralytics import YOLO

    register_custom_modules()
    model = YOLO(model_path)
    device = next(model.model.parameters()).device
    img_tensor = img_tensor.to(device)

    dwgsa_module = None
    for m in model.model.model.modules():
        if isinstance(m, DWGSA):
            dwgsa_module = m
            break

    if dwgsa_module is None:
        print("[WARNING] DWGSA module not found in model. Using random init.")
        return extract_wavelet_features(img_tensor)

    # Hook 捕获 DWGSA 接收到的真实输入特征图
    dwgsa_input_holder = {}

    def hook_input(module, inputs, output):
        dwgsa_input_holder["x"] = inputs[0].detach()

    # 同时 hook geo_estimator 拿到训练好的 geo mask
    geo_mask_holder = {}

    def hook_geo(module, inputs, output):
        geo_mask_holder["mask"] = output.detach()

    h_input = dwgsa_module.register_forward_hook(hook_input)
    h_geo = dwgsa_module.geometry_sparse_attn.geo_estimator.register_forward_hook(hook_geo)

    model.model.eval()
    model.model(img_tensor)

    h_input.remove()
    h_geo.remove()

    if "x" not in dwgsa_input_holder:
        print("[WARNING] Failed to capture DWGSA input. Using random init.")
        return extract_wavelet_features(img_tensor)

    # 在真实输入特征上复现 DWT 计算
    x_in = dwgsa_input_holder["x"]  # (1, 512, 20, 20) 来自 backbone P5
    proj = dwgsa_module.cv1(x_in)
    x_wave = proj.chunk(2, dim=1)[0]  # 真实进入 wavelet 分支的特征
    dwt = dwgsa_module.wavelet_branch.dwt
    ll1, lh1, hl1, hh1 = dwt(x_wave)

    geo_mask = geo_mask_holder.get("mask")
    if geo_mask is None:
        # 即使 hook miss 也能从 ll subband 重新跑 geo estimator
        geo_estimator = dwgsa_module.geometry_sparse_attn.geo_estimator
        ll2, _, _, _ = dwt(ll1)
        geo_mask = geo_estimator(ll2)

    return {
        "ll": ll1.mean(dim=1).squeeze(),
        "lh": lh1.mean(dim=1).squeeze().abs(),
        "hl": hl1.mean(dim=1).squeeze().abs(),
        "hh": hh1.mean(dim=1).squeeze().abs(),
        "geo_mask": geo_mask.squeeze(),
    }


def visualize_subbands(img_resized, features, output_path):
    """生成 6 格子可视化图。"""
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 6, figure=fig, wspace=0.08)

    titles = ["Input", "LL (Low-Freq)", "LH (Horizontal)", "HL (Vertical)", "HH (Diagonal)", "Geometry Prior"]
    cmaps = [None, "gray", "hot", "hot", "hot", "jet"]

    images = [
        img_resized,
        normalize_for_display(features["ll"]),
        normalize_for_display(features["lh"]),
        normalize_for_display(features["hl"]),
        normalize_for_display(features["hh"]),
        normalize_for_display(features["geo_mask"]),
    ]

    for i, (title, img, cmap) in enumerate(zip(titles, images, cmaps)):
        ax = fig.add_subplot(gs[0, i])
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Haar DWT Decomposition & Geometry Prior Estimation", fontsize=12, y=0.98)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"[INFO] Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Wavelet Subband Visualization for DWGSA")
    parser.add_argument("--image", type=str, required=True, help="Input PCB image path")
    parser.add_argument("--weights", type=str, default=None,
                        help="Trained DWGSA-YOLO weights (optional, uses random init if not provided)")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "results" / "wavelet_vis.png"))
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    img_resized, img_tensor = preprocess_image(args.image, args.imgsz)

    if args.weights and Path(args.weights).exists():
        print("[INFO] Extracting features from trained model...")
        features = extract_from_trained_model(img_tensor, args.weights)
    else:
        print("[INFO] Using random initialization for visualization...")
        features = extract_wavelet_features(img_tensor)

    visualize_subbands(img_resized, features, args.output)
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
