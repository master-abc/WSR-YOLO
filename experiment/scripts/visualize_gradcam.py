"""
Grad-CAM 可视化：对比 YOLO11m 基线和 DWGSA-YOLO 的注意力热力图。

Usage:
    python experiment/scripts/visualize_gradcam.py \
        --baseline runs/sota/yolo11m_baseline/weights/best.pt \
        --proposed runs/sota/dwgsa_yolo11m/weights/best.pt \
        --images datasets/DeepPCB/images/test \
        --output results/gradcam

输出:
    对每张测试图片生成三列对比图：原图 | 基线热力图 | DWGSA-YOLO 热力图
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # DWGSA-YOLO/
sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.register import register_custom_modules
from ultralytics import YOLO


class YOLOGradCAM:
    """YOLO 模型的 Grad-CAM 实现。

    在 backbone-neck 衔接处的 attention 模块（layer 10：C2PSA / DWGSA / FDSA / CBAM / ...）
    上计算梯度加权类激活映射。该位置兼顾语义层次和空间分辨率，比 dfl.conv 等末端单通道层
    更适合定位缺陷区域。
    """

    ATTN_MODULE_NAMES = ("DWGSA", "C2PSA", "FDSA", "CBAM", "CoordAtt", "EMA", "SimAM")

    def __init__(self, model, target_layer_idx=10):
        """
        Args:
            model: Ultralytics YOLO 模型
            target_layer_idx: 目标特征层索引（默认 10，对应 layer 10 的 attention 模块）
                              如果该位置不是 attention 模块，回退到自动搜索
        """
        self.model = model.model
        self.model.eval()
        self.device = next(self.model.parameters()).device

        self.target_layer = self._get_target_layer(target_layer_idx)
        self.gradients = None
        self.activations = None

        self.fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _get_target_layer(self, idx):
        """获取目标层：优先按 idx，否则在整个模型中搜索 attention 模块。"""
        layers = list(self.model.model.children())
        if 0 <= idx < len(layers):
            candidate = layers[idx]
            if candidate.__class__.__name__ in self.ATTN_MODULE_NAMES:
                return candidate

        # 全模型搜索 attention 模块（取第一个）
        for module in self.model.modules():
            if module.__class__.__name__ in self.ATTN_MODULE_NAMES:
                return module

        # 最终 fallback：SPPF 或最后一个 C3k2
        fallback = None
        for module in self.model.modules():
            if module.__class__.__name__ in ("SPPF", "C3k2"):
                fallback = module
        if fallback is not None:
            return fallback

        # 实在没有就用末尾 conv（旧逻辑）
        return layers[min(idx, len(layers) - 1)]

    def _forward_hook(self, module, input, output):
        """前向传播 hook：保存激活值。"""
        if isinstance(output, (list, tuple)):
            self.activations = output[0]
        else:
            self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        """反向传播 hook：保存梯度。"""
        if grad_output and grad_output[0] is not None:
            self.gradients = grad_output[0].detach()

    def __del__(self):
        try:
            self.fwd_handle.remove()
            self.bwd_handle.remove()
        except Exception:
            pass

    def generate_cam(self, image_tensor, target_class=None):
        """生成 Grad-CAM 热力图。

        Args:
            image_tensor: 预处理后的图片张量 (1, 3, H, W)
            target_class: 暂未启用（YOLO Detect 输出已包含所有类别 logits）

        Returns:
            cam: 归一化的热力图 (H, W)，值域 [0, 1]
        """
        image_tensor = image_tensor.to(self.device).clone().detach().requires_grad_(True)

        # 重置缓存
        self.activations = None
        self.gradients = None

        # 前向传播 (eval 模式)
        self.model.zero_grad()
        output = self.model(image_tensor)

        # eval 模式下 ultralytics 返回 (predictions, raw_heads) 元组
        score = None
        if isinstance(output, tuple) and len(output) >= 1 and torch.is_tensor(output[0]):
            score = output[0].max()
        elif isinstance(output, (list, tuple)):
            tensors = [o for o in output if torch.is_tensor(o) and o.requires_grad]
            if tensors:
                score = sum(t.max() for t in tensors)
        elif torch.is_tensor(output):
            score = output.max()

        if score is None or not score.requires_grad:
            print("[WARNING] Could not derive a differentiable score. Returning zero CAM.")
            return np.zeros((image_tensor.shape[2], image_tensor.shape[3]))

        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            print("[WARNING] No gradients captured. Falling back to activation magnitude.")
            if self.activations is not None:
                cam = self.activations.detach().abs().mean(dim=1, keepdim=True).squeeze()
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-8)
                return cam.cpu().numpy()
            return np.zeros((image_tensor.shape[2], image_tensor.shape[3]))

        # 计算 Grad-CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # GAP on spatial dims
        cam = (weights * self.activations.detach()).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze()

        # 归一化（梯度退化时回退到 activation magnitude）
        denom = cam.max() - cam.min()
        if denom < 1e-10:
            cam = self.activations.detach().abs().mean(dim=1, keepdim=True).squeeze()
            denom = cam.max() - cam.min()
            if denom < 1e-10:
                return np.zeros_like(cam.cpu().numpy())
        cam = (cam - cam.min()) / (denom + 1e-10)

        return cam.cpu().numpy()


def preprocess_image(image_path, imgsz=640):
    """预处理图片为模型输入格式。"""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img_resized = cv2.resize(img, (imgsz, imgsz))

    # Normalize and to tensor
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    return img, img_resized, img_tensor


def overlay_cam_on_image(image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """将 CAM 热力图叠加到原图上。"""
    # Resize CAM to image size
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # 转换为热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 叠加
    overlay = np.float32(heatmap) / 255.0 * alpha + np.float32(image) / 255.0 * (1 - alpha)
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return overlay


def visualize_comparison(image_path, baseline_model, dwgsa_model, output_path, imgsz=640):
    """生成单张图片的对比可视化。"""
    # 预处理
    img_orig, img_resized, img_tensor = preprocess_image(image_path, imgsz)

    # 生成 Grad-CAM
    baseline_cam = YOLOGradCAM(baseline_model)
    dwgsa_cam = YOLOGradCAM(dwgsa_model)

    cam_baseline = baseline_cam.generate_cam(img_tensor.clone())
    cam_dwgsa = dwgsa_cam.generate_cam(img_tensor.clone())

    # 叠加热力图
    overlay_baseline = overlay_cam_on_image(img_resized, cam_baseline)
    overlay_dwgsa = overlay_cam_on_image(img_resized, cam_dwgsa)

    # 绘制三列对比图
    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.05)

    titles = ["Input Image", "YOLO11m (Baseline)", "DWGSA-YOLO (Ours)"]
    images = [img_resized, overlay_baseline, overlay_dwgsa]

    for i, (title, img) in enumerate(zip(titles, images)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM Visualization for DWGSA-YOLO")
    parser.add_argument("--baseline", type=str,
                        default=str(PROJECT_ROOT / "experiment" / "exp1" / "runs" / "yolo11m_baseline" / "weights" / "best.pt"),
                        help="Path to baseline model weights")
    parser.add_argument("--proposed", type=str,
                        default=str(PROJECT_ROOT / "experiment" / "exp1" / "runs" / "dwgsa_yolo11m" / "weights" / "best.pt"),
                        help="Path to DWGSA-YOLO model weights")
    parser.add_argument("--images", type=str,
                        default=str(PROJECT_ROOT / "experiment" / "datasets" / "DeepPCB" / "images" / "test"),
                        help="Path to test images directory")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "results" / "gradcam"),
                        help="Output directory for visualizations")
    parser.add_argument("--num-images", type=int, default=10,
                        help="Number of images to visualize")
    args = parser.parse_args()

    # 注册自定义模块
    register_custom_modules()

    # 加载模型
    print("[INFO] Loading models...")
    baseline_model = YOLO(args.baseline)
    proposed_model = YOLO(args.proposed)

    # 获取测试图片
    images_dir = Path(args.images)
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    if not image_files:
        print(f"[ERROR] No images found in {images_dir}")
        sys.exit(1)

    # 选择子集
    image_files = image_files[:args.num_images]

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成可视化
    print(f"\n[INFO] Generating Grad-CAM visualizations for {len(image_files)} images...")
    for img_path in image_files:
        output_path = output_dir / f"gradcam_{img_path.stem}.png"
        try:
            visualize_comparison(img_path, baseline_model, proposed_model, output_path)
        except Exception as e:
            print(f"  [WARNING] Failed for {img_path.name}: {e}")

    print(f"\n[INFO] Visualizations saved to {output_dir}")
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
