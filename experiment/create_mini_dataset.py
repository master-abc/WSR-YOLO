"""
生成合成迷你 PCB 数据集用于 smoke test。

生成带有模拟缺陷标注的合成图像，确保实验脚本能正确运行。
实际论文实验应使用真实数据集（DeepPCB, DefectDet）。
"""

import os
import sys
import random
import numpy as np
from pathlib import Path

try:
    import cv2
except ImportError:
    print("[ERROR] opencv-python not installed. Run: pip install opencv-python")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
MINI_DEEPPCB_DIR = SCRIPT_DIR / "datasets_mini" / "mini_deeppcb"
MINI_DEFECTDET_DIR = SCRIPT_DIR / "datasets_mini" / "mini_defectdet"

DEEPPCB_CLASSES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]

DEFECTDET_CLASSES = ["missing_pad", "open_circuit", "short_circuit", "spur", "spurious_copper"]


def generate_pcb_image(width=640, height=640):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (35, 90, 35)

    for _ in range(random.randint(5, 15)):
        color = (random.randint(140, 180), random.randint(120, 160), random.randint(60, 100))
        thickness = random.randint(2, 8)
        if random.random() > 0.5:
            y = random.randint(50, height - 50)
            cv2.line(img, (0, y), (width, y), color, thickness)
        else:
            x = random.randint(50, width - 50)
            cv2.line(img, (x, 0), (x, height), color, thickness)

    for _ in range(random.randint(3, 8)):
        cx, cy = random.randint(50, width-50), random.randint(50, height-50)
        r = random.randint(8, 20)
        cv2.circle(img, (cx, cy), r, (160, 140, 80), -1)

    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def generate_labels(num_classes, max_objects=5):
    labels = []
    n = random.randint(1, max_objects)
    for _ in range(n):
        cls_id = random.randint(0, num_classes - 1)
        cx = random.uniform(0.1, 0.9)
        cy = random.uniform(0.1, 0.9)
        w = random.uniform(0.02, 0.15)
        h = random.uniform(0.02, 0.15)
        labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(labels)


def create_dataset(output_dir, num_classes, n_train=50, n_val=20):
    output_dir = Path(output_dir)
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    for split, count in [("train", n_train), ("val", n_val)]:
        for i in range(count):
            img = generate_pcb_image()
            img_path = output_dir / "images" / split / f"pcb_{i:04d}.jpg"
            cv2.imwrite(str(img_path), img)

            labels = generate_labels(num_classes)
            lbl_path = output_dir / "labels" / split / f"pcb_{i:04d}.txt"
            lbl_path.write_text(labels, encoding="utf-8")

    print(f"[OK] Created {n_train} train + {n_val} val images at {output_dir}")


def main():
    print("=" * 60)
    print("  Creating Mini Datasets for Smoke Testing")
    print("=" * 60)

    print("\n[1/2] Mini DeepPCB dataset (6 classes)...")
    create_dataset(MINI_DEEPPCB_DIR, num_classes=6, n_train=40, n_val=15)

    print("\n[2/2] Mini DefectDet dataset (5 classes)...")
    create_dataset(MINI_DEFECTDET_DIR, num_classes=5, n_train=40, n_val=15)

    mini_deeppcb_yaml = SCRIPT_DIR / "datasets_mini" / "mini_deeppcb.yaml"
    mini_deeppcb_yaml.write_text(
        f"path: {str(MINI_DEEPPCB_DIR).replace(chr(92), '/')}\n"
        "train: images/train\nval: images/val\ntest: images/val\n\n"
        "names:\n  0: open\n  1: short\n  2: mousebite\n"
        "  3: spur\n  4: copper\n  5: pin-hole\n",
        encoding="utf-8",
    )

    mini_defectdet_yaml = SCRIPT_DIR / "datasets_mini" / "mini_defectdet.yaml"
    mini_defectdet_yaml.write_text(
        f"path: {str(MINI_DEFECTDET_DIR).replace(chr(92), '/')}\n"
        "train: images/train\nval: images/val\ntest: images/val\n\n"
        "names:\n  0: missing_pad\n  1: open_circuit\n  2: short_circuit\n"
        "  3: spur\n  4: spurious_copper\n",
        encoding="utf-8",
    )

    print(f"\n[DONE] Dataset configs written:")
    print(f"  {mini_deeppcb_yaml}")
    print(f"  {mini_defectdet_yaml}")


if __name__ == "__main__":
    main()
