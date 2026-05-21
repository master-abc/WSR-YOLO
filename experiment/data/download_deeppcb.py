"""
下载 DeepPCB 数据集并转换为 YOLO 格式。

DeepPCB 数据集实际结构:
    DeepPCB/
    ├── PCBData/
    │   ├── trainval.txt          (训练+验证列表)
    │   ├── test.txt              (测试列表)
    │   ├── group00041/
    │   │   ├── 00041/            (图片目录)
    │   │   │   ├── 00041000_temp.jpg  (模板图)
    │   │   │   ├── 00041000_test.jpg  (测试图)
    │   │   │   └── 00041000.jpg       (或直接 .jpg)
    │   │   └── 00041_not/        (标注目录)
    │   │       ├── 00041000.txt
    │   │       └── ...
    │   └── ...

列表文件格式: group.../XXXXX/XXXXXXXX.jpg group.../XXXXX_not/XXXXXXXX.txt
标注格式 (原始): x1 y1 x2 y2 class_id (1-indexed, 1-6)
YOLO 格式 (目标): class_id cx cy w h (0-indexed, normalized)

图片尺寸: 640x640
"""

import os
import sys
import shutil
import random
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "DeepPCB"


def download_deeppcb():
    """从 GitHub 克隆 DeepPCB 数据集。"""
    repo_url = "https://github.com/tangsanli5201/DeepPCB.git"
    clone_dir = PROJECT_ROOT / "datasets" / "DeepPCB_raw"

    if clone_dir.exists():
        print(f"[INFO] Raw dataset already exists at {clone_dir}")
        return clone_dir

    print(f"[INFO] Cloning DeepPCB dataset from {repo_url}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
        check=True,
    )
    print(f"[INFO] Dataset cloned to {clone_dir}")
    return clone_dir


def parse_annotation(ann_path, img_width=640, img_height=640):
    """解析 DeepPCB 原始标注文件，转换为 YOLO 格式。"""
    labels = []
    if not os.path.exists(ann_path):
        return labels

    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            x1, y1, x2, y2 = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            class_id = int(parts[4]) - 1  # 1-indexed → 0-indexed

            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            cx = (x1 + x2) / 2.0 / img_width
            cy = (y1 + y2) / 2.0 / img_height
            w = (x2 - x1) / img_width
            h = (y2 - y1) / img_height

            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w = max(0, min(1, w))
            h = max(0, min(1, h))

            if w > 0 and h > 0:
                labels.append((class_id, cx, cy, w, h))

    return labels


def load_split_list(list_file, pcb_data_dir):
    """从 trainval.txt / test.txt 加载图片-标注对。

    列表文件中图片路径为 XXXXXXXX.jpg，但实际文件名为 XXXXXXXX_test.jpg。
    """
    samples = []
    with open(list_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            img_rel = parts[0]
            ann_rel = parts[1]

            # 标注文件直接匹配
            ann_path = pcb_data_dir / ann_rel
            if not ann_path.exists():
                continue

            # 图片文件：列表中是 XXXXXXXX.jpg，实际是 XXXXXXXX_test.jpg
            img_path = pcb_data_dir / img_rel
            if not img_path.exists():
                # 尝试 _test.jpg 后缀
                stem = Path(img_rel).stem
                parent = Path(img_rel).parent
                img_path = pcb_data_dir / parent / f"{stem}_test.jpg"

            if img_path.exists():
                samples.append((img_path, ann_path))

    return samples


def convert_to_yolo(raw_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """将 DeepPCB 数据集转换为 YOLO 格式并划分训练/验证/测试集。"""
    pcb_data_dir = Path(raw_dir) / "PCBData"
    if not pcb_data_dir.exists():
        print(f"[ERROR] PCBData directory not found in {raw_dir}")
        sys.exit(1)

    output_dir = Path(output_dir)

    # 创建输出目录结构
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 使用官方 trainval.txt 和 test.txt 划分
    trainval_file = pcb_data_dir / "trainval.txt"
    test_file = pcb_data_dir / "test.txt"

    if trainval_file.exists() and test_file.exists():
        print("[INFO] Using official train/test split files")
        trainval_samples = load_split_list(trainval_file, pcb_data_dir)
        test_samples = load_split_list(test_file, pcb_data_dir)

        # 从 trainval 中划分 train 和 val，目标比例 850/150（与 README 一致）
        random.seed(42)
        random.shuffle(trainval_samples)
        n_val = int(round(len(trainval_samples) * 150 / 1000))
        val_samples = trainval_samples[:n_val]
        train_samples = trainval_samples[n_val:]

        splits = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        }
    else:
        # 备用方案：手动搜索所有文件
        print("[INFO] Split files not found, scanning directories...")
        all_samples = []
        for ann_file in sorted(pcb_data_dir.rglob("*_not/*.txt")):
            # 从标注路径推断图片路径
            # 标注: groupXXXXX/XXXXX_not/XXXXXXXX.txt
            # 图片: groupXXXXX/XXXXX/XXXXXXXX_test.jpg 或 XXXXXXXX.jpg
            ann_dir = ann_file.parent  # XXXXX_not
            img_dir_name = ann_dir.name.replace("_not", "")
            img_dir = ann_dir.parent / img_dir_name
            stem = ann_file.stem

            # 尝试多种图片命名
            img_path = None
            for suffix in [f"{stem}_test.jpg", f"{stem}.jpg", f"{stem}_temp.jpg"]:
                candidate = img_dir / suffix
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path and ann_file.exists():
                all_samples.append((img_path, ann_file))

        print(f"[INFO] Found {len(all_samples)} annotated images by scanning")

        if len(all_samples) == 0:
            print("[ERROR] No samples found. Check dataset structure.")
            sys.exit(1)

        random.seed(42)
        random.shuffle(all_samples)
        n_train = int(len(all_samples) * train_ratio)
        n_val = int(len(all_samples) * val_ratio)

        splits = {
            "train": all_samples[:n_train],
            "val": all_samples[n_train:n_train + n_val],
            "test": all_samples[n_train + n_val:],
        }

    # 统计信息
    class_names = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]
    total_labels = {name: 0 for name in class_names}

    for split_name, split_samples in splits.items():
        for img_path, ann_path in split_samples:
            # 复制图片（统一命名避免冲突）
            unique_name = f"{img_path.parent.parent.name}_{img_path.stem}"
            dst_img = output_dir / "images" / split_name / f"{unique_name}.jpg"
            shutil.copy2(img_path, dst_img)

            # 转换并保存标注
            labels = parse_annotation(str(ann_path))
            dst_label = output_dir / "labels" / split_name / f"{unique_name}.txt"
            with open(dst_label, "w") as f:
                for cls_id, cx, cy, w, h in labels:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    if 0 <= cls_id < len(class_names):
                        total_labels[class_names[cls_id]] += 1

        print(f"[INFO] {split_name}: {len(split_samples)} images")

    # 打印统计
    print("\n[INFO] Class distribution:")
    for name, count in total_labels.items():
        print(f"  {name}: {count}")
    print(f"  Total: {sum(total_labels.values())}")

    return output_dir


def main():
    """主函数：下载并转换 DeepPCB 数据集。"""
    print("=" * 60)
    print("DeepPCB Dataset Preparation for DWGSA-YOLO")
    print("=" * 60)

    # Step 1: 下载
    raw_dir = download_deeppcb()

    # Step 2: 转换
    print(f"\n[INFO] Converting to YOLO format...")
    convert_to_yolo(raw_dir, DATASET_DIR)

    print(f"\n[INFO] Dataset ready at: {DATASET_DIR}")
    print("[INFO] You can now start training with configs/deeppcb.yaml")


if __name__ == "__main__":
    main()
