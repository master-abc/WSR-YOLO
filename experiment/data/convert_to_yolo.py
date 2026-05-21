"""
独立的 DeepPCB → YOLO 格式转换工具。

如果你已经手动下载了 DeepPCB 数据集，可以直接使用此脚本转换。

Usage:
    python data/convert_to_yolo.py --input /path/to/DeepPCB --output datasets/DeepPCB
"""

import argparse
from pathlib import Path
import sys

# 复用 download_deeppcb.py 中的转换逻辑
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_deeppcb import convert_to_yolo


def main():
    parser = argparse.ArgumentParser(description="Convert DeepPCB to YOLO format")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to raw DeepPCB dataset (containing PCBData/)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: ../datasets/DeepPCB)")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_dir = args.output or str(project_root / "datasets" / "DeepPCB")

    convert_to_yolo(args.input, output_dir, args.train_ratio, args.val_ratio)
    print(f"\n[Done] YOLO dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
