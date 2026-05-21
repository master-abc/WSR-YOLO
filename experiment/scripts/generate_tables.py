"""
绘制论文用的训练曲线对比图。

Usage:
    python experiment/scripts/generate_tables.py

从各 experiment/expN/runs/*/results.csv 中读取训练曲线，
绘制 mAP/Loss 多模型对比图，输出到 results/figure_training_curves.png。

注：LaTeX 表格生成由 experiment/scripts/eval_all.py 统一负责，本脚本不再生成。
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # DWGSA-YOLO/
sys.path.insert(0, str(PROJECT_ROOT))


def load_training_results(runs_dir):
    """从 runs 目录加载所有训练结果。"""
    results = {}
    runs_dir = Path(runs_dir)

    for exp_dir in runs_dir.rglob("*/results.csv"):
        exp_name = exp_dir.parent.name
        try:
            df = pd.read_csv(exp_dir)
            df.columns = df.columns.str.strip()
            results[exp_name] = df
        except Exception as e:
            print(f"[WARNING] Failed to load {exp_dir}: {e}")

    return results


def plot_training_curves(results, output_path):
    """绘制训练曲线对比图。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics = [
        ("metrics/mAP50(B)", "mAP@0.5"),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95"),
        ("train/box_loss", "Box Loss"),
    ]

    name_map = {
        "yolo11m_baseline": "YOLO11m (Baseline)",
        "dwgsa_yolo11m": "DWGSA-YOLO (Ours)",
        "fdsa_yolo11m": "YOLO11m + FDSA",
        "yolo11m_cbam": "YOLO11m + CBAM",
        "yolo11m_ema": "YOLO11m + EMA",
        "yolo11m_simam": "YOLO11m + SimAM",
        "yolo11m_coordatt": "YOLO11m + CoordAtt",
        "yolov9c_baseline": "YOLOv9-C",
        "dwgsa_wave_only": "Wavelet Only",
        "dwgsa_sparse_only": "Sparse Only",
        "dwgsa_no_geo_prior": "w/o GeoPrior",
        "dwgsa_no_adaptive": "w/o Adaptive",
        "dwgsa_single_level": "1-Level DWT",
    }

    colors = {
        "YOLO11m (Baseline)": "#1f77b4",
        "DWGSA-YOLO (Ours)": "#d62728",
        "YOLO11m + CBAM": "#2ca02c",
        "YOLO11m + EMA": "#ff7f0e",
        "YOLO11m + SimAM": "#9467bd",
        "YOLO11m + CoordAtt": "#8c564b",
        "YOLO11m + FDSA": "#e377c2",
    }

    for ax, (col_name, title) in zip(axes, metrics):
        for exp_name, df in results.items():
            display_name = name_map.get(exp_name, exp_name)
            color = colors.get(display_name, None)

            if col_name in df.columns:
                ax.plot(df["epoch"], df[col_name],
                        label=display_name, color=color, linewidth=1.5)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training curves saved to {output_path}")


def main():
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Plotting Training Curves")
    print("=" * 60)

    # 合并 exp1 + exp2 所有训练曲线
    all_results = {}
    for exp_subdir in ("exp1", "exp2", "exp3"):
        runs_dir = PROJECT_ROOT / "experiment" / exp_subdir / "runs"
        if runs_dir.exists():
            all_results.update(load_training_results(runs_dir))

    if not all_results:
        print("\n[WARNING] No training results found in experiment/exp*/runs/. "
              "Train models first.")
        return

    print(f"\n[INFO] Found {len(all_results)} experiment runs")
    plot_training_curves(all_results, results_dir / "figure_training_curves.png")

    print("\n[INFO] Done. For LaTeX tables, run: python experiment/scripts/eval_all.py")


if __name__ == "__main__":
    main()
