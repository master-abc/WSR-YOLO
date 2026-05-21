"""
实验2：消融实验 (Ablation Study)
================================
所有参数从 experiment/configs/experiment.yaml 读取，无需命令行参数。
直接运行: python experiment/exp2/run.py
"""

import sys
import json
import time
import gc
from pathlib import Path
from datetime import datetime

import yaml
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.register import register_custom_modules
from experiment.scripts.resume_utils import (
    load_existing_results, is_method_completed, resolve_resume,
    save_results_incremental,
)


def load_config():
    cfg_path = PROJECT_ROOT / "experiment" / "configs" / "experiment.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_single(variant, data_yaml, train_cfg, device, project_dir):
    """运行单个消融变体。"""
    from ultralytics import YOLO

    print(f"\n  Training: {variant['desc']}")
    print(f"  Config: {variant['config']}")

    # 训练级 resume
    runs_dir = Path(project_dir)
    resume_ckpt = resolve_resume(runs_dir, variant["name"])
    if resume_ckpt is not None:
        print(f"  [RESUME] Found incomplete training, resuming from {resume_ckpt.name}")
        model = YOLO(str(resume_ckpt))
        train_resume = True
    else:
        model = YOLO(variant["config"])
        train_resume = False
    start_time = time.time()

    model.train(
        data=data_yaml,
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        project=project_dir,
        name=variant["name"],
        device=device,
        optimizer=train_cfg["optimizer"],
        lr0=train_cfg["lr0"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
        warmup_epochs=min(train_cfg["warmup_epochs"], train_cfg["epochs"]),
        warmup_momentum=train_cfg["warmup_momentum"],
        warmup_bias_lr=train_cfg["warmup_bias_lr"],
        cos_lr=train_cfg["cos_lr"],
        amp=train_cfg["amp"],
        mosaic=train_cfg["mosaic"],
        mixup=train_cfg["mixup"],
        hsv_h=train_cfg["hsv_h"],
        hsv_s=train_cfg["hsv_s"],
        hsv_v=train_cfg["hsv_v"],
        flipud=train_cfg["flipud"],
        fliplr=train_cfg["fliplr"],
        seed=train_cfg["seed"],
        workers=train_cfg["workers"],
        plots=train_cfg["plots"],
        patience=0,
        exist_ok=True,
        verbose=False,
        resume=train_resume,
    )
    train_time = time.time() - start_time

    metrics = model.val(data=data_yaml, split="test", imgsz=train_cfg["imgsz"], batch=train_cfg["batch"])

    # 计算参数量
    params_M = sum(p.numel() for p in model.model.parameters()) / 1e6

    result = {
        "name": variant["name"],
        "desc": variant["desc"],
        "component": variant["component"],
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "f1": float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-8)),
        "params_M": params_M,
        "train_time_s": train_time,
    }

    print(f"    mAP@.5={result['map50']:.4f}  mAP@.5:.95={result['map50_95']:.4f}  Params={params_M:.2f}M")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def analyze_ablation(results):
    """分析消融结果。"""
    print(f"\n{'='*70}")
    print(f"  ABLATION ANALYSIS")
    print(f"{'='*70}")

    full = next((r for r in results if r.get("component") == "full" and "error" not in r), None)
    if not full:
        print("  [WARNING] Full DWGSA result not found, cannot compute deltas")
        return

    print(f"\n  Reference: {full['desc']} — mAP@.5={full['map50']:.4f}")
    print(f"\n  {'Variant':<30} {'Δ mAP@.5':>10} {'Δ mAP@.5:.95':>14} {'Interpretation'}")
    print(f"  {'-'*75}")

    interpretations = {
        "baseline": "No attention module",
        "wavelet": "Wavelet branch contributes this much",
        "sparse": "Sparse branch contributes this much",
        "geo_prior": "Geometric prior contributes this much",
        "adaptive": "Adaptive fusion contributes this much",
        "multi_level": "Multi-level DWT contributes this much",
    }

    for r in results:
        if r.get("component") == "full" or "error" in r:
            continue
        delta_50 = full["map50"] - r["map50"]
        delta_95 = full["map50_95"] - r["map50_95"]
        interp = interpretations.get(r["component"], "")
        print(f"  {r['desc']:<30} {delta_50:>+10.4f} {delta_95:>+14.4f} {interp}")


def generate_figures(results, output_dir):
    """生成消融实验图表。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    valid = [r for r in results if "error" not in r]
    if not valid:
        return

    full = next((r for r in valid if r.get("component") == "full"), None)
    if not full:
        return

    ablations = [r for r in valid if r.get("component") not in ("full", "baseline")]

    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r["desc"] for r in ablations]
    deltas = [(r["map50"] - full["map50"]) * 100 for r in ablations]
    colors = ["#d32f2f" if d < 0 else "#388e3c" for d in deltas]

    bars = ax.barh(range(len(names)), deltas, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Delta mAP@0.5 (%) relative to full DWGSA")
    ax.set_title("Exp2: Ablation Study - Component Contribution")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    for bar, d in zip(bars, deltas):
        ax.text(bar.get_width() + 0.1 * (1 if d >= 0 else -1), bar.get_y() + bar.get_height()/2,
                f"{d:+.2f}%", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "figure_ablation.png", dpi=150)
    plt.close()
    print(f"[FIGURES] Saved to {output_dir}/figure_ablation.png")


def main():
    cfg = load_config()
    train_cfg = cfg["train"].copy()
    device = ",".join(str(d) for d in cfg["hardware"]["device"])
    exp2_cfg = cfg["exp2"]
    is_smoke = cfg.get("mode", "full") == "smoke"

    if is_smoke:
        train_cfg["epochs"] = 2
        train_cfg["batch"] = 2
        train_cfg["workers"] = 0
        train_cfg["plots"] = False

    register_custom_modules()

    configs_dir = PROJECT_ROOT / "experiment" / "configs"

    mini_data = PROJECT_ROOT / "experiment" / "datasets_mini" / "mini_deeppcb.yaml"
    if is_smoke and mini_data.exists():
        data_yaml = str(mini_data)
    else:
        data_yaml = str(configs_dir / cfg["datasets"][exp2_cfg["data"]])
    output_dir = PROJECT_ROOT / "experiment" / exp2_cfg["output_dir"]
    project_dir = str(output_dir / "runs")

    variants = []
    for v in exp2_cfg["variants"]:
        variants.append({
            "name": v["name"],
            "desc": v["desc"],
            "config": str(configs_dir / v["config"]),
            "component": v["component"],
        })

    print(f"{'='*70}")
    print(f"  EXP2: ABLATION STUDY ({len(variants)} variants)")
    print(f"  Data: {exp2_cfg['data']} | Epochs: {train_cfg['epochs']} | "
          f"Batch: {train_cfg['batch']} | Device: {device}")
    print(f"{'='*70}")

    output_file = output_dir / "results.json"
    runs_dir = Path(project_dir)
    existing = load_existing_results(output_file)
    exp_meta = {
        "experiment": "exp2_ablation_study",
        "config": {"epochs": train_cfg["epochs"], "imgsz": train_cfg["imgsz"],
                   "batch": train_cfg["batch"], "device": device},
    }

    results = []
    completed_names = set()
    for variant in variants:
        if is_method_completed(variant["name"], existing, runs_dir):
            results.append(existing[variant["name"]])
            completed_names.add(variant["name"])
            print(f"  [SKIP] {variant['desc']} already completed "
                  f"(mAP@.5={existing[variant['name']].get('map50', 0):.4f})")

    for variant in variants:
        if variant["name"] in completed_names:
            continue
        try:
            result = run_single(variant, data_yaml, train_cfg, device, project_dir)
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] {variant['desc']}: {e}")
            results.append({
                "name": variant["name"], "desc": variant["desc"],
                "component": variant["component"],
                "map50": 0, "map50_95": 0, "precision": 0, "recall": 0, "f1": 0,
                "train_time_s": 0, "error": str(e),
            })
        save_results_incremental(output_file, results, exp_meta)

    analyze_ablation(results)

    print(f"\n[SAVED] Results -> {output_file}")
    generate_figures(results, output_dir)


if __name__ == "__main__":
    main()
