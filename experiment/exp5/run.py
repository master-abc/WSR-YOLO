"""
实验5：噪声鲁棒性实验 (Noise Robustness Study)
================================================
所有参数从 experiment/configs/experiment.yaml 读取，无需命令行参数。
直接运行: python experiment/exp5/run.py
"""

import sys
import json
import time
import gc
from pathlib import Path
from datetime import datetime

import yaml
import torch
import numpy as np
import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.register import register_custom_modules
from experiment.scripts.resume_utils import (
    load_existing_results, is_method_completed, resolve_resume,
    save_results_incremental,
)

NOISE_LEVELS = {
    "clean": {"sigma": 0, "desc": "Clean (No Noise)"},
    "low": {"sigma": 3, "desc": "Low Noise (σ=3, Realistic)"},
    "medium": {"sigma": 6, "desc": "Medium Noise (σ=6, 2x)"},
    "high": {"sigma": 9, "desc": "High Noise (σ=9, 3x)"},
    "very_high": {"sigma": 15, "desc": "Very High Noise (σ=15, 5x)"},
}


def add_gaussian_noise(image, sigma):
    if sigma == 0:
        return image
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def create_noisy_test_set(source_dir, output_dir, sigma):
    """只对测试集添加噪声，不复制train/val"""
    source_images = source_dir / "images" / "test"
    source_labels = source_dir / "labels" / "test"

    output_images = output_dir / "images" / "test"
    output_labels = output_dir / "labels" / "test"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    image_files = list(source_images.glob("*.jpg")) + list(source_images.glob("*.png"))
    print(f"  Creating noisy test set (σ={sigma}): {len(image_files)} images")

    for img_path in tqdm(image_files, desc=f"  σ={sigma}", leave=False):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        noisy_img = add_gaussian_noise(img, sigma)
        cv2.imwrite(str(output_images / img_path.name), noisy_img)

        label_path = source_labels / (img_path.stem + ".txt")
        if label_path.exists():
            (output_labels / label_path.name).write_text(label_path.read_text())


def create_noisy_yaml(base_yaml, output_yaml, source_dir, noisy_test_dir):
    """创建yaml：train/val指向原始数据，test指向噪声目录"""
    with open(base_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["path"] = str(source_dir)
    config["test"] = str(noisy_test_dir / "images" / "test")
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def load_config():
    cfg_path = PROJECT_ROOT / "experiment" / "configs" / "experiment.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_on_clean(method, data_yaml, train_cfg, device, project_dir):
    """在clean数据上训练模型（只训练一次）"""
    from ultralytics import YOLO

    run_name = method["name"]
    runs_dir = Path(project_dir)

    # 检查是否已训练完成
    best_pt = runs_dir / run_name / "weights" / "best.pt"
    if best_pt.exists():
        print(f"  [SKIP] {method['desc']} already trained, loading {best_pt}")
        return str(best_pt)

    print(f"\n{'='*60}")
    print(f"  Training on CLEAN data: {method['desc']}")
    print(f"{'='*60}")

    resume_ckpt = resolve_resume(runs_dir, run_name)
    if resume_ckpt is not None:
        print(f"  [RESUME] Resuming from {resume_ckpt.name}")
        model = YOLO(str(resume_ckpt))
        train_resume = True
    else:
        model = YOLO(method["config"])
        train_resume = False

    model.train(
        data=data_yaml,
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        project=project_dir,
        name=run_name,
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

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return str(best_pt)


def evaluate_on_noisy(weights_path, data_yaml, noise_level, noise_sigma, train_cfg, device):
    """用训练好的模型在噪声测试集上评估"""
    from ultralytics import YOLO

    model = YOLO(weights_path)
    metrics = model.val(
        data=data_yaml,
        split="test",
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        device=device,
        verbose=False,
    )

    result = {
        "noise_level": noise_level,
        "noise_sigma": noise_sigma,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "f1": float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-8)),
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def analyze_robustness(results):
    print(f"\n{'='*80}")
    print(f"  NOISE ROBUSTNESS ANALYSIS")
    print(f"{'='*80}")

    methods = {}
    for r in results:
        if "error" in r:
            continue
        method = r["method"]
        if method not in methods:
            methods[method] = []
        methods[method].append(r)

    for method_name, method_results in methods.items():
        method_results = sorted(method_results, key=lambda x: x["noise_sigma"])
        print(f"\n  Method: {method_results[0]['desc']}")
        print(f"  {'Noise Level':<20} {'σ':>5} {'mAP@.5':>8} {'Δ mAP':>8} {'Degradation':>12}")
        print(f"  {'-'*60}")

        clean_map = method_results[0]["map50"]
        for r in method_results:
            delta = r["map50"] - clean_map
            degradation = (delta / clean_map * 100) if clean_map > 0 else 0
            print(f"  {r['noise_level']:<20} {r['noise_sigma']:>5.1f} {r['map50']:>8.4f} "
                  f"{delta:>+8.4f} {degradation:>+11.2f}%")

    if "yolo11m_baseline" in methods and "dwgsa_yolo11m" in methods:
        print(f"\n{'='*80}")
        print(f"  ROBUSTNESS COMPARISON: Baseline vs DWGSA")
        print(f"{'='*80}")
        baseline = sorted(methods["yolo11m_baseline"], key=lambda x: x["noise_sigma"])
        dwgsa = sorted(methods["dwgsa_yolo11m"], key=lambda x: x["noise_sigma"])

        print(f"  {'Noise Level':<20} {'Baseline':>10} {'DWGSA':>10} {'Advantage':>10}")
        print(f"  {'-'*60}")
        for b, d in zip(baseline, dwgsa):
            advantage = d["map50"] - b["map50"]
            print(f"  {b['noise_level']:<20} {b['map50']:>10.4f} {d['map50']:>10.4f} {advantage:>+10.4f}")


def generate_figures(results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    valid = [r for r in results if "error" not in r]
    if not valid:
        return

    methods = {}
    for r in valid:
        method = r["method"]
        if method not in methods:
            methods[method] = []
        methods[method].append(r)

    colors = {"yolo11m_baseline": "#1f77b4", "dwgsa_yolo11m": "#d62728"}
    markers = {"yolo11m_baseline": "o", "dwgsa_yolo11m": "*"}
    labels = {"yolo11m_baseline": "YOLO11m Baseline", "dwgsa_yolo11m": "DWGSA-YOLO (Ours)"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for method_name, method_results in methods.items():
        method_results = sorted(method_results, key=lambda x: x["noise_sigma"])
        sigmas = [r["noise_sigma"] for r in method_results]
        maps = [r["map50"] * 100 for r in method_results]
        ax.plot(sigmas, maps, marker=markers.get(method_name, "s"), markersize=10,
                linewidth=2, color=colors.get(method_name, "#333"), label=labels.get(method_name, method_name))

    ax.set_xlabel("Noise Standard Deviation (σ)", fontsize=12)
    ax.set_ylabel("mAP@0.5 (%)", fontsize=12)
    ax.set_title("Noise Robustness: Clean-Trained Models on Noisy Test Data", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, 3, 6, 9, 15])
    plt.tight_layout()
    fig.savefig(output_dir / "figure_noise_robustness.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    for method_name, method_results in methods.items():
        method_results = sorted(method_results, key=lambda x: x["noise_sigma"])
        clean_map = method_results[0]["map50"]
        sigmas = [r["noise_sigma"] for r in method_results[1:]]
        degradations = [(r["map50"] - clean_map) / clean_map * 100 for r in method_results[1:]]
        ax.plot(sigmas, degradations, marker=markers.get(method_name, "s"), markersize=10,
                linewidth=2, color=colors.get(method_name, "#333"), label=labels.get(method_name, method_name))

    ax.set_xlabel("Noise Standard Deviation (σ)", fontsize=12)
    ax.set_ylabel("Performance Degradation (%)", fontsize=12)
    ax.set_title("Performance Drop Relative to Clean Baseline", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks([3, 6, 9, 15])
    plt.tight_layout()
    fig.savefig(output_dir / "figure_noise_degradation.png", dpi=150)
    plt.close()

    print(f"[FIGURES] Saved to {output_dir}/figure_noise_*.png")


def main():
    cfg = load_config()
    train_cfg = cfg["train"].copy()
    device = ",".join(str(d) for d in cfg["hardware"]["device"])
    is_smoke = cfg.get("mode", "full") == "smoke"

    if is_smoke:
        train_cfg["epochs"] = 2
        train_cfg["batch"] = 2
        train_cfg["workers"] = 0
        train_cfg["plots"] = False

    register_custom_modules()

    configs_dir = PROJECT_ROOT / "experiment" / "configs"
    clean_data_yaml = str(configs_dir / cfg["datasets"]["deeppcb"])
    source_dataset = PROJECT_ROOT / "datasets" / "DeepPCB"

    if is_smoke:
        mini_data = PROJECT_ROOT / "experiment" / "datasets_mini" / "mini_deeppcb.yaml"
        if mini_data.exists():
            clean_data_yaml = str(mini_data)
            source_dataset = PROJECT_ROOT / "experiment" / "datasets_mini" / "mini_deeppcb"

    output_dir = PROJECT_ROOT / "experiment" / "exp5"
    project_dir = str(output_dir / "runs")
    noisy_datasets_dir = output_dir / "noisy_datasets"

    methods = [
        {"name": "yolo11m_baseline", "desc": "YOLO11m Baseline",
         "config": str(configs_dir / "yolo11m_baseline.yaml")},
        {"name": "dwgsa_yolo11m", "desc": "DWGSA-YOLO (Ours)",
         "config": str(configs_dir / "dwgsa_yolo11m.yaml")},
    ]

    print(f"{'='*70}")
    print(f"  EXP5: NOISE ROBUSTNESS STUDY")
    print(f"  Strategy: Train on CLEAN → Test on NOISY")
    print(f"  Methods: {len(methods)} | Noise Levels: {len(NOISE_LEVELS)}")
    print(f"  Epochs: {train_cfg['epochs']} | Batch: {train_cfg['batch']} | Device: {device}")
    print(f"{'='*70}")

    # ================================================================
    # Phase 1: 在clean数据上训练（每个方法只训练1次）
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Training on clean data")
    print(f"{'='*70}")

    trained_weights = {}
    for method in methods:
        weights = train_on_clean(method, clean_data_yaml, train_cfg, device, project_dir)
        trained_weights[method["name"]] = weights
        print(f"  {method['desc']}: {weights}")

    # ================================================================
    # Phase 2: 生成噪声测试集
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Creating noisy test datasets")
    print(f"{'='*70}")

    noisy_yamls = {}
    for level_name, level_info in NOISE_LEVELS.items():
        sigma = level_info["sigma"]
        if sigma == 0:
            noisy_yamls[level_name] = clean_data_yaml
        else:
            noisy_path = noisy_datasets_dir / f"deeppcb_sigma_{sigma}"
            noisy_yaml = noisy_datasets_dir / f"deeppcb_sigma_{sigma}.yaml"
            if not noisy_yaml.exists():
                create_noisy_test_set(source_dataset, noisy_path, sigma)
                create_noisy_yaml(clean_data_yaml, str(noisy_yaml), source_dataset, noisy_path)
            noisy_yamls[level_name] = str(noisy_yaml)

    # ================================================================
    # Phase 3: 在各噪声水平上评估
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 3: Evaluating on noisy test sets")
    print(f"{'='*70}")

    output_file = output_dir / "results.json"
    results = []

    for method in methods:
        weights = trained_weights[method["name"]]
        print(f"\n  Evaluating: {method['desc']}")
        print(f"  Weights: {weights}")

        for level_name, level_info in NOISE_LEVELS.items():
            sigma = level_info["sigma"]
            data_yaml = noisy_yamls[level_name]

            print(f"    Testing σ={sigma} ({level_name})...", end=" ")
            try:
                r = evaluate_on_noisy(weights, data_yaml, level_name, sigma, train_cfg, device)
                r["name"] = f"{method['name']}_noise_{level_name}"
                r["method"] = method["name"]
                r["desc"] = method["desc"]
                results.append(r)
                print(f"mAP@.5={r['map50']:.4f}")
            except Exception as e:
                print(f"FAILED: {e}")
                results.append({
                    "name": f"{method['name']}_noise_{level_name}",
                    "method": method["name"], "desc": method["desc"],
                    "noise_level": level_name, "noise_sigma": sigma,
                    "map50": 0, "map50_95": 0, "precision": 0, "recall": 0, "f1": 0,
                    "error": str(e),
                })

    # 保存结果
    exp_meta = {
        "experiment": "exp5_noise_robustness",
        "strategy": "train_clean_test_noisy",
        "noise_levels": NOISE_LEVELS,
        "config": {"epochs": train_cfg["epochs"], "imgsz": train_cfg["imgsz"],
                   "batch": train_cfg["batch"], "device": device},
    }
    save_results_incremental(output_file, results, exp_meta)

    analyze_robustness(results)
    print(f"\n[SAVED] Results -> {output_file}")
    generate_figures(results, output_dir)


if __name__ == "__main__":
    main()
