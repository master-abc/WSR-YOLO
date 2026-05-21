"""
实验3：跨数据集泛化实验
========================
测试模型在不同来源数据集上的泛化能力
数据集: DefectDet (5类, 268张) + PKU_PCB (6类, 693张)

所有参数从 experiment/configs/experiment.yaml 读取。
直接运行: python experiment/exp3/run.py
Smoke test: python experiment/exp3/run.py --smoke
"""

import sys
import json
import time
import gc
import argparse
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


def run_single(exp, data_yaml, train_cfg, device, project_dir):
    """运行单个跨数据集实验。"""
    from ultralytics import YOLO

    print(f"\n  Training: {exp['desc']}")
    print(f"  Dataset: {exp['data']} ({exp['nc']} classes)")

    runs_dir = Path(project_dir)
    resume_ckpt = resolve_resume(runs_dir, exp["name"])
    if resume_ckpt is not None:
        print(f"  [RESUME] Found incomplete training, resuming from {resume_ckpt.name}")
        model = YOLO(str(resume_ckpt))
        train_resume = True
    else:
        model = YOLO(exp["config"])
        train_resume = False
    start_time = time.time()

    model.train(
        data=data_yaml,
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        project=project_dir,
        name=exp["name"],
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

    metrics = model.val(data=data_yaml, split="val", imgsz=train_cfg["imgsz"], batch=train_cfg["batch"])

    per_class_ap = {}
    if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
        for i, ap in enumerate(metrics.box.ap50):
            per_class_ap[f"class_{i}"] = float(ap)

    result = {
        "name": exp["name"],
        "desc": exp["desc"],
        "dataset": exp["data"],
        "nc": exp["nc"],
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "f1": float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-8)),
        "per_class_ap50": per_class_ap,
        "train_time_s": train_time,
    }

    print(f"    mAP@.5={result['map50']:.4f}  mAP@.5:.95={result['map50_95']:.4f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def analyze_generalization(results):
    """分析泛化能力。"""
    print(f"\n{'='*70}")
    print(f"  CROSS-DATASET GENERALIZATION ANALYSIS")
    print(f"{'='*70}")

    datasets = {}
    for r in results:
        if "error" in r:
            continue
        ds = r["dataset"]
        if ds not in datasets:
            datasets[ds] = {}
        if "baseline" in r["name"]:
            datasets[ds]["baseline"] = r
        else:
            datasets[ds]["dwgsa"] = r

    header = f"  {'Dataset':<15} {'Method':<25} {'mAP@.5':>8} {'mAP@.5:.95':>11} {'Delta':>10}"
    print(f"\n{header}")
    print(f"  {'-'*70}")

    for ds_name, methods in datasets.items():
        baseline = methods.get("baseline")
        dwgsa = methods.get("dwgsa")
        if baseline:
            print(f"  {ds_name:<15} {baseline['desc']:<25} "
                  f"{baseline['map50']:>8.4f} {baseline['map50_95']:>11.4f} {'--':>10}")
        if dwgsa and baseline:
            delta = dwgsa["map50"] - baseline["map50"]
            print(f"  {'':<15} {dwgsa['desc']:<25} "
                  f"{dwgsa['map50']:>8.4f} {dwgsa['map50_95']:>11.4f} {delta:>+10.4f}")
        elif dwgsa:
            print(f"  {ds_name:<15} {dwgsa['desc']:<25} "
                  f"{dwgsa['map50']:>8.4f} {dwgsa['map50_95']:>11.4f} {'--':>10}")

    print(f"\n  Conclusion:")
    for ds_name, methods in datasets.items():
        baseline = methods.get("baseline")
        dwgsa = methods.get("dwgsa")
        if baseline and dwgsa:
            delta = dwgsa["map50"] - baseline["map50"]
            if delta > 0.005:
                print(f"    [OK] {ds_name}: DWGSA improves by +{delta:.4f} mAP@.5")
            elif delta > 0:
                print(f"    [~]  {ds_name}: DWGSA marginally improves by +{delta:.4f}")
            else:
                print(f"    [!]  {ds_name}: DWGSA does not improve ({delta:+.4f})")


def generate_figures(results, output_dir):
    """生成跨数据集泛化对比图表。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    valid = [r for r in results if "error" not in r]
    if not valid:
        return

    datasets = {}
    for r in valid:
        ds = r["dataset"]
        if ds not in datasets:
            datasets[ds] = {}
        if "baseline" in r["name"]:
            datasets[ds]["baseline"] = r
        else:
            datasets[ds]["dwgsa"] = r

    ds_names = list(datasets.keys())
    baseline_map50 = [datasets[ds].get("baseline", {}).get("map50", 0) * 100 for ds in ds_names]
    dwgsa_map50 = [datasets[ds].get("dwgsa", {}).get("map50", 0) * 100 for ds in ds_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ds_names))
    w = 0.35
    ax.bar(x - w/2, baseline_map50, w, label="YOLO11m (Baseline)", color="#1f77b4")
    ax.bar(x + w/2, dwgsa_map50, w, label="DWGSA-YOLO (Ours)", color="#d62728")

    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_title("Exp3: Cross-Dataset Generalization")
    ax.set_xticks(x)
    ax.set_xticklabels([ds.replace("_", " ").upper() for ds in ds_names])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i in range(len(ds_names)):
        if baseline_map50[i] > 0 and dwgsa_map50[i] > 0:
            delta = dwgsa_map50[i] - baseline_map50[i]
            ax.annotate(f"+{delta:.1f}%", xy=(x[i] + w/2, dwgsa_map50[i]),
                        xytext=(0, 5), textcoords="offset points", ha="center",
                        fontsize=9, color="#d62728", fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "figure_generalization.png", dpi=150)
    plt.close()
    print(f"[FIGURES] Saved to {output_dir}/figure_generalization.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Exp3: Cross-Dataset Generalization")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (2 epochs, batch=2)")
    parser.add_argument("--full", action="store_true", help="Full training (uses experiment.yaml settings)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    train_cfg = cfg["train"].copy()
    device = cfg["hardware"]["device"]
    exp3_cfg = cfg["exp3"]

    if args.smoke:
        train_cfg["epochs"] = 2
        train_cfg["batch"] = 2
        train_cfg["workers"] = 0
        train_cfg["plots"] = False

    register_custom_modules()

    configs_dir = PROJECT_ROOT / "experiment" / "configs"
    output_dir = PROJECT_ROOT / "experiment" / exp3_cfg["output_dir"]
    project_dir = str(output_dir / "runs")

    # smoke 模式使用 mini 数据集
    use_mini = args.smoke and (PROJECT_ROOT / "experiment" / "datasets_mini" / "mini_defectdet.yaml").exists()

    experiments = []
    for e in exp3_cfg["experiments"]:
        experiments.append({
            "name": e["name"],
            "desc": e["desc"],
            "config": str(configs_dir / e["config"]),
            "data": e["data"],
            "nc": e["nc"],
        })

    print(f"{'='*70}")
    print(f"  EXP3: CROSS-DATASET GENERALIZATION ({len(experiments)} runs)")
    print(f"  Epochs: {train_cfg['epochs']} | Batch: {train_cfg['batch']} | Device: {device}")
    print(f"{'='*70}")

    output_file = output_dir / "results.json"
    runs_dir = Path(project_dir)
    existing = load_existing_results(output_file)
    exp_meta = {
        "experiment": "exp3_cross_dataset_generalization",
        "config": {"epochs": train_cfg["epochs"], "imgsz": train_cfg["imgsz"],
                   "batch": train_cfg["batch"], "device": device},
    }

    results = []
    completed_names = set()
    for exp in experiments:
        if is_method_completed(exp["name"], existing, runs_dir):
            results.append(existing[exp["name"]])
            completed_names.add(exp["name"])
            print(f"  [SKIP] {exp['desc']} already completed "
                  f"(mAP@.5={existing[exp['name']].get('map50', 0):.4f})")

    for exp in experiments:
        if exp["name"] in completed_names:
            continue
        try:
            if use_mini:
                mini_name = f"mini_{exp['data']}.yaml"
                data_yaml = str(PROJECT_ROOT / "experiment" / "datasets_mini" / mini_name)
            else:
                data_yaml = str(configs_dir / cfg["datasets"][exp["data"]])
            result = run_single(exp, data_yaml, train_cfg, device, project_dir)
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] {exp['desc']}: {e}")
            results.append({
                "name": exp["name"], "desc": exp["desc"],
                "dataset": exp["data"], "nc": exp["nc"],
                "map50": 0, "map50_95": 0, "precision": 0, "recall": 0, "f1": 0,
                "per_class_ap50": {}, "train_time_s": 0, "error": str(e),
            })
        save_results_incremental(output_file, results, exp_meta)

    analyze_generalization(results)

    print(f"\n[SAVED] Results -> {output_file}")
    generate_figures(results, output_dir)


if __name__ == "__main__":
    main()
