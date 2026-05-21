"""
实验1：SOTA 对比实验
====================
所有参数从 experiment/configs/experiment.yaml 读取，无需命令行参数。
直接运行: python experiment/exp1/run.py
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
    """加载实验配置。"""
    cfg_path = PROJECT_ROOT / "experiment" / "configs" / "experiment.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_model_complexity(model_cfg_path, imgsz):
    """计算模型参数量和 FLOPs（在可用时使用 CUDA 加速）。"""
    from ultralytics import YOLO
    try:
        model = YOLO(model_cfg_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.model.to(device)
        params = sum(p.numel() for p in model.model.parameters()) / 1e6
        dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
        try:
            from thop import profile
            flops, _ = profile(model.model, inputs=(dummy,), verbose=False)
            flops_g = flops / 1e9
        except ImportError:
            flops_g = -1
        del model, dummy
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return params, flops_g
    except Exception:
        return -1, -1


def run_single(method, data_yaml, train_cfg, device, project_dir):
    """运行单个方法的训练和评估。"""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"  Training: {method['desc']}")
    print(f"  Config: {method['config']}")
    print(f"{'='*60}")

    # 训练级 resume：若上次训练中断（last.pt 存在但 best.pt 不存在），从 last.pt 恢复
    runs_dir = Path(project_dir)
    resume_ckpt = resolve_resume(runs_dir, method["name"])
    if resume_ckpt is not None:
        print(f"  [RESUME] Found incomplete training, resuming from {resume_ckpt.name}")
        model = YOLO(str(resume_ckpt))
        train_resume = True
    else:
        model = YOLO(method["config"])
        train_resume = False
    start_time = time.time()

    model.train(
        data=data_yaml,
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        project=project_dir,
        name=method["name"],
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
    params, flops = compute_model_complexity(method["config"], train_cfg["imgsz"])

    # FPS 测量 (batch=1, 300 次推理取平均)
    fps = -1
    try:
        dev = next(model.model.parameters()).device
        dummy = torch.randn(1, 3, train_cfg["imgsz"], train_cfg["imgsz"]).to(dev)
        model.model.eval()
        with torch.no_grad():
            for _ in range(50):
                model.model(dummy)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(300):
                model.model(dummy)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            fps = 300.0 / (time.time() - t0)
        del dummy
    except Exception:
        pass

    result = {
        "name": method["name"],
        "desc": method["desc"],
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "f1": float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-8)),
        "params_M": params,
        "flops_G": flops,
        "fps": fps,
        "train_time_s": train_time,
        "epochs": train_cfg["epochs"],
    }

    print(f"    mAP@.5={result['map50']:.4f}  mAP@.5:.95={result['map50_95']:.4f}  "
          f"P={result['precision']:.4f}  R={result['recall']:.4f}  Params={params:.2f}M")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def print_results_table(results):
    """打印结果表格。"""
    print(f"\n{'='*90}")
    print(f"  EXP1 RESULTS: SOTA COMPARISON")
    print(f"{'='*90}")
    print(f"  {'Method':<25} {'mAP@.5':>8} {'mAP@.5:.95':>11} {'P':>7} {'R':>7} {'F1':>7} {'Params':>8} {'FLOPs':>8}")
    print(f"  {'-'*90}")
    for r in results:
        print(f"  {r['desc']:<25} {r['map50']:>8.4f} {r['map50_95']:>11.4f} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} "
              f"{r['params_M']:>7.2f}M {r['flops_G']:>7.1f}G")
    print(f"  {'-'*90}")


def generate_figures(results, output_dir):
    """生成 Exp1 对比图表。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    valid = [r for r in results if "error" not in r]
    if not valid:
        return

    names = [r["desc"] for r in valid]
    map50 = [r["map50"] * 100 for r in valid]
    map50_95 = [r["map50_95"] * 100 for r in valid]

    # Figure 1: mAP 对比柱状图
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, map50, w, label="mAP@0.5", color="#2196F3")
    bars2 = ax.bar(x + w/2, map50_95, w, label="mAP@0.5:0.95", color="#FF9800")
    ax.set_ylabel("mAP (%)")
    ax.set_title("Exp1: SOTA Comparison on DeepPCB")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in bars1 + bars2:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
    plt.tight_layout()
    fig.savefig(output_dir / "figure_sota_comparison.png", dpi=150)
    plt.close()

    # Figure 2: 精度-参数量散点图
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in valid:
        is_ours = "DWGSA" in r["desc"] or "Ours" in r["desc"]
        color = "#d62728" if is_ours else "#1f77b4"
        marker = "*" if is_ours else "o"
        size = 200 if is_ours else 80
        ax.scatter(r["params_M"], r["map50"] * 100, c=color, marker=marker, s=size, zorder=3)
        ax.annotate(r["desc"], (r["params_M"], r["map50"] * 100),
                    xytext=(5, 5), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_title("Accuracy vs. Model Size")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "figure_accuracy_vs_params.png", dpi=150)
    plt.close()

    print(f"[FIGURES] Saved to {output_dir}/figure_*.png")


def main():
    cfg = load_config()
    train_cfg = cfg["train"].copy()
    device = ",".join(str(d) for d in cfg["hardware"]["device"])
    exp1_cfg = cfg["exp1"]
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
        data_yaml = str(configs_dir / cfg["datasets"][exp1_cfg["data"]])
    output_dir = PROJECT_ROOT / "experiment" / exp1_cfg["output_dir"]
    project_dir = str(output_dir / "runs")

    methods = []
    for m in exp1_cfg["methods"]:
        methods.append({
            "name": m["name"],
            "desc": m["desc"],
            "config": str(configs_dir / m["config"]),
        })

    print(f"{'='*70}")
    print(f"  EXP1: SOTA COMPARISON")
    print(f"  Data: {exp1_cfg['data']} | Epochs: {train_cfg['epochs']} | "
          f"Batch: {train_cfg['batch']} | Device: {device}")
    print(f"  Methods: {len(methods)}")
    print(f"{'='*70}")

    output_file = output_dir / "results.json"
    runs_dir = Path(project_dir)
    existing = load_existing_results(output_file)
    exp_meta = {
        "experiment": "exp1_sota_comparison",
        "config": {"epochs": train_cfg["epochs"], "imgsz": train_cfg["imgsz"],
                   "batch": train_cfg["batch"], "device": device},
    }

    results = []
    # 预填已完成方法（保持顺序与 methods 一致）
    completed_names = set()
    for method in methods:
        if is_method_completed(method["name"], existing, runs_dir):
            results.append(existing[method["name"]])
            completed_names.add(method["name"])
            print(f"  [SKIP] {method['desc']} already completed "
                  f"(mAP@.5={existing[method['name']].get('map50', 0):.4f})")

    for method in methods:
        if method["name"] in completed_names:
            continue
        try:
            result = run_single(method, data_yaml, train_cfg, device, project_dir)
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] {method['desc']} failed: {e}")
            results.append({
                "name": method["name"], "desc": method["desc"],
                "map50": 0, "map50_95": 0, "precision": 0, "recall": 0, "f1": 0,
                "params_M": -1, "flops_G": -1, "train_time_s": 0,
                "epochs": train_cfg["epochs"], "error": str(e),
            })
        # 增量保存：每完成一个方法就落盘，崩溃也不丢已得结果
        save_results_incremental(output_file, results, exp_meta)

    print_results_table(results)

    print(f"\n[SAVED] Results -> {output_file}")
    generate_figures(results, output_dir)


if __name__ == "__main__":
    main()
