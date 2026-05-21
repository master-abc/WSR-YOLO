"""
统一评估所有已训练模型，生成论文表格数据。

Usage:
    python experiment/scripts/eval_all.py

输出:
    - results/sota_comparison.csv    (Table 1: SOTA 对比)
    - results/ablation_table.csv     (Table 2: 消融实验)
    - results/generalization.csv     (Table 3: 泛化实验)
    - results/tables.tex             (LaTeX 格式表格)
"""

import sys
import time
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # DWGSA-YOLO/
sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.register import register_custom_modules
from ultralytics import YOLO


def count_parameters(model):
    """统计模型参数量 (M)。"""
    return sum(p.numel() for p in model.model.parameters()) / 1e6


def measure_gflops(model, imgsz=640):
    """估算模型 GFLOPs。"""
    try:
        from thop import profile
        device = next(model.model.parameters()).device
        dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
        flops, _ = profile(model.model, inputs=(dummy,), verbose=False)
        return flops / 1e9
    except ImportError:
        pass
    try:
        from ultralytics.utils.torch_utils import get_flops
        return get_flops(model.model, imgsz)
    except Exception:
        return 0.0


def measure_fps(model, imgsz=640, n_warmup=50, n_test=200):
    """测量推理 FPS (单张图片)。"""
    device = next(model.model.parameters()).device
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)

    model.model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model.model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_test):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    return 1.0 / np.mean(times)


def evaluate_model(weight_path, data_cfg, model_name):
    """评估单个模型，返回指标字典。"""
    print(f"\n{'─'*50}")
    print(f"  Evaluating: {model_name}")
    print(f"  Weights:    {weight_path}")
    print(f"{'─'*50}")

    model = YOLO(weight_path)
    metrics = model.val(data=data_cfg, split="test", verbose=False)

    params = count_parameters(model)
    gflops = measure_gflops(model)
    fps = measure_fps(model)

    precision = metrics.box.mp
    recall = metrics.box.mr
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    result = {
        "Model": model_name,
        "mAP@.5": metrics.box.map50,
        "mAP@.5:.95": metrics.box.map,
        "F1-Score": f1,
        "Params (M)": params,
        "GFLOPs": gflops,
        "FPS": fps,
    }

    print(f"  mAP@.5={result['mAP@.5']:.4f}  mAP@.5:.95={result['mAP@.5:.95']:.4f}  "
          f"F1={result['F1-Score']:.3f}  Params={result['Params (M)']:.1f}M  "
          f"GFLOPs={result['GFLOPs']:.1f}  FPS={result['FPS']:.0f}")

    return result


def load_experiment_config():
    """从 experiment.yaml 加载实验配置。"""
    cfg_path = PROJECT_ROOT / "experiment" / "configs" / "experiment.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_models(cfg, exp_key):
    """从实验配置中发现已训练模型的权重路径。"""
    exp_cfg = cfg[exp_key]
    datasets_cfg = cfg.get("datasets", {})
    default_data = datasets_cfg.get("deeppcb", "deeppcb.yaml")
    output_dir = exp_cfg.get("output_dir", exp_key)
    project = f"experiment/{output_dir}/runs"

    if "methods" in exp_cfg:
        experiments = exp_cfg["methods"]
    elif "variants" in exp_cfg:
        experiments = exp_cfg["variants"]
    elif "experiments" in exp_cfg:
        experiments = exp_cfg["experiments"]
    else:
        return []

    models = []
    for exp in experiments:
        weight_path = PROJECT_ROOT / project / exp["name"] / "weights" / "best.pt"
        exp_data = exp.get("data", exp_cfg.get("data", "deeppcb"))
        data_cfg = datasets_cfg.get(exp_data, exp_data)
        if not Path(data_cfg).is_absolute():
            data_cfg = str(PROJECT_ROOT / "experiment" / "configs" / data_cfg)
        models.append({
            "name": exp["desc"],
            "exp_name": exp["name"],
            "weight_path": weight_path,
            "data_cfg": data_cfg,
        })

    return models


def generate_latex_sota(df, output_lines):
    """生成 SOTA 对比 LaTeX 表格。"""
    output_lines.append("% Table 1: SOTA Comparison on DeepPCB")
    output_lines.append("\\begin{table}[htbp]")
    output_lines.append("    \\centering")
    output_lines.append("    \\caption{Performance comparison on DeepPCB dataset.}")
    output_lines.append("    \\label{tab:sota}")
    output_lines.append("    \\setlength{\\tabcolsep}{2.5pt}")
    output_lines.append("    \\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lcccccc@{}}")
    output_lines.append("        \\toprule")
    output_lines.append("        \\textbf{Model} & \\textbf{mAP@.5} & \\textbf{mAP@.5:.95} & "
                        "\\textbf{F1} & \\textbf{Params(M)} & \\textbf{GFLOPs} & \\textbf{FPS} \\\\")
    output_lines.append("        \\midrule")

    for _, row in df.iterrows():
        is_ours = "DWGSA" in row["Model"]
        prefix = "\\textbf{" if is_ours else ""
        suffix = "}" if is_ours else ""
        output_lines.append(
            f"        {prefix}{row['Model']}{suffix} & "
            f"{prefix}{row['mAP@.5']:.1%}{suffix} & "
            f"{prefix}{row['mAP@.5:.95']:.1%}{suffix} & "
            f"{prefix}{row['F1-Score']:.2f}{suffix} & "
            f"{prefix}{row['Params (M)']:.1f}{suffix} & "
            f"{prefix}{row['GFLOPs']:.1f}{suffix} & "
            f"{prefix}{row['FPS']:.0f}{suffix} \\\\"
        )

    output_lines.append("        \\bottomrule")
    output_lines.append("    \\end{tabular*}")
    output_lines.append("\\end{table}")
    output_lines.append("")


def generate_latex_ablation(df, output_lines):
    """生成消融实验 LaTeX 表格。"""
    output_lines.append("% Table 2: Ablation Study")
    output_lines.append("\\begin{table}[htbp]")
    output_lines.append("    \\centering")
    output_lines.append("    \\caption{Ablation study of DWGSA components on DeepPCB.}")
    output_lines.append("    \\label{tab:ablation}")
    output_lines.append("    \\setlength{\\tabcolsep}{3pt}")
    output_lines.append("    \\begin{tabular}{l c c c c c}")
    output_lines.append("        \\toprule")
    output_lines.append("        \\textbf{Model} & \\textbf{mAP@.5} & \\textbf{mAP@.5:.95} & "
                        "\\textbf{F1} & \\textbf{Params(M)} & \\textbf{$\\Delta$mAP} \\\\")
    output_lines.append("        \\midrule")

    baseline_map = df.iloc[0]["mAP@.5"] if len(df) > 0 else 0

    for _, row in df.iterrows():
        imp = row["mAP@.5"] - baseline_map
        imp_str = f"+{imp:.1%}" if imp > 0 else "-"
        is_full = "Full" in row["Model"] or "DWGSA-YOLO" in row["Model"]
        prefix = "\\textbf{" if is_full else ""
        suffix = "}" if is_full else ""

        output_lines.append(
            f"        {prefix}{row['Model']}{suffix} & "
            f"{prefix}{row['mAP@.5']:.1%}{suffix} & "
            f"{prefix}{row['mAP@.5:.95']:.1%}{suffix} & "
            f"{prefix}{row['F1-Score']:.2f}{suffix} & "
            f"{prefix}{row['Params (M)']:.1f}{suffix} & "
            f"{prefix}{imp_str}{suffix} \\\\"
        )

    output_lines.append("        \\bottomrule")
    output_lines.append("    \\end{tabular}")
    output_lines.append("\\end{table}")
    output_lines.append("")


def main():
    register_custom_modules()

    cfg = load_experiment_config()
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DWGSA-YOLO: Unified Model Evaluation")
    print("=" * 60)

    latex_lines = []

    # Exp1: SOTA Comparison
    print("\n[Exp1] SOTA Comparison")
    sota_models = discover_models(cfg, "exp1")
    sota_results = []
    for m in sota_models:
        if m["weight_path"].exists():
            result = evaluate_model(str(m["weight_path"]), m["data_cfg"], m["name"])
            sota_results.append(result)
        else:
            print(f"  [SKIP] {m['name']}: {m['weight_path']} not found")

    if sota_results:
        sota_df = pd.DataFrame(sota_results)
        sota_df.to_csv(results_dir / "sota_comparison.csv", index=False)
        generate_latex_sota(sota_df, latex_lines)
        print(f"\n  Saved: {results_dir / 'sota_comparison.csv'}")

    # Exp2: Ablation Study
    print("\n[Exp2] Ablation Study")
    ablation_models = discover_models(cfg, "exp2")
    ablation_results = []
    for m in ablation_models:
        if m["weight_path"].exists():
            result = evaluate_model(str(m["weight_path"]), m["data_cfg"], m["name"])
            ablation_results.append(result)
        else:
            print(f"  [SKIP] {m['name']}: {m['weight_path']} not found")

    if ablation_results:
        ablation_df = pd.DataFrame(ablation_results)
        ablation_df.to_csv(results_dir / "ablation_table.csv", index=False)
        generate_latex_ablation(ablation_df, latex_lines)
        print(f"\n  Saved: {results_dir / 'ablation_table.csv'}")

    # Exp3: Generalization
    print("\n[Exp3] Cross-Dataset Generalization")
    gen_models = discover_models(cfg, "exp3")
    gen_results = []
    for m in gen_models:
        if m["weight_path"].exists():
            result = evaluate_model(str(m["weight_path"]), m["data_cfg"], m["name"])
            gen_results.append(result)
        else:
            print(f"  [SKIP] {m['name']}: {m['weight_path']} not found")

    if gen_results:
        gen_df = pd.DataFrame(gen_results)
        gen_df.to_csv(results_dir / "generalization.csv", index=False)
        print(f"\n  Saved: {results_dir / 'generalization.csv'}")

    # Save LaTeX
    if latex_lines:
        tex_path = results_dir / "tables.tex"
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("\n".join(latex_lines))
        print(f"\n  LaTeX tables: {tex_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    total = len(sota_results) + len(ablation_results) + len(gen_results)
    print(f"  Models evaluated: {total}")
    print(f"  Results directory: {results_dir}")


if __name__ == "__main__":
    main()
