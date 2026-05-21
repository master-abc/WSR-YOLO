"""Smoke test: 顺序运行关键方法，每次训练后清理 GPU 内存。"""
import sys
import json
import gc
from pathlib import Path
from datetime import datetime

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.register import register_custom_modules
register_custom_modules()

from ultralytics import YOLO

data = str(PROJECT_ROOT / "experiment" / "configs" / "deeppcb.yaml")
device = "0" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("  COMPREHENSIVE SMOKE TEST")
print("  Testing: Exp1 (SOTA), Exp2 (Ablation), Exp3 (Cross-dataset)")
print(f"  Device: {device}")
print("=" * 70)

all_results = {}

# --- EXP1: SOTA Comparison (3 key methods) ---
print(f"\n{'='*70}")
print("  EXP1: SOTA COMPARISON (key methods)")
print(f"{'='*70}")

exp1_methods = [
    ("yolo11m_baseline", "experiment/configs/yolo11m_baseline.yaml", "YOLO11m (C2PSA)"),
    ("yolo11m_cbam", "experiment/configs/yolo11m_cbam.yaml", "YOLO11m + CBAM"),
    ("dwgsa_yolo11m", "experiment/configs/dwgsa_yolo11m.yaml", "DWGSA-YOLO (Ours)"),
]

exp1_results = []
for name, cfg, desc in exp1_methods:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n  [{name}] Training: {desc}...")
    try:
        model = YOLO(str(PROJECT_ROOT / cfg))
        model.train(
            data=data, epochs=2, imgsz=640, batch=2,
            project=str(PROJECT_ROOT / "experiment" / "exp1" / "runs"), name=name,
            device=device, optimizer="SGD", lr0=0.01, seed=42,
            exist_ok=True, verbose=False, workers=0, plots=False,
        )
        metrics = model.val(data=data, split="test", imgsz=640, batch=2, workers=0)
        r = {
            "name": name, "desc": desc,
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        exp1_results.append(r)
        print(f"    OK: mAP@.5={r['map50']:.4f}")
        del model
    except Exception as e:
        print(f"    FAILED: {e}")
        exp1_results.append({"name": name, "desc": desc, "error": str(e)})

all_results["exp1"] = exp1_results

# --- EXP2: Ablation (3 key variants) ---
print(f"\n{'='*70}")
print("  EXP2: ABLATION STUDY (key variants)")
print(f"{'='*70}")

exp2_methods = [
    ("baseline_c2psa", "experiment/configs/yolo11m_baseline.yaml", "Baseline (C2PSA)"),
    ("dwgsa_wave_only", "experiment/configs/dwgsa_wave_only.yaml", "+ Wave Only"),
    ("dwgsa_full", "experiment/configs/dwgsa_yolo11m.yaml", "DWGSA Full"),
]

exp2_results = []
for name, cfg, desc in exp2_methods:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n  [{name}] Training: {desc}...")
    try:
        model = YOLO(str(PROJECT_ROOT / cfg))
        model.train(
            data=data, epochs=2, imgsz=640, batch=2,
            project=str(PROJECT_ROOT / "experiment" / "exp2" / "runs"), name=name,
            device=device, optimizer="SGD", lr0=0.01, seed=42,
            exist_ok=True, verbose=False, workers=0, plots=False,
        )
        metrics = model.val(data=data, split="test", imgsz=640, batch=2, workers=0)
        r = {
            "name": name, "desc": desc,
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        }
        exp2_results.append(r)
        print(f"    OK: mAP@.5={r['map50']:.4f}")
        del model
    except Exception as e:
        print(f"    FAILED: {e}")
        exp2_results.append({"name": name, "desc": desc, "error": str(e)})

all_results["exp2"] = exp2_results

# --- EXP3: Cross-dataset (DefectDet) ---
print(f"\n{'='*70}")
print("  EXP3: CROSS-DATASET (mini DefectDet)")
print(f"{'='*70}")

data_defectdet = str(PROJECT_ROOT / "experiment" / "configs" / "defectdet.yaml")

exp3_methods = [
    ("defectdet_baseline", "experiment/configs/yolo11m_baseline.yaml", "Baseline on DefectDet"),
    ("defectdet_dwgsa", "experiment/configs/dwgsa_yolo11m.yaml", "DWGSA on DefectDet"),
]

exp3_results = []
for name, cfg, desc in exp3_methods:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n  [{name}] Training: {desc}...")
    try:
        model = YOLO(str(PROJECT_ROOT / cfg))
        model.train(
            data=data_defectdet, epochs=2, imgsz=640, batch=2,
            project=str(PROJECT_ROOT / "experiment" / "exp3" / "runs"), name=name,
            device=device, optimizer="SGD", lr0=0.01, seed=42,
            exist_ok=True, verbose=False, workers=0, plots=False,
        )
        metrics = model.val(data=data_defectdet, split="val", imgsz=640, batch=2, workers=0)
        r = {
            "name": name, "desc": desc,
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        }
        exp3_results.append(r)
        print(f"    OK: mAP@.5={r['map50']:.4f}")
        del model
    except Exception as e:
        print(f"    FAILED: {e}")
        exp3_results.append({"name": name, "desc": desc, "error": str(e)})

all_results["exp3"] = exp3_results

# --- Summary ---
print(f"\n{'='*70}")
print("  SMOKE TEST SUMMARY")
print(f"{'='*70}")

total = 0
passed = 0
for exp_name, results in all_results.items():
    for r in results:
        total += 1
        if "error" not in r:
            passed += 1
            print(f"  [PASS] {exp_name}/{r['name']}: {r['desc']} — mAP@.5={r['map50']:.4f}")
        else:
            print(f"  [FAIL] {exp_name}/{r['name']}: {r['desc']} — {r['error'][:60]}")

print(f"\n  Total: {passed}/{total} passed")

# Save all results
output = PROJECT_ROOT / "experiment" / "smoke_test_results.json"
with open(output, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
        "summary": f"{passed}/{total} passed",
    }, f, indent=2, ensure_ascii=False)

print(f"  Results saved to: {output}")

if passed == total:
    print("\n  [ALL PASSED] All experiments verified successfully!")
else:
    print(f"\n  [PARTIAL] {total - passed} experiments failed — check GPU memory")
