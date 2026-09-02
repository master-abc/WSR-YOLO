from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from .common import PROJECT_DIR, atomic_json_dump, class_names, load_yaml, sha256_file
    from .run import metrics_payload
except ImportError:
    from common import PROJECT_DIR, atomic_json_dump, class_names, load_yaml, sha256_file
    from run import metrics_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one frozen checkpoint on a counterfactual suite")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from algorithm.register import register_custom_modules

    register_custom_modules()
    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()))
    evaluations = []
    for data_yaml in sorted(args.suite_root.resolve().rglob("dataset.yaml")):
        data = load_yaml(data_yaml)
        metrics = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=0.001,
            iou=0.7,
            max_det=300,
            plots=False,
            verbose=False,
        )
        evaluations.append(
            {
                "data_yaml": str(data_yaml),
                "condition": data.get("metadata", {}),
                "metrics": metrics_payload(metrics, class_names(data)),
            }
        )
        print(f"[eval] {data_yaml.parent.name}: AP={evaluations[-1]['metrics']['map50_95']:.4f}")
    payload = {
        "schema_version": 1,
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "weights": str(args.weights.resolve()),
        "weights_sha256": sha256_file(args.weights),
        "suite_root": str(args.suite_root.resolve()),
        "evaluations": evaluations,
    }
    atomic_json_dump(payload, args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

