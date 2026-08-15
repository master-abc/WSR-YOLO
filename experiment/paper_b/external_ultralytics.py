from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

PAPER_B_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PAPER_B_DIR.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from experiment.paper_b.coco_evaluator import evaluate_coco_predictions, predict_yolo_to_coco
from experiment.paper_b.common import atomic_json_dump, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Official-fork adapter for YOLOv10/YOLOv12")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--eval-batch", type=int, default=1)
    args = parser.parse_args()

    from ultralytics import YOLO, __version__

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=args.device,
        deterministic=True,
        amp=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        patience=50,
        close_mosaic=10,
        project=str(args.output.parent),
        name=args.output.name,
        exist_ok=True,
    )
    weights = args.output / "weights" / "best.pt"
    evaluated = YOLO(str(weights))
    dataset_name = args.data.resolve().parent.name
    generated_root = args.data.resolve().parent.parent.parent
    coco_root = generated_root / "coco" / dataset_name
    annotations = coco_root / "annotations" / "instances_test.json"
    predictions = predict_yolo_to_coco(
        evaluated,
        annotations,
        coco_root / "test",
        args.output / "test_predictions.json",
        args.imgsz,
        args.eval_batch,
        args.device,
        0.001,
        0.7,
        300,
    )
    evaluation = evaluate_coco_predictions(annotations, predictions, 100)
    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = float(get_flops(evaluated.model, imgsz=args.imgsz))
    except Exception:
        gflops = None
    payload = {
        "schema_version": 2,
        "track": "sota",
        "dataset": dataset_name,
        "model": args.name,
        "seed": args.seed,
        "data_yaml": str(args.data.resolve()),
        "weights": str(weights.resolve()),
        "weights_sha256": sha256_file(weights),
        "training_policy": "common_yolo_200e_sgd_coco_pretrained",
        "evaluation_batch": args.eval_batch,
        "metrics": evaluation["metrics"],
        "unified_evaluation": evaluation,
        "complexity": {
            "parameters": sum(parameter.numel() for parameter in evaluated.model.parameters()),
            "gflops": gflops,
        },
        "environment": {"python": platform.python_version(), "ultralytics": __version__},
    }
    atomic_json_dump(payload, args.output / "standardized_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
