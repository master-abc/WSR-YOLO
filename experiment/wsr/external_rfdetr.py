from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPRODUCIBILITY_DIR = Path(__file__).resolve().parent
PROJECT_DIR = REPRODUCIBILITY_DIR.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from experiment.wsr.coco_evaluator import evaluate_coco_predictions
from experiment.wsr.common import atomic_json_dump, environment_snapshot, sha256_file


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def materialize_rfdetr_dataset(coco_root: Path, output: Path) -> Path:
    mapping = {"train": "train", "val": "valid", "test": "test"}
    for source_split, target_split in mapping.items():
        source_annotations = coco_root / "annotations" / f"instances_{source_split}.json"
        if not source_annotations.exists():
            raise FileNotFoundError(source_annotations)
        target_dir = output / target_split
        target_dir.mkdir(parents=True, exist_ok=True)
        link_or_copy(source_annotations, target_dir / "_annotations.coco.json")
        payload = json.loads(source_annotations.read_text(encoding="utf-8"))
        for image in payload["images"]:
            link_or_copy(coco_root / source_split / image["file_name"], target_dir / image["file_name"])
    return output


def export_predictions(model, annotations: Path, image_dir: Path, output: Path, threshold: float) -> Path:
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    categories = [item["id"] for item in sorted(payload["categories"], key=lambda item: item["id"])]
    rows = []
    for image in sorted(payload["images"], key=lambda item: int(item["id"])):
        detections = model.predict(str(image_dir / image["file_name"]), threshold=threshold)
        for coordinates, score, class_id in zip(
            detections.xyxy, detections.confidence, detections.class_id
        ):
            x1, y1, x2, y2 = (float(value) for value in coordinates)
            class_index = int(class_id)
            # rfdetr 输出头含第 N+1 个 no-object 通道,低阈值下会漏出,类别表外的一律丢弃
            if class_index >= len(categories):
                continue
            rows.append(
                {
                    "image_id": int(image["id"]),
                    "category_id": int(categories[class_index]),
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score),
                }
            )
    atomic_json_dump(rows, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Official RF-DETR adapter for the paper protocol")
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--pretrain-weights",
        default=None,
        help="本地 COCO 预训练权重路径;不提供时 rfdetr 会尝试联网下载(离线服务器必须显式给出)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="跳过训练,直接用 --output 目录下已有的 checkpoint_best_total.pth 执行导出评测",
    )
    args = parser.parse_args()

    from importlib.metadata import version

    from rfdetr import RFDETRMedium

    rfdetr_version = version("rfdetr")

    args.output.mkdir(parents=True, exist_ok=True)
    dataset_dir = materialize_rfdetr_dataset(
        args.coco_root.resolve(), args.output / "dataset_adapter"
    )
    train_annotations = dataset_dir / "train" / "_annotations.coco.json"
    metadata = json.loads(train_annotations.read_text(encoding="utf-8"))
    num_classes = len(metadata["categories"])
    extra = {"pretrain_weights": args.pretrain_weights} if args.pretrain_weights else {}
    if not args.skip_train:
        model = RFDETRMedium(num_classes=num_classes, resolution=args.resolution, **extra)
        model.train(
            dataset_dir=str(dataset_dir),
            output_dir=str(args.output),
            epochs=args.epochs,
            batch_size=args.batch,
            grad_accum_steps=args.grad_accum,
            lr=1e-4,
            seed=args.seed,
            device=args.device,
            early_stopping=True,
            early_stopping_patience=15,
            run_test=False,
        )
    checkpoint = args.output / "checkpoint_best_total.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    evaluated = RFDETRMedium(
        num_classes=num_classes,
        resolution=args.resolution,
        pretrain_weights=str(checkpoint),
    )
    annotations = args.coco_root.resolve() / "annotations" / "instances_test.json"
    predictions = export_predictions(
        evaluated,
        annotations,
        args.coco_root.resolve() / "test",
        args.output / "test_predictions.json",
        threshold=0.001,
    )
    evaluation = evaluate_coco_predictions(annotations, predictions, max_det=100)
    network = evaluated.model.model
    result = {
        "schema_version": 2,
        "track": "sota",
        "dataset": args.coco_root.resolve().name,
        "model": "rf_detr_m",
        "seed": args.seed,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "weights_sha256": sha256_file(checkpoint),
        "protocol_sha256": sha256_file(REPRODUCIBILITY_DIR / "protocol.yaml"),
        "training_policy": "official_rfdetr_finetune_recipe",
        "train_args": {
            "epochs": args.epochs,
            "resolution": args.resolution,
            "batch": args.batch,
            "grad_accum": args.grad_accum,
            "lr": 1e-4,
        },
        "metrics": evaluation["metrics"],
        "unified_evaluation": evaluation,
        "complexity": {"parameters": sum(parameter.numel() for parameter in network.parameters())},
        "environment": environment_snapshot() | {"rfdetr": rfdetr_version},
    }
    atomic_json_dump(result, args.output / "standardized_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
