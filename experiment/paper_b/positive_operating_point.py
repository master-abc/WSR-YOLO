from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .coco_evaluator import evaluate_coco_predictions
    from .common import atomic_json_dump, sha256_file
    from .operating_point import operational_metrics, validation_f1_thresholds
except ImportError:
    from coco_evaluator import evaluate_coco_predictions
    from common import atomic_json_dump, sha256_file
    from operating_point import operational_metrics, validation_f1_thresholds


def evaluate_positive_classwise(
    validation_annotations_path: Path,
    validation_predictions_path: Path,
    test_annotations_path: Path,
    test_predictions_path: Path,
    output_path: Path,
    reference_threshold: float = 0.25,
    match_iou: float = 0.5,
    beta: float = 1.0,
) -> dict:
    validation_annotations = json.loads(
        validation_annotations_path.read_text(encoding="utf-8")
    )
    validation_predictions = json.loads(
        validation_predictions_path.read_text(encoding="utf-8")
    )
    test_annotations = json.loads(test_annotations_path.read_text(encoding="utf-8"))
    test_predictions = json.loads(test_predictions_path.read_text(encoding="utf-8"))
    thresholds = validation_f1_thresholds(
        validation_annotations,
        validation_predictions,
        iou_threshold=match_iou,
        beta=beta,
    )
    filtered = [
        row
        for row in test_predictions
        if float(row["score"]) >= thresholds.get(int(row["category_id"]), 1.0)
    ]
    filtered_path = output_path.with_name(f"{output_path.stem}_filtered_predictions.json")
    atomic_json_dump(filtered, filtered_path)
    category_names = {
        int(row["id"]): str(row["name"])
        for row in validation_annotations.get("categories", [])
    }
    payload = {
        "schema_version": 1,
        "selection": {
            "policy": "per-class threshold maximizing positive-validation F-beta",
            "beta": float(beta),
            "selection_split": "val",
            "test_evaluated_during_selection": False,
            "thresholds": {
                category_names.get(key, str(key)): value
                for key, value in sorted(thresholds.items())
            },
        },
        "validation": operational_metrics(
            validation_annotations, validation_predictions, thresholds, match_iou
        ),
        "test": {
            "reference": operational_metrics(
                test_annotations, test_predictions, reference_threshold, match_iou
            ),
            "classwise": operational_metrics(
                test_annotations, test_predictions, thresholds, match_iou
            ),
            "classwise_coco": evaluate_coco_predictions(
                test_annotations_path, filtered_path, max_det=100
            ),
        },
        "inputs": {
            "validation_annotations_sha256": sha256_file(validation_annotations_path),
            "validation_predictions_sha256": sha256_file(validation_predictions_path),
            "test_annotations_sha256": sha256_file(test_annotations_path),
            "test_predictions_sha256": sha256_file(test_predictions_path),
        },
    }
    atomic_json_dump(payload, output_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select classwise thresholds on validation and evaluate positive test images"
    )
    parser.add_argument("--validation-annotations", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--test-annotations", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-threshold", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()
    payload = evaluate_positive_classwise(
        args.validation_annotations.resolve(),
        args.validation_predictions.resolve(),
        args.test_annotations.resolve(),
        args.test_predictions.resolve(),
        args.output.resolve(),
        args.reference_threshold,
        args.match_iou,
        args.beta,
    )
    print(json.dumps(payload["selection"], ensure_ascii=False))
    print(json.dumps(payload["test"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
