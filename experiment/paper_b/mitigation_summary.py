from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    from .coco_evaluator import evaluate_coco_predictions
    from .common import atomic_json_dump, sha256_file
    from .operating_point import _negative_rows, negative_metrics, operational_metrics
except ImportError:
    from coco_evaluator import evaluate_coco_predictions
    from common import atomic_json_dump, sha256_file
    from operating_point import _negative_rows, negative_metrics, operational_metrics


def _negative_sample_key(image: str) -> str:
    """Return the official DeepPCB sample id independent of local path/suffix."""
    stem = Path(image).stem
    match = re.search(r"(\d{8})(?:_(?:negative|temp|test))?$", stem, re.IGNORECASE)
    return match.group(1) if match else Path(image).name.casefold()


def _key_negative_rows(rows: dict[str, list[dict]], label: str) -> dict[str, list[dict]]:
    keyed: dict[str, list[dict]] = {}
    for image, detections in rows.items():
        key = _negative_sample_key(image)
        if key in keyed:
            raise ValueError(f"Duplicate negative sample key {key!r} in {label}")
        keyed[key] = detections
    return keyed


def summarize_mitigation(
    annotations_path: Path,
    original_predictions_path: Path,
    mitigated_predictions_path: Path,
    original_negatives_path: Path,
    mitigated_negatives_path: Path,
    output_path: Path,
    thresholds: list[float],
    match_iou: float = 0.5,
) -> dict:
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    original_predictions = json.loads(
        original_predictions_path.read_text(encoding="utf-8")
    )
    mitigated_predictions = json.loads(
        mitigated_predictions_path.read_text(encoding="utf-8")
    )
    original_negatives = _key_negative_rows(
        _negative_rows(original_negatives_path), "original negative audit"
    )
    mitigated_negatives = _key_negative_rows(
        _negative_rows(mitigated_negatives_path), "mitigated negative audit"
    )
    if not set(original_negatives).issubset(mitigated_negatives):
        raise ValueError("Mitigated negatives do not contain the original held-out images")
    # A mitigation audit may record all 1,500 templates so its non-test subset
    # can be used for calibration. Compare only the exact 500-image held-out
    # manifest carried by the original formal negative audit.
    mitigated_negatives = {
        image: mitigated_negatives[image] for image in original_negatives
    }

    operating_points = {}
    for threshold in thresholds:
        operating_points[str(threshold)] = {
            "original": {
                "positive": operational_metrics(
                    annotations, original_predictions, threshold, match_iou
                ),
                "negative": negative_metrics(original_negatives, threshold),
            },
            "mitigated": {
                "positive": operational_metrics(
                    annotations, mitigated_predictions, threshold, match_iou
                ),
                "negative": negative_metrics(mitigated_negatives, threshold),
            },
        }
    payload = {
        "schema_version": 1,
        "comparison": "original positive-only training versus defect-free-template-aware training",
        "single_seed_exploratory": True,
        "operating_points": operating_points,
        "coco": {
            "original": evaluate_coco_predictions(
                annotations_path, original_predictions_path, max_det=100
            ),
            "mitigated": evaluate_coco_predictions(
                annotations_path, mitigated_predictions_path, max_det=100
            ),
        },
        "inputs": {
            "annotations_sha256": sha256_file(annotations_path),
            "original_predictions_sha256": sha256_file(original_predictions_path),
            "mitigated_predictions_sha256": sha256_file(mitigated_predictions_path),
            "original_negatives_sha256": sha256_file(original_negatives_path),
            "mitigated_negatives_sha256": sha256_file(mitigated_negatives_path),
        },
    }
    atomic_json_dump(payload, output_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize positive/negative trade-offs for false-alarm mitigation"
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--original-predictions", type=Path, required=True)
    parser.add_argument("--mitigated-predictions", type=Path, required=True)
    parser.add_argument("--original-negatives", type=Path, required=True)
    parser.add_argument("--mitigated-negatives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.50])
    parser.add_argument("--match-iou", type=float, default=0.5)
    args = parser.parse_args()
    payload = summarize_mitigation(
        args.annotations.resolve(),
        args.original_predictions.resolve(),
        args.mitigated_predictions.resolve(),
        args.original_negatives.resolve(),
        args.mitigated_negatives.resolve(),
        args.output.resolve(),
        args.thresholds,
        args.match_iou,
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
