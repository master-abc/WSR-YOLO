from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .coco_evaluator import evaluate_coco_predictions
    from .common import atomic_json_dump, sha256_file
except ImportError:
    from coco_evaluator import evaluate_coco_predictions
    from common import atomic_json_dump, sha256_file


def _negative_rows(path: Path) -> dict[str, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, list[float]] = {}
    for row in payload.get("per_image", []):
        image = str(row["image"])
        if image in rows:
            raise ValueError(f"Duplicate negative image in {path}: {image}")
        rows[image] = [float(score) for score in row.get("scores", [])]
    if not rows:
        raise ValueError(f"No per-image negative predictions in {path}")
    return rows


def select_threshold(rows: dict[str, list[float]], target_board_fpr: float) -> float:
    """Choose the lowest observed-score boundary meeting a board-level FPR cap."""

    if not 0.0 <= target_board_fpr < 1.0:
        raise ValueError("target_board_fpr must be in [0, 1)")
    maxima = sorted(
        (max(scores) if scores else float("-inf") for scores in rows.values()),
        reverse=True,
    )
    allowed_positive_boards = math.floor(target_board_fpr * len(maxima))
    if allowed_positive_boards >= len(maxima):
        return 0.0
    boundary = maxima[allowed_positive_boards]
    if not math.isfinite(boundary):
        return 0.0
    return math.nextafter(float(boundary), math.inf)


def negative_metrics(rows: dict[str, list[float]], threshold: float) -> dict[str, Any]:
    counts = [sum(score >= threshold for score in scores) for scores in rows.values()]
    return {
        "images": len(counts),
        "threshold": float(threshold),
        "boards_with_false_positives": sum(count > 0 for count in counts),
        "board_false_positive_rate": sum(count > 0 for count in counts) / len(counts),
        "false_positives": sum(counts),
        "false_positives_per_image": sum(counts) / len(counts),
    }


def _rich_negative_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, list[dict[str, Any]]] = {}
    for row in payload.get("per_image", []):
        image = str(row["image"])
        detections = row.get("detections")
        if detections is None:
            raise ValueError(f"Rich negative detections are missing for {image}")
        rows[image] = [
            {
                "score": float(detection["score"]),
                "class_id": int(detection["class_id"]),
            }
            for detection in detections
        ]
    if not rows:
        raise ValueError(f"No rich negative predictions in {path}")
    return rows


def classwise_negative_metrics(
    rows: dict[str, list[dict[str, Any]]], thresholds: dict[int, float]
) -> dict[str, Any]:
    counts = [
        sum(
            detection["score"] >= thresholds.get(detection["class_id"], 1.0)
            for detection in detections
        )
        for detections in rows.values()
    ]
    return {
        "images": len(counts),
        "class_thresholds": {str(key): value for key, value in sorted(thresholds.items())},
        "boards_with_false_positives": sum(count > 0 for count in counts),
        "board_false_positive_rate": sum(count > 0 for count in counts) / len(counts),
        "false_positives": sum(counts),
        "false_positives_per_image": sum(counts) / len(counts),
    }


def _iou(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = (float(value) for value in left)
    rx, ry, rw, rh = (float(value) for value in right)
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, lw) * max(0.0, lh) + max(0.0, rw) * max(0.0, rh) - intersection
    return intersection / union if union > 0.0 else 0.0


def operational_metrics(
    annotations: dict[str, Any],
    predictions: list[dict[str, Any]],
    threshold: float | dict[int, float],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute one-to-one class-aware detection metrics at a fixed operating point."""

    ground_truth: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    for annotation in annotations.get("annotations", []):
        if int(annotation.get("iscrowd", 0)):
            continue
        ground_truth[(int(annotation["image_id"]), int(annotation["category_id"]))].append(
            [float(value) for value in annotation["bbox"]]
        )

    detections: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        category_id = int(prediction["category_id"])
        category_threshold = (
            float(threshold.get(category_id, 1.0))
            if isinstance(threshold, dict)
            else float(threshold)
        )
        if float(prediction["score"]) >= category_threshold:
            detections[(int(prediction["image_id"]), int(prediction["category_id"]))].append(
                prediction
            )

    category_names = {
        int(category["id"]): str(category["name"])
        for category in annotations.get("categories", [])
    }
    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_category: dict[int, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for key in sorted(set(ground_truth) | set(detections)):
        category_id = key[1]
        targets = ground_truth.get(key, [])
        matched: set[int] = set()
        ordered = sorted(detections.get(key, []), key=lambda row: float(row["score"]), reverse=True)
        for detection in ordered:
            candidates = [
                (index, _iou(detection["bbox"], target))
                for index, target in enumerate(targets)
                if index not in matched
            ]
            if candidates:
                index, overlap = max(candidates, key=lambda item: item[1])
            else:
                index, overlap = -1, 0.0
            if overlap >= iou_threshold:
                matched.add(index)
                totals["tp"] += 1
                per_category[category_id]["tp"] += 1
            else:
                totals["fp"] += 1
                per_category[category_id]["fp"] += 1
        missed = len(targets) - len(matched)
        totals["fn"] += missed
        per_category[category_id]["fn"] += missed

    def rates(counts: dict[str, int]) -> dict[str, Any]:
        precision_denominator = counts["tp"] + counts["fp"]
        recall_denominator = counts["tp"] + counts["fn"]
        precision = counts["tp"] / precision_denominator if precision_denominator else 0.0
        recall = counts["tp"] / recall_denominator if recall_denominator else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {**counts, "precision": precision, "recall": recall, "f1": f1}

    return {
        "score_threshold": (
            {str(key): value for key, value in sorted(threshold.items())}
            if isinstance(threshold, dict)
            else float(threshold)
        ),
        "iou_threshold": float(iou_threshold),
        "overall": rates(totals),
        "per_class": {
            category_names.get(category_id, str(category_id)): rates(counts)
            for category_id, counts in sorted(per_category.items())
        },
    }


def validation_f1_thresholds(
    annotations: dict[str, Any],
    predictions: list[dict[str, Any]],
    minimum_threshold: float = 0.05,
    maximum_threshold: float = 0.95,
    step: float = 0.01,
    iou_threshold: float = 0.5,
    beta: float = 1.0,
) -> dict[int, float]:
    """Select one threshold per class using positive-validation F-beta only."""

    if step <= 0.0:
        raise ValueError("step must be positive")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    candidates = []
    value = minimum_threshold
    while value <= maximum_threshold + 1e-12:
        candidates.append(round(value, 10))
        value += step
    selected: dict[int, float] = {}
    for category in annotations.get("categories", []):
        category_id = int(category["id"])
        category_annotations = {
            **annotations,
            "categories": [category],
            "annotations": [
                row
                for row in annotations.get("annotations", [])
                if int(row["category_id"]) == category_id
            ],
        }
        category_predictions = [
            row for row in predictions if int(row["category_id"]) == category_id
        ]
        scored = []
        for candidate in candidates:
            result = operational_metrics(
                category_annotations, category_predictions, candidate, iou_threshold
            )["overall"]
            precision, recall = result["precision"], result["recall"]
            denominator = beta * beta * precision + recall
            f_beta = (
                (1.0 + beta * beta) * precision * recall / denominator
                if denominator
                else 0.0
            )
            scored.append((f_beta, result["recall"], -candidate, candidate))
        selected[category_id] = max(scored)[-1]
    return selected


def constrain_class_thresholds(
    base_thresholds: dict[int, float],
    negative_rows: dict[str, list[dict[str, Any]]],
    target_board_fpr: float,
) -> tuple[dict[int, float], float]:
    """Raise class thresholds by one shared logit-like interpolation factor."""

    if not 0.0 <= target_board_fpr < 1.0:
        raise ValueError("target_board_fpr must be in [0, 1)")

    def shifted(alpha: float) -> dict[int, float]:
        return {
            category_id: threshold + alpha * (1.0 - threshold)
            for category_id, threshold in base_thresholds.items()
        }

    if classwise_negative_metrics(negative_rows, base_thresholds)[
        "board_false_positive_rate"
    ] <= target_board_fpr:
        return dict(base_thresholds), 0.0
    low, high = 0.0, 1.0
    for _ in range(64):
        middle = (low + high) / 2.0
        rate = classwise_negative_metrics(negative_rows, shifted(middle))[
            "board_false_positive_rate"
        ]
        if rate <= target_board_fpr:
            high = middle
        else:
            low = middle
    return shifted(high), high


def calibrate_and_evaluate(
    negative_pool_path: Path,
    negative_holdout_path: Path,
    positive_annotations_path: Path,
    positive_predictions_path: Path,
    output_path: Path,
    target_board_fpr: float = 0.05,
    reference_threshold: float = 0.25,
    match_iou: float = 0.5,
    rich_negative_pool_path: Path | None = None,
    rich_negative_holdout_path: Path | None = None,
    positive_val_annotations_path: Path | None = None,
    positive_val_predictions_path: Path | None = None,
) -> dict[str, Any]:
    pool = _negative_rows(negative_pool_path)
    holdout = _negative_rows(negative_holdout_path)
    overlap = set(pool) & set(holdout)
    if overlap and overlap != set(holdout):
        raise ValueError("Negative pool and holdout have a partial, ambiguous overlap")
    if set(holdout).issubset(pool):
        calibration = {image: scores for image, scores in pool.items() if image not in holdout}
        pool_layout = "combined_pool_with_holdout_removed"
    else:
        # A pre-split calibration file is preferable when available.
        calibration = dict(pool)
        pool_layout = "explicit_disjoint_calibration_and_holdout"
    if not calibration:
        raise ValueError("Negative calibration set is empty after excluding holdout images")

    threshold = select_threshold(calibration, target_board_fpr)
    annotations = json.loads(positive_annotations_path.read_text(encoding="utf-8"))
    predictions = json.loads(positive_predictions_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("Positive COCO predictions must be a JSON list")

    filtered_path = output_path.with_name(f"{output_path.stem}_filtered_predictions.json")
    filtered = [row for row in predictions if float(row["score"]) >= threshold]
    atomic_json_dump(filtered, filtered_path)
    payload = {
        "schema_version": 1,
        "selection": {
            "policy": "lowest global confidence threshold meeting the calibration board-FPR cap",
            "target_calibration_board_fpr": float(target_board_fpr),
            "selected_threshold": threshold,
            "calibration_images": len(calibration),
            "holdout_images_excluded_from_selection": len(holdout),
            "negative_pool_layout": pool_layout,
        },
        "negative_calibration": negative_metrics(calibration, threshold),
        "negative_holdout": {
            "reference": negative_metrics(holdout, reference_threshold),
            "calibrated": negative_metrics(holdout, threshold),
        },
        "positive_test": {
            "reference": operational_metrics(
                annotations, predictions, reference_threshold, match_iou
            ),
            "calibrated": operational_metrics(annotations, predictions, threshold, match_iou),
            "calibrated_coco": evaluate_coco_predictions(
                positive_annotations_path, filtered_path, max_det=100
            ),
        },
        "inputs": {
            "negative_pool": str(negative_pool_path.resolve()),
            "negative_pool_sha256": sha256_file(negative_pool_path),
            "negative_holdout": str(negative_holdout_path.resolve()),
            "negative_holdout_sha256": sha256_file(negative_holdout_path),
            "positive_annotations": str(positive_annotations_path.resolve()),
            "positive_annotations_sha256": sha256_file(positive_annotations_path),
            "positive_predictions": str(positive_predictions_path.resolve()),
            "positive_predictions_sha256": sha256_file(positive_predictions_path),
        },
    }
    optional_paths = (
        rich_negative_pool_path,
        positive_val_annotations_path,
        positive_val_predictions_path,
    )
    if any(path is not None for path in optional_paths):
        if not all(path is not None for path in optional_paths):
            raise ValueError(
                "Classwise calibration requires rich negatives and both positive-val files"
            )
        assert rich_negative_pool_path is not None
        assert positive_val_annotations_path is not None
        assert positive_val_predictions_path is not None
        rich_pool = _rich_negative_rows(rich_negative_pool_path)
        if rich_negative_holdout_path is not None:
            rich_holdout = _rich_negative_rows(rich_negative_holdout_path)
            if set(rich_pool) != set(pool):
                raise ValueError(
                    "Rich and score-only negative calibration files do not align"
                )
            if set(rich_holdout) != set(holdout):
                raise ValueError("Rich and score-only negative holdout files do not align")
            rich_calibration = dict(rich_pool)
        else:
            if set(rich_pool) != set(pool):
                raise ValueError("Rich and score-only negative pools do not contain the same images")
            rich_calibration = {
                image: detections
                for image, detections in rich_pool.items()
                if image not in holdout
            }
            rich_holdout = {
                image: detections for image, detections in rich_pool.items() if image in holdout
            }
        val_annotations = json.loads(
            positive_val_annotations_path.read_text(encoding="utf-8")
        )
        val_predictions = json.loads(
            positive_val_predictions_path.read_text(encoding="utf-8")
        )
        base_thresholds = validation_f1_thresholds(
            val_annotations, val_predictions, iou_threshold=match_iou
        )
        class_thresholds, shift = constrain_class_thresholds(
            base_thresholds, rich_calibration, target_board_fpr
        )
        classwise_filtered_path = output_path.with_name(
            f"{output_path.stem}_classwise_filtered_predictions.json"
        )
        classwise_filtered = [
            row
            for row in predictions
            if float(row["score"])
            >= class_thresholds.get(int(row["category_id"]), 1.0)
        ]
        atomic_json_dump(classwise_filtered, classwise_filtered_path)
        payload["classwise_calibration"] = {
            "policy": (
                "per-class validation-F1 thresholds, uniformly raised toward one until "
                "the disjoint negative-calibration board-FPR cap is met"
            ),
            "base_validation_thresholds": {
                str(key): value for key, value in sorted(base_thresholds.items())
            },
            "shared_interpolation_shift": shift,
            "selected_thresholds": {
                str(key): value for key, value in sorted(class_thresholds.items())
            },
            "negative_calibration": classwise_negative_metrics(
                rich_calibration, class_thresholds
            ),
            "negative_holdout": classwise_negative_metrics(
                rich_holdout, class_thresholds
            ),
            "positive_validation": {
                "base": operational_metrics(
                    val_annotations, val_predictions, base_thresholds, match_iou
                ),
                "constrained": operational_metrics(
                    val_annotations, val_predictions, class_thresholds, match_iou
                ),
            },
            "positive_test": {
                "operational": operational_metrics(
                    annotations, predictions, class_thresholds, match_iou
                ),
                "coco": evaluate_coco_predictions(
                    positive_annotations_path, classwise_filtered_path, max_det=100
                ),
            },
            "inputs": {
                "rich_negative_pool": str(rich_negative_pool_path.resolve()),
                "rich_negative_pool_sha256": sha256_file(rich_negative_pool_path),
                "rich_negative_holdout": (
                    str(rich_negative_holdout_path.resolve())
                    if rich_negative_holdout_path is not None
                    else None
                ),
                "rich_negative_holdout_sha256": (
                    sha256_file(rich_negative_holdout_path)
                    if rich_negative_holdout_path is not None
                    else None
                ),
                "positive_val_annotations": str(positive_val_annotations_path.resolve()),
                "positive_val_annotations_sha256": sha256_file(
                    positive_val_annotations_path
                ),
                "positive_val_predictions": str(positive_val_predictions_path.resolve()),
                "positive_val_predictions_sha256": sha256_file(
                    positive_val_predictions_path
                ),
            },
        }
    atomic_json_dump(payload, output_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate a detector confidence threshold on disjoint negative images"
    )
    parser.add_argument("--negative-pool", type=Path, required=True)
    parser.add_argument("--negative-holdout", type=Path, required=True)
    parser.add_argument("--positive-annotations", type=Path, required=True)
    parser.add_argument("--positive-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-board-fpr", type=float, default=0.05)
    parser.add_argument("--reference-threshold", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--rich-negative-pool", type=Path)
    parser.add_argument("--rich-negative-holdout", type=Path)
    parser.add_argument("--positive-val-annotations", type=Path)
    parser.add_argument("--positive-val-predictions", type=Path)
    args = parser.parse_args()
    payload = calibrate_and_evaluate(
        args.negative_pool.resolve(),
        args.negative_holdout.resolve(),
        args.positive_annotations.resolve(),
        args.positive_predictions.resolve(),
        args.output.resolve(),
        args.target_board_fpr,
        args.reference_threshold,
        args.match_iou,
        args.rich_negative_pool.resolve() if args.rich_negative_pool else None,
        (
            args.rich_negative_holdout.resolve()
            if args.rich_negative_holdout
            else None
        ),
        args.positive_val_annotations.resolve() if args.positive_val_annotations else None,
        args.positive_val_predictions.resolve() if args.positive_val_predictions else None,
    )
    print(json.dumps(payload["selection"], ensure_ascii=False))
    print(json.dumps(payload["negative_holdout"], ensure_ascii=False))
    print(json.dumps(payload["positive_test"], ensure_ascii=False))
    if "classwise_calibration" in payload:
        print(json.dumps(payload["classwise_calibration"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
