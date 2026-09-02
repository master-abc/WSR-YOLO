from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import beta

try:
    from .common import atomic_json_dump, sha256_file
    from .operating_point import operational_metrics
except ImportError:
    from common import atomic_json_dump, sha256_file
    from operating_point import operational_metrics


def _iou_xyxy(left: list[float], right: list[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    x1, y1 = max(lx1, rx1), max(ly1, ry1)
    x2, y2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _xywh_to_xyxy(box: Iterable[float]) -> list[float]:
    x, y, width, height = (float(value) for value in box)
    return [x, y, x + width, y + height]


def _xyxy_to_xywh(box: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _load_positive(path: Path) -> dict[tuple[int, int], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in payload:
        rows[(int(row["image_id"]), int(row["category_id"]))].append(
            {
                "score": float(row["score"]),
                "box": _xywh_to_xyxy(row["bbox"]),
            }
        )
    return rows


def _load_negative(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for image_row in payload.get("per_image", []):
        image = Path(str(image_row["image"])).name
        for detection in image_row.get("detections", []):
            rows[(image, int(detection["class_id"]))].append(
                {
                    "score": float(detection["score"]),
                    "box": [float(value) for value in detection["xyxy"]],
                }
            )
    return rows


def _weighted_box(members: list[dict[str, Any]]) -> list[float]:
    total = sum(max(float(member["score"]), 1e-12) for member in members)
    return [
        sum(float(member["box"][index]) * float(member["score"]) for member in members)
        / total
        for index in range(4)
    ]


def consensus_detections(
    model_rows: list[dict[Any, list[dict[str, Any]]]],
    base_confidence: float,
    match_iou: float,
    minimum_votes: int,
    nms_iou: float = 0.5,
) -> dict[Any, list[dict[str, Any]]]:
    """Fuse same-class boxes only when distinct models agree spatially."""

    if not 2 <= minimum_votes <= len(model_rows):
        raise ValueError("minimum_votes must be between two and the model count")
    keys = sorted(set().union(*(set(rows) for rows in model_rows)))
    fused: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for key in keys:
        nodes: list[dict[str, Any]] = []
        for model_index, rows in enumerate(model_rows):
            for detection_index, detection in enumerate(rows.get(key, [])):
                if float(detection["score"]) >= base_confidence:
                    nodes.append(
                        {
                            **detection,
                            "model_index": model_index,
                            "detection_index": detection_index,
                        }
                    )
        nodes.sort(key=lambda row: float(row["score"]), reverse=True)
        candidates: list[dict[str, Any]] = []
        for anchor in nodes:
            members = [anchor]
            for model_index in range(len(model_rows)):
                if model_index == int(anchor["model_index"]):
                    continue
                matches = [
                    row
                    for row in nodes
                    if int(row["model_index"]) == model_index
                    and _iou_xyxy(anchor["box"], row["box"]) >= match_iou
                ]
                if matches:
                    members.append(max(matches, key=lambda row: float(row["score"])))
            voters = {int(member["model_index"]) for member in members}
            if len(voters) < minimum_votes:
                continue
            candidates.append(
                {
                    "box": _weighted_box(members),
                    "score": sum(float(member["score"]) for member in members) / len(members),
                    "votes": len(voters),
                }
            )

        # Greedy same-class NMS also collapses candidates produced from different anchors.
        kept: list[dict[str, Any]] = []
        for candidate in sorted(
            candidates,
            key=lambda row: (int(row["votes"]), float(row["score"])),
            reverse=True,
        ):
            if all(_iou_xyxy(candidate["box"], row["box"]) < nms_iou for row in kept):
                kept.append(candidate)
        if kept:
            fused[key] = kept
    return dict(fused)


def _positive_predictions(
    fused: dict[tuple[int, int], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    return [
        {
            "image_id": image_id,
            "category_id": category_id,
            "bbox": _xyxy_to_xywh(row["box"]),
            "score": float(row["score"]),
            "votes": int(row["votes"]),
        }
        for (image_id, category_id), rows in sorted(fused.items())
        for row in rows
    ]


def _negative_metrics(
    fused: dict[tuple[str, int], list[dict[str, Any]]], image_names: set[str]
) -> dict[str, Any]:
    counts = {
        image: sum(len(rows) for (name, _), rows in fused.items() if name == image)
        for image in image_names
    }
    positive_boards = sum(count > 0 for count in counts.values())
    image_count = len(counts)
    ci_low = (
        float(beta.ppf(0.025, positive_boards, image_count - positive_boards + 1))
        if positive_boards
        else 0.0
    )
    ci_high = (
        float(beta.ppf(0.975, positive_boards + 1, image_count - positive_boards))
        if positive_boards < image_count
        else 1.0
    )
    return {
        "images": image_count,
        "boards_with_false_positives": positive_boards,
        "board_false_positive_rate": positive_boards / image_count,
        "board_false_positive_rate_ci95_exact": [ci_low, ci_high],
        "false_positives": sum(counts.values()),
        "false_positives_per_image": sum(counts.values()) / len(counts),
    }


def _negative_image_names(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = {Path(str(row["image"])).name for row in payload.get("per_image", [])}
    if not names:
        raise ValueError(f"No negative images in {path}")
    return names


def _configuration_metrics(
    positive_models: list[dict[Any, list[dict[str, Any]]]],
    negative_models: list[dict[Any, list[dict[str, Any]]]],
    annotations: dict[str, Any],
    negative_images: set[str],
    configuration: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arguments = {
        "base_confidence": float(configuration["base_confidence"]),
        "match_iou": float(configuration["match_iou"]),
        "minimum_votes": int(configuration["minimum_votes"]),
        "nms_iou": float(configuration.get("nms_iou", 0.5)),
    }
    positive = _positive_predictions(consensus_detections(positive_models, **arguments))
    negative = consensus_detections(negative_models, **arguments)
    metrics = {
        "configuration": arguments,
        "negative": _negative_metrics(negative, negative_images),
        "positive": operational_metrics(annotations, positive, threshold=0.0),
    }
    return metrics, positive


def select_policies(
    positive_annotation_path: Path,
    positive_prediction_paths: list[Path],
    negative_paths: list[Path],
    output: Path,
    target_board_fprs: list[float],
    base_confidences: list[float],
    match_ious: list[float],
    minimum_votes: list[int],
) -> dict[str, Any]:
    if len(positive_prediction_paths) != len(negative_paths):
        raise ValueError("Positive and negative model input counts must match")
    annotations = json.loads(positive_annotation_path.read_text(encoding="utf-8"))
    positives = [_load_positive(path) for path in positive_prediction_paths]
    negatives = [_load_negative(path) for path in negative_paths]
    image_sets = [_negative_image_names(path) for path in negative_paths]
    if any(names != image_sets[0] for names in image_sets[1:]):
        raise ValueError("Negative model inputs must contain identical image names")

    candidates = []
    for confidence in base_confidences:
        for match_iou in match_ious:
            for votes in minimum_votes:
                configuration = {
                    "base_confidence": confidence,
                    "match_iou": match_iou,
                    "minimum_votes": votes,
                    "nms_iou": 0.5,
                }
                metrics, _ = _configuration_metrics(
                    positives, negatives, annotations, image_sets[0], configuration
                )
                candidates.append(metrics)

    selected = {}
    for target in target_board_fprs:
        feasible = [
            row
            for row in candidates
            if row["negative"]["board_false_positive_rate"] <= target
        ]
        if not feasible:
            raise RuntimeError(f"No consensus policy meets validation FPR target {target}")
        best = max(
            feasible,
            key=lambda row: (
                row["positive"]["overall"]["f1"],
                row["positive"]["overall"]["recall"],
                -row["negative"]["board_false_positive_rate"],
            ),
        )
        selected[str(target)] = best

    payload = {
        "schema_version": 1,
        "stage": "validation_policy_selection",
        "policy": "same-class spatial consensus among independently seeded models",
        "selection_rule": "maximize positive-validation F1 subject to board-FPR cap",
        "model_count": len(positives),
        "targets": selected,
        "grid": {
            "base_confidences": base_confidences,
            "match_ious": match_ious,
            "minimum_votes": minimum_votes,
            "candidate_count": len(candidates),
        },
        "inputs": {
            "positive_annotations": str(positive_annotation_path.resolve()),
            "positive_annotations_sha256": sha256_file(positive_annotation_path),
            "positive_predictions": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in positive_prediction_paths
            ],
            "negative_predictions": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in negative_paths
            ],
        },
    }
    atomic_json_dump(payload, output.resolve())
    return payload


def evaluate_policies(
    policy_path: Path,
    positive_annotation_path: Path,
    positive_prediction_paths: list[Path],
    negative_paths: list[Path],
    output: Path,
) -> dict[str, Any]:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if len(positive_prediction_paths) != int(policy["model_count"]):
        raise ValueError("Test model count differs from the frozen policy")
    if len(positive_prediction_paths) != len(negative_paths):
        raise ValueError("Positive and negative model input counts must match")
    annotations = json.loads(positive_annotation_path.read_text(encoding="utf-8"))
    positives = [_load_positive(path) for path in positive_prediction_paths]
    negatives = [_load_negative(path) for path in negative_paths]
    image_sets = [_negative_image_names(path) for path in negative_paths]
    if any(names != image_sets[0] for names in image_sets[1:]):
        raise ValueError("Negative model inputs must contain identical image names")

    results = {}
    prediction_paths = {}
    for target, validation in policy["targets"].items():
        metrics, predictions = _configuration_metrics(
            positives,
            negatives,
            annotations,
            image_sets[0],
            validation["configuration"],
        )
        prediction_path = output.with_name(f"{output.stem}_target_{target}_predictions.json")
        atomic_json_dump(predictions, prediction_path.resolve())
        results[target] = {"validation": validation, "test": metrics}
        prediction_paths[target] = {
            "path": str(prediction_path.resolve()),
            "sha256": sha256_file(prediction_path),
        }

    payload = {
        "schema_version": 1,
        "stage": "frozen_policy_test_evaluation",
        "policy_path": str(policy_path.resolve()),
        "policy_sha256": sha256_file(policy_path),
        "results": results,
        "prediction_outputs": prediction_paths,
        "inputs": {
            "positive_annotations": str(positive_annotation_path.resolve()),
            "positive_annotations_sha256": sha256_file(positive_annotation_path),
            "positive_predictions": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in positive_prediction_paths
            ],
            "negative_predictions": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in negative_paths
            ],
        },
        "protocol_note": (
            "The consensus family was introduced after inspecting prior single-model test "
            "results; treat this result as exploratory until confirmed on a fresh holdout."
        ),
    }
    atomic_json_dump(payload, output.resolve())
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Select and evaluate a seeded consensus detector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--positive-annotations", type=Path, required=True)
    select.add_argument("--positive-predictions", type=Path, nargs="+", required=True)
    select.add_argument("--negative-predictions", type=Path, nargs="+", required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--target-board-fprs", type=float, nargs="+", default=[0.01, 0.02])
    select.add_argument(
        "--base-confidences",
        type=float,
        nargs="+",
        default=[
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.40,
            0.50,
            0.60,
            0.65,
            0.70,
            0.72,
            0.74,
            0.76,
            0.78,
            0.80,
            0.82,
            0.85,
            0.90,
        ],
    )
    select.add_argument("--match-ious", type=float, nargs="+", default=[0.3, 0.5])
    select.add_argument("--minimum-votes", type=int, nargs="+", default=[2, 3])

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--policy", type=Path, required=True)
    evaluate.add_argument("--positive-annotations", type=Path, required=True)
    evaluate.add_argument("--positive-predictions", type=Path, nargs="+", required=True)
    evaluate.add_argument("--negative-predictions", type=Path, nargs="+", required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "select":
        payload = select_policies(
            args.positive_annotations,
            args.positive_predictions,
            args.negative_predictions,
            args.output,
            args.target_board_fprs,
            args.base_confidences,
            args.match_ious,
            args.minimum_votes,
        )
    else:
        payload = evaluate_policies(
            args.policy,
            args.positive_annotations,
            args.positive_predictions,
            args.negative_predictions,
            args.output,
        )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
