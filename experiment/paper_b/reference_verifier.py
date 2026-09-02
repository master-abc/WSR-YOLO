from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .coco_evaluator import evaluate_coco_predictions
    from .common import atomic_json_dump, sha256_file
    from .operating_point import operational_metrics
except ImportError:
    from coco_evaluator import evaluate_coco_predictions
    from common import atomic_json_dump, sha256_file
    from operating_point import operational_metrics


SAMPLE_KEY = re.compile(r"(\d{8})(?:_(?:test|temp))?\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def sample_key(path_or_name: str | Path) -> str:
    match = SAMPLE_KEY.search(Path(path_or_name).name)
    if match is None:
        raise ValueError(f"Cannot recover the DeepPCB sample key from {path_or_name}")
    return match.group(1)


def paired_image_index(raw_root: Path) -> dict[str, tuple[Path, Path]]:
    targets = {sample_key(path): path.resolve() for path in raw_root.rglob("*_test.jpg")}
    templates = {sample_key(path): path.resolve() for path in raw_root.rglob("*_temp.jpg")}
    keys = sorted(set(targets) & set(templates))
    if len(keys) != 1500:
        raise ValueError(f"Expected 1,500 complete DeepPCB pairs, found {len(keys)}")
    return {key: (targets[key], templates[key]) for key in keys}


def materialize_coco_targets(
    annotations_path: Path, raw_root: Path, output: Path
) -> dict[str, Any]:
    """Hard-link raw targets under the exact names used by a COCO annotation file."""

    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    pairs = paired_image_index(raw_root)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for image in annotations["images"]:
        key = sample_key(image["file_name"])
        source = pairs[key][0]
        destination = output / image["file_name"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        records.append(
            {
                "image_id": int(image["id"]),
                "key": key,
                "source": str(source),
                "destination": str(destination.resolve()),
                "sha256": sha256_file(destination),
            }
        )
    manifest = {
        "schema_version": 1,
        "annotations": str(annotations_path.resolve()),
        "annotations_sha256": sha256_file(annotations_path),
        "raw_root": str(raw_root.resolve()),
        "images": records,
    }
    atomic_json_dump(manifest, output / "manifest.json")
    return manifest


def normalized_difference(target_path: Path, template_path: Path) -> np.ndarray:
    """Photometrically normalize an aligned template, then return a denoised difference."""

    target = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if target is None or template is None:
        raise FileNotFoundError(f"Cannot read pair: {target_path}, {template_path}")
    if target.shape != template.shape:
        raise ValueError(f"Pair shape mismatch: {target.shape} versus {template.shape}")
    target_f = target.astype(np.float32)
    template_f = template.astype(np.float32)
    target_mean, target_std = float(target_f.mean()), float(target_f.std())
    template_mean, template_std = float(template_f.mean()), float(template_f.std())
    scale = np.clip(target_std / max(template_std, 1e-6), 0.5, 2.0)
    adjusted = (template_f - template_mean) * scale + target_mean
    difference = np.abs(target_f - np.clip(adjusted, 0.0, 255.0))
    return cv2.GaussianBlur(difference, (3, 3), 0)


def box_change_score(difference: np.ndarray, bbox: list[float]) -> float:
    """Return a robust local change score in [0, 1] for a COCO xywh box."""

    height, width = difference.shape
    x, y, box_width, box_height = (float(value) for value in bbox)
    margin_x = max(2.0, box_width * 0.10)
    margin_y = max(2.0, box_height * 0.10)
    x1 = max(0, int(math.floor(x - margin_x)))
    y1 = max(0, int(math.floor(y - margin_y)))
    x2 = min(width, int(math.ceil(x + box_width + margin_x)))
    y2 = min(height, int(math.ceil(y + box_height + margin_y)))
    crop = difference[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    # Defects may occupy only part of a predicted box. A high quantile is more
    # stable than the maximum and less diluted than a whole-box mean.
    return float(np.quantile(crop, 0.90) / 255.0)


def score_predictions(
    annotations: dict[str, Any],
    predictions: list[dict[str, Any]],
    pairs: dict[str, tuple[Path, Path]],
) -> list[dict[str, Any]]:
    image_keys = {int(row["id"]): sample_key(row["file_name"]) for row in annotations["images"]}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_image[int(prediction["image_id"])].append(prediction)
    scored = []
    for image_id, rows in sorted(by_image.items()):
        key = image_keys[image_id]
        difference = normalized_difference(*pairs[key])
        for row in rows:
            scored.append({**row, "change_score": box_change_score(difference, row["bbox"])})
    return scored


def _filtered(
    rows: list[dict[str, Any]], confidence: float, change_threshold: float
) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "change_score"}
        for row in rows
        if float(row["score"]) >= confidence
        and float(row["change_score"]) >= change_threshold
    ]


def select_change_threshold(
    annotations: dict[str, Any],
    scored_predictions: list[dict[str, Any]],
    confidence: float,
    beta: float = 1.0,
    iou_threshold: float = 0.5,
) -> tuple[float, dict[str, Any]]:
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    eligible_scores = sorted(
        {
            round(float(row["change_score"]), 6)
            for row in scored_predictions
            if float(row["score"]) >= confidence
        }
    )
    candidates = [0.0]
    if eligible_scores:
        indexes = np.linspace(0, len(eligible_scores) - 1, min(201, len(eligible_scores)))
        candidates.extend(eligible_scores[int(round(index))] for index in indexes)
    candidates = sorted(set(candidates))
    evaluated = []
    for threshold in candidates:
        metrics = operational_metrics(
            annotations,
            _filtered(scored_predictions, confidence, threshold),
            0.0,
            iou_threshold,
        )["overall"]
        precision, recall = metrics["precision"], metrics["recall"]
        denominator = beta * beta * precision + recall
        f_beta = (
            (1.0 + beta * beta) * precision * recall / denominator
            if denominator
            else 0.0
        )
        evaluated.append((f_beta, recall, precision, -threshold, threshold, metrics))
    selected = max(evaluated)
    return selected[4], {**selected[5], "f_beta": selected[0], "beta": float(beta)}


def evaluate_reference_verifier(
    validation_annotations_path: Path,
    validation_predictions_path: Path,
    test_annotations_path: Path,
    test_predictions_path: Path,
    raw_root: Path,
    output_path: Path,
    confidence: float = 0.25,
    beta: float = 1.0,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    validation_annotations = json.loads(validation_annotations_path.read_text(encoding="utf-8"))
    validation_predictions = json.loads(validation_predictions_path.read_text(encoding="utf-8"))
    test_annotations = json.loads(test_annotations_path.read_text(encoding="utf-8"))
    test_predictions = json.loads(test_predictions_path.read_text(encoding="utf-8"))
    pairs = paired_image_index(raw_root)
    scored_validation = score_predictions(validation_annotations, validation_predictions, pairs)
    threshold, validation_selected = select_change_threshold(
        validation_annotations,
        scored_validation,
        confidence,
        beta,
        iou_threshold,
    )
    scored_test = score_predictions(test_annotations, test_predictions, pairs)
    filtered_test = _filtered(scored_test, confidence, threshold)
    filtered_path = output_path.with_name(f"{output_path.stem}_filtered_predictions.json")
    atomic_json_dump(filtered_test, filtered_path)
    payload = {
        "schema_version": 1,
        "method": "paired-template photometric-normalized local-difference verification",
        "scope": "DeepPCB paired-input auxiliary analysis; not applicable to single-image DsPCBSD+",
        "selection": {
            "split": "val",
            "test_evaluated_during_selection": False,
            "confidence": float(confidence),
            "change_threshold": float(threshold),
            "policy": "maximize validation F-beta after a fixed detector confidence threshold",
            "validation_selected": validation_selected,
        },
        "test": {
            "detector_only": operational_metrics(
                test_annotations, test_predictions, confidence, iou_threshold
            ),
            "reference_verified": operational_metrics(
                test_annotations, filtered_test, 0.0, iou_threshold
            ),
            "reference_verified_coco": evaluate_coco_predictions(
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
    parser = argparse.ArgumentParser(description="DeepPCB paired-template false-positive verifier")
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--annotations", type=Path, required=True)
    materialize.add_argument("--raw-root", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--validation-annotations", type=Path, required=True)
    evaluate.add_argument("--validation-predictions", type=Path, required=True)
    evaluate.add_argument("--test-annotations", type=Path, required=True)
    evaluate.add_argument("--test-predictions", type=Path, required=True)
    evaluate.add_argument("--raw-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--confidence", type=float, default=0.25)
    evaluate.add_argument("--beta", type=float, default=1.0)
    evaluate.add_argument("--match-iou", type=float, default=0.5)
    args = parser.parse_args()
    if args.command == "materialize":
        payload = materialize_coco_targets(
            args.annotations.resolve(), args.raw_root.resolve(), args.output.resolve()
        )
        print(json.dumps({"images": len(payload["images"])}, ensure_ascii=False))
        return 0
    payload = evaluate_reference_verifier(
        args.validation_annotations.resolve(),
        args.validation_predictions.resolve(),
        args.test_annotations.resolve(),
        args.test_predictions.resolve(),
        args.raw_root.resolve(),
        args.output.resolve(),
        args.confidence,
        args.beta,
        args.match_iou,
    )
    print(json.dumps(payload["selection"], ensure_ascii=False))
    print(json.dumps(payload["test"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
