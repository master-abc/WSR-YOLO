"""Attach deterministic golden-reference change evidence to detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .common import atomic_json_dump
except ImportError:
    from experiment.wsr.common import atomic_json_dump


def box_change_evidence(encoded_bgr: np.ndarray, xyxy: list[float]) -> float:
    """Return the 90th percentile signed-change magnitude inside a box."""

    if encoded_bgr.ndim != 3 or encoded_bgr.shape[2] != 3:
        raise ValueError("encoded_bgr must have exactly three channels")
    height, width = encoded_bgr.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    left = max(0, min(width - 1, int(np.floor(x1))))
    top = max(0, min(height - 1, int(np.floor(y1))))
    right = max(left + 1, min(width, int(np.ceil(x2))))
    bottom = max(top + 1, min(height, int(np.ceil(y2))))
    patch = encoded_bgr[top:bottom, left:right]
    # Context encoding stores candidate+change and candidate-change in G/R.
    change = np.abs(patch[:, :, 1].astype(np.float32) - patch[:, :, 2]) / 2.0
    return float(np.percentile(change, 90.0))


def box_structural_change_evidence(
    encoded_bgr: np.ndarray, xyxy: list[float]
) -> float:
    """Return largest opened change component as percent of box area."""

    if encoded_bgr.ndim != 3 or encoded_bgr.shape[2] != 3:
        raise ValueError("encoded_bgr must have exactly three channels")
    height, width = encoded_bgr.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    left = max(0, min(width - 1, int(np.floor(x1))))
    top = max(0, min(height - 1, int(np.floor(y1))))
    right = max(left + 1, min(width, int(np.ceil(x2))))
    bottom = max(top + 1, min(height, int(np.ceil(y2))))
    patch = encoded_bgr[top:bottom, left:right]
    change = np.abs(patch[:, :, 1].astype(np.float32) - patch[:, :, 2]) / 2.0
    mask = (change >= 12.0).astype(np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    components, _, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    largest = 0 if components <= 1 else int(stats[1:, cv2.CC_STAT_AREA].max())
    return 100.0 * largest / float(mask.size)


EVIDENCE_STATISTICS = {
    "p90_channel_separation": box_change_evidence,
    "opened_component_area_pct": box_structural_change_evidence,
}


def _read_encoded(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read encoded image: {path}")
    return image


def enrich_coco_predictions(
    annotations_path: Path,
    predictions_path: Path,
    image_dir: Path,
    output_path: Path,
    statistic: str = "p90_channel_separation",
) -> list[dict[str, Any]]:
    if statistic not in EVIDENCE_STATISTICS:
        raise ValueError(f"Unsupported evidence statistic: {statistic}")
    evidence_function = EVIDENCE_STATISTICS[statistic]
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    files = {int(row["id"]): image_dir / row["file_name"] for row in annotations["images"]}
    cache: dict[int, np.ndarray] = {}
    for detection in predictions:
        image_id = int(detection["image_id"])
        if image_id not in cache:
            cache[image_id] = _read_encoded(files[image_id])
        x, y, width, height = (float(value) for value in detection["bbox"])
        detection["change_evidence"] = evidence_function(
            cache[image_id], [x, y, x + width, y + height]
        )
    atomic_json_dump(predictions, output_path)
    return predictions


def enrich_negative_audit(
    audit_path: Path,
    output_path: Path,
    statistic: str = "p90_channel_separation",
) -> dict[str, Any]:
    if statistic not in EVIDENCE_STATISTICS:
        raise ValueError(f"Unsupported evidence statistic: {statistic}")
    evidence_function = EVIDENCE_STATISTICS[statistic]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for row in audit["per_image"]:
        image = _read_encoded(Path(row["image"]))
        for detection in row.get("detections", []):
            detection["change_evidence"] = evidence_function(
                image, detection["xyxy"]
            )
    audit["change_evidence"] = {
        "source": "context-encoding channel separation",
        "statistic": statistic,
        "uses_second_model": False,
    }
    atomic_json_dump(audit, output_path)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Attach reference-change evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    coco = subparsers.add_parser("coco")
    coco.add_argument("--annotations", type=Path, required=True)
    coco.add_argument("--predictions", type=Path, required=True)
    coco.add_argument("--image-dir", type=Path, required=True)
    coco.add_argument("--output", type=Path, required=True)
    coco.add_argument(
        "--statistic", choices=sorted(EVIDENCE_STATISTICS), default="p90_channel_separation"
    )
    negative = subparsers.add_parser("negative")
    negative.add_argument("--audit", type=Path, required=True)
    negative.add_argument("--output", type=Path, required=True)
    negative.add_argument(
        "--statistic", choices=sorted(EVIDENCE_STATISTICS), default="p90_channel_separation"
    )
    args = parser.parse_args()
    if args.command == "coco":
        enrich_coco_predictions(
            args.annotations.resolve(),
            args.predictions.resolve(),
            args.image_dir.resolve(),
            args.output.resolve(),
            args.statistic,
        )
    else:
        enrich_negative_audit(
            args.audit.resolve(), args.output.resolve(), args.statistic
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
