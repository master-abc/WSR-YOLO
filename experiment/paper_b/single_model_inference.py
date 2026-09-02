"""Run the frozen paired-reference detector with one model forward pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

try:
    from .common import atomic_json_dump, sha256_file
    from .paired_difference import encode_context_difference, encode_paired_difference
    from .reference_evidence import EVIDENCE_STATISTICS
except ImportError:
    from experiment.paper_b.common import atomic_json_dump, sha256_file
    from experiment.paper_b.paired_difference import (
        encode_context_difference,
        encode_paired_difference,
    )
    from experiment.paper_b.reference_evidence import EVIDENCE_STATISTICS


ENCODERS = {
    "context": encode_context_difference,
    "residual": encode_paired_difference,
}


def _read_grayscale(path: Path, role: str):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read {role}: {path}")
    # Ultralytics patches cv2.imread process-wide and keeps an explicit channel
    # axis for grayscale inputs. Normalize both OpenCV behaviours to H x W.
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise ValueError(f"{role.capitalize()} must be readable as one grayscale plane")
    return image


def load_frozen_policy(policy_path: Path) -> dict[str, Any]:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("test_evaluated_during_selection") is not False:
        raise ValueError("Policy is not validation-locked")
    model_input = policy.get("model_input")
    if not isinstance(model_input, dict):
        raise ValueError("Policy does not contain a frozen model_input contract")
    encoding = model_input.get("encoding")
    if encoding not in ENCODERS:
        raise ValueError(f"Unsupported frozen encoding: {encoding}")
    if model_input.get("single_model_forward_pass") is not True:
        raise ValueError("Policy is not a single-forward-pass contract")
    return policy


def encode_pair(candidate_path: Path, reference_path: Path, policy: dict[str, Any]):
    candidate = _read_grayscale(candidate_path, "candidate")
    reference = _read_grayscale(reference_path, "reference")
    contract = policy["model_input"]
    return ENCODERS[str(contract["encoding"])](
        candidate,
        reference,
        noise_floor=float(contract["noise_floor"]),
        gain=float(contract["gain"]),
    )


def infer_pair(
    candidate_path: Path,
    reference_path: Path,
    policy_path: Path,
    output_path: Path,
    annotated_image_path: Path | None = None,
    device: str = "0",
    imgsz: int = 640,
    iou: float = 0.7,
) -> dict[str, Any]:
    policy = load_frozen_policy(policy_path)
    weights_path = Path(policy["weights"])
    if not weights_path.is_file():
        raise FileNotFoundError(f"Cannot find frozen weights: {weights_path}")
    if sha256_file(weights_path) != policy["weights_sha256"]:
        raise ValueError("Frozen weights hash does not match policy")
    # Ultralytics replaces a few OpenCV helpers at import time. Read every raw
    # deployment image first so IMREAD_GRAYSCALE retains its standard meaning.
    encoded = encode_pair(candidate_path, reference_path, policy)
    candidate_gray = _read_grayscale(candidate_path, "candidate")
    from algorithm.register import register_custom_modules

    register_custom_modules()
    from ultralytics import YOLO

    thresholds = {int(key): float(value) for key, value in policy["thresholds"].items()}
    minimum_change_evidence = float(policy.get("minimum_change_evidence", 0.0))
    evidence_statistic = str(
        policy.get("change_evidence_statistic", "p90_channel_separation")
    )
    if evidence_statistic not in EVIDENCE_STATISTICS:
        raise ValueError(f"Unsupported frozen evidence statistic: {evidence_statistic}")
    result = YOLO(str(weights_path)).predict(
        source=encoded,
        conf=min(thresholds.values()),
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]
    detections: list[dict[str, Any]] = []
    if result.boxes is not None:
        for xyxy, score, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            category = int(class_id)
            if float(score) < thresholds.get(category, 1.0):
                continue
            change_evidence = EVIDENCE_STATISTICS[evidence_statistic](
                encoded, [float(value) for value in xyxy]
            )
            if change_evidence < minimum_change_evidence:
                continue
            detections.append(
                {
                    "class_id": category,
                    "class_name": str(result.names[category]),
                    "score": float(score),
                    "change_evidence": change_evidence,
                    "xyxy": [float(value) for value in xyxy],
                }
            )
    payload = {
        "schema_version": 1,
        "single_model": True,
        "forward_passes": 1,
        "candidate": str(candidate_path.resolve()),
        "reference": str(reference_path.resolve()),
        "policy": str(policy_path.resolve()),
        "policy_sha256": sha256_file(policy_path),
        "weights": str(weights_path.resolve()),
        "weights_sha256": policy["weights_sha256"],
        "alarmed": bool(detections),
        "detections": detections,
    }
    atomic_json_dump(payload, output_path)
    if annotated_image_path is not None:
        canvas = cv2.cvtColor(candidate_gray, cv2.COLOR_GRAY2BGR)
        for detection in detections:
            x1, y1, x2, y2 = [int(round(value)) for value in detection["xyxy"]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f'{detection["class_name"]} {detection["score"]:.2f}'
            cv2.putText(
                canvas,
                label,
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        annotated_image_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(annotated_image_path), canvas):
            raise RuntimeError(f"Failed to write {annotated_image_path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-pass inference for the frozen paired-reference detector"
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotated-image", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()
    payload = infer_pair(
        args.candidate.resolve(),
        args.reference.resolve(),
        args.policy.resolve(),
        args.output.resolve(),
        args.annotated_image.resolve() if args.annotated_image else None,
        args.device,
        args.imgsz,
        args.iou,
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
