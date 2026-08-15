from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .common import IMAGE_SUFFIXES, PROJECT_DIR, atomic_json_dump
except ImportError:
    from common import IMAGE_SUFFIXES, PROJECT_DIR, atomic_json_dump


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure false alarms on verified defect-free PCB templates")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Directory containing defect-free images only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.25, 0.50])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    images = sorted(
        path for path in args.images.resolve().rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No images found under {args.images}")
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from algorithm.register import register_custom_modules

    register_custom_modules()
    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()))
    results = model.predict(
        source=[str(path) for path in images],
        conf=min(args.thresholds),
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        verbose=False,
    )
    scores = []
    per_image = []
    for image, result in zip(images, results):
        confidence = result.boxes.conf.detach().cpu().tolist()
        scores.extend(confidence)
        per_image.append({"image": str(image), "scores": confidence})
    metrics = {}
    for threshold in args.thresholds:
        counts = [sum(score >= threshold for score in row["scores"]) for row in per_image]
        metrics[str(threshold)] = {
            "board_false_positive_rate": sum(count > 0 for count in counts) / len(counts),
            "false_positives_per_image": sum(counts) / len(counts),
            "false_positives": sum(counts),
            "images": len(counts),
        }
    atomic_json_dump(
        {
            "warning": "Valid only if every supplied image is independently verified defect-free.",
            "weights": str(args.weights.resolve()),
            "image_root": str(args.images.resolve()),
            "metrics": metrics,
            "per_image": per_image,
        },
        args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

