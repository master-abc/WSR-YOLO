from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .common import (
        IMAGE_SUFFIXES,
        PROJECT_DIR,
        atomic_json_dump,
        environment_snapshot,
        sha256_file,
        sha256_text,
    )
except ImportError:
    from common import IMAGE_SUFFIXES, PROJECT_DIR, atomic_json_dump, environment_snapshot, sha256_file, sha256_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure false alarms on verified defect-free PCB templates")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Directory containing defect-free images only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.25, 0.50])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name-pattern", default="*")
    parser.add_argument("--image-list", type=Path)
    parser.add_argument("--expected-images", type=int, default=0)
    args = parser.parse_args()
    if args.image_list:
        images = sorted(
            Path(line.strip()).resolve()
            for line in args.image_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        missing = [path for path in images if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Image list contains missing paths: {missing[:3]}")
    else:
        images = sorted(
            path
            for path in args.images.resolve().rglob(args.name_pattern)
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if not images:
        raise ValueError(f"No images found under {args.images}")
    if args.expected_images and len(images) != args.expected_images:
        raise ValueError(f"Expected {args.expected_images} images, found {len(images)}")
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from algorithm.register import register_custom_modules

    register_custom_modules()
    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()))
    scores = []
    per_image = []
    if args.batch < 1:
        raise ValueError("batch must be at least 1")
    # Ultralytics may treat a path list as one in-memory batch regardless of
    # ``batch``. Explicit chunks prevent a 1,500-image negative audit from
    # exhausting GPU memory.
    for offset in range(0, len(images), args.batch):
        chunk = images[offset : offset + args.batch]
        results = model.predict(
            source=[str(path) for path in chunk],
            conf=min(args.thresholds),
            imgsz=args.imgsz,
            device=args.device,
            stream=True,
            batch=len(chunk),
            iou=0.7,
            max_det=300,
            verbose=False,
        )
        observed = 0
        for image, result in zip(chunk, results):
            observed += 1
            confidence = [float(value) for value in result.boxes.conf.detach().cpu().tolist()]
            classes = [int(value) for value in result.boxes.cls.detach().cpu().tolist()]
            boxes = [
                [float(value) for value in coordinates]
                for coordinates in result.boxes.xyxy.detach().cpu().tolist()
            ]
            detections = [
                {
                    "score": score,
                    "class_id": class_id,
                    "class_name": str(result.names.get(class_id, class_id)),
                    "xyxy": coordinates,
                }
                for score, class_id, coordinates in zip(confidence, classes, boxes)
            ]
            scores.extend(confidence)
            per_image.append(
                {"image": str(image), "scores": confidence, "detections": detections}
            )
        if observed != len(chunk):
            raise RuntimeError(
                f"Ultralytics returned {observed} results for {len(chunk)} negative images"
            )
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
            "schema_version": 3,
            "warning": "Valid only if every supplied image is independently verified defect-free.",
            "weights": str(args.weights.resolve()),
            "weights_sha256": sha256_file(args.weights),
            "image_root": str(args.images.resolve()),
            "image_pattern": args.name_pattern,
            "image_list": str(args.image_list.resolve()) if args.image_list else None,
            "image_manifest_sha256": sha256_text(
                "\n".join(
                    f"{path.relative_to(args.images.resolve()).as_posix()} {sha256_file(path)}"
                    for path in images
                )
            ),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "iou": 0.7,
            "max_det": 300,
            "metrics": metrics,
            "per_image": per_image,
            "environment": environment_snapshot(),
        },
        args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

