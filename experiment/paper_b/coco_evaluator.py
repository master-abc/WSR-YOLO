from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any

try:
    from .common import atomic_json_dump, sha256_file
except ImportError:
    from common import atomic_json_dump, sha256_file


COCO_STAT_NAMES = (
    "map50_95",
    "map50",
    "map75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar_1",
    "ar_10",
    "ar_100",
    "ar_small",
    "ar_medium",
    "ar_large",
)


def load_coco_api():
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError(
            "Unified evaluation requires pycocotools. Install requirements-paper-b.txt."
        ) from exc
    return COCO, COCOeval


def _empty_detection_api(coco_gt, coco_class):
    coco_dt = coco_class()
    coco_dt.dataset = {
        "images": list(coco_gt.dataset.get("images", [])),
        "categories": list(coco_gt.dataset.get("categories", [])),
        "annotations": [],
    }
    coco_dt.createIndex()
    return coco_dt


def evaluate_coco_predictions(
    annotations: str | Path,
    predictions: str | Path,
    max_det: int = 100,
) -> dict[str, Any]:
    """Evaluate any detector's prediction JSON with one pinned COCO API."""

    annotations = Path(annotations).resolve()
    predictions = Path(predictions).resolve()
    prediction_rows = json.loads(predictions.read_text(encoding="utf-8"))
    if not isinstance(prediction_rows, list):
        raise ValueError("COCO predictions must be a JSON list")

    COCO, COCOeval = load_coco_api()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(annotations))
        coco_dt = coco_gt.loadRes(prediction_rows) if prediction_rows else _empty_detection_api(coco_gt, COCO)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = sorted(coco_gt.getImgIds())
        # COCOeval.summarize() is defined for the canonical [1, 10, 100]
        # limits. A custom final value makes the public ``stats`` slots return
        # -1 even for perfect detections. Prediction export may retain more
        # candidates; the evaluator deterministically truncates them to 100.
        if int(max_det) != 100:
            raise ValueError("Unified COCO evaluation requires max_det=100")
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    stats = [float(value) for value in evaluator.stats]
    metrics: dict[str, Any] = {
        name: (stats[index] if index < len(stats) and stats[index] >= 0.0 else None)
        for index, name in enumerate(COCO_STAT_NAMES)
    }
    category_ids = list(evaluator.params.catIds)
    categories = {item["id"]: item["name"] for item in coco_gt.dataset.get("categories", [])}
    precision = evaluator.eval.get("precision")
    per_class: dict[str, float | None] = {}
    if precision is not None:
        for category_index, category_id in enumerate(category_ids):
            values = precision[:, :, category_index, 0, -1]
            valid = values[values > -1]
            per_class[categories.get(category_id, str(category_id))] = (
                float(valid.mean()) if valid.size else None
            )
    metrics["per_class_ap50_95"] = per_class

    try:
        import pycocotools

        version = getattr(pycocotools, "__version__", "unknown")
    except Exception:
        version = "unknown"
    return {
        "evaluator": "pycocotools.COCOeval",
        "pycocotools": version,
        "annotations": str(annotations),
        "annotations_sha256": sha256_file(annotations),
        "predictions": str(predictions),
        "predictions_sha256": sha256_file(predictions),
        "prediction_count": len(prediction_rows),
        "image_count": len(coco_gt.getImgIds()),
        "max_det": int(max_det),
        "metrics": metrics,
    }


def predict_yolo_to_coco(
    yolo: Any,
    annotations: str | Path,
    image_dir: str | Path,
    output: str | Path,
    imgsz: int,
    batch: int,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
) -> Path:
    """Export Ultralytics predictions without using its AP implementation."""

    if int(batch) < 1:
        raise ValueError("Evaluation batch must be at least 1")

    annotations = Path(annotations).resolve()
    image_dir = Path(image_dir).resolve()
    output = Path(output).resolve()
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    images = sorted(payload["images"], key=lambda item: int(item["id"]))
    paths = [str(image_dir / item["file_name"]) for item in images]
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"COCO evaluation images are missing, first={missing[0]}")
    category_ids = [item["id"] for item in sorted(payload["categories"], key=lambda item: item["id"])]

    predictions: list[dict[str, Any]] = []
    # Ultralytics treats a Python list as an in-memory source and may use the
    # entire list as one batch, irrespective of the requested ``batch`` value.
    # Explicit chunks keep low-confidence COCO export bounded in GPU memory.
    batch_size = int(batch)
    for offset in range(0, len(paths), batch_size):
        chunk_images = images[offset : offset + batch_size]
        chunk_paths = paths[offset : offset + batch_size]
        results = yolo.predict(
            source=chunk_paths,
            stream=True,
            imgsz=int(imgsz),
            batch=len(chunk_paths),
            device=device,
            conf=float(conf),
            iou=float(iou),
            max_det=int(max_det),
            verbose=False,
        )
        observed = 0
        for image, result in zip(chunk_images, results):
            observed += 1
            if not hasattr(result, "boxes"):
                raise TypeError(
                    "Ultralytics prediction did not return a Results object. "
                    "Load end-to-end models such as YOLOv10 with their dedicated "
                    "model class instead of silently dropping every detection."
                )
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().tolist()
            scores = boxes.conf.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            for coordinates, score, class_value in zip(xyxy, scores, classes):
                x1, y1, x2, y2 = (float(value) for value in coordinates)
                class_index = int(class_value)
                if not 0 <= class_index < len(category_ids):
                    raise ValueError(
                        f"Predicted class {class_index} is outside {len(category_ids)} classes"
                    )
                predictions.append(
                    {
                        "image_id": int(image["id"]),
                        "category_id": int(category_ids[class_index]),
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    }
                )
        if observed != len(chunk_images):
            raise RuntimeError(
                f"Ultralytics returned {observed} results for {len(chunk_images)} evaluation images"
            )
    atomic_json_dump(predictions, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="One COCO evaluator for every paper model")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--annotations", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--max-det", type=int, default=100)
    predict = subparsers.add_parser("predict-yolo")
    predict.add_argument("--weights", type=Path, required=True)
    predict.add_argument("--annotations", type=Path, required=True)
    predict.add_argument("--image-dir", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--batch", type=int, default=1)
    predict.add_argument("--device", default="0")
    predict.add_argument("--conf", type=float, default=0.001)
    predict.add_argument("--iou", type=float, default=0.7)
    predict.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()

    if args.command == "evaluate":
        report = evaluate_coco_predictions(args.annotations, args.predictions, args.max_det)
        atomic_json_dump(report, args.output.resolve())
        print(args.output.resolve())
        return 0

    from ultralytics import YOLO

    yolo = YOLO(str(args.weights.resolve()))
    predict_yolo_to_coco(
        yolo,
        args.annotations,
        args.image_dir,
        args.output,
        args.imgsz,
        args.batch,
        args.device,
        args.conf,
        args.iou,
        args.max_det,
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
