from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

try:
    from .audit_dataset import parse_label
    from .common import (
        PROJECT_DIR,
        atomic_json_dump,
        class_names,
        dataset_sources,
        image_to_label_path,
        sha256_file,
    )
except ImportError:
    from audit_dataset import parse_label
    from common import PROJECT_DIR, atomic_json_dump, class_names, dataset_sources, image_to_label_path, sha256_file


COLORS = ("#009E73", "#0072B2", "#D55E00")


def display_name(name: str) -> str:
    return {
        "conductor_foreign_object": "conductor FO",
        "base_material_foreign_object": "base FO",
        "conductor_scratch": "scratch",
        "hole_breakout": "breakout",
        "spurious_copper": "spurious",
        "mouse_bite": "mouse bite",
    }.get(name, name.replace("_", " "))


def model_blind_selection(images: list[Path], nc: int, count: int, seed: int) -> list[Path]:
    ranked = sorted(
        images,
        key=lambda path: hashlib.sha256(f"{seed}:{path.name}".encode("utf-8")).hexdigest(),
    )
    selected = []
    covered: set[int] = set()
    for image in ranked:
        boxes, errors = parse_label(image_to_label_path(image), nc)
        if errors:
            continue
        classes = {int(box[0]) for box in boxes}
        if classes - covered:
            selected.append(image)
            covered.update(classes)
        if len(selected) == count:
            return selected
    for image in ranked:
        if image not in selected:
            selected.append(image)
        if len(selected) == count:
            break
    return selected


def draw_ground_truth(axis, image: Path, names: list[str]) -> list[dict]:
    boxes, errors = parse_label(image_to_label_path(image), len(names))
    if errors:
        raise ValueError(f"Invalid label for {image}: {errors}")
    width, height = Image.open(image).size
    rows = []
    for class_id, center_x, center_y, box_w, box_h in boxes:
        x1 = (center_x - box_w / 2) * width
        y1 = (center_y - box_h / 2) * height
        pixel_w = box_w * width
        pixel_h = box_h * height
        axis.add_patch(Rectangle((x1, y1), pixel_w, pixel_h, fill=False, edgecolor=COLORS[0], linewidth=1.2))
        axis.text(
            x1,
            max(1, y1 - 2),
            display_name(names[class_id]),
            color="white",
            fontsize=5.5,
            bbox={"facecolor": COLORS[0], "alpha": 0.9, "pad": 0.8, "edgecolor": "none"},
        )
        rows.append({"class_id": class_id, "class_name": names[class_id], "xyxy": [x1, y1, x1 + pixel_w, y1 + pixel_h]})
    return rows


def draw_predictions(axis, result, names: list[str], color: str) -> list[dict]:
    rows = []
    if result.boxes is None:
        return rows
    xyxy = result.boxes.xyxy.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().int().tolist()
    confidences = result.boxes.conf.detach().cpu().tolist()
    for box, class_id, confidence in zip(xyxy, classes, confidences):
        x1, y1, x2, y2 = box
        axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.2))
        axis.text(
            x1,
            max(1, y1 - 2),
            f"{display_name(names[class_id])} {confidence:.2f}",
            color="white",
            fontsize=5.5,
            bbox={"facecolor": color, "alpha": 0.9, "pad": 0.8, "edgecolor": "none"},
        )
        rows.append({"class_id": class_id, "class_name": names[class_id], "confidence": confidence, "xyxy": box})
    return rows


def render(
    baseline_weights: Path,
    candidate_weights: Path,
    data: Path,
    output: Path,
    manifest: Path,
    device: str,
    split: str,
    count: int,
    seed: int,
    confidence: float,
) -> dict:
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from algorithm.register import register_custom_modules

    register_custom_modules()
    from ultralytics import YOLO

    dataset, _, sources = dataset_sources(data.resolve())
    names = class_names(dataset)
    selected = model_blind_selection(sources[split], len(names), count, seed)
    models = {
        "YOLO11s": YOLO(str(baseline_weights.resolve())),
        "WSR-YOLO11s": YOLO(str(candidate_weights.resolve())),
    }
    figure, axes = plt.subplots(count, 3, figsize=(7.05, 1.72 * count), squeeze=False)
    rows = []
    for row_index, image in enumerate(selected):
        opened = Image.open(image).convert("RGB")
        for axis in axes[row_index]:
            axis.imshow(opened)
            axis.set_axis_off()
        ground_truth = draw_ground_truth(axes[row_index, 0], image, names)
        item = {"image": str(image), "image_sha256": sha256_file(image), "ground_truth": ground_truth, "predictions": {}}
        for column, (name, model) in enumerate(models.items(), 1):
            result = model.predict(
                source=str(image),
                imgsz=640,
                conf=confidence,
                iou=0.7,
                max_det=100,
                device=device,
                verbose=False,
            )[0]
            item["predictions"][name] = draw_predictions(axes[row_index, column], result, names, COLORS[column])
        rows.append(item)
    for column, title in enumerate(("Ground truth", "YOLO11s", "WSR-YOLO11s")):
        axes[0, column].set_title(title, fontsize=9)
    figure.tight_layout(pad=0.25, w_pad=0.25, h_pad=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    payload = {
        "schema_version": 1,
        "selection": "SHA-256 rank followed by greedy ground-truth class coverage; predictions unused",
        "selection_seed": seed,
        "confidence_threshold": confidence,
        "data_yaml": str(data.resolve()),
        "data_yaml_sha256": sha256_file(data),
        "baseline_weights_sha256": sha256_file(baseline_weights),
        "candidate_weights_sha256": sha256_file(candidate_weights),
        "rows": rows,
    }
    atomic_json_dump(payload, manifest.resolve())
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a model-blind qualitative comparison")
    parser.add_argument("--baseline-weights", type=Path, required=True)
    parser.add_argument("--candidate-weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    render(
        args.baseline_weights,
        args.candidate_weights,
        args.data,
        args.output,
        args.manifest,
        args.device,
        args.split,
        args.count,
        args.seed,
        args.confidence,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
