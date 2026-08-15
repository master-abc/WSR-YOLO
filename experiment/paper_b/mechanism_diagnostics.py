from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .audit_dataset import parse_label
    from .common import PROJECT_DIR, atomic_json_dump, dataset_sources, image_to_label_path, sha256_file
except ImportError:
    from audit_dataset import parse_label
    from common import PROJECT_DIR, atomic_json_dump, dataset_sources, image_to_label_path, sha256_file


def ground_truth_mask(image: Path, shape: tuple[int, int], imgsz: int, nc: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    with Image.open(image) as opened:
        width, height = opened.size
    feature_h, feature_w = shape
    ratio = min(imgsz / width, imgsz / height)
    resized_w, resized_h = round(width * ratio), round(height * ratio)
    pad_x, pad_y = (imgsz - resized_w) / 2.0, (imgsz - resized_h) / 2.0
    boxes, errors = parse_label(image_to_label_path(image), nc)
    if errors:
        raise ValueError(f"Invalid labels for {image}: {errors}")
    mask = np.zeros((feature_h, feature_w), dtype=bool)
    centers = []
    for _, center_x, center_y, box_w, box_h in boxes:
        x1 = ((center_x - box_w / 2) * width * ratio + pad_x) / imgsz * feature_w
        x2 = ((center_x + box_w / 2) * width * ratio + pad_x) / imgsz * feature_w
        y1 = ((center_y - box_h / 2) * height * ratio + pad_y) / imgsz * feature_h
        y2 = ((center_y + box_h / 2) * height * ratio + pad_y) / imgsz * feature_h
        ix1, ix2 = max(0, int(np.floor(x1))), min(feature_w, max(int(np.ceil(x2)), int(np.floor(x1)) + 1))
        iy1, iy2 = max(0, int(np.floor(y1))), min(feature_h, max(int(np.ceil(y2)), int(np.floor(y1)) + 1))
        mask[iy1:iy2, ix1:ix2] = True
        centers.append(
            (
                min(feature_h - 1, max(0, int(((center_y * height * ratio + pad_y) / imgsz) * feature_h))),
                min(feature_w - 1, max(0, int(((center_x * width * ratio + pad_x) / imgsz) * feature_w))),
            )
        )
    return mask, centers


def summarize(rows: list[dict]) -> dict:
    keys = ("route_density", "route_recall", "route_precision", "route_enrichment", "center_hit_rate", "wave_inside", "wave_outside")
    output = {}
    for key in keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        output[key] = {
            "n": len(values),
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.stdev(values) if len(values) >= 2 else None,
        }
    return output


def diagnose_routes(
    weights: Path,
    data: Path,
    output: Path,
    split: str,
    imgsz: int,
    device: str,
    limit: int = 0,
) -> dict:
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from algorithm.register import register_custom_modules
    from algorithm.dwgsa import DWGSARouter

    register_custom_modules()
    from ultralytics import YOLO

    _, _, sources = dataset_sources(data.resolve())
    if split not in sources:
        raise KeyError(f"Dataset has no '{split}' split: {data.resolve()}")
    images = sources[split][: limit or None]
    model = YOLO(str(weights.resolve()))
    routers = [module for module in model.model.modules() if isinstance(module, DWGSARouter)]
    if not routers:
        raise ValueError("The checkpoint contains no DWGSARouter module")
    for router in routers:
        router.enable_diagnostics(True)
    rows = []
    for image in images:
        model.predict(source=str(image), imgsz=imgsz, device=device, rect=False, verbose=False)
        for module_index, router in enumerate(routers):
            route = router.last_route_mask[0, 0].cpu().numpy().astype(bool)
            wave = router.last_wave_attention[0, 0].cpu().numpy()
            ground_truth, centers = ground_truth_mask(image, route.shape, imgsz, len(model.names))
            intersection = np.logical_and(route, ground_truth).sum()
            route_density = float(route.mean())
            gt_area = int(ground_truth.sum())
            route_area = int(route.sum())
            route_recall = float(intersection / gt_area) if gt_area else None
            route_precision = float(intersection / route_area) if route_area else None
            center_hit = float(np.mean([route[y, x] for y, x in centers])) if centers else None
            rows.append(
                {
                    "image": str(image),
                    "module_index": module_index,
                    "feature_shape": list(route.shape),
                    "gt_pixels": gt_area,
                    "route_density": route_density,
                    "route_recall": route_recall,
                    "route_precision": route_precision,
                    "route_enrichment": route_recall / route_density if route_recall is not None and route_density else None,
                    "center_hit_rate": center_hit,
                    "wave_inside": float(wave[ground_truth].mean()) if gt_area else None,
                    "wave_outside": float(wave[~ground_truth].mean()) if (~ground_truth).any() else None,
                    "fusion_weights": router.last_fusion_weights[0].cpu().tolist(),
                }
            )
    payload = {
        "schema_version": 1,
        "weights": str(weights.resolve()),
        "weights_sha256": sha256_file(weights),
        "data_yaml": str(data.resolve()),
        "data_yaml_sha256": sha256_file(data),
        "selection_split": split,
        "imgsz": int(imgsz),
        "summary": summarize(rows),
        "images": len(images),
        "rows": rows,
    }
    atomic_json_dump(payload, output.resolve())
    print(f"Analyzed {len(images)} images and {len(rows)} router outputs")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure whether sparse routes concentrate on real defects")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    diagnose_routes(
        args.weights,
        args.data,
        args.output,
        args.split,
        args.imgsz,
        args.device,
        args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
