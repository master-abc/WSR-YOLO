from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
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


def scalar_summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "std": None, "median": None, "ci95": [None, None]}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else None
    if std is None:
        interval = [mean, mean]
    else:
        try:
            from scipy.stats import t

            half_width = float(t.ppf(0.975, len(values) - 1)) * std / np.sqrt(len(values))
        except ImportError:
            half_width = 1.96 * std / np.sqrt(len(values))
        interval = [mean - half_width, mean + half_width]
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "median": statistics.median(values),
        "ci95": interval,
    }


def summarize(rows: list[dict]) -> dict:
    keys = ("route_density", "route_recall", "route_precision", "route_enrichment", "center_hit_rate", "wave_inside", "wave_outside")
    return {
        key: scalar_summary([row[key] for row in rows if row.get(key) is not None])
        for key in keys
    }


def topk_mask(score: np.ndarray, k: int) -> np.ndarray:
    flat = np.asarray(score).reshape(-1)
    k = min(flat.size, max(1, int(k)))
    indices = np.argpartition(flat, flat.size - k)[-k:]
    mask = np.zeros(flat.size, dtype=bool)
    mask[indices] = True
    return mask.reshape(score.shape)


def center_prior(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    y = (yy + 0.5) / height - 0.5
    x = (xx + 0.5) / width - 0.5
    return -(x * x + y * y)


def mask_metrics(route: np.ndarray, ground_truth: np.ndarray, centers: list[tuple[int, int]]) -> dict:
    intersection = int(np.logical_and(route, ground_truth).sum())
    density = float(route.mean())
    gt_area = int(ground_truth.sum())
    route_area = int(route.sum())
    recall = float(intersection / gt_area) if gt_area else None
    return {
        "route_density": density,
        "route_recall": recall,
        "route_precision": float(intersection / route_area) if route_area else None,
        "route_enrichment": recall / density if recall is not None and density else None,
        "center_hit_rate": float(np.mean([route[y, x] for y, x in centers])) if centers else None,
    }


def training_occupancy_prior(
    images: list[Path], shape: tuple[int, int], imgsz: int, nc: int
) -> np.ndarray:
    occupancy = np.zeros(shape, dtype=np.float64)
    for image in images:
        ground_truth, _ = ground_truth_mask(image, shape, imgsz, nc)
        occupancy += ground_truth
    if images:
        occupancy /= len(images)
    return occupancy


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_control_tests(rows: list[dict]) -> dict:
    names = sorted({name for row in rows for name in row.get("controls", {}) if name != "actual"})
    tests = []
    for name in names:
        actual = []
        control = []
        for row in rows:
            actual_value = row.get("controls", {}).get("actual", {}).get("route_enrichment")
            control_value = row.get("controls", {}).get(name, {}).get("route_enrichment")
            if actual_value is not None and control_value is not None:
                actual.append(actual_value)
                control.append(control_value)
        difference = np.asarray(actual) - np.asarray(control)
        if difference.size and np.any(difference != 0):
            try:
                from scipy.stats import wilcoxon

                result = wilcoxon(actual, control, zero_method="wilcox", alternative="two-sided", method="auto")
                p_value = float(result.pvalue)
                statistic = float(result.statistic)
            except ImportError:
                p_value = None
                statistic = None
        else:
            p_value = 1.0 if difference.size else None
            statistic = 0.0 if difference.size else None
        tests.append(
            {
                "control": name,
                "n": int(difference.size),
                "mean_paired_difference": float(difference.mean()) if difference.size else None,
                "median_paired_difference": float(np.median(difference)) if difference.size else None,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    valid = [index for index, item in enumerate(tests) if item["p_value"] is not None]
    if valid:
        adjusted = holm_adjust([tests[index]["p_value"] for index in valid])
        for index, value in zip(valid, adjusted):
            tests[index]["holm_adjusted_p"] = value
    return {item["control"]: item for item in tests}


def summarize_controls(rows: list[dict]) -> dict:
    names = sorted({name for row in rows for name in row.get("controls", {})})
    keys = ("route_density", "route_recall", "route_precision", "route_enrichment", "center_hit_rate")
    return {
        name: {
            key: scalar_summary(
                [
                    row["controls"][name][key]
                    for row in rows
                    if name in row.get("controls", {}) and row["controls"][name].get(key) is not None
                ]
            )
            for key in keys
        }
        for name in names
    }


def shuffled_route_summary(
    route_masks: dict[int, list[np.ndarray]],
    ground_truth_masks: dict[int, list[np.ndarray]],
    repeats: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    output = {}
    for module_index in sorted(route_masks):
        routes = np.stack(route_masks[module_index])
        ground_truth = np.stack(ground_truth_masks[module_index])
        repeat_means = []
        for _ in range(repeats):
            permutation = rng.permutation(len(routes))
            if len(routes) > 1 and np.array_equal(permutation, np.arange(len(routes))):
                permutation = np.roll(permutation, 1)
            values = []
            for route, truth in zip(routes[permutation], ground_truth):
                gt_area = int(truth.sum())
                density = float(route.mean())
                if gt_area and density:
                    recall = float(np.logical_and(route, truth).sum() / gt_area)
                    values.append(recall / density)
            repeat_means.append(statistics.fmean(values))
        output[str(module_index)] = {
            "repeats": repeats,
            "permutation_seed": seed,
            "mean_route_enrichment_by_permutation": scalar_summary(repeat_means),
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
    controls: bool = False,
    control_split: str = "train",
    control_limit: int = 0,
    shuffle_repeats: int = 32,
    seed: int = 2026,
) -> dict:
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from algorithm.register import register_custom_modules
    from algorithm.dwgsa import DWGSARouter, StableWaveletContextRouter

    register_custom_modules()
    import torch
    import torch.nn.functional as F
    from ultralytics import YOLO

    _, _, sources = dataset_sources(data.resolve())
    if split not in sources:
        raise KeyError(f"Dataset has no '{split}' split: {data.resolve()}")
    if controls and control_split not in sources:
        raise KeyError(f"Dataset has no control split '{control_split}': {data.resolve()}")
    images = sources[split][: limit or None]
    control_images = sources.get(control_split, [])[: control_limit or None]
    model = YOLO(str(weights.resolve()))
    routers = [module for module in model.model.modules() if isinstance(module, DWGSARouter)]
    if not routers:
        raise ValueError("The checkpoint contains no DWGSARouter module")
    for router in routers:
        router.enable_diagnostics(True)

    captured: dict[int, dict[str, np.ndarray]] = {}
    handles = []

    def capture_input(module_index: int):
        def hook(router, inputs):
            if not controls:
                return
            with torch.no_grad():
                projected = router.project(inputs[0])
                wave, _ = projected.chunk(2, dim=1)
                activation = wave.abs().mean(1, keepdim=True)
                context_router = router.context_router
                ll, lh, hl, hh = context_router.dwt(wave)
                directional = torch.cat(
                    [lh.abs().mean(1, keepdim=True), hl.abs().mean(1, keepdim=True), hh.abs().mean(1, keepdim=True)],
                    dim=1,
                )
                ll_context = ll.abs().mean(1, keepdim=True)
                ll_residual = (ll_context - F.avg_pool2d(ll_context, 3, 1, 1)).abs()
                size = wave.shape[2:]
                if isinstance(context_router, StableWaveletContextRouter):
                    directional = context_router._spatial_ratio(directional)
                    hf_value = directional.mean(1, keepdim=True)
                    geo_value = context_router._spatial_ratio(ll_residual)
                    hf_full = F.interpolate(hf_value, size=size, mode="bilinear", align_corners=False)
                    geo_full = F.interpolate(geo_value, size=size, mode="bilinear", align_corners=False)
                else:
                    hf_value = directional.mean(1, keepdim=True)
                    hf_full = F.interpolate(hf_value, size=size, mode="bilinear", align_corners=False)
                    geo_full = F.interpolate(ll_residual, size=size, mode="bilinear", align_corners=False)
                    hf_full = hf_full / (hf_full.mean(dim=(2, 3), keepdim=True) + 1e-6)
                    geo_full = geo_full / (geo_full.mean(dim=(2, 3), keepdim=True) + 1e-6)
                if not context_router.use_hf:
                    hf_full = torch.zeros_like(hf_full)
                if not context_router.use_ll:
                    geo_full = torch.zeros_like(geo_full)
                fixed_haar = torch.log1p(hf_full + geo_full)
                captured[module_index] = {
                    "activation_energy": activation[0, 0].detach().float().cpu().numpy(),
                    "fixed_haar": fixed_haar[0, 0].detach().float().cpu().numpy(),
                }

        return hook

    if controls:
        handles = [router.register_forward_pre_hook(capture_input(index)) for index, router in enumerate(routers)]

    rows = []
    route_masks: dict[int, list[np.ndarray]] = defaultdict(list)
    ground_truth_masks: dict[int, list[np.ndarray]] = defaultdict(list)
    occupancy_cache: dict[tuple[int, int], np.ndarray] = {}
    center_cache: dict[tuple[int, int], np.ndarray] = {}
    rng = np.random.default_rng(seed)
    try:
        for image in images:
            model.predict(source=str(image), imgsz=imgsz, device=device, rect=False, verbose=False)
            for module_index, router in enumerate(routers):
                route = router.last_route_mask[0, 0].cpu().numpy().astype(bool)
                wave = router.last_wave_attention[0, 0].cpu().numpy()
                ground_truth, centers = ground_truth_mask(image, route.shape, imgsz, len(model.names))
                actual_metrics = mask_metrics(route, ground_truth, centers)
                row = {
                    "image": str(image),
                    "module_index": module_index,
                    "feature_shape": list(route.shape),
                    "gt_pixels": int(ground_truth.sum()),
                    **actual_metrics,
                    "wave_inside": float(wave[ground_truth].mean()) if ground_truth.any() else None,
                    "wave_outside": float(wave[~ground_truth].mean()) if (~ground_truth).any() else None,
                    "fusion_weights": router.last_fusion_weights[0].cpu().tolist(),
                }
                if controls:
                    k = int(route.sum())
                    shape = tuple(route.shape)
                    if shape not in occupancy_cache:
                        print(f"Building {control_split}-only occupancy prior at {shape} from {len(control_images)} images")
                        occupancy_cache[shape] = training_occupancy_prior(control_images, shape, imgsz, len(model.names))
                        center_cache[shape] = center_prior(shape)
                    random_score = rng.random(shape)
                    control_masks = {
                        "actual": route,
                        "uniform_random": topk_mask(random_score, k),
                        "center_prior": topk_mask(center_cache[shape], k),
                        "train_occupancy_prior": topk_mask(occupancy_cache[shape], k),
                        "activation_energy": topk_mask(captured[module_index]["activation_energy"], k),
                        "fixed_haar": topk_mask(captured[module_index]["fixed_haar"], k),
                    }
                    row["controls"] = {
                        name: mask_metrics(mask, ground_truth, centers)
                        for name, mask in control_masks.items()
                    }
                    route_masks[module_index].append(route)
                    ground_truth_masks[module_index].append(ground_truth)
                rows.append(row)
    finally:
        for handle in handles:
            handle.remove()

    payload = {
        "schema_version": 2 if controls else 1,
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
    if controls:
        payload.update(
            {
                "control_split": control_split,
                "control_images": len(control_images),
                "control_seed": seed,
                "control_summary": summarize_controls(rows),
                "paired_control_tests": paired_control_tests(rows),
                "shuffled_route_control": shuffled_route_summary(
                    route_masks, ground_truth_masks, shuffle_repeats, seed
                ),
            }
        )
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
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--control-split", choices=("train", "val"), default="train")
    parser.add_argument("--control-limit", type=int, default=0)
    parser.add_argument("--shuffle-repeats", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    diagnose_routes(
        args.weights,
        args.data,
        args.output,
        args.split,
        args.imgsz,
        args.device,
        args.limit,
        args.controls,
        args.control_split,
        args.control_limit,
        args.shuffle_repeats,
        args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
