from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .common import PROJECT_DIR, atomic_json_dump, dataset_sources, environment_snapshot, sha256_file
except ImportError:
    from common import PROJECT_DIR, atomic_json_dump, dataset_sources, environment_snapshot, sha256_file


def timing_summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "fps_from_mean": 1000.0 / statistics.fmean(values),
    }


def synchronize(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_weights(
    weights: Path,
    data: Path,
    output: Path,
    device_name: str,
    imgsz: int = 640,
    warmup: int = 50,
    repetitions: int = 200,
    half: bool = False,
    split: str = "test",
    maximum_images: int = 50,
) -> dict:
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from algorithm.register import register_custom_modules

    register_custom_modules()
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    yolo = YOLO(str(weights.resolve()))
    torch_device = torch.device(
        f"cuda:{device_name}"
        if torch.cuda.is_available() and device_name != "cpu"
        else "cpu"
    )
    network = yolo.model.to(torch_device).eval()
    if half and torch_device.type == "cuda":
        network.half()
    dtype = torch.float16 if half and torch_device.type == "cuda" else torch.float32
    torch.manual_seed(2026)
    sample = torch.randn(1, 3, imgsz, imgsz, device=torch_device, dtype=dtype)
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
    with torch.inference_mode():
        for _ in range(warmup):
            network(sample)
        synchronize(torch)
        model_times = []
        for _ in range(repetitions):
            start = time.perf_counter()
            network(sample)
            synchronize(torch)
            model_times.append((time.perf_counter() - start) * 1000.0)

    _, _, sources = dataset_sources(data.resolve())
    if split not in sources:
        raise KeyError(f"Dataset has no '{split}' split: {data.resolve()}")
    selected_paths = sources[split][: min(maximum_images, len(sources[split]))]
    images = []
    for path in selected_paths:
        with Image.open(path) as opened:
            images.append(np.asarray(opened.convert("RGB")).copy())
    if not images:
        raise ValueError(f"No images available for latency benchmark split '{split}'")
    for image in images[: min(warmup, len(images))]:
        yolo.predict(image, imgsz=imgsz, device=device_name, half=half, verbose=False)
    end_to_end = []
    for index in range(repetitions):
        image = images[index % len(images)]
        start = time.perf_counter()
        yolo.predict(image, imgsz=imgsz, device=device_name, half=half, verbose=False)
        synchronize(torch)
        end_to_end.append((time.perf_counter() - start) * 1000.0)

    peak_memory = None
    if torch_device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(torch_device))
    payload = {
        "schema_version": 2,
        "weights": str(weights.resolve()),
        "weights_sha256": sha256_file(weights),
        "data_yaml": str(data.resolve()),
        "data_yaml_sha256": sha256_file(data),
        "selection_split": split,
        "device": str(torch_device),
        "precision": "fp16" if dtype == torch.float16 else "fp32",
        "imgsz": imgsz,
        "batch": 1,
        "parameters": sum(parameter.numel() for parameter in network.parameters()),
        "gflops": float(get_flops(network, imgsz=imgsz)),
        "peak_memory_bytes": peak_memory,
        "model_only": timing_summary(model_times),
        "preprocess_forward_nms": timing_summary(end_to_end),
        "environment": environment_snapshot(),
    }
    atomic_json_dump(payload, output.resolve())
    print(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure model-only and preprocess+forward+NMS latency")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--maximum-images", type=int, default=50)
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()
    benchmark_weights(
        args.weights,
        args.data,
        args.output,
        args.device,
        args.imgsz,
        args.warmup,
        args.repetitions,
        args.half,
        args.split,
        args.maximum_images,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
