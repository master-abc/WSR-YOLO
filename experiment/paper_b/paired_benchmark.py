from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

try:
    from .common import PROJECT_DIR, atomic_json_dump, dataset_sources, environment_snapshot, sha256_file
except ImportError:
    from common import PROJECT_DIR, atomic_json_dump, dataset_sources, environment_snapshot, sha256_file


def synchronize(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timing_summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "std_ms": statistics.stdev(values) if len(values) >= 2 else None,
        "p95_ms": ordered[min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)],
        "fps_from_mean": 1000.0 / statistics.fmean(values),
    }


def ratio_summary(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else None
    if std is None:
        ci95 = [mean, mean]
    else:
        try:
            from scipy.stats import t

            critical = float(t.ppf(0.975, len(values) - 1))
        except ImportError:
            critical = 1.96
        half_width = critical * std / np.sqrt(len(values))
        ci95 = [mean - half_width, mean + half_width]
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "median": statistics.median(values),
        "ci95": ci95,
    }


def run_abba(
    callbacks: dict[str, Callable[[int], None]],
    torch,
    cycles: int,
    repetitions_per_segment: int,
) -> dict:
    raw = {"baseline": [], "candidate": []}
    cycle_rows = []
    for cycle in range(cycles):
        order = (
            ("baseline", "candidate", "candidate", "baseline")
            if cycle % 2 == 0
            else ("candidate", "baseline", "baseline", "candidate")
        )
        cycle_values = {"baseline": [], "candidate": []}
        for name in order:
            callback = callbacks[name]
            for repetition in range(repetitions_per_segment):
                synchronize(torch)
                start = time.perf_counter()
                callback(repetition)
                synchronize(torch)
                elapsed = (time.perf_counter() - start) * 1000.0
                raw[name].append(elapsed)
                cycle_values[name].append(elapsed)
        baseline_mean = statistics.fmean(cycle_values["baseline"])
        candidate_mean = statistics.fmean(cycle_values["candidate"])
        cycle_rows.append(
            {
                "cycle": cycle,
                "order": list(order),
                "baseline_mean_ms": baseline_mean,
                "candidate_mean_ms": candidate_mean,
                "candidate_over_baseline": candidate_mean / baseline_mean,
            }
        )
    baseline_summary = timing_summary(raw["baseline"])
    candidate_summary = timing_summary(raw["candidate"])
    return {
        "design": "alternating ABBA/BAAB; two segments per model and cycle",
        "cycles": cycles,
        "repetitions_per_segment": repetitions_per_segment,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "candidate_over_baseline_ratio_of_means": candidate_summary["mean_ms"] / baseline_summary["mean_ms"],
        "paired_cycle_ratio": ratio_summary([row["candidate_over_baseline"] for row in cycle_rows]),
        "cycle_rows": cycle_rows,
        "raw_ms": raw,
    }


def paired_benchmark(
    baseline_weights: Path,
    candidate_weights: Path,
    data: Path,
    output: Path,
    device_name: str,
    imgsz: int = 640,
    warmup: int = 50,
    cycles: int = 10,
    repetitions_per_segment: int = 25,
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

    if not torch.cuda.is_available() and device_name != "cpu":
        raise RuntimeError("CUDA was requested but is unavailable")
    torch_device = torch.device(f"cuda:{device_name}" if device_name != "cpu" else "cpu")
    yolo = {
        "baseline": YOLO(str(baseline_weights.resolve())),
        "candidate": YOLO(str(candidate_weights.resolve())),
    }
    networks = {name: value.model.to(torch_device).eval() for name, value in yolo.items()}
    if half and torch_device.type == "cuda":
        for network in networks.values():
            network.half()
    dtype = torch.float16 if half and torch_device.type == "cuda" else torch.float32
    torch.manual_seed(2026)
    sample = torch.randn(1, 3, imgsz, imgsz, device=torch_device, dtype=dtype)

    with torch.inference_mode():
        for repetition in range(warmup):
            networks["baseline" if repetition % 2 == 0 else "candidate"](sample)
        for network in networks.values():
            for _ in range(max(1, warmup // 2)):
                network(sample)
        synchronize(torch)
        if torch_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch_device)
        model_only = run_abba(
            {name: (lambda _index, network=network: network(sample)) for name, network in networks.items()},
            torch,
            cycles,
            repetitions_per_segment,
        )
        model_only_peak_memory = (
            int(torch.cuda.max_memory_allocated(torch_device)) if torch_device.type == "cuda" else None
        )

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
    for repetition in range(warmup):
        name = "baseline" if repetition % 2 == 0 else "candidate"
        yolo[name].predict(
            images[repetition % len(images)], imgsz=imgsz, device=device_name, half=half, verbose=False
        )
    synchronize(torch)
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
    pipeline = run_abba(
        {
            name: (
                lambda index, detector=detector: detector.predict(
                    images[index % len(images)], imgsz=imgsz, device=device_name, half=half, verbose=False
                )
            )
            for name, detector in yolo.items()
        },
        torch,
        cycles,
        repetitions_per_segment,
    )
    pipeline_peak_memory = (
        int(torch.cuda.max_memory_allocated(torch_device)) if torch_device.type == "cuda" else None
    )

    payload = {
        "schema_version": 1,
        "design": "paired same-process, same-physical-GPU timing",
        "baseline_weights": str(baseline_weights.resolve()),
        "baseline_weights_sha256": sha256_file(baseline_weights),
        "candidate_weights": str(candidate_weights.resolve()),
        "candidate_weights_sha256": sha256_file(candidate_weights),
        "data_yaml": str(data.resolve()),
        "data_yaml_sha256": sha256_file(data),
        "selection_split": split,
        "device": str(torch_device),
        "precision": "fp16" if dtype == torch.float16 else "fp32",
        "imgsz": imgsz,
        "batch": 1,
        "warmup": warmup,
        "maximum_images": len(images),
        "models": {
            name: {
                "parameters": sum(parameter.numel() for parameter in network.parameters()),
                "gflops": float(get_flops(network, imgsz=imgsz)),
            }
            for name, network in networks.items()
        },
        "model_only": model_only,
        "preprocess_forward_nms": pipeline,
        "model_only_peak_memory_bytes": model_only_peak_memory,
        "pipeline_peak_memory_bytes": pipeline_peak_memory,
        "environment": environment_snapshot(),
    }
    atomic_json_dump(payload, output.resolve())
    print(
        f"model-only ratio={model_only['candidate_over_baseline_ratio_of_means']:.4f}; "
        f"pipeline ratio={pipeline['candidate_over_baseline_ratio_of_means']:.4f}"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired ABBA latency benchmark on one physical device")
    parser.add_argument("--baseline-weights", type=Path, required=True)
    parser.add_argument("--candidate-weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--repetitions-per-segment", type=int, default=25)
    parser.add_argument("--maximum-images", type=int, default=50)
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()
    paired_benchmark(
        args.baseline_weights,
        args.candidate_weights,
        args.data,
        args.output,
        args.device,
        args.imgsz,
        args.warmup,
        args.cycles,
        args.repetitions_per_segment,
        args.half,
        args.split,
        args.maximum_images,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
