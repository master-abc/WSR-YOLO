from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    from .common import (
        PAPER_B_DIR,
        atomic_json_dump,
        class_names,
        dataset_sources,
        image_to_label_path,
        stable_int,
        write_lines,
    )
except ImportError:
    from common import (
        PAPER_B_DIR,
        atomic_json_dump,
        class_names,
        dataset_sources,
        image_to_label_path,
        stable_int,
        write_lines,
    )


CORRUPTIONS = (
    "gaussian",
    "poisson",
    "speckle",
    "motion_blur",
    "defocus_blur",
    "brightness",
    "contrast",
    "jpeg",
)


def corrupt(image: np.ndarray, kind: str, severity: int, rng: np.random.Generator) -> np.ndarray:
    if severity not in range(1, 6):
        raise ValueError("Severity must be between 1 and 5")
    value = image.astype(np.float32)
    if kind == "gaussian":
        value += rng.normal(0.0, [5, 10, 15, 20, 25][severity - 1], value.shape)
    elif kind == "poisson":
        peak = [60, 30, 15, 8, 4][severity - 1]
        value = rng.poisson(np.clip(value, 0, 255) / 255.0 * peak) / peak * 255.0
    elif kind == "speckle":
        sigma = [0.04, 0.08, 0.12, 0.18, 0.25][severity - 1]
        value += value * rng.normal(0.0, sigma, value.shape)
    elif kind == "motion_blur":
        size = [3, 5, 7, 9, 13][severity - 1]
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, :] = 1.0 / size
        value = cv2.filter2D(value, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    elif kind == "defocus_blur":
        sigma = [0.8, 1.4, 2.0, 3.0, 4.5][severity - 1]
        value = cv2.GaussianBlur(value, (0, 0), sigmaX=sigma, sigmaY=sigma)
    elif kind == "brightness":
        value *= [0.85, 0.70, 0.55, 0.40, 0.25][severity - 1]
    elif kind == "contrast":
        mean = value.mean(axis=(0, 1), keepdims=True)
        value = (value - mean) * [0.85, 0.70, 0.55, 0.40, 0.25][severity - 1] + mean
    elif kind == "jpeg":
        quality = [70, 50, 35, 20, 10][severity - 1]
        ok, encoded = cv2.imencode(".jpg", np.clip(value, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("OpenCV JPEG encoding failed")
        value = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    else:
        raise ValueError(f"Unknown corruption: {kind}")
    return np.clip(value, 0, 255).astype(np.uint8)


def materialize(data_yaml: Path, output_root: Path, kinds: list[str], seed: int) -> None:
    data, _, sources = dataset_sources(data_yaml)
    names = class_names(data)
    if "test" not in sources:
        raise ValueError("An independent test split is required")
    manifest = []
    for kind in kinds:
        if kind not in CORRUPTIONS:
            raise ValueError(f"Unsupported corruption {kind}; choose from {CORRUPTIONS}")
        for severity in range(1, 6):
            level_root = output_root / kind / f"severity_{severity}"
            image_dir = level_root / "images" / "test"
            label_dir = level_root / "labels" / "test"
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            generated = []
            for index, source in enumerate(sources["test"], 1):
                target = image_dir / f"{index:08d}_{source.name}"
                label_target = label_dir / target.with_suffix(".txt").name
                per_image_seed = seed ^ stable_int(f"{kind}:{severity}:{source.as_posix()}")
                rng = np.random.default_rng(per_image_seed)
                image = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Cannot read image: {source}")
                transformed = corrupt(image, kind, severity, rng)
                if not cv2.imwrite(str(target), transformed):
                    raise RuntimeError(f"Cannot write image: {target}")
                label_source = image_to_label_path(source)
                if label_source.exists():
                    shutil.copy2(label_source, label_target)
                else:
                    label_target.write_text("", encoding="utf-8")
                generated.append(target.resolve())
                manifest.append(
                    {
                        "source": source.as_posix(),
                        "target": target.as_posix(),
                        "corruption": kind,
                        "severity": severity,
                        "seed": int(per_image_seed),
                    }
                )
            write_lines((path.as_posix() for path in generated), level_root / "test.txt")
            dataset_payload = {
                "path": level_root.resolve().as_posix(),
                "train": str((data_yaml.parent / "train.txt").resolve().as_posix()),
                "val": str((data_yaml.parent / "val.txt").resolve().as_posix()),
                "test": "test.txt",
                "names": {index: name for index, name in enumerate(names)},
                "metadata": {"corruption": kind, "severity": severity, "seed": seed},
            }
            (level_root / "dataset.yaml").write_text(
                yaml.safe_dump(dataset_payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
    atomic_json_dump(manifest, output_root / "manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize deterministic PCB-C corruptions")
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--corruptions", nargs="+", choices=CORRUPTIONS, default=list(CORRUPTIONS))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    materialize(args.data.resolve(), args.output.resolve(), args.corruptions, args.seed)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

