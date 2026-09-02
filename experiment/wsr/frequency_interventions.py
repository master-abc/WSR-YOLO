from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

try:
    from .common import atomic_json_dump, class_names, dataset_sources, image_to_label_path, write_lines
except ImportError:
    from common import atomic_json_dump, class_names, dataset_sources, image_to_label_path, write_lines


INTERVENTIONS = ("low_only", "high_only", "remove_lh", "remove_hl", "remove_hh")


def haar_filters(channels: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    scale = 0.5
    filters = torch.tensor(
        [
            [[1, 1], [1, 1]],
            [[-1, -1], [1, 1]],
            [[-1, 1], [-1, 1]],
            [[1, -1], [-1, 1]],
        ],
        dtype=dtype,
        device=device,
    ) * scale
    return filters.unsqueeze(1).repeat(channels, 1, 1, 1)


def intervene(image: np.ndarray, mode: str) -> np.ndarray:
    if mode not in INTERVENTIONS:
        raise ValueError(mode)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    _, channels, height, width = tensor.shape
    pad_h, pad_w = height % 2, width % 2
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    weight = haar_filters(channels, tensor.dtype, tensor.device)
    bands = F.conv2d(tensor, weight, stride=2, groups=channels).view(
        1, channels, 4, tensor.shape[2] // 2, tensor.shape[3] // 2
    )
    if mode == "low_only":
        bands[:, :, 1:] = 0
    elif mode == "high_only":
        bands[:, :, 0] = 0
    else:
        band_index = {"remove_lh": 1, "remove_hl": 2, "remove_hh": 3}[mode]
        bands[:, :, band_index] = 0
    packed = bands.reshape(1, channels * 4, bands.shape[-2], bands.shape[-1])
    reconstructed = F.conv_transpose2d(packed, weight, stride=2, groups=channels)
    reconstructed = reconstructed[:, :, :height, :width].clamp(0, 1)
    return (reconstructed.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def materialize(data_yaml: Path, output_root: Path, modes: list[str]) -> None:
    data, _, sources = dataset_sources(data_yaml)
    names = class_names(data)
    manifest = []
    for mode in modes:
        level_root = output_root / mode
        image_dir = level_root / "images" / "test"
        label_dir = level_root / "labels" / "test"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        generated = []
        for index, source in enumerate(sources["test"], 1):
            target = image_dir / f"{index:08d}_{source.with_suffix('.png').name}"
            image = np.asarray(Image.open(source).convert("RGB"))
            Image.fromarray(intervene(image, mode)).save(target)
            label_source = image_to_label_path(source)
            label_target = label_dir / target.with_suffix(".txt").name
            if label_source.exists():
                shutil.copy2(label_source, label_target)
            else:
                label_target.write_text("", encoding="utf-8")
            generated.append(target.resolve())
            manifest.append({"source": source.as_posix(), "target": target.as_posix(), "mode": mode})
        write_lines((path.as_posix() for path in generated), level_root / "test.txt")
        dataset_payload = {
            "path": level_root.resolve().as_posix(),
            "train": str((data_yaml.parent / "train.txt").resolve().as_posix()),
            "val": str((data_yaml.parent / "val.txt").resolve().as_posix()),
            "test": "test.txt",
            "names": {index: name for index, name in enumerate(names)},
            "metadata": {"frequency_intervention": mode, "haar_level": 1},
        }
        (level_root / "dataset.yaml").write_text(
            yaml.safe_dump(dataset_payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
    atomic_json_dump(manifest, output_root / "manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Haar-band counterfactual test sets")
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--modes", nargs="+", choices=INTERVENTIONS, default=list(INTERVENTIONS))
    args = parser.parse_args()
    materialize(args.data.resolve(), args.output.resolve(), args.modes)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

