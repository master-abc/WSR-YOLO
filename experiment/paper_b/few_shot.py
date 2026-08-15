from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml

try:
    from .common import atomic_json_dump, class_names, dataset_sources, write_lines
    from .split_dataset import multilabel_stratified_split
except ImportError:
    from common import atomic_json_dump, class_names, dataset_sources, write_lines
    from split_dataset import multilabel_stratified_split


def materialize(data_yaml: Path, output_root: Path, fractions: list[float], seed: int) -> None:
    data, _, sources = dataset_sources(data_yaml)
    names = class_names(data)
    train_images = sources["train"]
    manifest = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError(f"Fraction must be in (0,1], got {fraction}")
        target = min(len(train_images), max(len(names), math.ceil(len(train_images) * fraction)))
        if target == len(train_images):
            selected = train_images
        else:
            selected, _ = multilabel_stratified_split(
                train_images, [target / len(train_images), 1 - target / len(train_images)], seed, len(names)
            )
        key = f"{fraction * 100:g}pct".replace(".", "p")
        level = output_root / key
        write_lines((path.as_posix() for path in selected), level / "train.txt")
        write_lines((path.as_posix() for path in sources["val"]), level / "val.txt")
        write_lines((path.as_posix() for path in sources["test"]), level / "test.txt")
        payload = {
            "path": level.resolve().as_posix(),
            "train": "train.txt",
            "val": "val.txt",
            "test": "test.txt",
            "names": {index: name for index, name in enumerate(names)},
            "metadata": {
                "requested_fraction": fraction,
                "actual_fraction": len(selected) / len(train_images),
                "images": len(selected),
                "subset_seed": seed,
                "test_unchanged": True,
            },
        }
        (level / "dataset.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        manifest.append(payload["metadata"] | {"dataset_yaml": str(level / "dataset.yaml")})
    atomic_json_dump(manifest, output_root / "manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic data-efficiency subsets")
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.25, 0.50, 1.0])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    materialize(args.data.resolve(), args.output.resolve(), args.fractions, args.seed)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

