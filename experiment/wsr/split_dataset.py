from __future__ import annotations

import argparse
import collections
import os
import random
import shutil
from pathlib import Path
from typing import Iterable

import yaml

try:
    from .audit_dataset import parse_label
    from .common import class_names, dataset_sources, image_to_label_path, write_lines
except ImportError:
    from audit_dataset import parse_label
    from common import class_names, dataset_sources, image_to_label_path, write_lines


def image_classes(image: Path, nc: int) -> set[int]:
    boxes, _ = parse_label(image_to_label_path(image), nc)
    return {box[0] for box in boxes}


def multilabel_stratified_split(
    images: Iterable[Path], fractions: list[float], seed: int, nc: int
) -> list[list[Path]]:
    """Deterministic greedy multilabel split without an extra dependency.

    Samples containing rare classes are placed first.  Each assignment minimizes
    normalized class and split-size deficits, so the result is substantially more
    stable than a random split for small PCB datasets.
    """

    images = list(dict.fromkeys(Path(path).resolve() for path in images))
    if not images:
        raise ValueError("No images were provided")
    if len(fractions) < 2 or any(value <= 0 for value in fractions):
        raise ValueError("At least two positive split fractions are required")
    total_fraction = sum(fractions)
    fractions = [value / total_fraction for value in fractions]

    labels = {image: image_classes(image, nc) for image in images}
    class_totals = collections.Counter(class_id for value in labels.values() for class_id in value)
    rng = random.Random(seed)
    jitter = {image: rng.random() for image in images}
    ordered = sorted(
        images,
        key=lambda image: (
            min((class_totals[class_id] for class_id in labels[image]), default=len(images) + 1),
            -len(labels[image]),
            jitter[image],
        ),
    )

    target_sizes = [len(images) * value for value in fractions]
    target_classes = [
        {class_id: class_totals[class_id] * value for class_id in range(nc)}
        for value in fractions
    ]
    splits: list[list[Path]] = [[] for _ in fractions]
    split_classes = [collections.Counter() for _ in fractions]

    for image in ordered:
        candidates: list[tuple[float, float, int]] = []
        for split_index in range(len(fractions)):
            size_deficit = (target_sizes[split_index] - len(splits[split_index])) / max(
                target_sizes[split_index], 1.0
            )
            class_deficit = sum(
                (target_classes[split_index][class_id] - split_classes[split_index][class_id])
                / max(target_classes[split_index][class_id], 1.0)
                for class_id in labels[image]
            )
            overfill = max(0.0, (len(splits[split_index]) + 1 - target_sizes[split_index]))
            score = class_deficit + 0.35 * size_deficit - 4.0 * overfill
            candidates.append((score, -len(splits[split_index]), -split_index))
        selected = max(range(len(candidates)), key=candidates.__getitem__)
        splits[selected].append(image)
        split_classes[selected].update(labels[image])

    return [sorted(split) for split in splits]


def write_dataset(
    output_dir: Path,
    splits: dict[str, list[Path]],
    names: list[str],
    source_yaml: Path,
    split_seed: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized: dict[str, list[Path]] = {}
    for split, images in splits.items():
        materialized[split] = []
        for index, image in enumerate(images, 1):
            target = output_dir / "images" / split / f"{index:08d}_{image.name}"
            label_source = image_to_label_path(image)
            label_target = output_dir / "labels" / split / target.with_suffix(".txt").name
            target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    os.link(image, target)
                except OSError:
                    shutil.copy2(image, target)
            if label_source.exists() and not label_target.exists():
                try:
                    os.link(label_source, label_target)
                except OSError:
                    shutil.copy2(label_source, label_target)
            elif not label_source.exists() and not label_target.exists():
                label_target.write_text("", encoding="utf-8")
            materialized[split].append(target.resolve())
        write_lines((path.as_posix() for path in materialized[split]), output_dir / f"{split}.txt")
    payload = {
        "path": output_dir.resolve().as_posix(),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {index: name for index, name in enumerate(names)},
        "metadata": {
            "source_yaml": source_yaml.resolve().as_posix(),
            "split_seed": split_seed,
            "counts": {key: len(value) for key, value in materialized.items()},
            "storage": "hardlink_with_copy_fallback",
        },
    }
    output_yaml = output_dir / "dataset.yaml"
    output_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output_yaml


def prepare_split(
    dataset_yaml: Path,
    output_dir: Path,
    mode: str,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> Path:
    data, _, sources = dataset_sources(dataset_yaml)
    names = class_names(data)
    if mode == "resplit-all":
        pool = sources.get("train", []) + sources.get("val", []) + sources.get("test", [])
        pool = list(dict.fromkeys(pool))
        test_fraction = 1.0 - train_fraction - val_fraction
        if test_fraction <= 0:
            raise ValueError("train_fraction + val_fraction must be below one")
        train, val, test = multilabel_stratified_split(
            pool, [train_fraction, val_fraction, test_fraction], seed, len(names)
        )
    elif mode == "official-val-as-test":
        if not sources.get("train") or not sources.get("val"):
            raise ValueError("This mode requires official train and validation splits")
        train, val = multilabel_stratified_split(
            sources["train"], [train_fraction, 1.0 - train_fraction], seed, len(names)
        )
        test = sources["val"]
    elif mode == "preserve":
        missing = [split for split in ("train", "val", "test") if not sources.get(split)]
        if missing:
            raise ValueError(f"Preserve mode requires all splits; missing {missing}")
        train, val, test = sources["train"], sources["val"], sources["test"]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return write_dataset(
        output_dir,
        {"train": train, "val": val, "test": test},
        names,
        dataset_yaml,
        seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create one fixed, stratified PCB dataset split")
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--mode", choices=("resplit-all", "official-val-as-test", "preserve"), default="resplit-all"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    output = prepare_split(
        args.data.resolve(), args.output.resolve(), args.mode, args.seed, args.train_fraction, args.val_fraction
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
