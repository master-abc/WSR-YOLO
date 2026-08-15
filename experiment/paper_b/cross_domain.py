from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import yaml

try:
    from .audit_dataset import parse_label
    from .common import atomic_json_dump, class_names, dataset_sources, image_to_label_path, write_lines
except ImportError:
    from audit_dataset import parse_label
    from common import atomic_json_dump, class_names, dataset_sources, image_to_label_path, write_lines


CANONICAL = ["open", "short", "mouse_bite", "spur", "spurious_copper"]
ALIASES = {
    "open": {"open", "opencircuit"},
    "short": {"short", "shortcircuit"},
    "mouse_bite": {"mousebite"},
    "spur": {"spur"},
    "spurious_copper": {"copper", "spuriouscopper"},
}


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def remap(data_yaml: Path, output: Path) -> Path:
    data, _, sources = dataset_sources(data_yaml)
    old_names = class_names(data)
    mapping = {}
    for old_id, name in enumerate(old_names):
        normalized = normalize(name)
        for new_id, canonical in enumerate(CANONICAL):
            if normalized in ALIASES[canonical]:
                mapping[old_id] = new_id
                break
    if len(mapping) < 5:
        raise ValueError(f"Dataset does not contain all five shared classes: {old_names}, mapping={mapping}")
    split_paths = {}
    kept_counts = {name: 0 for name in CANONICAL}
    split_statistics = {}
    for split, images in sources.items():
        targets = []
        skipped_non_shared = 0
        for index, source in enumerate(images, 1):
            boxes, errors = parse_label(image_to_label_path(source), len(old_names))
            if errors:
                raise ValueError(f"Invalid labels for {source}: {errors}")
            # Never create an unlabelled-object benchmark: an image containing
            # any non-shared defect is excluded rather than treating that defect
            # as background and counting a reasonable prediction as a false alarm.
            if not boxes or any(old_id not in mapping for old_id, *_ in boxes):
                skipped_non_shared += 1
                continue
            target = output / "images" / split / f"{index:08d}_{source.name}"
            hardlink(source, target)
            lines = []
            for old_id, x, y, width, height in boxes:
                new_id = mapping[old_id]
                lines.append(f"{new_id} {x:.8f} {y:.8f} {width:.8f} {height:.8f}")
                kept_counts[CANONICAL[new_id]] += 1
            label = output / "labels" / split / target.with_suffix(".txt").name
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            targets.append(target.resolve())
        write_lines((path.as_posix() for path in targets), output / f"{split}.txt")
        split_paths[split] = f"{split}.txt"
        split_statistics[split] = {
            "kept_images": len(targets),
            "skipped_images_with_non_shared_or_empty_labels": skipped_non_shared,
        }
    payload = {
        "path": output.resolve().as_posix(),
        **split_paths,
        "names": {index: name for index, name in enumerate(CANONICAL)},
        "metadata": {
            "source": str(data_yaml.resolve()),
            "class_mapping": mapping,
            "kept_boxes": kept_counts,
            "strict_shared_only": True,
            "split_statistics": split_statistics,
        },
    }
    result = output / "dataset.yaml"
    result.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return result


def make_pair(source_yaml: Path, target_yaml: Path, output: Path) -> Path:
    _, _, source = dataset_sources(source_yaml)
    _, _, target = dataset_sources(target_yaml)
    for split, images in (("train", source["train"]), ("val", source["val"]), ("test", target["test"])):
        write_lines((path.as_posix() for path in images), output / f"{split}.txt")
    payload = {
        "path": output.resolve().as_posix(),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {index: name for index, name in enumerate(CANONICAL)},
        "metadata": {
            "source_domain": str(source_yaml.resolve()),
            "target_domain": str(target_yaml.resolve()),
            "zero_shot_test": True,
        },
    }
    result = output / "dataset.yaml"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a common-five-class zero-shot transfer benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    remap_parser = subparsers.add_parser("remap")
    remap_parser.add_argument("data", type=Path)
    remap_parser.add_argument("output", type=Path)
    pair_parser = subparsers.add_parser("pair")
    pair_parser.add_argument("source", type=Path)
    pair_parser.add_argument("target", type=Path)
    pair_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "remap":
        result = remap(args.data.resolve(), args.output.resolve())
    else:
        result = make_pair(args.source.resolve(), args.target.resolve(), args.output.resolve())
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
