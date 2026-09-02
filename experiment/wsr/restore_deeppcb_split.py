from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

try:
    from .common import atomic_json_dump, sha256_file
except ImportError:
    from common import atomic_json_dump, sha256_file


KEY = re.compile(r"(\d{8})(?:_(?:test|temp))?\.(?:jpg|jpeg|png)$", re.IGNORECASE)
NAMES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]


def sample_key(value: str | Path) -> str:
    match = KEY.search(Path(value).name)
    if match is None:
        raise ValueError(f"Cannot recover sample key from {value}")
    return match.group(1)


def coco_keys(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {sample_key(row["file_name"]) for row in payload["images"]}


def parse_official_list(path: Path) -> list[str]:
    return [
        sample_key(line.split()[0])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def yolo_rows(annotation: Path, width: int = 640, height: int = 640) -> list[str]:
    rows = []
    for line in annotation.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        x1, y1, x2, y2, category = line.split()[:5]
        left, right = sorted((float(x1), float(x2)))
        top, bottom = sorted((float(y1), float(y2)))
        box_width, box_height = right - left, bottom - top
        if box_width <= 0.0 or box_height <= 0.0:
            continue
        rows.append(
            f"{int(category) - 1} "
            f"{(left + right) / 2.0 / width:.8f} "
            f"{(top + bottom) / 2.0 / height:.8f} "
            f"{box_width / width:.8f} {box_height / height:.8f}"
        )
    return rows


def _link(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def restore_split(
    raw_root: Path,
    validation_annotations: Path,
    test_annotations: Path,
    output: Path,
) -> Path:
    validation = coco_keys(validation_annotations)
    test = coco_keys(test_annotations)
    official_trainval = parse_official_list(raw_root / "trainval.txt")
    official_test = parse_official_list(raw_root / "test.txt")
    train = set(official_trainval) - validation
    if len(train) != 850 or len(validation) != 150 or len(test) != 500:
        raise ValueError(
            f"Unexpected split sizes: train={len(train)}, val={len(validation)}, test={len(test)}"
        )
    if set(official_test) != test:
        raise ValueError("The supplied test annotations do not match official DeepPCB test.txt")
    targets = {sample_key(path): path.resolve() for path in raw_root.rglob("*_test.jpg")}
    annotations = {
        path.stem: path.resolve()
        for path in raw_root.rglob("*_not/*.txt")
    }
    records: dict[str, list[dict[str, Any]]] = {}
    for split, keys in (("train", train), ("val", validation), ("test", test)):
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        records[split] = []
        for index, key in enumerate(sorted(keys), start=1):
            source_image = targets[key]
            source_label = annotations[key]
            # The one-prefix name is intentional: negative_aware._target_key
            # recovers the final eight-digit template key after the first '_'.
            stem = f"paper{index:05d}_{key}"
            image = image_dir / f"{stem}.jpg"
            label = label_dir / f"{stem}.txt"
            _link(source_image, image)
            if not label.exists():
                label.write_text("\n".join(yolo_rows(source_label)) + "\n", encoding="utf-8")
            records[split].append(
                {
                    "key": key,
                    "image": str(image.resolve()),
                    "image_sha256": sha256_file(image),
                    "source_annotation": str(source_label),
                    "source_annotation_sha256": sha256_file(source_label),
                }
            )
    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(NAMES)},
    }
    data_path = output / "dataset.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    templates = sorted(
        path.resolve()
        for path in raw_root.rglob("*_temp.jpg")
        if sample_key(path) in targets
    )
    if len(templates) != 1500:
        raise ValueError(f"Expected 1,500 target-matched templates, found {len(templates)}")
    template_list = output / "templates.txt"
    template_list.write_text("\n".join(map(str, templates)) + "\n", encoding="utf-8")
    atomic_json_dump(
        {
            "schema_version": 1,
            "protocol": "reconstruction of the frozen 850/150/500 paper split",
            "validation_annotations": str(validation_annotations.resolve()),
            "validation_annotations_sha256": sha256_file(validation_annotations),
            "test_annotations": str(test_annotations.resolve()),
            "test_annotations_sha256": sha256_file(test_annotations),
            "splits": records,
            "template_list": str(template_list.resolve()),
            "template_list_sha256": sha256_file(template_list),
        },
        output / "manifest.json",
    )
    return data_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore the exact DeepPCB paper split")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--validation-annotations", type=Path, required=True)
    parser.add_argument("--test-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = restore_split(
        args.raw_root.resolve(),
        args.validation_annotations.resolve(),
        args.test_annotations.resolve(),
        args.output.resolve(),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
