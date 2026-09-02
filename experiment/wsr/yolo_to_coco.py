from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from PIL import Image

try:
    from .audit_dataset import parse_label
    from .common import atomic_json_dump, class_names, dataset_sources, image_to_label_path
except ImportError:
    from audit_dataset import parse_label
    from common import atomic_json_dump, class_names, dataset_sources, image_to_label_path


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def convert_split(images: list[Path], names: list[str], image_dir: Path, output_json: Path) -> None:
    coco = {
        "info": {"description": "Exact YOLO-to-COCO conversion for the paper protocol"},
        "licenses": [],
        "images": [],
        "annotations": [],
        # Zero-based category IDs are intentional: DETR configs use remap_mscoco_category=false.
        "categories": [{"id": index, "name": name} for index, name in enumerate(names)],
    }
    manifest = []
    annotation_id = 1
    for image_id, source in enumerate(images, 1):
        target_name = f"{image_id:08d}_{source.name}"
        target = image_dir / target_name
        mode = link_or_copy(source, target)
        with Image.open(source) as opened:
            width, height = opened.size
        coco["images"].append(
            {"id": image_id, "file_name": target_name, "width": width, "height": height}
        )
        boxes, errors = parse_label(image_to_label_path(source), len(names))
        if errors:
            raise ValueError(f"Invalid label for {source}: {errors}")
        for class_id, center_x, center_y, box_width, box_height in boxes:
            x = (center_x - box_width / 2.0) * width
            y = (center_y - box_height / 2.0) * height
            w = box_width * width
            h = box_height * height
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
        manifest.append({"source": source.as_posix(), "target": target.as_posix(), "mode": mode})
    atomic_json_dump(coco, output_json)
    atomic_json_dump(manifest, output_json.with_name(f"{output_json.stem}_manifest.json"))


def convert_dataset(dataset_yaml: Path, output_dir: Path) -> None:
    data, _, sources = dataset_sources(dataset_yaml)
    names = class_names(data)
    for split in ("train", "val", "test"):
        if split not in sources:
            continue
        convert_split(
            sources[split], names, output_dir / split, output_dir / "annotations" / f"instances_{split}.json"
        )
    metadata = {
        "source_yaml": dataset_yaml.resolve().as_posix(),
        "num_classes": len(names),
        "names": names,
        "category_ids": "zero_based",
    }
    atomic_json_dump(metadata, output_dir / "conversion.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert exact YOLO splits to COCO for DETR baselines")
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert_dataset(args.data.resolve(), args.output.resolve())
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

