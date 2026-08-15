from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import yaml

try:
    from .common import PROJECT_DIR, atomic_json_dump
except ImportError:
    from common import PROJECT_DIR, atomic_json_dump


FIGSHARE_URL = "https://ndownloader.figshare.com/files/44069552"
ARCHIVE_MD5 = "508334b65bdaea7336f4c1b5d5a80a81"
CATEGORY_NAMES = {
    1: "short",
    2: "spur",
    3: "spurious_copper",
    4: "open",
    5: "mouse_bite",
    6: "hole_breakout",
    7: "conductor_scratch",
    8: "conductor_foreign_object",
    9: "base_material_foreign_object",
}


def md5(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324 - dataset integrity, not security
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
        while True:
            chunk = response.read(4 * 1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)
    temporary.replace(target)


def safe_extract(archive: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    root = output.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (output / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(output)


def hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        import shutil

        shutil.copy2(source, target)


def convert_split(coco_json: Path, image_dir: Path, output_root: Path, split: str) -> dict:
    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    declared = {int(item["id"]): item["name"] for item in coco["categories"]}
    expected = {1: "SH", 2: "SP", 3: "SC", 4: "OP", 5: "MB", 6: "HB", 7: "CS", 8: "CFO", 9: "BMFO"}
    if declared != expected:
        raise ValueError(f"Unexpected category mapping: {declared}")
    annotations = collections.defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)
    class_counts = collections.Counter()
    invalid = []
    clipped_rounding_boxes = 0
    for image in coco["images"]:
        image_id = int(image["id"])
        width, height = int(image["width"]), int(image["height"])
        source = image_dir / image["file_name"]
        if not source.exists():
            raise FileNotFoundError(source)
        hardlink(source, output_root / "images" / split / image["file_name"])
        lines = []
        for annotation in annotations[image_id]:
            category_id = int(annotation["category_id"])
            x, y, box_width, box_height = map(float, annotation["bbox"])
            overflow = max(0.0, -x, -y, x + box_width - width, y + box_height - height)
            if (
                category_id not in CATEGORY_NAMES
                or box_width <= 0
                or box_height <= 0
                or overflow > 0.5
            ):
                invalid.append({"image_id": image_id, "annotation": annotation})
                continue
            if overflow > 1e-6:
                clipped_rounding_boxes += 1
            x1 = min(float(width), max(0.0, x))
            y1 = min(float(height), max(0.0, y))
            x2 = min(float(width), max(0.0, x + box_width))
            y2 = min(float(height), max(0.0, y + box_height))
            box_width, box_height = x2 - x1, y2 - y1
            x, y = x1, y1
            class_id = category_id - 1
            lines.append(
                f"{class_id} {(x + box_width / 2) / width:.8f} "
                f"{(y + box_height / 2) / height:.8f} {box_width / width:.8f} {box_height / height:.8f}"
            )
            class_counts[class_id] += 1
        label = output_root / "labels" / split / Path(image["file_name"]).with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if invalid:
        raise ValueError(f"Found {len(invalid)} invalid COCO boxes; first={invalid[0]}")
    return {
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "clipped_rounding_boxes": clipped_rounding_boxes,
        "class_counts": {CATEGORY_NAMES[index + 1]: class_counts[index] for index in range(9)},
    }


def prepare(root: Path, keep_archive: bool = True) -> Path:
    archive = root / "DsPCBSD+.zip"
    official = root / "official"
    coco_root = official / "Data_COCO"
    output = root / "yolo_source"
    if not archive.exists():
        print(f"Downloading {FIGSHARE_URL} -> {archive}")
        download(FIGSHARE_URL, archive)
    actual_md5 = md5(archive)
    if actual_md5 != ARCHIVE_MD5:
        raise ValueError(f"Archive MD5 mismatch: expected {ARCHIVE_MD5}, got {actual_md5}")
    if not (coco_root / "annotations" / "instances_train2017.json").exists():
        print(f"Extracting {archive} -> {official}")
        safe_extract(archive, official)

    summaries = {}
    for source_split, target_split in (("train", "train"), ("val", "val")):
        summaries[target_split] = convert_split(
            coco_root / "annotations" / f"instances_{source_split}2017.json",
            coco_root / f"{source_split}2017",
            output,
            target_split,
        )
    dataset = {
        "path": output.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(CATEGORY_NAMES.values())},
        "metadata": {
            "source": "DsPCBSD+ official Figshare article 24970329, file 44069552",
            "source_format": "COCO",
            "official_val_usage": "reserved as final test by paper_b split protocol",
            "archive_md5": actual_md5,
        },
    }
    dataset_yaml = output / "dataset.yaml"
    dataset_yaml.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    atomic_json_dump(
        {
            "figshare_url": FIGSHARE_URL,
            "archive_md5": actual_md5,
            "category_mapping": CATEGORY_NAMES,
            "splits": summaries,
        },
        output / "conversion.json",
    )
    if not keep_archive:
        archive.unlink()
    return dataset_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and losslessly convert official DsPCBSD+ COCO data")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "datasets" / "DsPCBSD_plus")
    parser.add_argument("--remove-archive", action="store_true")
    args = parser.parse_args()
    output = prepare(args.root.resolve(), keep_archive=not args.remove_archive)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
