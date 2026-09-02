from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from .common import (
        atomic_json_dump,
        class_names,
        dataset_sources,
        image_to_label_path,
        sha256_file,
    )
except ImportError:
    from common import atomic_json_dump, class_names, dataset_sources, image_to_label_path, sha256_file


def parse_label(path: Path, nc: int) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    errors: list[str] = []
    if not path.exists():
        return boxes, ["missing_label"]
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 5:
            errors.append(f"line_{line_number}:expected_5_fields")
            continue
        try:
            class_value = float(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"line_{line_number}:non_numeric")
            continue
        class_id = int(class_value)
        if class_value != class_id or not 0 <= class_id < nc:
            errors.append(f"line_{line_number}:invalid_class")
        x, y, width, height = values
        if not all(math.isfinite(value) for value in values):
            errors.append(f"line_{line_number}:non_finite")
        if not (0 < width <= 1 and 0 < height <= 1 and 0 <= x <= 1 and 0 <= y <= 1):
            errors.append(f"line_{line_number}:invalid_normalized_box")
        if x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6:
            errors.append(f"line_{line_number}:x_outside_image")
        if y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6:
            errors.append(f"line_{line_number}:y_outside_image")
        boxes.append((class_id, x, y, width, height))
    return boxes, errors


def perceptual_signatures(path: Path, thumbnail_size: int = 32) -> tuple[int, bytes]:
    """Decode once and return dHash plus a compact grayscale thumbnail."""

    with Image.open(path) as image:
        grayscale = image.convert("L")
        pixels = list(
            grayscale.resize((9, 8), Image.Resampling.LANCZOS).getdata()
        )
        thumbnail = grayscale.resize(
            (thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS
        ).tobytes()
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(
                pixels[row * 9 + column] > pixels[row * 9 + column + 1]
            )
    return value, thumbnail


def difference_hash(path: Path) -> int:
    """Return a 64-bit perceptual hash for cross-split near-duplicate screening."""

    return perceptual_signatures(path)[0]


def grayscale_thumbnail(path: Path, size: int = 32) -> bytes:
    """Return a compact intensity signature used to reject dHash collisions."""

    return perceptual_signatures(path, size)[1]


def near_duplicate_pairs(
    left: list[tuple[int, str, bytes]],
    right: list[tuple[int, str, bytes]],
    threshold: int = 4,
    maximum_thumbnail_mae: float = 0.05,
    max_examples: int = 100,
) -> tuple[int, list[dict[str, Any]]]:
    # Eight 8-bit bands keep candidate generation sub-quadratic. With a
    # Hamming threshold of four, at least one band must be identical.
    buckets: dict[tuple[int, int], list[tuple[int, str, bytes]]] = collections.defaultdict(list)
    for value, path, thumbnail in right:
        for band in range(8):
            buckets[(band, (value >> (band * 8)) & 0xFF)].append(
                (value, path, thumbnail)
            )
    count = 0
    examples: list[dict[str, Any]] = []
    for left_value, left_path, left_thumbnail in left:
        candidates: dict[str, tuple[int, bytes]] = {}
        for band in range(8):
            for right_value, right_path, right_thumbnail in buckets.get(
                (band, (left_value >> (band * 8)) & 0xFF), []
            ):
                candidates[right_path] = (right_value, right_thumbnail)
        for right_path, (right_value, right_thumbnail) in candidates.items():
            distance = (left_value ^ right_value).bit_count()
            if distance > threshold:
                continue
            thumbnail_mae = sum(
                abs(left_pixel - right_pixel)
                for left_pixel, right_pixel in zip(left_thumbnail, right_thumbnail)
            ) / (255.0 * len(left_thumbnail))
            if thumbnail_mae > maximum_thumbnail_mae:
                continue
            count += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "left": left_path,
                        "right": right_path,
                        "hamming_distance": distance,
                        "thumbnail_mae": thumbnail_mae,
                    }
                )
    return count, examples


def audit_dataset(dataset_yaml: str | Path, hash_images: bool = True) -> dict[str, Any]:
    data, _, sources = dataset_sources(dataset_yaml)
    names = class_names(data)
    if not names:
        raise ValueError("Dataset YAML must define class names")

    report: dict[str, Any] = {
        "dataset_yaml": str(Path(dataset_yaml).resolve()),
        "class_names": names,
        "content_hashes_checked": bool(hash_images),
        "splits": {},
        "cross_split": {},
        "fatal": [],
        "warnings": [],
    }
    hashes: dict[str, dict[str, list[str]]] = {}
    files_by_size: dict[str, dict[int, list[Path]]] = {}
    perceptual_hashes: dict[str, list[tuple[int, str, bytes]]] = {}
    source_sets = {split: {str(path.resolve()) for path in images} for split, images in sources.items()}

    for split, images in sources.items():
        class_counts = collections.Counter()
        signatures = collections.Counter()
        invalid: dict[str, list[str]] = {}
        missing_images: list[str] = []
        empty_labels = 0
        areas: list[float] = []
        hashes[split] = collections.defaultdict(list)
        files_by_size[split] = collections.defaultdict(list)
        perceptual_hashes[split] = []

        for image in images:
            if not image.exists():
                missing_images.append(str(image))
                continue
            label = image_to_label_path(image)
            boxes, errors = parse_label(label, len(names))
            if errors:
                invalid[str(label)] = errors
            if not boxes:
                empty_labels += 1
            for class_id, x, y, width, height in boxes:
                class_counts[class_id] += 1
                signatures[(class_id, round(x, 6), round(y, 6), round(width, 6), round(height, 6))] += 1
                areas.append(width * height)
            if hash_images:
                files_by_size[split][image.stat().st_size].append(image)
                try:
                    perceptual_hash, thumbnail = perceptual_signatures(image)
                    perceptual_hashes[split].append(
                        (perceptual_hash, str(image), thumbnail)
                    )
                except Exception as exc:
                    invalid.setdefault(str(image), []).append(f"image_decode:{exc}")

        box_count = sum(class_counts.values())
        dominant_signature, dominant_count = (signatures.most_common(1)[0] if signatures else (None, 0))
        dominant_fraction = dominant_count / box_count if box_count else 0.0
        split_report = {
            "images": len(images),
            "boxes": box_count,
            "empty_labels": empty_labels,
            "missing_images": missing_images,
            "invalid_label_files": invalid,
            "class_box_counts": {names[key]: class_counts[key] for key in range(len(names))},
            "normalized_area": {
                "min": min(areas) if areas else None,
                "median": sorted(areas)[len(areas) // 2] if areas else None,
                "max": max(areas) if areas else None,
            },
            "dominant_box_signature": dominant_signature,
            "dominant_signature_fraction": dominant_fraction,
        }
        report["splits"][split] = split_report

        if missing_images:
            report["fatal"].append(f"{split}: {len(missing_images)} image paths do not exist")
        if invalid:
            report["fatal"].append(f"{split}: {len(invalid)} label files are invalid")
        if box_count and dominant_fraction >= 0.90 and len(signatures) <= max(3, len(names)):
            report["fatal"].append(
                f"{split}: {dominant_fraction:.1%} boxes share one geometry; likely pseudo labels"
            )

    # Exact duplicates must have the same byte size. Hash only sizes occurring
    # in more than one split; this keeps the leakage check practical at 10k+ images.
    if hash_images:
        candidate_sizes = set()
        for left, right in itertools.combinations(sources, 2):
            candidate_sizes.update(set(files_by_size[left]) & set(files_by_size[right]))
        for split in sources:
            for size in candidate_sizes:
                for image in files_by_size[split].get(size, []):
                    hashes[split][sha256_file(image)].append(str(image))

    for left, right in itertools.combinations(sources, 2):
        same_paths = sorted(source_sets[left] & source_sets[right])
        duplicate_hashes: list[dict[str, Any]] = []
        if hash_images:
            for digest in set(hashes[left]) & set(hashes[right]):
                duplicate_hashes.append(
                    {"sha256": digest, left: hashes[left][digest], right: hashes[right][digest]}
                )
        key = f"{left}_vs_{right}"
        report["cross_split"][key] = {
            "same_paths": same_paths,
            "duplicate_content": duplicate_hashes,
        }
        if hash_images:
            near_count, near_examples = near_duplicate_pairs(
                perceptual_hashes[left], perceptual_hashes[right]
            )
            report["cross_split"][key]["near_duplicate_content"] = {
                "screen": "dhash64+gray_thumbnail_mae32",
                "hamming_threshold": 4,
                "maximum_thumbnail_mae": 0.05,
                "count": near_count,
                "examples": near_examples,
            }
            if near_count:
                report["warnings"].append(
                    f"{key}: {near_count} perceptually similar image pairs require board/lot review"
                )
        if same_paths:
            report["fatal"].append(f"{key}: {len(same_paths)} identical image paths")
        if duplicate_hashes:
            report["fatal"].append(f"{key}: {len(duplicate_hashes)} duplicate image hashes")

    if "test" not in sources:
        report["fatal"].append("No independent test split is configured")
    if not sources.get("train") or not sources.get("val"):
        report["fatal"].append("Both train and validation splits are required")

    group_regex = data.get("metadata", {}).get("group_regex")
    if group_regex:
        compiled = re.compile(str(group_regex))
        groups: dict[str, set[str]] = {}
        for split, images in sources.items():
            groups[split] = {
                match.group(1) if (match := compiled.search(image.name)) else image.name
                for image in images
            }
        for left, right in itertools.combinations(groups, 2):
            overlap = sorted(groups[left] & groups[right])
            report["cross_split"][f"{left}_vs_{right}"]["shared_groups"] = overlap
            if overlap:
                report["fatal"].append(
                    f"{left}_vs_{right}: {len(overlap)} source board/lot groups overlap"
                )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a YOLO detection dataset before training")
    parser.add_argument("data", type=Path, help="YOLO dataset YAML")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-hashes", action="store_true", help="Skip slower duplicate-content check")
    parser.add_argument("--allow-fatal", action="store_true", help="Report errors without failing")
    args = parser.parse_args()

    report = audit_dataset(args.data, hash_images=not args.skip_hashes)
    if args.output:
        atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["fatal"] and not args.allow_fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
