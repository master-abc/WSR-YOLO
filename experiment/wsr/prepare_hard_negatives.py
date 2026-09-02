from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

try:
    from .common import atomic_json_dump, sha256_file
except ImportError:
    from common import atomic_json_dump, sha256_file


def prepare(
    base_data_path: Path,
    negative_audit_path: Path,
    output: Path,
    hard_fraction: float,
    repeat: int,
) -> dict[str, Any]:
    if not 0.0 < hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be in (0, 1]")
    if repeat < 2:
        raise ValueError("repeat must be at least two")
    dataset = yaml.safe_load(base_data_path.read_text(encoding="utf-8"))
    train_sources = dataset.get("train")
    if not isinstance(train_sources, list) or len(train_sources) != 2:
        raise ValueError("Expected [positive_train, negative_train] sources")
    positive_train = Path(train_sources[0]).resolve()
    negative_train = Path(train_sources[1]).resolve()
    negative_paths = [
        str(Path(line.strip()).resolve())
        for line in negative_train.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audit = json.loads(negative_audit_path.read_text(encoding="utf-8"))
    score_by_image = {
        str(Path(row["image"]).resolve()): max(
            (float(score) for score in row.get("scores", [])), default=0.0
        )
        for row in audit.get("per_image", [])
    }
    if set(score_by_image) != set(negative_paths):
        missing = len(set(negative_paths) - set(score_by_image))
        extra = len(set(score_by_image) - set(negative_paths))
        raise ValueError(f"Negative audit mismatch: missing={missing}, extra={extra}")
    hard_count = max(1, math.ceil(len(negative_paths) * hard_fraction))
    ranked = sorted(negative_paths, key=lambda path: (-score_by_image[path], path))
    hard_paths = ranked[:hard_count]
    oversampled = negative_paths + hard_paths * (repeat - 1)

    output.mkdir(parents=True, exist_ok=True)
    negative_output = output / "negative_train_oversampled.txt"
    negative_output.write_text("\n".join(oversampled) + "\n", encoding="utf-8")
    derived = dict(dataset)
    derived["train"] = [str(positive_train), str(negative_output.resolve())]
    metadata = dict(derived.get("metadata", {}))
    metadata["hard_negative_mining"] = {
        "selection_split": "train_negative",
        "hard_fraction": hard_fraction,
        "hard_images": hard_count,
        "repeat": repeat,
        "unique_negative_images": len(negative_paths),
        "effective_negative_samples": len(oversampled),
        "negative_audit": str(negative_audit_path.resolve()),
        "negative_audit_sha256": sha256_file(negative_audit_path),
        "test_images_used_for_selection": False,
    }
    derived["metadata"] = metadata
    data_output = output / "dataset.yaml"
    data_output.write_text(
        yaml.safe_dump(derived, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    payload = {
        "schema_version": 1,
        "base_data": str(base_data_path.resolve()),
        "base_data_sha256": sha256_file(base_data_path),
        "negative_audit": str(negative_audit_path.resolve()),
        "negative_audit_sha256": sha256_file(negative_audit_path),
        "hard_fraction": hard_fraction,
        "hard_images": hard_count,
        "repeat": repeat,
        "unique_negative_images": len(negative_paths),
        "effective_negative_samples": len(oversampled),
        "score_boundary": score_by_image[hard_paths[-1]],
        "maximum_score": score_by_image[hard_paths[0]],
        "dataset_yaml": str(data_output.resolve()),
        "dataset_yaml_sha256": sha256_file(data_output),
        "negative_list": str(negative_output.resolve()),
        "negative_list_sha256": sha256_file(negative_output),
        "test_images_used_for_selection": False,
    }
    atomic_json_dump(payload, output / "manifest.json")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Oversample high-scoring training templates without using test negatives"
    )
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--negative-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-fraction", type=float, default=0.25)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    payload = prepare(
        args.base_data.resolve(),
        args.negative_audit.resolve(),
        args.output.resolve(),
        args.hard_fraction,
        args.repeat,
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
