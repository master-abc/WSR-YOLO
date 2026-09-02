"""Generate one unseen defect-free perturbation replicate for final audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

try:
    from .common import atomic_json_dump, dataset_sources, sha256_file, write_lines
    from .paired_difference import (
        _read_gray,
        _synthetic_clean_candidate,
        encode_context_difference,
        encode_paired_difference,
    )
    from .reference_verifier import sample_key
except ImportError:
    from experiment.paper_b.common import (
        atomic_json_dump,
        dataset_sources,
        sha256_file,
        write_lines,
    )
    from experiment.paper_b.paired_difference import (
        _read_gray,
        _synthetic_clean_candidate,
        encode_context_difference,
        encode_paired_difference,
    )
    from experiment.paper_b.reference_verifier import sample_key


def prepare_negative_holdout(
    base_data_yaml: Path,
    template_list: Path,
    output: Path,
    split: str,
    replicate: int,
    encoding: str = "context",
    noise_floor: float = 3.0,
    gain: float = 4.0,
) -> Path:
    if replicate < 1:
        raise ValueError("Holdout replicate must be at least one")
    encoders = {
        "context": encode_context_difference,
        "residual": encode_paired_difference,
    }
    if encoding not in encoders:
        raise ValueError(f"Unsupported encoding: {encoding}")
    _, _, sources = dataset_sources(base_data_yaml)
    if split not in sources or not sources[split]:
        raise ValueError(f"Dataset split is empty: {split}")
    templates = [
        Path(row).resolve()
        for row in template_list.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    template_by_key = {sample_key(path): path for path in templates}
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    images: list[Path] = []
    for target in sources[split]:
        key = sample_key(target)
        reference_path = template_by_key.get(key)
        if reference_path is None:
            raise KeyError(f"No reference for {target}")
        destination = image_dir / f"{key}_r{replicate}.png"
        if not destination.exists():
            reference = _read_gray(reference_path)
            clean_candidate = _synthetic_clean_candidate(
                reference, key, split, replicate
            )
            encoded = encoders[encoding](
                clean_candidate, reference, noise_floor, gain
            )
            if not cv2.imwrite(str(destination), encoded):
                raise RuntimeError(f"Failed to write {destination}")
        images.append(destination.resolve())
    image_list = output / "images.txt"
    write_lines([str(path) for path in images], image_list)
    atomic_json_dump(
        {
            "schema_version": 1,
            "purpose": "fresh defect-free perturbation holdout",
            "excluded_from_training_and_policy_selection": True,
            "base_data_yaml": str(base_data_yaml.resolve()),
            "base_data_yaml_sha256": sha256_file(base_data_yaml),
            "template_list": str(template_list.resolve()),
            "template_list_sha256": sha256_file(template_list),
            "split": split,
            "replicate": replicate,
            "encoding": encoding,
            "noise_floor": noise_floor,
            "gain": gain,
            "images": len(images),
            "image_list": str(image_list.resolve()),
        },
        output / "manifest.json",
    )
    return image_list


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a fresh negative holdout")
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--template-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--encoding", choices=("context", "residual"), default="context")
    parser.add_argument("--noise-floor", type=float, default=3.0)
    parser.add_argument("--gain", type=float, default=4.0)
    args = parser.parse_args()
    path = prepare_negative_holdout(
        args.base_data.resolve(),
        args.template_list.resolve(),
        args.output.resolve(),
        args.split,
        args.replicate,
        args.encoding,
        args.noise_floor,
        args.gain,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
