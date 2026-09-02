"""Build a single-pass detector input from a PCB image and golden reference.

The resulting detector is still one model with one forward pass.  Two encodings
are supported: a pure three-channel residual and a context-preserving signed
difference representation.  This is intentionally a different operating
contract from target-only detection: deployment must provide a registered
golden image for each inspected board.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

try:
    from .common import (
        atomic_json_dump,
        dataset_sources,
        image_to_label_path,
        sha256_file,
        stable_int,
        write_lines,
    )
    from .reference_verifier import sample_key
except ImportError:
    from experiment.wsr.common import (
        atomic_json_dump,
        dataset_sources,
        image_to_label_path,
        sha256_file,
        stable_int,
        write_lines,
    )
    from experiment.wsr.reference_verifier import sample_key


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def encode_paired_difference(
    candidate: np.ndarray,
    reference: np.ndarray,
    noise_floor: float = 3.0,
    gain: float = 4.0,
) -> np.ndarray:
    """Encode registered grayscale images as three residual channels."""

    if candidate.ndim != 2 or reference.ndim != 2:
        raise ValueError("candidate and reference must be grayscale images")
    if candidate.shape != reference.shape:
        raise ValueError(
            f"Pair shape mismatch: {candidate.shape} versus {reference.shape}"
        )
    if noise_floor < 0.0 or gain <= 0.0:
        raise ValueError("noise_floor must be non-negative and gain must be positive")
    candidate_f = candidate.astype(np.float32)
    reference_f = reference.astype(np.float32)
    candidate_mean, candidate_std = float(candidate_f.mean()), float(candidate_f.std())
    reference_mean, reference_std = float(reference_f.mean()), float(reference_f.std())
    scale = np.clip(candidate_std / max(reference_std, 1e-6), 0.5, 2.0)
    adjusted_reference = (
        (reference_f - reference_mean) * scale + candidate_mean
    )
    residual = cv2.GaussianBlur(
        candidate_f - np.clip(adjusted_reference, 0.0, 255.0), (3, 3), 0
    )
    dark = np.maximum(-residual - noise_floor, 0.0) * gain
    bright = np.maximum(residual - noise_floor, 0.0) * gain
    magnitude = np.maximum(np.abs(residual) - noise_floor, 0.0) * gain
    # OpenCV/Ultralytics reads the saved image as BGR. Channel order is part of
    # the model contract and is recorded in the manifest.
    return np.clip(np.stack((dark, bright, magnitude), axis=2), 0.0, 255.0).astype(
        np.uint8
    )


def encode_context_difference(
    candidate: np.ndarray,
    reference: np.ndarray,
    noise_floor: float = 3.0,
    gain: float = 4.0,
) -> np.ndarray:
    """Preserve target context while colour-coding signed reference change.

    The channel mean remains close to the original grayscale candidate, which
    makes this representation compatible with a target-only pretrained model.
    Green and red move in opposite directions only where the aligned reference
    provides evidence of a real change.
    """

    if candidate.ndim != 2 or reference.ndim != 2:
        raise ValueError("candidate and reference must be grayscale images")
    if candidate.shape != reference.shape:
        raise ValueError(
            f"Pair shape mismatch: {candidate.shape} versus {reference.shape}"
        )
    if noise_floor < 0.0 or gain <= 0.0:
        raise ValueError("noise_floor must be non-negative and gain must be positive")
    candidate_f = candidate.astype(np.float32)
    reference_f = reference.astype(np.float32)
    candidate_mean, candidate_std = float(candidate_f.mean()), float(candidate_f.std())
    reference_mean, reference_std = float(reference_f.mean()), float(reference_f.std())
    scale = np.clip(candidate_std / max(reference_std, 1e-6), 0.5, 2.0)
    adjusted_reference = (
        (reference_f - reference_mean) * scale + candidate_mean
    )
    residual = cv2.GaussianBlur(
        candidate_f - np.clip(adjusted_reference, 0.0, 255.0), (3, 3), 0
    )
    signed_change = np.sign(residual) * np.maximum(
        np.abs(residual) - noise_floor, 0.0
    ) * gain
    blue = candidate_f
    green = candidate_f + signed_change
    red = candidate_f - signed_change
    return np.clip(np.stack((blue, green, red), axis=2), 0.0, 255.0).astype(
        np.uint8
    )


def _synthetic_clean_candidate(
    reference: np.ndarray, key: str, split: str, replicate: int = 0
) -> np.ndarray:
    """Apply deterministic acquisition noise without introducing a defect."""

    if replicate < 0:
        raise ValueError("replicate must be non-negative")
    suffix = "" if replicate == 0 else f":r{replicate}"
    rng = np.random.default_rng(
        stable_int(f"paired-clean:{split}:{key}{suffix}") % (2**32)
    )
    image = reference.astype(np.float32)
    gain = float(rng.uniform(0.985, 1.015))
    bias = float(rng.uniform(-3.0, 3.0))
    image = image * gain + bias
    image += rng.normal(0.0, float(rng.uniform(0.5, 1.5)), size=image.shape)
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    if stable_int(f"paired-blur:{split}:{key}{suffix}") % 2:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    quality = 90 + stable_int(f"paired-jpeg:{split}:{key}{suffix}") % 11
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Failed to synthesize clean candidate for {key}")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise RuntimeError(f"Failed to decode clean candidate for {key}")
    return decoded


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_dataset(
    base_data_yaml: Path,
    template_list: Path,
    output: Path,
    noise_floor: float = 3.0,
    gain: float = 4.0,
    encoding: str = "residual",
    train_negative_replicates: int = 1,
    val_negative_replicates: int = 1,
    test_negative_replicates: int = 1,
    train_positive_repeats: int = 1,
) -> Path:
    encoders = {
        "residual": encode_paired_difference,
        "context": encode_context_difference,
    }
    if encoding not in encoders:
        raise ValueError(f"Unsupported encoding: {encoding}")
    encoder = encoders[encoding]
    replicate_counts = {
        "train": int(train_negative_replicates),
        "val": int(val_negative_replicates),
        "test": int(test_negative_replicates),
    }
    if any(value < 1 for value in replicate_counts.values()):
        raise ValueError("Every split must have at least one negative replicate")
    if train_positive_repeats < 1:
        raise ValueError("train_positive_repeats must be at least one")
    data, _, sources = dataset_sources(base_data_yaml)
    expected = {"train": 850, "val": 150, "test": 500}
    for split, count in expected.items():
        if len(sources.get(split, [])) != count:
            raise ValueError(
                f"Expected DeepPCB {split}={count}, found {len(sources.get(split, []))}"
            )
    templates = [
        Path(row).resolve()
        for row in template_list.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    template_by_key = {sample_key(path): path for path in templates}
    if len(template_by_key) != 1500:
        raise ValueError(f"Expected 1,500 unique references, found {len(template_by_key)}")

    output.mkdir(parents=True, exist_ok=True)
    lists: dict[str, dict[str, Path]] = {}
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        positive_images: list[Path] = []
        negative_images: list[Path] = []
        negative_by_replicate: list[list[Path]] = [
            [] for _ in range(replicate_counts[split])
        ]
        for target in sources[split]:
            key = sample_key(target)
            reference_path = template_by_key.get(key)
            if reference_path is None:
                raise KeyError(f"No reference for {target}")
            reference = _read_gray(reference_path)
            candidate = _read_gray(target)
            encoded_positive = encoder(
                candidate, reference, noise_floor, gain
            )
            positive_path = output / "images" / f"{split}_positive" / f"{key}.png"
            positive_path.parent.mkdir(parents=True, exist_ok=True)
            if not positive_path.exists() and not cv2.imwrite(
                str(positive_path), encoded_positive
            ):
                raise RuntimeError(f"Failed to write {positive_path}")
            label_source = image_to_label_path(target)
            label_target = output / "labels" / f"{split}_positive" / f"{key}.txt"
            _link_or_copy(label_source, label_target)
            positive_images.append(positive_path.resolve())

            for replicate in range(replicate_counts[split]):
                clean_candidate = _synthetic_clean_candidate(
                    reference, key, split, replicate
                )
                encoded_negative = encoder(
                    clean_candidate, reference, noise_floor, gain
                )
                negative_path = (
                    output
                    / "images"
                    / f"{split}_negative"
                    / f"{key}_r{replicate}.png"
                )
                negative_path.parent.mkdir(parents=True, exist_ok=True)
                if not negative_path.exists() and not cv2.imwrite(
                    str(negative_path), encoded_negative
                ):
                    raise RuntimeError(f"Failed to write {negative_path}")
                negative_label = (
                    output
                    / "labels"
                    / f"{split}_negative"
                    / f"{key}_r{replicate}.txt"
                )
                negative_label.parent.mkdir(parents=True, exist_ok=True)
                negative_label.touch(exist_ok=True)
                resolved_negative = negative_path.resolve()
                negative_images.append(resolved_negative)
                negative_by_replicate[replicate].append(resolved_negative)

        lists[split] = {}
        effective_positive_images = (
            positive_images * train_positive_repeats
            if split == "train"
            else positive_images
        )
        for role, images in (
            ("positive", effective_positive_images),
            ("negative", negative_images),
        ):
            list_path = output / f"{role}_{split}.txt"
            write_lines((str(path) for path in images), list_path)
            lists[split][role] = list_path.resolve()
        for replicate, images in enumerate(negative_by_replicate):
            write_lines(
                (str(path) for path in images),
                output / f"negative_{split}_r{replicate}.txt",
            )
        counts[split] = {
            "unique_positive": len(positive_images),
            "effective_positive": len(effective_positive_images),
            "negative_replicates": replicate_counts[split],
            "effective_negative": len(negative_images),
        }

    dataset: dict[str, Any] = {
        "path": str(output.resolve()),
        "train": [str(lists["train"]["positive"]), str(lists["train"]["negative"])],
        "val": [str(lists["val"]["positive"]), str(lists["val"]["negative"])],
        "test": str(lists["test"]["positive"]),
        "names": data["names"],
        "metadata": {
            "input_contract": "one inspected PCB plus its aligned golden reference",
            "single_model_forward_pass": True,
            "encoding": encoding,
            "channels_bgr": (
                ["dark_change", "bright_change", "absolute_change"]
                if encoding == "residual"
                else ["candidate", "candidate_plus_change", "candidate_minus_change"]
            ),
            "noise_floor": float(noise_floor),
            "gain": float(gain),
            "negative_policy": "deterministic defect-free acquisition perturbation",
            "train_positive_repeats": int(train_positive_repeats),
            "negative_replicates": replicate_counts,
            "test_excluded_from_training_and_selection": True,
            "base_data_yaml": str(base_data_yaml.resolve()),
            "base_data_yaml_sha256": sha256_file(base_data_yaml),
            "template_list": str(template_list.resolve()),
            "template_list_sha256": sha256_file(template_list),
            "counts": counts,
        },
    }
    dataset_path = output / "dataset.yaml"
    dataset_path.write_text(
        yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    atomic_json_dump(dataset["metadata"], output / "manifest.json")
    return dataset_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare paired-reference PCB residual inputs")
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--template-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise-floor", type=float, default=3.0)
    parser.add_argument("--gain", type=float, default=4.0)
    parser.add_argument("--encoding", choices=("residual", "context"), default="residual")
    parser.add_argument("--train-negative-replicates", type=int, default=1)
    parser.add_argument("--val-negative-replicates", type=int, default=1)
    parser.add_argument("--test-negative-replicates", type=int, default=1)
    parser.add_argument("--train-positive-repeats", type=int, default=1)
    args = parser.parse_args()
    path = prepare_dataset(
        args.base_data.resolve(),
        args.template_list.resolve(),
        args.output.resolve(),
        args.noise_floor,
        args.gain,
        args.encoding,
        args.train_negative_replicates,
        args.val_negative_replicates,
        args.test_negative_replicates,
        args.train_positive_repeats,
    )
    print(json.dumps({"dataset": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
