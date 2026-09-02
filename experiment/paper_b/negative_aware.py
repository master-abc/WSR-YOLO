from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

try:
    from .coco_evaluator import evaluate_coco_predictions, predict_yolo_to_coco
    from .common import (
        PROJECT_DIR,
        atomic_json_dump,
        class_names,
        dataset_sources,
        environment_snapshot,
        sha256_file,
        sha256_text,
        stable_int,
        write_lines,
    )
    from .pretrained import transfer_pretrained
except ImportError:
    from experiment.paper_b.coco_evaluator import (
        evaluate_coco_predictions,
        predict_yolo_to_coco,
    )
    from experiment.paper_b.common import (
        PROJECT_DIR,
        atomic_json_dump,
        class_names,
        dataset_sources,
        environment_snapshot,
        sha256_file,
        sha256_text,
        stable_int,
        write_lines,
    )
    from experiment.paper_b.pretrained import transfer_pretrained


TRAIN_ARGS: dict[str, Any] = {
    "epochs": 80,
    "imgsz": 640,
    "batch": 8,
    "nbs": 64,
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "cos_lr": True,
    "amp": True,
    "deterministic": True,
    "patience": 15,
    "close_mosaic": 10,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "workers": 2,
    "plots": True,
}


def _target_key(path: Path) -> str:
    stem = path.stem
    return stem.split("_", 1)[1] if "_" in stem else stem


def _template_key(path: Path) -> str:
    suffix = "_temp"
    if not path.stem.endswith(suffix):
        raise ValueError(f"Expected a DeepPCB template name ending in {suffix}: {path}")
    return path.stem[: -len(suffix)]


def _materialize_backgrounds(
    target_images: list[Path],
    template_by_key: dict[str, Path],
    output: Path,
    split: str,
) -> list[Path]:
    image_dir = output / "images" / f"{split}_negative"
    label_dir = output / "labels" / f"{split}_negative"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    backgrounds = []
    for target in target_images:
        key = _target_key(target)
        template = template_by_key.get(key)
        if template is None:
            raise KeyError(f"No defect-free template matches {target}")
        destination = image_dir / f"{key}_negative{template.suffix.lower()}"
        if not destination.exists():
            try:
                os.link(template, destination)
            except OSError:
                shutil.copy2(template, destination)
        label = label_dir / f"{destination.stem}.txt"
        if not label.exists():
            label.touch()
        backgrounds.append(destination.resolve())
    return backgrounds


def prepare_negative_aware_dataset(
    base_data_yaml: Path,
    template_list: Path,
    output: Path,
    train_negative_fraction: float = 1.0,
) -> Path:
    if not 0.0 < train_negative_fraction <= 1.0:
        raise ValueError("train_negative_fraction must be in (0, 1]")
    data, _, sources = dataset_sources(base_data_yaml)
    required = {"train": 850, "val": 150, "test": 500}
    for split, expected in required.items():
        observed = len(sources.get(split, []))
        if observed != expected:
            raise ValueError(f"Expected DeepPCB {split}={expected}, found {observed}")
    templates = [
        Path(line.strip()).resolve()
        for line in template_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    template_by_key = {_template_key(path): path for path in templates}
    if len(template_by_key) != 1500 or any(not path.is_file() for path in templates):
        raise ValueError("The official DeepPCB template list must contain 1,500 existing images")

    output.mkdir(parents=True, exist_ok=True)
    lists: dict[str, Path] = {}
    manifests: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        backgrounds = _materialize_backgrounds(
            sources[split], template_by_key, output, split
        )
        if split == "train" and train_negative_fraction < 1.0:
            count = max(1, round(len(backgrounds) * train_negative_fraction))
            backgrounds = sorted(
                backgrounds, key=lambda path: stable_int(f"negative-aware:{path.name}")
            )[:count]
        path = output / f"negative_{split}.txt"
        write_lines((str(image) for image in backgrounds), path)
        lists[split] = path.resolve()
        manifests[split] = {
            "positive_images": len(sources[split]),
            "negative_images": len(backgrounds),
            "negative_list": str(path.resolve()),
            "negative_list_sha256": sha256_file(path),
        }

    base_lists: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        path = output / f"positive_{split}.txt"
        write_lines((str(image) for image in sources[split]), path)
        base_lists[split] = path.resolve()

    dataset = {
        "path": str(output.resolve()),
        "train": [str(base_lists["train"]), str(lists["train"])],
        "val": [str(base_lists["val"]), str(lists["val"])],
        # Positive test is kept for checkpoint-independent AP evaluation. The
        # disjoint negative test list is evaluated separately and never used by training.
        "test": str(base_lists["test"]),
        "names": data["names"],
        "metadata": {
            "protocol": "one paired defect-free template per positive train/val image",
            "train_negative_fraction": float(train_negative_fraction),
            "base_data_yaml": str(base_data_yaml.resolve()),
            "base_data_yaml_sha256": sha256_file(base_data_yaml),
            "template_list": str(template_list.resolve()),
            "template_list_sha256": sha256_file(template_list),
            "test_negatives_excluded_from_training_and_validation": True,
            "splits": manifests,
        },
    }
    dataset_yaml = output / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    atomic_json_dump(dataset["metadata"], output / "manifest.json")
    return dataset_yaml


def finetune_negative_aware(
    data_yaml: Path,
    initial_weights: Path,
    output: Path,
    seed: int,
    device: str,
    epochs: int = 20,
    lr0: float = 0.001,
    batch: int = 8,
    workers: int = 2,
) -> Path:
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from algorithm.register import register_custom_modules

    register_custom_modules()
    from ultralytics import YOLO

    model = YOLO(str(initial_weights.resolve()))
    args = dict(TRAIN_ARGS)
    args.update(
        {
            "epochs": int(epochs),
            "batch": int(batch),
            "workers": int(workers),
            "lr0": float(lr0),
            "lrf": 0.01,
            "warmup_epochs": 1.0,
            "patience": 0,
            "mosaic": 0.0,
            "close_mosaic": 0,
            "plots": True,
            "data": str(data_yaml.resolve()),
            "seed": int(seed),
            "device": device,
            "project": str(output.parent.resolve()),
            "name": output.name,
            "exist_ok": True,
            "verbose": True,
        }
    )
    manifest = {
        "schema_version": 1,
        "track": "negative_aware_finetuning_validation_selection",
        "selection_split": "positive_and_defect_free_validation",
        "test_evaluated": False,
        "seed": int(seed),
        "initial_weights": str(initial_weights.resolve()),
        "initial_weights_sha256": sha256_file(initial_weights),
        "data_yaml": str(data_yaml.resolve()),
        "data_yaml_sha256": sha256_file(data_yaml),
        "train_args": args,
        "environment_at_start": environment_snapshot(),
    }
    atomic_json_dump(manifest, output / "run_manifest.json")
    model.train(**args)
    best = output / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Fine-tuning ended without best weights: {best}")
    payload = {
        **manifest,
        "weights": str(best.resolve()),
        "weights_sha256": sha256_file(best),
        "environment": environment_snapshot(),
    }
    result = output / "negative_aware_finetune_result.json"
    atomic_json_dump(payload, result)
    print(result.resolve())
    return result


def select_negative_aware(candidates: list[Path], output: Path) -> dict[str, Any]:
    rows = []
    initial_hashes = set()
    for directory in candidates:
        result_path = directory / "negative_aware_finetune_result.json"
        history_path = directory / "results.csv"
        if not result_path.is_file() or not history_path.is_file():
            raise FileNotFoundError(f"Incomplete negative-aware candidate: {directory}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("test_evaluated") is not False:
            raise ValueError(f"Candidate is not validation-only: {directory}")
        initial_hashes.add(result["initial_weights_sha256"])
        with history_path.open("r", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        metric_key = "metrics/mAP50-95(B)"
        best = max(history, key=lambda row: float(row[metric_key]))
        dataset = yaml.safe_load(Path(result["data_yaml"]).read_text(encoding="utf-8"))
        rows.append(
            {
                "directory": str(directory.resolve()),
                "weights": result["weights"],
                "weights_sha256": result["weights_sha256"],
                "train_negative_fraction": float(
                    dataset["metadata"]["train_negative_fraction"]
                ),
                "best_mixed_validation_map50_95": float(best[metric_key]),
                "best_mixed_validation_epoch": int(best["epoch"]),
                "history_sha256": sha256_file(history_path),
                "result_sha256": sha256_file(result_path),
            }
        )
    if len(initial_hashes) != 1:
        raise ValueError("Candidates do not start from the same frozen checkpoint")
    selected = max(
        rows,
        key=lambda row: (
            row["best_mixed_validation_map50_95"],
            -row["train_negative_fraction"],
        ),
    )
    payload = {
        "schema_version": 1,
        "policy": (
            "highest mixed positive-plus-negative validation AP50:95; ties prefer "
            "the lower negative fraction"
        ),
        "selection_split": "val",
        "test_evaluated": False,
        "initial_weights_sha256": next(iter(initial_hashes)),
        "candidates": rows,
        "selected": selected,
    }
    atomic_json_dump(payload, output)
    return payload


def finalize_validation_selected_run(
    run: Path,
    data_yaml: Path,
    initial_weights: Path,
    output: Path,
    seed: int,
) -> dict[str, Any]:
    """Freeze an early-stopped local run using validation history only."""

    history_path = run / "results.csv"
    weights = run / "weights" / "best.pt"
    if not history_path.is_file() or not weights.is_file():
        raise FileNotFoundError(f"Incomplete run: {run}")
    with history_path.open("r", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    if not history:
        raise ValueError(f"No completed validation epochs in {history_path}")
    metric_key = "metrics/mAP50-95(B)"
    best = max(history, key=lambda row: float(row[metric_key]))
    payload = {
        "schema_version": 1,
        "track": "stage2_full_negative_validation_selection",
        "selection_split": "positive_and_defect_free_validation",
        "test_evaluated": False,
        "stopping_policy": "stop after three completed epochs fail to improve epoch one",
        "completed_epochs": len(history),
        "best_epoch": int(best["epoch"]),
        "best_mixed_validation_map50_95": float(best[metric_key]),
        "seed": int(seed),
        "run": str(run.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "data_yaml_sha256": sha256_file(data_yaml),
        "initial_weights": str(initial_weights.resolve()),
        "initial_weights_sha256": sha256_file(initial_weights),
        "weights": str(weights.resolve()),
        "weights_sha256": sha256_file(weights),
        "history": str(history_path.resolve()),
        "history_sha256": sha256_file(history_path),
        "environment": environment_snapshot(),
    }
    atomic_json_dump(payload, output)
    return payload


def train_negative_aware(
    data_yaml: Path,
    architecture: str,
    pretrained: Path,
    output: Path,
    seed: int,
    device: str,
    positive_test_annotations: Path,
    positive_test_images: Path,
) -> Path:
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from algorithm.register import register_custom_modules

    register_custom_modules()
    from ultralytics import YOLO

    architecture_path = Path(architecture)
    resolved_architecture = (
        str(architecture_path.resolve()) if architecture_path.is_file() else architecture
    )
    model = YOLO(resolved_architecture)
    transfer = transfer_pretrained(model, pretrained, YOLO, minimum_parameter_fraction=0.99)
    args = dict(TRAIN_ARGS)
    args.update(
        {
            "data": str(data_yaml.resolve()),
            "seed": int(seed),
            "device": device,
            "project": str(output.parent.resolve()),
            "name": output.name,
            "exist_ok": True,
            "verbose": True,
        }
    )
    start = environment_snapshot()
    manifest = {
        "schema_version": 1,
        "track": "negative_aware_mitigation",
        "selection_split": "positive_and_defect_free_validation",
        "test_evaluated_during_selection": False,
        "seed": int(seed),
        "architecture": resolved_architecture,
        "architecture_sha256": (
            sha256_file(architecture_path) if architecture_path.is_file() else None
        ),
        "architecture_definition_sha256": sha256_text(
            json.dumps(model.model.yaml, sort_keys=True, ensure_ascii=False, default=str)
        ),
        "pretrained": str(pretrained.resolve()),
        "pretrained_sha256": sha256_file(pretrained),
        "pretrained_transfer": transfer,
        "data_yaml": str(data_yaml.resolve()),
        "data_yaml_sha256": sha256_file(data_yaml),
        "train_args": args,
        "environment_at_start": start,
    }
    atomic_json_dump(manifest, output / "run_manifest.json")
    model.train(**args)
    best = output / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training ended without best weights: {best}")

    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    evaluated = YOLO(str(best))
    prediction_path = output / "positive_test_predictions.json"
    predict_yolo_to_coco(
        evaluated,
        positive_test_annotations,
        positive_test_images,
        prediction_path,
        640,
        1,
        device,
        0.001,
        0.7,
        300,
    )
    report = evaluate_coco_predictions(
        positive_test_annotations, prediction_path, max_det=100
    )
    payload = {
        **manifest,
        "weights": str(best.resolve()),
        "weights_sha256": sha256_file(best),
        "positive_test": report,
        "environment": environment_snapshot(),
    }
    result = output / "negative_aware_result.json"
    atomic_json_dump(payload, result)
    print(f"positive-test AP50:95={report['metrics']['map50_95']:.6f}")
    print(result.resolve())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and train a leakage-safe DeepPCB negative-aware detector"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--base-data", type=Path, required=True)
    prepare.add_argument("--template-list", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--train-negative-fraction", type=float, default=1.0)
    train = subparsers.add_parser("train")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--architecture", required=True)
    train.add_argument("--pretrained", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--device", default="0")
    train.add_argument("--positive-test-annotations", type=Path, required=True)
    train.add_argument("--positive-test-images", type=Path, required=True)
    finetune = subparsers.add_parser("finetune")
    finetune.add_argument("--data", type=Path, required=True)
    finetune.add_argument("--initial-weights", type=Path, required=True)
    finetune.add_argument("--output", type=Path, required=True)
    finetune.add_argument("--seed", type=int, required=True)
    finetune.add_argument("--device", default="0")
    finetune.add_argument("--epochs", type=int, default=20)
    finetune.add_argument("--lr0", type=float, default=0.001)
    finetune.add_argument("--batch", type=int, default=8)
    finetune.add_argument("--workers", type=int, default=2)
    select = subparsers.add_parser("select")
    select.add_argument("--candidates", type=Path, nargs="+", required=True)
    select.add_argument("--output", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--run", type=Path, required=True)
    finalize.add_argument("--data", type=Path, required=True)
    finalize.add_argument("--initial-weights", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        path = prepare_negative_aware_dataset(
            args.base_data.resolve(),
            args.template_list.resolve(),
            args.output.resolve(),
            args.train_negative_fraction,
        )
        print(path)
        return 0
    if args.command == "finetune":
        finetune_negative_aware(
            args.data.resolve(),
            args.initial_weights.resolve(),
            args.output.resolve(),
            args.seed,
            args.device,
            args.epochs,
            args.lr0,
            args.batch,
            args.workers,
        )
        return 0
    if args.command == "select":
        payload = select_negative_aware(
            [path.resolve() for path in args.candidates], args.output.resolve()
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    if args.command == "finalize":
        payload = finalize_validation_selected_run(
            args.run.resolve(),
            args.data.resolve(),
            args.initial_weights.resolve(),
            args.output.resolve(),
            args.seed,
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    train_negative_aware(
        args.data.resolve(),
        args.architecture,
        args.pretrained.resolve(),
        args.output.resolve(),
        args.seed,
        args.device,
        args.positive_test_annotations.resolve(),
        args.positive_test_images.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
