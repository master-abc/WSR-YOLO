from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from typing import Any

import yaml

try:
    from .common import (
        PAPER_B_DIR,
        PROJECT_DIR,
        atomic_json_dump,
        class_names,
        dataset_sources,
        environment_snapshot,
        file_hashes,
        load_yaml,
        resolve_path,
        sha256_file,
        sha256_text,
        write_lines,
    )
    from .coco_evaluator import evaluate_coco_predictions, predict_yolo_to_coco
    from .pretrained import transfer_pretrained
    from .yolo_to_coco import convert_dataset
    from .run import (
        assert_formal_run_provenance,
        generated_dataset_yaml,
        import_ultralytics,
        load_protocol,
        model_complexity,
    )
    from .split_dataset import image_classes, multilabel_stratified_split
except ImportError:
    from common import PAPER_B_DIR, PROJECT_DIR, atomic_json_dump, class_names, dataset_sources, environment_snapshot, file_hashes, load_yaml, resolve_path, sha256_file, sha256_text, write_lines
    from coco_evaluator import evaluate_coco_predictions, predict_yolo_to_coco
    from pretrained import transfer_pretrained
    from yolo_to_coco import convert_dataset
    from run import assert_formal_run_provenance, generated_dataset_yaml, import_ultralytics, load_protocol, model_complexity
    from split_dataset import image_classes, multilabel_stratified_split


def materialize_validation_subset(
    protocol: dict[str, Any],
    dataset_name: str,
    base_data_yaml: Path,
    fraction: float,
    subset_seed: int,
) -> tuple[Path, dict[str, Any]]:
    """Create a deterministic multilabel-stratified train subset without a test key."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"Validation-only training fraction must be in (0, 1], got {fraction}")
    data, _, sources = dataset_sources(base_data_yaml)
    training = sources.get("train", [])
    validation = sources.get("val", [])
    if not training or not validation:
        raise ValueError(f"Validation-only training requires train and val splits: {base_data_yaml}")
    names = class_names(data)
    if fraction < 1.0:
        selected, _ = multilabel_stratified_split(
            training, [fraction, 1.0 - fraction], subset_seed, len(names)
        )
    else:
        selected = training

    tag = f"f{fraction:.4f}_seed{subset_seed}".replace(".", "p")
    output_dir = base_data_yaml.parent / "validation_subsets"
    train_list = output_dir / f"train_{tag}.txt"
    val_list = output_dir / f"val_{tag}.txt"
    write_lines((path.as_posix() for path in selected), train_list)
    write_lines((path.as_posix() for path in validation), val_list)
    subset_yaml = output_dir / f"dataset_{tag}.yaml"
    subset_payload = {
        "path": base_data_yaml.parent.resolve().as_posix(),
        "train": train_list.resolve().as_posix(),
        "val": val_list.resolve().as_posix(),
        "names": {index: name for index, name in enumerate(names)},
        "metadata": {
            "purpose": "validation_only_architecture_selection",
            "base_data_yaml": base_data_yaml.resolve().as_posix(),
            "fraction": fraction,
            "subset_seed": subset_seed,
            "test_omitted": True,
        },
    }
    subset_yaml.parent.mkdir(parents=True, exist_ok=True)
    subset_yaml.write_text(
        yaml.safe_dump(subset_payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    selected_class_images = {name: 0 for name in names}
    for image in selected:
        for class_id in image_classes(image, len(names)):
            selected_class_images[names[class_id]] += 1
    report = {
        "strategy": "deterministic_greedy_multilabel_stratified",
        "fraction": fraction,
        "subset_seed": subset_seed,
        "base_train_images": len(training),
        "selected_train_images": len(selected),
        "validation_images": len(validation),
        "selected_class_image_counts": selected_class_images,
        "train_list_sha256": sha256_file(train_list),
        "val_list_sha256": sha256_file(val_list),
        "dataset_yaml_sha256": sha256_file(subset_yaml),
        "test_omitted": True,
    }
    atomic_json_dump(report, output_dir / f"manifest_{tag}.json")
    return subset_yaml, report


def materialize_models(protocol: dict) -> dict[str, str | Path]:
    output = protocol["_output_root"] / "models" / "ablation"
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str | Path] = {}
    pilot = protocol.get("pilot_gate", {})
    baseline_name = pilot.get("baseline")
    if baseline_name:
        paths[str(baseline_name)] = str(pilot.get("baseline_architecture", "yolo11s.yaml"))
    for variant in protocol["ablation_track"]["models"]:
        if "architecture" in variant:
            paths[variant["name"]] = resolve_path(variant["architecture"], protocol["_path"].parent)
            continue
        template = resolve_path(variant["template"], protocol["_path"].parent)
        model_yaml = load_yaml(template)
        found = 0
        for section in ("backbone", "head"):
            for layer in model_yaml.get(section, []):
                if len(layer) >= 4 and layer[2] in {"DWGSARouter", "WSR"}:
                    layer[3] = list(variant["router_args"])
                    found += 1
        if not found:
            raise ValueError(f"No DWGSARouter found in {template}")
        target = output / f"{variant['name']}.yaml"
        target.write_text(yaml.safe_dump(model_yaml, sort_keys=False), encoding="utf-8")
        paths[variant["name"]] = target
    atomic_json_dump(
        {
            key: str(value.resolve()) if isinstance(value, Path) else str(value)
            for key, value in paths.items()
        },
        output / "manifest.json",
    )
    return paths


def subset_coco_annotations(source: Path, target: Path, maximum_images: int) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    images = sorted(payload.get("images", []), key=lambda item: int(item["id"]))[:maximum_images]
    image_ids = {int(item["id"]) for item in images}
    subset = dict(payload)
    subset["images"] = images
    subset["annotations"] = [
        item for item in payload.get("annotations", []) if int(item["image_id"]) in image_ids
    ]
    atomic_json_dump(subset, target)
    return target


def train_validation_only(
    protocol: dict[str, Any],
    variant_name: str,
    seed: int,
    device: str,
    smoke: bool,
    force: bool = False,
    budget_profile: str = "ablation",
    resume: bool = False,
) -> Path:
    if force and resume:
        raise ValueError("--force and --resume are mutually exclusive")
    dataset_name = protocol["ablation_track"]["dataset"]
    data_yaml = generated_dataset_yaml(protocol, dataset_name)
    paths = materialize_models(protocol)
    if variant_name not in paths:
        raise KeyError(f"Unknown ablation variant: {variant_name}")
    registered_seeds = [int(value) for value in protocol["ablation_track"]["seeds"]]
    if not smoke and seed not in registered_seeds:
        raise ValueError(f"Seed {seed} is outside registered ablation seeds: {registered_seeds}")
    audit_path = data_yaml.parent / "audit.json"
    if not audit_path.exists():
        raise FileNotFoundError(f"Run the full dataset audit first: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("fatal") or not audit.get("content_hashes_checked"):
        raise RuntimeError(f"Dataset audit is not valid for pilot selection: {audit_path}")
    scope = "smoke" if smoke else "runs"
    track_directory = "pilot" if budget_profile == "pilot" else "ablation"
    run_dir = (
        protocol["_output_root"]
        / scope
        / track_directory
        / dataset_name
        / variant_name
        / f"seed_{seed}"
    )
    result_file = run_dir / "ablation_result.json"
    if result_file.exists() and not force:
        existing = json.loads(result_file.read_text(encoding="utf-8"))
        if (
            existing.get("budget_profile") != budget_profile
            or existing.get("protocol_sha256") != sha256_file(protocol["_path"])
        ):
            raise RuntimeError(
                f"Existing result uses a different budget or protocol; pass --force to replace it: {result_file}"
            )
        print(f"[skip] {result_file}")
        return result_file
    train_args = copy.deepcopy(protocol["controlled_track"]["train"])
    if budget_profile not in {"pilot", "ablation"}:
        raise ValueError(f"Unknown validation-only budget profile: {budget_profile}")
    profile_config = (
        protocol["pilot_gate"] if budget_profile == "pilot" else protocol["ablation_track"]
    )
    train_args.update(copy.deepcopy(profile_config.get("train", {})))
    if smoke:
        train_args.update(
            {"epochs": 2, "imgsz": 320, "batch": 2, "workers": 0, "fraction": 0.02}
        )
    requested_fraction = float(train_args.pop("fraction", 1.0))
    subset_seed = int(train_args.pop("subset_seed", int(protocol["split_seed"]) + 1))
    training_data_yaml, training_subset = materialize_validation_subset(
        protocol, dataset_name, data_yaml, requested_fraction, subset_seed
    )
    train_args["fraction"] = 1.0
    environment = assert_formal_run_provenance(protocol, smoke)
    YOLO = import_ultralytics()
    architecture = str(paths[variant_name])
    pretrained = str(
        profile_config.get("pretrained", protocol["ablation_track"].get("pretrained", "yolo11s.pt"))
    )
    protocol_hash = sha256_file(protocol["_path"])
    base_data_hash = sha256_file(data_yaml)
    training_data_hash = sha256_file(training_data_yaml)
    manifest_file = run_dir / "run_manifest.json"
    transfer_file = run_dir / "pretrained_transfer.json"
    last = run_dir / "weights" / "last.pt"
    source_files = [
        PROJECT_DIR / "algorithm" / "dwgsa.py",
        PROJECT_DIR / "algorithm" / "register.py",
        Path(__file__),
        PAPER_B_DIR / "coco_evaluator.py",
        PAPER_B_DIR / "pretrained.py",
        PAPER_B_DIR / "split_dataset.py",
        protocol["_path"],
    ]
    architecture_path = Path(architecture)
    if architecture_path.is_file():
        source_files.append(architecture_path)

    resumed = bool(resume and last.exists())
    training_complete_file = run_dir / "training_complete.json"
    if resumed:
        if not manifest_file.exists() or not transfer_file.exists():
            raise FileNotFoundError("Cannot audit validation-only resume without manifest and transfer report")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        expected = {
            "dataset": dataset_name,
            "model": variant_name,
            "seed": int(seed),
            "budget_profile": budget_profile,
            "protocol_sha256": protocol_hash,
            "base_data_yaml_sha256": base_data_hash,
            "training_data_yaml_sha256": training_data_hash,
            "train_list_sha256": training_subset["train_list_sha256"],
            "git_commit": environment.get("git_commit"),
        }
        observed = {
            "dataset": manifest.get("dataset"),
            "model": manifest.get("model"),
            "seed": manifest.get("seed"),
            "budget_profile": manifest.get("budget_profile"),
            "protocol_sha256": manifest.get("protocol_sha256"),
            "base_data_yaml_sha256": manifest.get("base_data_yaml_sha256"),
            "training_data_yaml_sha256": manifest.get("training_data_yaml_sha256"),
            "train_list_sha256": manifest.get("training_subset", {}).get("train_list_sha256"),
            "git_commit": manifest.get("environment_at_start", {}).get("git_commit"),
        }
        if observed != expected:
            raise RuntimeError(
                "Validation-only resume provenance mismatch; refusing to mix runs. "
                f"expected={expected}, observed={observed}"
            )
        environment = manifest["environment_at_start"]
        architecture_definition_sha256 = manifest["architecture_definition_sha256"]
        transfer = json.loads(transfer_file.read_text(encoding="utf-8"))
        model = YOLO(str(last))
        train_call = {"resume": True}
        results_csv = run_dir / "results.csv"
        observed_epochs = 0
        if results_csv.exists():
            observed_epochs = max(0, len(results_csv.read_text(encoding="utf-8-sig").splitlines()) - 1)
        should_train = not training_complete_file.exists() and observed_epochs < int(
            train_args["epochs"]
        )
        if not should_train:
            print(f"[resume] training already complete; continuing with evaluation: {run_dir}")
    else:
        if resume:
            print(f"[resume] no checkpoint exists; starting a fresh audited run: {run_dir}")
        model = YOLO(architecture)
        architecture_definition_sha256 = sha256_text(
            json.dumps(model.model.yaml, sort_keys=True, ensure_ascii=False, default=str)
        )
        transfer = transfer_pretrained(
            model,
            pretrained,
            YOLO,
            float(protocol.get("reproducibility", {}).get("minimum_pretrained_fraction", 0.99)),
        )
        atomic_json_dump(transfer, transfer_file)
        manifest = {
            "schema_version": 1,
            "dataset": dataset_name,
            "model": variant_name,
            "seed": int(seed),
            "smoke": bool(smoke),
            "budget_profile": budget_profile,
            "architecture": architecture,
            "architecture_definition_sha256": architecture_definition_sha256,
            "pretrained": pretrained,
            "protocol_sha256": protocol_hash,
            "base_data_yaml_sha256": base_data_hash,
            "training_data_yaml_sha256": training_data_hash,
            "training_subset": training_subset,
            "train_args": train_args,
            "environment_at_start": environment,
            "source_files_sha256": file_hashes(source_files),
        }
        atomic_json_dump(manifest, manifest_file)
        train_call = dict(train_args)
        train_call.update(
            {
                "data": str(training_data_yaml),
                "seed": seed,
                "device": device,
                "project": str(run_dir.parent),
                "name": run_dir.name,
                "exist_ok": True,
            }
        )
        should_train = True
    if should_train:
        model.train(**train_call)
        completed = {
            "schema_version": 1,
            "weights_last": str(last.resolve()),
            "weights_last_sha256": sha256_file(last),
            "protocol_sha256": protocol_hash,
            "environment_at_start": environment,
        }
        best_after_train = run_dir / "weights" / "best.pt"
        if best_after_train.exists():
            completed["weights_best"] = str(best_after_train.resolve())
            completed["weights_best_sha256"] = sha256_file(best_after_train)
        atomic_json_dump(completed, training_complete_file)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    evaluated = YOLO(str(best))
    coco_root = protocol["_output_root"] / "coco" / dataset_name
    annotations = coco_root / "annotations" / "instances_val.json"
    if not annotations.exists():
        convert_dataset(data_yaml, coco_root)
    evaluation_image_count = len(json.loads(annotations.read_text(encoding="utf-8"))["images"])
    if smoke:
        evaluation_image_count = int(
            protocol.get("reproducibility", {}).get("smoke_validation_images", 50)
        )
        annotations = subset_coco_annotations(
            annotations, run_dir / "smoke_instances_val.json", evaluation_image_count
        )
    predictions = predict_yolo_to_coco(
        evaluated,
        annotations,
        coco_root / "val",
        run_dir / "val_predictions.json",
        int(train_args["imgsz"]),
        int(protocol["evaluation"].get("batch", 1)),
        device,
        float(protocol["evaluation"]["conf"]),
        float(protocol["evaluation"]["iou"]),
        int(protocol["evaluation"]["prediction_max_det"]),
    )
    evaluation = evaluate_coco_predictions(
        annotations, predictions, int(protocol["evaluation"]["max_det"])
    )
    payload = {
        "schema_version": 2,
        "track": "ablation_validation_only",
        "smoke": bool(smoke),
        "dataset": dataset_name,
        "base_data_yaml": str(data_yaml.resolve()),
        "base_data_yaml_sha256": sha256_file(data_yaml),
        "training_data_yaml": str(training_data_yaml.resolve()),
        "training_data_yaml_sha256": sha256_file(training_data_yaml),
        "training_subset": training_subset,
        "model": variant_name,
        "seed": seed,
        "budget_profile": budget_profile,
        "resumed": resumed,
        "selection_split": "val",
        "test_evaluated": False,
        "evaluation_image_count": evaluation_image_count,
        "architecture": architecture,
        "pretrained": pretrained,
        "architecture_definition_sha256": architecture_definition_sha256,
        "pretrained_transfer": transfer,
        "weights": str(best.resolve()),
        "weights_sha256": sha256_file(best),
        "train_args": train_args,
        "metrics": evaluation["metrics"],
        "unified_evaluation": evaluation,
        "complexity": model_complexity(evaluated, int(train_args["imgsz"])),
        "environment_at_start": environment,
        "environment": environment_snapshot(),
        "protocol_sha256": protocol_hash,
        "source_files_sha256": file_hashes(source_files),
    }
    atomic_json_dump(payload, result_file)
    return result_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and run validation-only router ablations")
    parser.add_argument("--config", type=Path, default=PAPER_B_DIR / "paper_b.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("materialize")
    train = subparsers.add_parser("train")
    train.add_argument("--variant", required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--device", default="0")
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--force", action="store_true")
    train.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.config.resolve())
    if args.command == "materialize":
        for name, path in materialize_models(protocol).items():
            print(f"{name}: {path}")
        return 0
    train_validation_only(
        protocol, args.variant, args.seed, args.device, args.smoke, args.force, "ablation", args.resume
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
