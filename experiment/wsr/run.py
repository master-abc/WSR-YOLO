from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from .audit_dataset import audit_dataset
    from .common import (
        REPRODUCIBILITY_DIR,
        PROJECT_DIR,
        atomic_json_dump,
        class_names,
        environment_snapshot,
        file_hashes,
        load_yaml,
        dataset_sources,
        resolve_path,
        sha256_file,
        sha256_text,
    )
    from .split_dataset import prepare_split
    from .yolo_to_coco import convert_dataset
    from .coco_evaluator import evaluate_coco_predictions, predict_yolo_to_coco
    from .pretrained import transfer_pretrained
except ImportError:
    from audit_dataset import audit_dataset
    from common import (
        REPRODUCIBILITY_DIR,
        PROJECT_DIR,
        atomic_json_dump,
        class_names,
        environment_snapshot,
        file_hashes,
        load_yaml,
        dataset_sources,
        resolve_path,
        sha256_file,
        sha256_text,
    )
    from split_dataset import prepare_split
    from yolo_to_coco import convert_dataset
    from coco_evaluator import evaluate_coco_predictions, predict_yolo_to_coco
    from pretrained import transfer_pretrained


DEFAULT_CONFIG = REPRODUCIBILITY_DIR / "protocol.yaml"


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_yaml(path)
    protocol["_path"] = path.resolve()
    protocol["_output_root"] = resolve_path(protocol.get("output_root", "generated"), path.parent)
    return protocol


def enabled_datasets(protocol: dict[str, Any], selected: str | None = None):
    for name, config in protocol["datasets"].items():
        if selected and name != selected:
            continue
        if not config.get("enabled", True):
            if selected:
                raise ValueError(f"Dataset '{name}' is disabled: {config.get('note', 'no reason given')}")
            continue
        yield name, config


def generated_dataset_yaml(protocol: dict[str, Any], name: str) -> Path:
    return protocol["_output_root"] / "datasets" / name / "dataset.yaml"


def prepare_datasets(protocol: dict[str, Any], selected: str | None, coco: bool) -> None:
    for name, config in enabled_datasets(protocol, selected):
        source = resolve_path(config["source"], protocol["_path"].parent)
        output = protocol["_output_root"] / "datasets" / name
        print(f"[prepare] {name}: {source}")
        dataset_yaml = prepare_split(
            source,
            output,
            config.get("split_mode", "preserve"),
            int(protocol.get("split_seed", 2026)),
            float(config.get("train_fraction", 0.70)),
            float(config.get("val_fraction", 0.15)),
        )
        report = audit_dataset(dataset_yaml, hash_images=True)
        atomic_json_dump(report, output / "audit.json")
        if report["fatal"]:
            raise RuntimeError(f"Dataset audit failed for {name}: {report['fatal']}")
        if coco:
            convert_dataset(dataset_yaml, protocol["_output_root"] / "coco" / name)
        print(
            f"[ready] {name}: "
            + ", ".join(f"{split}={values['images']}" for split, values in report["splits"].items())
        )


def audit_datasets(protocol: dict[str, Any], selected: str | None, skip_hashes: bool) -> int:
    exit_code = 0
    for name, _ in enabled_datasets(protocol, selected):
        dataset_yaml = generated_dataset_yaml(protocol, name)
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Prepare the dataset first: {dataset_yaml}")
        report = audit_dataset(dataset_yaml, hash_images=not skip_hashes)
        # A quick structural check must never downgrade the full hash audit
        # required by formal training.
        report_name = "audit_quick.json" if skip_hashes else "audit.json"
        atomic_json_dump(report, dataset_yaml.parent / report_name)
        status = "FAIL" if report["fatal"] else "PASS"
        mode = "quick" if skip_hashes else "full"
        print(f"[{status} {mode}] {name}: {report['fatal'] or 'no fatal issue'}")
        exit_code |= int(bool(report["fatal"]))
    return exit_code


def controlled_models(protocol: dict[str, Any], selected: str | None = None):
    for model in protocol["controlled_track"]["models"]:
        if selected and model["name"] != selected:
            continue
        yield model


def controlled_seeds(protocol: dict[str, Any], dataset_config: dict[str, Any]) -> list[int]:
    if dataset_config.get("role") == "primary_industrial":
        return [int(value) for value in protocol["controlled_track"]["seeds"]]
    return [
        int(value)
        for value in protocol["controlled_track"].get(
            "secondary_seeds", protocol["controlled_track"]["seeds"]
        )
    ]


def resolve_architecture(model_config: dict[str, Any], protocol: dict[str, Any]) -> str:
    value = str(model_config["architecture"])
    if value.startswith("yolo") and not any(separator in value for separator in ("/", "\\")):
        return value
    return str(resolve_path(value, protocol["_path"].parent))


def training_plan(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    plan = []
    for dataset, dataset_config in enabled_datasets(protocol):
        for model in controlled_models(protocol):
            for seed in controlled_seeds(protocol, dataset_config):
                plan.append(
                    {
                        "track": "controlled",
                        "dataset": dataset,
                        "model": model["name"],
                        "seed": seed,
                        "architecture": resolve_architecture(model, protocol),
                        "data": str(generated_dataset_yaml(protocol, dataset)),
                    }
                )
    return plan


def import_ultralytics():
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from algorithm.register import register_custom_modules

    register_custom_modules()
    from ultralytics import YOLO

    return YOLO


def model_complexity(yolo_model, imgsz: int) -> dict[str, Any]:
    model = yolo_model.model
    requires_grad = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    result: dict[str, Any] = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        # Loaded Ultralytics inference checkpoints can set requires_grad=False.
        # Keep this as measurement metadata, not as a trainable-parameter claim.
        "requires_grad_parameters_at_measurement": requires_grad,
    }
    try:
        from ultralytics.utils.torch_utils import get_flops

        result["gflops"] = float(get_flops(model, imgsz=imgsz))
    except Exception as exc:
        result["gflops"] = None
        result["gflops_error"] = str(exc)
    return result


def assert_formal_run_provenance(protocol: dict[str, Any], smoke: bool) -> dict[str, Any]:
    snapshot = environment_snapshot()
    policy = protocol.get("reproducibility", {})
    if not smoke and not snapshot.get("git_commit"):
        raise RuntimeError("Formal runs require a resolvable Git commit")
    if not smoke and policy.get("require_clean_git_for_formal", True) and snapshot.get("git_dirty"):
        preview = "\n".join(snapshot.get("git_status", [])[:12])
        raise RuntimeError(
            "Formal runs require a clean Git worktree so result provenance is truthful. "
            "Commit the experiment code first, or explicitly disable the policy in protocol.yaml.\n"
            f"Dirty paths:\n{preview}"
        )
    return snapshot


def assert_frozen_pilot_selection(
    protocol: dict[str, Any], smoke: bool
) -> dict[str, Any] | None:
    if smoke:
        return None
    path = REPRODUCIBILITY_DIR / "selection" / "pilot_decision.json"
    if not path.exists():
        raise RuntimeError(
            "Formal test runs require a frozen validation-only pilot decision. "
            "Run `python -m experiment.wsr.pilot freeze`, commit it, then retry."
        )
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("status") != "PASS":
        raise RuntimeError("The frozen pilot gate did not pass")
    if decision.get("selection_split") != "val" or decision.get("test_evaluated") is not False:
        raise RuntimeError("The frozen pilot decision is not validation-only")
    if decision.get("protocol_sha256") != sha256_file(protocol["_path"]):
        raise RuntimeError("The experiment protocol changed after pilot selection; rerun the pilot")
    proposed = next(
        (
            item["name"]
            for item in protocol["controlled_track"]["models"]
            if item.get("role") == "proposed"
        ),
        None,
    )
    if decision.get("formal_model") != proposed:
        raise RuntimeError("Frozen pilot selection does not match the registered proposed model")
    return decision


def metrics_payload(metrics, names: list[str]) -> dict[str, Any]:
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    all_ap = metrics.box.all_ap
    ap75 = float(all_ap[:, 5].mean()) if getattr(all_ap, "size", 0) else None
    per_class: dict[str, float] = {}
    indices = list(getattr(metrics.box, "ap_class_index", []))
    if getattr(all_ap, "size", 0):
        for row, class_id in enumerate(indices):
            per_class[names[int(class_id)]] = float(all_ap[row].mean())
    speed = {key: float(value) for key, value in getattr(metrics, "speed", {}).items()}
    return {
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": ap75,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "per_class_ap50_95": per_class,
        "speed_ms_per_image": speed,
    }


def train_one(
    protocol: dict[str, Any],
    dataset: str,
    model_name: str,
    seed: int,
    device: str,
    smoke: bool,
    force: bool,
    resume: bool,
) -> Path:
    model_config = next(controlled_models(protocol, model_name), None)
    if model_config is None:
        raise KeyError(f"Unknown controlled model: {model_name}")
    dataset_config = next((config for name, config in enabled_datasets(protocol, dataset) if name == dataset), None)
    if dataset_config is None:
        raise KeyError(f"Unknown or disabled dataset: {dataset}")
    if not smoke and seed not in controlled_seeds(protocol, dataset_config):
        raise ValueError(
            f"Seed {seed} is outside the registered controlled seeds for {dataset}: "
            f"{controlled_seeds(protocol, dataset_config)}"
        )
    data_yaml = generated_dataset_yaml(protocol, dataset)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Prepare dataset first: {data_yaml}")
    audit = audit_dataset(data_yaml, hash_images=False)
    if audit["fatal"]:
        raise RuntimeError(f"Refusing to train on a failed dataset audit: {audit['fatal']}")
    if not smoke:
        stored_audit_path = data_yaml.parent / "audit.json"
        if not stored_audit_path.exists():
            raise FileNotFoundError(
                f"Formal runs require a full content-hash audit: {stored_audit_path}"
            )
        stored_audit = json.loads(stored_audit_path.read_text(encoding="utf-8"))
        if stored_audit.get("fatal") or not stored_audit.get("content_hashes_checked"):
            raise RuntimeError(
                "Re-run `python -m experiment.wsr.run audit --dataset "
                f"{dataset}` without --skip-hashes before a formal run"
            )
    start_environment = assert_formal_run_provenance(protocol, smoke)
    pilot_selection = assert_frozen_pilot_selection(protocol, smoke)

    if smoke:
        run_dir = protocol["_output_root"] / "smoke" / dataset / model_name / f"seed_{seed}"
    else:
        run_dir = protocol["_output_root"] / "runs" / "controlled" / dataset / model_name / f"seed_{seed}"
    result_file = run_dir / "standardized_result.json"
    if result_file.exists() and not force:
        print(f"[skip] completed result exists: {result_file}")
        return result_file

    train_args = copy.deepcopy(protocol["controlled_track"]["train"])
    if smoke:
        train_args.update({"epochs": 2, "imgsz": 320, "batch": 2, "workers": 0, "patience": 2})
    architecture = resolve_architecture(model_config, protocol)
    pretrained = str(model_config.get("pretrained", ""))
    data, _, _ = dataset_sources(data_yaml)
    names = class_names(data)
    YOLO = import_ultralytics()
    pretrained_transfer: dict[str, Any] | None = None
    architecture_definition_sha256: str | None = None

    last_weights = run_dir / "weights" / "last.pt"
    manifest_file = run_dir / "run_manifest.json"
    protocol_hash = sha256_file(protocol["_path"])
    data_yaml_hash = sha256_file(data_yaml)
    selection_path = REPRODUCIBILITY_DIR / "selection" / "pilot_decision.json"
    selection_hash = sha256_file(selection_path) if selection_path.exists() else None
    source_files = [
        PROJECT_DIR / "algorithm" / "dwgsa.py",
        PROJECT_DIR / "algorithm" / "register.py",
        Path(__file__),
        REPRODUCIBILITY_DIR / "common.py",
        REPRODUCIBILITY_DIR / "pretrained.py",
        REPRODUCIBILITY_DIR / "coco_evaluator.py",
        protocol["_path"],
    ]
    architecture_path = Path(architecture)
    if architecture_path.is_file():
        source_files.append(architecture_path)
    if selection_path.exists():
        source_files.append(selection_path)

    if resume:
        if not last_weights.exists():
            raise FileNotFoundError(f"--resume requested but checkpoint is missing: {last_weights}")
        if not manifest_file.exists():
            raise FileNotFoundError(f"Cannot audit resume without run manifest: {manifest_file}")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        expected = {
            "dataset": dataset,
            "model": model_name,
            "seed": int(seed),
            "protocol_sha256": protocol_hash,
            "data_yaml_sha256": data_yaml_hash,
            "pilot_selection_sha256": selection_hash,
            "git_commit": start_environment.get("git_commit"),
        }
        observed = {
            "dataset": manifest.get("dataset"),
            "model": manifest.get("model"),
            "seed": manifest.get("seed"),
            "protocol_sha256": manifest.get("protocol_sha256"),
            "data_yaml_sha256": manifest.get("data_yaml_sha256"),
            "pilot_selection_sha256": manifest.get("pilot_selection_sha256"),
            "git_commit": manifest.get("environment_at_start", {}).get("git_commit"),
        }
        if observed != expected:
            raise RuntimeError(
                "Resume provenance mismatch; refusing to mix code, data, protocol or selection. "
                f"expected={expected}, observed={observed}"
            )
        start_environment = manifest["environment_at_start"]
        architecture_definition_sha256 = manifest.get("architecture_definition_sha256")
        transfer_file = run_dir / "pretrained_transfer.json"
        if pretrained and not transfer_file.exists():
            raise FileNotFoundError(f"Resume is missing pretrained transfer report: {transfer_file}")
        pretrained_transfer = (
            json.loads(transfer_file.read_text(encoding="utf-8")) if transfer_file.exists() else None
        )
        yolo = YOLO(str(last_weights))
        train_call = {"resume": True}
    else:
        yolo = YOLO(architecture)
        architecture_definition_sha256 = sha256_text(
            json.dumps(yolo.model.yaml, sort_keys=True, ensure_ascii=False, default=str)
        )
        if model_config.get("role") == "proposed" and pilot_selection:
            if (
                architecture_definition_sha256
                != pilot_selection.get("candidate_architecture_definition_sha256")
            ):
                raise RuntimeError(
                    "The proposed architecture differs from the validation-selected pilot model"
                )
        if pretrained:
            minimum_fraction = float(
                model_config.get(
                    "minimum_pretrained_fraction",
                    protocol.get("reproducibility", {}).get("minimum_pretrained_fraction", 0.99),
                )
            )
            pretrained_transfer = transfer_pretrained(
                yolo, pretrained, YOLO, minimum_parameter_fraction=minimum_fraction
            )
            atomic_json_dump(pretrained_transfer, run_dir / "pretrained_transfer.json")
        train_call = dict(train_args)
        train_call.update(
            {
                "data": str(data_yaml),
                "seed": int(seed),
                "device": device,
                "project": str(run_dir.parent),
                "name": run_dir.name,
                "exist_ok": True,
                "verbose": True,
            }
        )
        manifest = {
            "schema_version": 1,
            "dataset": dataset,
            "model": model_name,
            "seed": int(seed),
            "smoke": bool(smoke),
            "architecture": architecture,
            "architecture_definition_sha256": architecture_definition_sha256,
            "pretrained": pretrained,
            "protocol_sha256": protocol_hash,
            "data_yaml_sha256": data_yaml_hash,
            "pilot_selection_sha256": selection_hash,
            "train_args": train_args,
            "environment_at_start": start_environment,
            "source_files_sha256": file_hashes(source_files),
        }
        atomic_json_dump(manifest, manifest_file)
    print(f"[train] {dataset}/{model_name}/seed={seed} -> {run_dir}")
    yolo.train(**train_call)

    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Training ended without best weights: {best_weights}")
    # The YOLO wrapper retains its trainer, optimizer and training model.
    # Release them before loading a second model for full-set prediction.
    del yolo
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    evaluated = YOLO(str(best_weights))
    eval_config = protocol["evaluation"]
    coco_root = protocol["_output_root"] / "coco" / dataset
    annotations = coco_root / "annotations" / "instances_test.json"
    if not annotations.exists():
        convert_dataset(data_yaml, coco_root)
    predictions = predict_yolo_to_coco(
        evaluated,
        annotations,
        coco_root / "test",
        run_dir / "test_predictions.json",
        int(train_args["imgsz"]),
        int(eval_config.get("batch", 1)),
        device,
        float(eval_config["conf"]),
        float(eval_config["iou"]),
        int(eval_config["prediction_max_det"]),
    )
    coco_report = evaluate_coco_predictions(annotations, predictions, int(eval_config["max_det"]))
    atomic_json_dump(coco_report, run_dir / "coco_evaluation.json")
    payload = {
        "schema_version": 2,
        "track": "smoke" if smoke else "controlled",
        "smoke": bool(smoke),
        "dataset": dataset,
        "model": model_name,
        "role": model_config.get("role"),
        "seed": int(seed),
        "split_seed": int(protocol["split_seed"]),
        "data_yaml": str(data_yaml.resolve()),
        "data_yaml_sha256": sha256_file(data_yaml),
        "architecture": architecture,
        "architecture_definition_sha256": architecture_definition_sha256,
        "pretrained": pretrained,
        "pretrained_transfer": pretrained_transfer,
        "pilot_selection": pilot_selection,
        "weights": str(best_weights.resolve()),
        "weights_sha256": sha256_file(best_weights),
        "train_args": train_args,
        "evaluation": eval_config,
        "metrics": coco_report["metrics"],
        "unified_evaluation": coco_report,
        # Measured after the dataset-specific detection head has been built, so
        # baseline and custom YAMLs use the same number of classes.
        "complexity": model_complexity(evaluated, int(train_args["imgsz"])),
        "environment_at_start": start_environment,
        "environment": environment_snapshot(),
        "protocol_sha256": protocol_hash,
        "source_files_sha256": file_hashes(source_files),
    }
    atomic_json_dump(payload, result_file)
    print(f"[done] COCO AP50-95={payload['metrics']['map50_95']:.4f}: {result_file}")
    return result_file


def main() -> int:
    parser = argparse.ArgumentParser(description="WSR-YOLO paper experiment orchestrator")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Print the complete controlled experiment matrix")
    plan_parser.add_argument("--json", action="store_true")

    prepare_parser = subparsers.add_parser("prepare", help="Create fixed splits and audit them")
    prepare_parser.add_argument("--dataset")
    prepare_parser.add_argument("--coco", action="store_true", help="Also materialize exact COCO copies")

    audit_parser = subparsers.add_parser("audit", help="Re-run leakage and annotation checks")
    audit_parser.add_argument("--dataset")
    audit_parser.add_argument("--skip-hashes", action="store_true")

    train_parser = subparsers.add_parser("train", help="Train and test one controlled run")
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--model", required=True)
    train_parser.add_argument("--seed", required=True, type=int)
    train_parser.add_argument("--device", default="0")
    train_parser.add_argument("--smoke", action="store_true")
    train_parser.add_argument("--force", action="store_true")
    train_parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    protocol = load_protocol(args.config.resolve())
    if args.command == "plan":
        plan = training_plan(protocol)
        if args.json:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
        else:
            for item in plan:
                print(f"{item['dataset']:<16} {item['model']:<24} seed={item['seed']}")
            print(f"Total controlled runs: {len(plan)}")
        return 0
    if args.command == "prepare":
        prepare_datasets(protocol, args.dataset, args.coco)
        return 0
    if args.command == "audit":
        return audit_datasets(protocol, args.dataset, args.skip_hashes)
    if args.command == "train":
        train_one(
            protocol,
            args.dataset,
            args.model,
            args.seed,
            args.device,
            args.smoke,
            args.force,
            args.resume,
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
