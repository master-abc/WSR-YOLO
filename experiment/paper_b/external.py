from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

try:
    from .common import PAPER_B_DIR, PROJECT_DIR, class_names, load_yaml, resolve_path
    from .run import generated_dataset_yaml, load_protocol
except ImportError:
    from common import PAPER_B_DIR, PROJECT_DIR, class_names, load_yaml, resolve_path
    from run import generated_dataset_yaml, load_protocol


def registry(protocol: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(protocol["sota_track"]["external_registry"], protocol["_path"].parent)
    return load_yaml(path)


def bootstrap_commands(models: dict[str, Any], repos_root: Path) -> list[str]:
    commands = [f"New-Item -ItemType Directory -Force -Path '{repos_root}' | Out-Null"]
    for name, model in models.items():
        if model["family"] == "ultralytics_native":
            continue
        target = repos_root / name
        commands.extend(
            [
                f"git clone '{model['repository']}' '{target}'",
                f"git -C '{target}' checkout --detach '{model['commit']}'",
            ]
        )
    return commands


def assert_revision(repo: Path, expected: str) -> None:
    if not (repo / ".git").exists():
        raise FileNotFoundError(f"Clone the pinned repository first: {repo}")
    actual = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if actual != expected:
        raise RuntimeError(f"Revision mismatch for {repo}: expected {expected}, found {actual}")


def yaml_path(path: Path) -> str:
    return path.resolve().as_posix()


def detr_overlays(
    name: str,
    model: dict[str, Any],
    repo: Path,
    protocol: dict[str, Any],
    dataset: str,
    num_classes: int,
    seed: int,
) -> tuple[Path, Path, Path]:
    workdir = repo / model.get("workdir", "")
    overlay_dir = workdir / "configs" / "paper_b"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    original = workdir / model["config"]
    relative_original = Path(os.path.relpath(original, overlay_dir)).as_posix()
    coco = protocol["_output_root"] / "coco" / dataset
    output = protocol["_output_root"] / "runs" / "sota" / dataset / name / f"seed_{seed}"

    common = {
        "num_classes": num_classes,
        "remap_mscoco_category": False,
        "output_dir": yaml_path(output),
        "train_dataloader": {
            "dataset": {
                "img_folder": yaml_path(coco / "train"),
                "ann_file": yaml_path(coco / "annotations" / "instances_train.json"),
            }
        },
        "val_dataloader": {
            "dataset": {
                "img_folder": yaml_path(coco / "val"),
                "ann_file": yaml_path(coco / "annotations" / "instances_val.json"),
            }
        },
    }
    # 单卡 24GB 放不下官方多卡 total_batch_size 的模型,在注册表里显式降档
    if model.get("train_total_batch_size"):
        common["train_dataloader"]["total_batch_size"] = model["train_total_batch_size"]
    train_payload = {"__include__": [relative_original], **common}
    train_config = overlay_dir / f"{dataset}_{name}_seed{seed}_train.yml"
    train_config.write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    test_payload = {
        "__include__": [f"./{train_config.name}"],
        "val_dataloader": {
            "dataset": {
                "img_folder": yaml_path(coco / "test"),
                "ann_file": yaml_path(coco / "annotations" / "instances_test.json"),
            }
        },
    }
    test_config = overlay_dir / f"{dataset}_{name}_seed{seed}_test.yml"
    test_config.write_text(yaml.safe_dump(test_payload, sort_keys=False), encoding="utf-8")
    return workdir, train_config, test_config


def materialize(protocol: dict[str, Any], dataset: str, repos_root: Path, device: str) -> Path:
    dataset_yaml = generated_dataset_yaml(protocol, dataset)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Prepare the dataset first: {dataset_yaml}")
    coco = protocol["_output_root"] / "coco" / dataset / "conversion.json"
    if not coco.exists():
        raise FileNotFoundError(
            f"COCO conversion is missing. Run: python -m experiment.paper_b.run prepare --dataset {dataset} --coco"
        )
    num_classes = len(class_names(load_yaml(dataset_yaml)))
    registered = registry(protocol)["models"]
    seeds = protocol["sota_track"]["seeds"]
    commands: list[str] = ["$ErrorActionPreference = 'Stop'"]
    adapter = PAPER_B_DIR / "external_ultralytics.py"
    rfdetr_adapter = PAPER_B_DIR / "external_rfdetr.py"

    for name, model in registered.items():
        family = model["family"]
        if family == "ultralytics_native":
            repo = PROJECT_DIR
        else:
            repo = (repos_root / name).resolve()
            assert_revision(repo, model["commit"])
        if family in {"ultralytics_fork", "ultralytics_native"}:
            for seed in seeds:
                output = protocol["_output_root"] / "runs" / "sota" / dataset / name / f"seed_{seed}"
                commands.append(
                    f"Push-Location '{repo}'; python '{adapter}' --model '{model['model']}' "
                    f"--data '{dataset_yaml}' --output '{output}' --name '{name}' "
                    f"--seed {seed} --device '{device}'; Pop-Location"
                )
            continue
        if family == "rfdetr":
            coco_root = protocol["_output_root"] / "coco" / dataset
            for seed in seeds:
                output = protocol["_output_root"] / "runs" / "sota" / dataset / name / f"seed_{seed}"
                commands.append(
                    f"Push-Location '{repo}'; python '{rfdetr_adapter}' --coco-root '{coco_root}' "
                    f"--output '{output}' --seed {seed} --device cuda; Pop-Location"
                )
            continue

        pretrained = None
        if model.get("pretrained_file"):
            pretrained = repo / "pretrained" / model["pretrained_file"]
            commands.append(f"New-Item -ItemType Directory -Force -Path '{pretrained.parent}' | Out-Null")
            if model.get("pretrained_url"):
                commands.append(
                    f"if (-not (Test-Path -LiteralPath '{pretrained}')) {{ "
                    f"Invoke-WebRequest -UseBasicParsing -Uri '{model['pretrained_url']}' -OutFile '{pretrained}' }}"
                )
            elif model.get("pretrained_gdrive_id"):
                commands.append(
                    f"if (-not (Test-Path -LiteralPath '{pretrained}')) {{ "
                    f"python -m gdown '{model['pretrained_gdrive_id']}' -O '{pretrained}' }}"
                )
        for seed in seeds:
            workdir, train_config, test_config = detr_overlays(
                name, model, repo, protocol, dataset, num_classes, seed
            )
            entrypoint = workdir / model["entrypoint"]
            tune = f" -t '{pretrained}'" if pretrained else ""
            commands.append(
                f"Push-Location '{workdir}'; python '{entrypoint}' -c '{train_config}' "
                f"--use-amp --seed={seed}{tune}; Pop-Location"
            )
            commands.append(
                f"# Test seed {seed} only after selecting its best validation checkpoint:\n"
                f"# Push-Location '{workdir}'; python '{entrypoint}' -c '{test_config}' "
                f"--test-only -r '<BEST_VALIDATION_CHECKPOINT>'; Pop-Location"
            )

    output = protocol["_output_root"] / "external" / dataset / "commands.ps1"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(commands) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Pin and materialize official SOTA comparison jobs")
    parser.add_argument("--config", type=Path, default=PAPER_B_DIR / "paper_b.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("bootstrap", help="Print exact clone/checkout commands")
    plan_parser.add_argument("--repos-root", type=Path, required=True)
    materialize_parser = subparsers.add_parser("materialize", help="Write official-recipe job configs")
    materialize_parser.add_argument("--dataset", required=True)
    materialize_parser.add_argument("--repos-root", type=Path, required=True)
    materialize_parser.add_argument("--device", default="0")
    args = parser.parse_args()
    protocol = load_protocol(args.config.resolve())
    models = registry(protocol)["models"]
    if args.command == "bootstrap":
        print("\n".join(bootstrap_commands(models, args.repos_root.resolve())))
        return 0
    output = materialize(protocol, args.dataset, args.repos_root.resolve(), args.device)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
