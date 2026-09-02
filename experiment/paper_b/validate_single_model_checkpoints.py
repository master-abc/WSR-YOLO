"""Select a paired-reference checkpoint without consulting the test split."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from .single_model_operating_point import choose_checkpoint, select_from_files
except ImportError:
    from experiment.paper_b.single_model_operating_point import (
        choose_checkpoint,
        select_from_files,
    )


def _run(command: list[str], project_root: Path) -> None:
    subprocess.run(command, cwd=project_root, check=True)


def validate_checkpoints(
    run_dir: Path,
    annotations: Path,
    positive_images: Path,
    negative_images: Path,
    negative_list: Path,
    input_manifest: Path,
    output_dir: Path,
    maximum_board_fpr: float,
    device: str,
    batch: int,
) -> dict:
    project_root = Path(__file__).resolve().parents[2]
    checkpoints = sorted(
        (run_dir / "weights").glob("epoch*.pt"),
        key=lambda path: int(path.stem.removeprefix("epoch")),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoints found under {run_dir / 'weights'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    policies: list[Path] = []
    for checkpoint in checkpoints:
        stem = checkpoint.stem
        predictions = output_dir / f"{stem}_positive_predictions.json"
        negative_audit = output_dir / f"{stem}_negative_audit.json"
        policy = output_dir / f"{stem}_policy.json"
        _run(
            [
                sys.executable,
                "-m",
                "experiment.paper_b.coco_evaluator",
                "predict-yolo",
                "--weights",
                str(checkpoint.resolve()),
                "--annotations",
                str(annotations.resolve()),
                "--image-dir",
                str(positive_images.resolve()),
                "--output",
                str(predictions.resolve()),
                "--conf",
                "0.001",
                "--iou",
                "0.7",
                "--batch",
                str(batch),
                "--device",
                device,
            ],
            project_root,
        )
        _run(
            [
                sys.executable,
                "-m",
                "experiment.paper_b.false_positive",
                "--weights",
                str(checkpoint.resolve()),
                "--images",
                str(negative_images.resolve()),
                "--image-list",
                str(negative_list.resolve()),
                "--expected-images",
                str(
                    sum(
                        bool(line.strip())
                        for line in negative_list.read_text(encoding="utf-8").splitlines()
                    )
                ),
                "--output",
                str(negative_audit.resolve()),
                "--thresholds",
                "0.001",
                "0.05",
                "0.1",
                "0.2",
                "0.3",
                "0.4",
                "0.5",
                "0.6",
                "0.7",
                "0.8",
                "0.9",
                "--batch",
                str(batch),
                "--device",
                device,
            ],
            project_root,
        )
        selection = select_from_files(
            annotations.resolve(),
            predictions.resolve(),
            negative_audit.resolve(),
            checkpoint.resolve(),
            policy.resolve(),
            maximum_board_fpr,
            input_manifest.resolve(),
        )
        overall = selection["positive_validation"]["overall"]
        negative = selection["negative_validation"]
        print(
            json.dumps(
                {
                    "checkpoint": checkpoint.name,
                    "positive_f1": overall["f1"],
                    "positive_precision": overall["precision"],
                    "positive_recall": overall["recall"],
                    "negative_board_fpr": negative["board_false_positive_rate"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        policies.append(policy)
    return choose_checkpoint(policies, output_dir / "frozen_policy.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validation-only checkpoint and class-threshold selection"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--positive-images", type=Path, required=True)
    parser.add_argument("--negative-images", type=Path, required=True)
    parser.add_argument("--negative-list", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-board-fpr", type=float, default=0.005)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    result = validate_checkpoints(
        args.run_dir.resolve(),
        args.annotations.resolve(),
        args.positive_images.resolve(),
        args.negative_images.resolve(),
        args.negative_list.resolve(),
        args.input_manifest.resolve(),
        args.output_dir.resolve(),
        args.maximum_board_fpr,
        args.device,
        args.batch,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
