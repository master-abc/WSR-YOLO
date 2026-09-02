"""Validation-locked single-model refinement for PCB false alarms and misses."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterator

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

try:
    from .common import atomic_json_dump, environment_snapshot, sha256_file
except ImportError:
    from experiment.wsr.common import (
        atomic_json_dump,
        environment_snapshot,
        sha256_file,
    )


@contextlib.contextmanager
def asymmetric_detection_loss(
    gamma_negative: float,
    gamma_positive: float,
    negative_weight: float,
) -> Iterator[None]:
    """Temporarily install asymmetric BCE in Ultralytics DetectionModel.

    A class-level patch is used only while the trainer constructs its criterion.
    It avoids storing an unpicklable bound callback inside checkpoints; inference
    checkpoints therefore remain ordinary Ultralytics models.
    """

    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import v8DetectionLoss

    from algorithm.asymmetric_loss import AsymmetricFocalBCE

    original = DetectionModel.init_criterion

    def init_criterion(model: Any) -> Any:
        criterion = original(model)
        if not isinstance(criterion, v8DetectionLoss):
            raise TypeError(
                "Single-model refinement currently supports standard detection "
                f"models, received {type(criterion).__name__}"
            )
        criterion.bce = AsymmetricFocalBCE(
            gamma_negative=gamma_negative,
            gamma_positive=gamma_positive,
            negative_weight=negative_weight,
        )
        return criterion

    DetectionModel.init_criterion = init_criterion
    try:
        yield
    finally:
        DetectionModel.init_criterion = original


def train(
    data_yaml: Path,
    initial_weights: Path,
    output: Path,
    seed: int,
    device: str,
    epochs: int,
    lr0: float,
    batch: int,
    workers: int,
    gamma_negative: float,
    gamma_positive: float,
    negative_weight: float,
    cls_gain: float,
    save_period: int,
    semantic_channels: bool,
) -> dict[str, Any]:
    from algorithm.register import register_custom_modules

    register_custom_modules()
    from ultralytics import YOLO

    output.mkdir(parents=True, exist_ok=True)
    args: dict[str, Any] = {
        "data": str(data_yaml.resolve()),
        "epochs": int(epochs),
        "imgsz": 640,
        "batch": int(batch),
        "nbs": 64,
        "optimizer": "SGD",
        "lr0": float(lr0),
        "lrf": 0.05,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 1.0,
        "cos_lr": True,
        "amp": True,
        "deterministic": True,
        "patience": 0,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "workers": int(workers),
        "plots": True,
        "seed": int(seed),
        "device": device,
        "project": str(output.parent.resolve()),
        "name": output.name,
        "exist_ok": True,
        "verbose": True,
        "cls": float(cls_gain),
        "save_period": int(save_period),
    }
    if semantic_channels:
        # Paired-difference channels encode dark residual, bright residual and
        # magnitude. Colour jitter or channel swapping would destroy those
        # meanings rather than provide a valid augmentation.
        args.update(
            {
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
                "bgr": 0.0,
                "erasing": 0.0,
            }
        )
    manifest = {
        "schema_version": 1,
        "track": "single_model_false_alarm_and_miss_refinement",
        "selection_split": "positive_and_defect_free_validation",
        "test_evaluated": False,
        "seed": int(seed),
        "initial_weights": str(initial_weights.resolve()),
        "initial_weights_sha256": sha256_file(initial_weights),
        "data_yaml": str(data_yaml.resolve()),
        "data_yaml_sha256": sha256_file(data_yaml),
        "loss": {
            "name": "asymmetric_focal_bce",
            "gamma_negative": float(gamma_negative),
            "gamma_positive": float(gamma_positive),
            "negative_weight": float(negative_weight),
            "positive_gradient_policy": "unfocused" if gamma_positive == 0 else "focused",
        },
        "train_args": args,
        "semantic_channels": bool(semantic_channels),
        "environment_at_start": environment_snapshot(),
    }
    atomic_json_dump(manifest, output / "run_manifest.json")

    model = YOLO(str(initial_weights.resolve()))
    with asymmetric_detection_loss(
        gamma_negative, gamma_positive, negative_weight
    ):
        model.train(**args)

    best = output / "weights" / "best.pt"
    last = output / "weights" / "last.pt"
    history_path = output / "results.csv"
    if not best.is_file() or not last.is_file() or not history_path.is_file():
        raise FileNotFoundError(f"Incomplete refinement run: {output}")
    with history_path.open("r", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    metric_key = "metrics/mAP50-95(B)"
    best_row = max(history, key=lambda row: float(row[metric_key]))
    result = {
        **manifest,
        "completed_epochs": len(history),
        "best_epoch": int(best_row["epoch"]),
        "best_mixed_validation_map50_95": float(best_row[metric_key]),
        "weights": str(best.resolve()),
        "weights_sha256": sha256_file(best),
        "last_weights": str(last.resolve()),
        "last_weights_sha256": sha256_file(last),
        "history_sha256": sha256_file(history_path),
        "environment": environment_snapshot(),
    }
    atomic_json_dump(result, output / "single_model_refinement_result.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refine one detector against hard negatives without sacrificing positives"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr0", type=float, default=5e-4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gamma-negative", type=float, default=2.0)
    parser.add_argument("--gamma-positive", type=float, default=0.0)
    parser.add_argument("--negative-weight", type=float, default=2.0)
    parser.add_argument("--cls-gain", type=float, default=0.75)
    parser.add_argument("--save-period", type=int, default=5)
    parser.add_argument(
        "--semantic-channels",
        action="store_true",
        help="Disable colour augmentations for non-RGB semantic input channels",
    )
    args = parser.parse_args()
    payload = train(
        args.data.resolve(),
        args.initial_weights.resolve(),
        args.output.resolve(),
        args.seed,
        args.device,
        args.epochs,
        args.lr0,
        args.batch,
        args.workers,
        args.gamma_negative,
        args.gamma_positive,
        args.negative_weight,
        args.cls_gain,
        args.save_period,
        args.semantic_channels,
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
