from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PAPER_B_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PAPER_B_DIR.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from experiment.paper_b.coco_evaluator import evaluate_coco_predictions
from experiment.paper_b.common import atomic_json_dump, environment_snapshot, sha256_file

try:
    import resource
except ImportError:  # pragma: no cover - resource is Unix-only
    resource = None


def raise_nofile_limit() -> None:
    """Let multi-worker PyTorch loaders pass tensors without exhausting file descriptors."""

    if resource is None:
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def prediction_rows(evaluator: Any) -> list[dict[str, Any]]:
    """Extract framework-neutral COCO rows from an official DETR evaluator."""

    try:
        annotations = evaluator.coco_eval["bbox"].cocoDt.dataset["annotations"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise RuntimeError("The official DETR evaluator did not retain bbox predictions") from exc
    rows = []
    for annotation in annotations:
        bbox = annotation.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("A DETR prediction has an invalid COCO bbox")
        rows.append(
            {
                "image_id": int(annotation["image_id"]),
                "category_id": int(annotation["category_id"]),
                "bbox": [float(value) for value in bbox],
                "score": float(annotation["score"]),
            }
        )
    return rows


def git_commit(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def run_official_evaluator(
    family: str, repository: Path, config: Path, checkpoint: Path, device: str
):
    os.chdir(repository)
    sys.path.insert(0, str(repository))
    if family == "deim":
        from engine.core import YAMLConfig
        from engine.data import CocoEvaluator
        from engine.solver import TASKS
    else:
        from src.core import YAMLConfig
        from src.data import CocoEvaluator
        from src.solver import TASKS

    cfg = YAMLConfig(str(config), resume=str(checkpoint), device=device, use_amp=True)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    captured: list[dict[str, Any]] = []
    original_update = CocoEvaluator.update

    def capture_update(evaluator, predictions):
        value = original_update(evaluator, predictions)
        # Official evaluators replace cocoDt on every batch. Capture each batch
        # immediately; reading cocoDt after val() would retain only the final one.
        captured.extend(prediction_rows(evaluator))
        return value

    CocoEvaluator.update = capture_update
    try:
        solver.val()
    finally:
        CocoEvaluator.update = original_update
    return solver, captured


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export official RT-DETR/D-FINE/DEIM predictions for unified COCOeval"
    )
    parser.add_argument("--family", choices=("rtdetrv2", "dfine", "deim"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    raise_nofile_limit()

    repository = args.repo.resolve()
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    coco_root = args.coco_root.resolve()
    output = args.output.resolve()
    for path in (repository, config, checkpoint, coco_root):
        if not path.exists():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)

    solver, rows = run_official_evaluator(
        args.family, repository, config, checkpoint, args.device
    )
    predictions = output / "test_predictions.json"
    atomic_json_dump(rows, predictions)
    annotations = coco_root / "annotations" / "instances_test.json"
    evaluation = evaluate_coco_predictions(annotations, predictions, max_det=100)
    network = solver.ema.module if getattr(solver, "ema", None) is not None else solver.model
    payload = {
        "schema_version": 2,
        "track": "sota",
        "dataset": coco_root.name,
        "model": args.model,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "weights_sha256": sha256_file(checkpoint),
        "protocol_sha256": sha256_file(PAPER_B_DIR / "paper_b.yaml"),
        "training_policy": "official_family_specific_finetune_recipe",
        "metrics": evaluation["metrics"],
        "unified_evaluation": evaluation,
        "complexity": {"parameters": sum(parameter.numel() for parameter in network.parameters())},
        "environment": environment_snapshot()
        | {"external_repository": str(repository), "external_commit": git_commit(repository)},
    }
    atomic_json_dump(payload, output / "standardized_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
