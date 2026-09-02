from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import atomic_json_dump, sha256_file
    from .coco_evaluator import evaluate_coco_predictions
except ImportError:
    from common import atomic_json_dump, sha256_file
    from coco_evaluator import evaluate_coco_predictions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize DETR predictions with the paper's unified COCO evaluator"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--max-det", type=int, default=100)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    repository = args.repository.resolve()
    import subprocess

    actual_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if actual_commit != args.commit:
        raise ValueError(f"Repository revision mismatch: expected {args.commit}, found {actual_commit}")
    predictions = args.predictions.resolve()
    if not predictions.exists():
        raise FileNotFoundError(predictions)
    evaluation = evaluate_coco_predictions(
        args.annotations.resolve(), predictions, max_det=args.max_det
    )
    payload = {
        "schema_version": 2,
        "track": "sota",
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "repository": str(repository),
        "commit": actual_commit,
        "predictions": str(predictions),
        "predictions_sha256": sha256_file(predictions),
        "metrics": evaluation["metrics"],
        "unified_evaluation": evaluation,
        "provenance_note": "Metrics were computed from raw predictions; no values were copied manually.",
    }
    atomic_json_dump(payload, args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
