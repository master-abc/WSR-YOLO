"""Select and apply a joint positive/negative single-model operating point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .common import atomic_json_dump, sha256_file
    from .operating_point import operational_metrics
except ImportError:
    from experiment.paper_b.common import atomic_json_dump, sha256_file
    from experiment.paper_b.operating_point import operational_metrics


def negative_metrics(
    audit: dict[str, Any],
    thresholds: dict[int, float],
    minimum_change_evidence: float = 0.0,
) -> dict[str, Any]:
    counts = [
        sum(
            float(detection["score"])
            >= thresholds.get(int(detection["class_id"]), 1.0)
            and float(detection.get("change_evidence", float("inf")))
            >= minimum_change_evidence
            for detection in row.get("detections", [])
        )
        for row in audit["per_image"]
    ]
    images = len(counts)
    if not images:
        raise ValueError("Negative audit contains no images")
    return {
        "images": images,
        "alarmed_boards": sum(count > 0 for count in counts),
        "board_false_positive_rate": sum(count > 0 for count in counts) / images,
        "false_positives": sum(counts),
        "false_positives_per_image": sum(counts) / images,
    }


def _evidence_filtered_predictions(
    predictions: list[dict[str, Any]], minimum_change_evidence: float
) -> list[dict[str, Any]]:
    return [
        row
        for row in predictions
        if float(row.get("change_evidence", float("inf")))
        >= minimum_change_evidence
    ]


def _objective(
    annotations: dict[str, Any],
    predictions: list[dict[str, Any]],
    negative_audit: dict[str, Any],
    thresholds: dict[int, float],
    maximum_board_fpr: float,
    minimum_change_evidence: float = 0.0,
) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    positive = operational_metrics(
        annotations,
        _evidence_filtered_predictions(predictions, minimum_change_evidence),
        thresholds,
    )
    negative = negative_metrics(
        negative_audit, thresholds, minimum_change_evidence
    )
    overall = positive["overall"]
    feasible = negative["board_false_positive_rate"] <= maximum_board_fpr + 1e-12
    key = (
        feasible,
        overall["f1"] if feasible else -negative["board_false_positive_rate"],
        overall["recall"],
        overall["precision"],
        -negative["board_false_positive_rate"],
        -sum(thresholds.values()),
    )
    return key, positive, negative


def select_classwise_operating_point(
    annotations: dict[str, Any],
    predictions: list[dict[str, Any]],
    negative_audit: dict[str, Any],
    maximum_board_fpr: float = 0.01,
    minimum_threshold: float = 0.20,
    maximum_threshold: float = 0.90,
    threshold_step: float = 0.02,
    maximum_iterations: int = 10,
    minimum_change_evidence: float = 0.0,
) -> dict[str, Any]:
    if not 0.0 <= maximum_board_fpr <= 1.0:
        raise ValueError("maximum_board_fpr must be in [0, 1]")
    if threshold_step <= 0.0 or minimum_threshold > maximum_threshold:
        raise ValueError("Invalid threshold grid")
    categories = sorted(int(row["id"]) for row in annotations["categories"])
    names = {int(row["id"]): str(row["name"]) for row in annotations["categories"]}
    count = int(round((maximum_threshold - minimum_threshold) / threshold_step))
    grid = [round(minimum_threshold + index * threshold_step, 10) for index in range(count + 1)]

    # Begin at the best feasible global threshold, then improve one class at a
    # time. This is deterministic and keeps the complete search off test data.
    global_candidates = []
    for threshold in grid:
        candidate = {category: threshold for category in categories}
        global_candidates.append(
            (
                _objective(
                    annotations,
                    predictions,
                    negative_audit,
                    candidate,
                    maximum_board_fpr,
                    minimum_change_evidence,
                )[0],
                candidate,
            )
        )
    thresholds = max(global_candidates, key=lambda row: row[0])[1]
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        previous = dict(thresholds)
        for category in categories:
            candidates = []
            for threshold in grid:
                candidate = {**thresholds, category: threshold}
                candidates.append(
                    (
                        _objective(
                            annotations,
                            predictions,
                            negative_audit,
                            candidate,
                            maximum_board_fpr,
                            minimum_change_evidence,
                        )[0],
                        candidate,
                    )
                )
            thresholds = max(candidates, key=lambda row: row[0])[1]
        if thresholds == previous:
            break
    _, positive, negative = _objective(
        annotations,
        predictions,
        negative_audit,
        thresholds,
        maximum_board_fpr,
        minimum_change_evidence,
    )
    return {
        "policy": "deterministic coordinate search maximizing positive-validation F1 under a negative-board FPR cap",
        "maximum_board_fpr": float(maximum_board_fpr),
        "grid": {
            "minimum": float(minimum_threshold),
            "maximum": float(maximum_threshold),
            "step": float(threshold_step),
        },
        "iterations": iterations,
        "minimum_change_evidence": float(minimum_change_evidence),
        "thresholds": {str(category): thresholds[category] for category in categories},
        "thresholds_by_name": {names[category]: thresholds[category] for category in categories},
        "positive_validation": positive,
        "negative_validation": negative,
    }


def select_from_files(
    annotations_path: Path,
    predictions_path: Path,
    negative_audit_path: Path,
    weights_path: Path,
    output_path: Path,
    maximum_board_fpr: float,
    input_manifest_path: Path | None = None,
    change_evidence_grid: list[float] | None = None,
    change_evidence_statistic: str | None = None,
) -> dict[str, Any]:
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    negative_audit = json.loads(negative_audit_path.read_text(encoding="utf-8"))
    if change_evidence_grid:
        selections = [
            select_classwise_operating_point(
                annotations,
                predictions,
                negative_audit,
                maximum_board_fpr,
                minimum_change_evidence=float(evidence),
            )
            for evidence in change_evidence_grid
        ]
        selection = max(
            selections,
            key=lambda row: (
                row["negative_validation"]["board_false_positive_rate"]
                <= maximum_board_fpr + 1e-12,
                row["positive_validation"]["overall"]["f1"],
                row["positive_validation"]["overall"]["recall"],
                row["positive_validation"]["overall"]["precision"],
                -row["negative_validation"]["board_false_positive_rate"],
            ),
        )
        selection["change_evidence_grid"] = [
            float(value) for value in change_evidence_grid
        ]
        selection["change_evidence_statistic"] = (
            change_evidence_statistic or "p90_channel_separation"
        )
    else:
        selection = select_classwise_operating_point(
            annotations, predictions, negative_audit, maximum_board_fpr
        )
    inputs = {
        "annotations": str(annotations_path.resolve()),
        "annotations_sha256": sha256_file(annotations_path),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": sha256_file(predictions_path),
        "negative_audit": str(negative_audit_path.resolve()),
        "negative_audit_sha256": sha256_file(negative_audit_path),
    }
    payload = {
        "schema_version": 1,
        "selection_split": "validation",
        "test_evaluated_during_selection": False,
        "single_model": True,
        "weights": str(weights_path.resolve()),
        "weights_sha256": sha256_file(weights_path),
        **selection,
        "inputs": inputs,
    }
    if input_manifest_path is not None:
        manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
        required = (
            "input_contract",
            "single_model_forward_pass",
            "encoding",
            "channels_bgr",
            "noise_floor",
            "gain",
        )
        missing = [key for key in required if key not in manifest]
        if missing:
            raise ValueError(f"Input manifest is missing: {', '.join(missing)}")
        payload["model_input"] = {key: manifest[key] for key in required}
        inputs["input_manifest"] = str(input_manifest_path.resolve())
        inputs["input_manifest_sha256"] = sha256_file(input_manifest_path)
    atomic_json_dump(payload, output_path)
    return payload


def choose_checkpoint(policy_paths: list[Path], output_path: Path) -> dict[str, Any]:
    """Freeze the best validation-only weight/threshold policy."""

    if not policy_paths:
        raise ValueError("At least one checkpoint policy is required")
    candidates: list[tuple[tuple[Any, ...], Path, dict[str, Any]]] = []
    summaries: list[dict[str, Any]] = []
    for path in policy_paths:
        policy = json.loads(path.read_text(encoding="utf-8"))
        if policy.get("test_evaluated_during_selection") is not False:
            raise ValueError(f"Checkpoint policy is not validation-locked: {path}")
        positive = policy["positive_validation"]["overall"]
        negative = policy["negative_validation"]
        maximum_board_fpr = float(policy["maximum_board_fpr"])
        board_fpr = float(negative["board_false_positive_rate"])
        feasible = board_fpr <= maximum_board_fpr + 1e-12
        key = (
            feasible,
            float(positive["f1"]) if feasible else -board_fpr,
            float(positive["recall"]),
            float(positive["precision"]),
            -board_fpr,
        )
        candidates.append((key, path, policy))
        summaries.append(
            {
                "policy": str(path.resolve()),
                "policy_sha256": sha256_file(path),
                "weights": policy["weights"],
                "weights_sha256": policy["weights_sha256"],
                "feasible": feasible,
                "positive_f1": float(positive["f1"]),
                "positive_precision": float(positive["precision"]),
                "positive_recall": float(positive["recall"]),
                "negative_board_fpr": board_fpr,
            }
        )
    _, selected_path, selected = max(candidates, key=lambda row: row[0])
    payload = {
        **selected,
        "checkpoint_selection": {
            "split": "validation",
            "test_evaluated": False,
            "objective": "maximum positive F1 among checkpoint policies satisfying their board-FPR cap",
            "selected_policy": str(selected_path.resolve()),
            "selected_policy_sha256": sha256_file(selected_path),
            "candidates": summaries,
        },
    }
    atomic_json_dump(payload, output_path)
    return payload


def evaluate_frozen_policy(
    policy_path: Path,
    annotations_path: Path,
    predictions_path: Path,
    negative_audit_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("test_evaluated_during_selection") is not False:
        raise ValueError("Policy is not validation-locked")
    thresholds = {int(key): float(value) for key, value in policy["thresholds"].items()}
    minimum_change_evidence = float(policy.get("minimum_change_evidence", 0.0))
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    negative_audit = json.loads(negative_audit_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "stage": "frozen_single_model_policy_test_evaluation",
        "single_model": True,
        "policy": str(policy_path.resolve()),
        "policy_sha256": sha256_file(policy_path),
        "weights": policy["weights"],
        "weights_sha256": policy["weights_sha256"],
        "thresholds": policy["thresholds"],
        "minimum_change_evidence": minimum_change_evidence,
        "positive_test": operational_metrics(
            annotations,
            _evidence_filtered_predictions(predictions, minimum_change_evidence),
            thresholds,
        ),
        "negative_test": negative_metrics(
            negative_audit, thresholds, minimum_change_evidence
        ),
        "inputs": {
            "annotations_sha256": sha256_file(annotations_path),
            "predictions_sha256": sha256_file(predictions_path),
            "negative_audit_sha256": sha256_file(negative_audit_path),
        },
    }
    atomic_json_dump(payload, output_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Joint positive/negative single-model operating point")
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--annotations", type=Path, required=True)
    select.add_argument("--predictions", type=Path, required=True)
    select.add_argument("--negative-audit", type=Path, required=True)
    select.add_argument("--weights", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--maximum-board-fpr", type=float, default=0.01)
    select.add_argument("--input-manifest", type=Path)
    select.add_argument("--change-evidence-grid", type=float, nargs="+")
    select.add_argument("--change-evidence-statistic")
    choose = subparsers.add_parser("choose")
    choose.add_argument("--policies", type=Path, nargs="+", required=True)
    choose.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--policy", type=Path, required=True)
    evaluate.add_argument("--annotations", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--negative-audit", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        payload = select_from_files(
            args.annotations.resolve(),
            args.predictions.resolve(),
            args.negative_audit.resolve(),
            args.weights.resolve(),
            args.output.resolve(),
            args.maximum_board_fpr,
            args.input_manifest.resolve() if args.input_manifest else None,
            args.change_evidence_grid,
            args.change_evidence_statistic,
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    if args.command == "choose":
        payload = choose_checkpoint(
            [path.resolve() for path in args.policies], args.output.resolve()
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    payload = evaluate_frozen_policy(
        args.policy.resolve(),
        args.annotations.resolve(),
        args.predictions.resolve(),
        args.negative_audit.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
