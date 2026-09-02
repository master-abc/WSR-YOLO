from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .coco_evaluator import evaluate_coco_predictions
    from .common import atomic_json_dump, sha256_file
    from .operating_point import operational_metrics, validation_f1_thresholds
except ImportError:
    from coco_evaluator import evaluate_coco_predictions
    from common import atomic_json_dump, sha256_file
    from operating_point import operational_metrics, validation_f1_thresholds


def box_iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lw, lh = (float(value) for value in left)
    rx1, ry1, rw, rh = (float(value) for value in right)
    lx2, ly2 = lx1 + max(lw, 0.0), ly1 + max(lh, 0.0)
    rx2, ry2 = rx1 + max(rw, 0.0), ry1 + max(rh, 0.0)
    overlap = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = max(lw, 0.0) * max(lh, 0.0) + max(rw, 0.0) * max(rh, 0.0) - overlap
    return overlap / union if union > 0.0 else 0.0


def suppress_overlaps(
    predictions: list[dict[str, Any]], iou_threshold: float, class_agnostic: bool
) -> list[dict[str, Any]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_image[int(prediction["image_id"])].append(prediction)
    kept = []
    for image_predictions in by_image.values():
        accepted: list[dict[str, Any]] = []
        for prediction in sorted(
            image_predictions, key=lambda row: float(row["score"]), reverse=True
        ):
            duplicate = any(
                (
                    class_agnostic
                    or int(prediction["category_id"]) == int(previous["category_id"])
                )
                and box_iou(prediction["bbox"], previous["bbox"]) >= iou_threshold
                for previous in accepted
            )
            if not duplicate:
                accepted.append(prediction)
        kept.extend(accepted)
    return kept


def apply_candidate(
    predictions: list[dict[str, Any]], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    thresholds = candidate["thresholds"]
    filtered = [
        row
        for row in predictions
        if float(row["score"]) >= thresholds.get(int(row["category_id"]), 1.0)
    ]
    if candidate["suppression"] == "none":
        return filtered
    return suppress_overlaps(
        filtered,
        float(candidate["nms_iou"]),
        class_agnostic=candidate["suppression"] == "class_agnostic",
    )


def select_postprocessing(
    validation_annotations_path: Path,
    validation_predictions_path: Path,
    test_annotations_path: Path,
    test_predictions_path: Path,
    output_path: Path,
    betas: list[float],
    nms_ious: list[float],
    reference_threshold: float = 0.25,
    recall_tolerance: float = 0.03,
    match_iou: float = 0.5,
) -> dict[str, Any]:
    validation_annotations = json.loads(
        validation_annotations_path.read_text(encoding="utf-8")
    )
    validation_predictions = json.loads(
        validation_predictions_path.read_text(encoding="utf-8")
    )
    test_annotations = json.loads(test_annotations_path.read_text(encoding="utf-8"))
    test_predictions = json.loads(test_predictions_path.read_text(encoding="utf-8"))
    category_names = {
        int(row["id"]): str(row["name"])
        for row in validation_annotations.get("categories", [])
    }
    reference_validation = operational_metrics(
        validation_annotations, validation_predictions, reference_threshold, match_iou
    )
    reference_test = operational_metrics(
        test_annotations, test_predictions, reference_threshold, match_iou
    )

    candidates = []
    for beta in betas:
        thresholds = validation_f1_thresholds(
            validation_annotations,
            validation_predictions,
            iou_threshold=match_iou,
            beta=beta,
        )
        configurations = [("none", None)]
        configurations.extend(
            (mode, iou)
            for mode in ("classwise", "class_agnostic")
            for iou in nms_ious
        )
        for suppression, nms_iou in configurations:
            candidate = {
                "beta": float(beta),
                "suppression": suppression,
                "nms_iou": nms_iou,
                "thresholds": thresholds,
            }
            filtered = apply_candidate(validation_predictions, candidate)
            metrics = operational_metrics(
                validation_annotations, filtered, 0.0, match_iou
            )
            candidates.append({**candidate, "validation": metrics["overall"]})

    best_f1 = max(
        candidates,
        key=lambda row: (
            row["validation"]["f1"],
            row["validation"]["precision"],
            -row["validation"]["fp"],
        ),
    )
    recall_floor = reference_validation["overall"]["recall"] - recall_tolerance
    feasible = [row for row in candidates if row["validation"]["recall"] >= recall_floor]
    constraint_satisfied = bool(feasible)
    if feasible:
        low_fp = max(
            feasible,
            key=lambda row: (
                -row["validation"]["fp"],
                row["validation"]["f1"],
                row["validation"]["precision"],
            ),
        )
    else:
        # Retain a complete audit record instead of failing after an expensive
        # evaluation. This fallback is explicitly marked and must not be
        # described as satisfying the requested recall constraint.
        low_fp = max(candidates, key=lambda row: row["validation"]["recall"])

    policies = {}
    for policy, selected in (("max_f1", best_f1), ("recall_constrained", low_fp)):
        test_filtered = apply_candidate(test_predictions, selected)
        filtered_path = output_path.with_name(
            f"{output_path.stem}_{policy}_predictions.json"
        )
        atomic_json_dump(test_filtered, filtered_path)
        policies[policy] = {
            "selection": {
                "beta": selected["beta"],
                "suppression": selected["suppression"],
                "nms_iou": selected["nms_iou"],
                "thresholds": {
                    category_names.get(key, str(key)): value
                    for key, value in sorted(selected["thresholds"].items())
                },
                "validation": selected["validation"],
                "recall_constraint_satisfied": (
                    None if policy == "max_f1" else constraint_satisfied
                ),
            },
            "test": operational_metrics(
                test_annotations, test_filtered, 0.0, match_iou
            ),
            "test_coco": evaluate_coco_predictions(
                test_annotations_path, filtered_path, max_det=100
            ),
        }

    payload = {
        "schema_version": 1,
        "post_hoc_exploratory": True,
        "selection_split": "validation",
        "test_evaluated_during_selection": False,
        "reference_threshold": reference_threshold,
        "recall_tolerance": recall_tolerance,
        "reference": {
            "validation": reference_validation,
            "test": reference_test,
        },
        "candidate_grid": candidates,
        "policies": policies,
        "inputs": {
            "validation_annotations_sha256": sha256_file(validation_annotations_path),
            "validation_predictions_sha256": sha256_file(validation_predictions_path),
            "test_annotations_sha256": sha256_file(test_annotations_path),
            "test_predictions_sha256": sha256_file(test_predictions_path),
        },
    }
    atomic_json_dump(payload, output_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select confidence and duplicate-suppression post-processing on validation"
    )
    parser.add_argument("--validation-annotations", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--test-annotations", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    parser.add_argument(
        "--nms-ious", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7]
    )
    parser.add_argument("--reference-threshold", type=float, default=0.25)
    parser.add_argument("--recall-tolerance", type=float, default=0.03)
    parser.add_argument("--match-iou", type=float, default=0.5)
    args = parser.parse_args()
    payload = select_postprocessing(
        args.validation_annotations.resolve(),
        args.validation_predictions.resolve(),
        args.test_annotations.resolve(),
        args.test_predictions.resolve(),
        args.output.resolve(),
        args.betas,
        args.nms_ious,
        args.reference_threshold,
        args.recall_tolerance,
        args.match_iou,
    )
    for name, policy in payload["policies"].items():
        print(json.dumps({name: policy["selection"]}, ensure_ascii=False))
        print(json.dumps({name: policy["test"]["overall"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
