from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from scipy.stats import t

try:
    from .common import atomic_json_dump, sha256_file
except ImportError:
    from common import atomic_json_dump, sha256_file


SEEDS = (13, 42, 3407)
MODELS = (
    "confirm_yolo11s",
    "confirm_wsr_p3_r25",
    "confirm_wsr_p3_r25_no_hf",
    "confirm_wsr_p3_r25_no_ll",
    "confirm_wsr_p3_r25_fixed_haar",
    "confirm_wsr_p3_r25_equal_fusion",
    "confirm_wsr_p3_r25_random_route",
    "confirm_matched_conv_p3",
    "confirm_scale_only_p3",
)


def describe(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else None
    if std is None:
        interval = [mean, mean]
    else:
        half_width = float(t.ppf(0.975, len(values) - 1)) * std / len(values) ** 0.5
        interval = [mean - half_width, mean + half_width]
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "ci95": interval,
        "values": values,
    }


def load_records(directory: Path) -> dict[str, dict[int, dict]]:
    grouped: dict[str, dict[int, dict]] = {}
    for path in sorted(directory.glob("*__seed_*.json")):
        model, raw_seed = path.stem.rsplit("__seed_", 1)
        record = json.loads(path.read_text(encoding="utf-8"))
        seed = int(raw_seed)
        if record.get("model") != model or int(record.get("seed", -1)) != seed:
            raise ValueError(f"Filename and payload identity disagree: {path}")
        if (
            record.get("budget_profile") != "confirmatory"
            or record.get("selection_split") != "val"
            or record.get("test_evaluated") is not False
        ):
            raise ValueError(f"Result is not confirmatory validation-only evidence: {path}")
        grouped.setdefault(model, {})[seed] = record
    if set(grouped) != set(MODELS):
        raise ValueError(
            f"Expected models {sorted(MODELS)}, found {sorted(grouped)}"
        )
    for model, records in grouped.items():
        if set(records) != set(SEEDS):
            raise ValueError(
                f"{model} has seeds {sorted(records)}, expected {sorted(SEEDS)}"
            )
    return grouped


def summarize(directory: Path, output: Path) -> dict:
    directory = directory.resolve()
    grouped = load_records(directory)
    protocol_hashes = {
        record["protocol_sha256"]
        for records in grouped.values()
        for record in records.values()
    }
    commits = {
        record["environment_at_start"]["git_commit"]
        for records in grouped.values()
        for record in records.values()
    }
    if len(protocol_hashes) != 1 or len(commits) != 1:
        raise ValueError("Confirmatory records do not share one protocol and Git commit")
    baseline = grouped["confirm_yolo11s"]
    proposed = grouped["confirm_wsr_p3_r25"]
    summary = {
        "schema_version": 2,
        "selection_split": "val",
        "test_evaluated": False,
        "seeds": list(SEEDS),
        "protocol_sha256": next(iter(protocol_hashes)),
        "git_commit": next(iter(commits)),
        "models": {},
    }
    for model in MODELS:
        records = grouped[model]
        values = [records[seed]["metrics"]["map50_95"] * 100 for seed in SEEDS]
        baseline_difference = [
            records[seed]["metrics"]["map50_95"] * 100
            - baseline[seed]["metrics"]["map50_95"] * 100
            for seed in SEEDS
        ]
        proposed_difference = [
            records[seed]["metrics"]["map50_95"] * 100
            - proposed[seed]["metrics"]["map50_95"] * 100
            for seed in SEEDS
        ]
        first = records[SEEDS[0]]
        baseline_first = baseline[SEEDS[0]]
        summary["models"][model] = {
            "map50_95_points": describe(values),
            "paired_delta_vs_yolo11s_points": describe(baseline_difference),
            "paired_delta_vs_full_wsr_points": describe(proposed_difference),
            "wins_vs_yolo11s": sum(value > 0 for value in baseline_difference),
            "parameters": first["complexity"]["parameters"],
            "added_parameters_vs_yolo11s": (
                first["complexity"]["parameters"]
                - baseline_first["complexity"]["parameters"]
            ),
            "gflops": first["complexity"]["gflops"],
            "added_gflops_vs_yolo11s": (
                first["complexity"]["gflops"]
                - baseline_first["complexity"]["gflops"]
            ),
            "pretrained_fraction": first["pretrained_transfer"][
                "loaded_parameter_fraction"
            ],
            "result_sha256": {
                str(seed): sha256_file(directory / f"{model}__seed_{seed}.json")
                for seed in SEEDS
            },
        }
    atomic_json_dump(summary, output.resolve())
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and summarize full-budget component controls"
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.results, args.output)
    for model, record in summary["models"].items():
        metric = record["map50_95_points"]
        delta = record["paired_delta_vs_yolo11s_points"]
        print(
            f"{model}: {metric['mean']:.3f} +/- {metric['std']:.3f}; "
            f"delta={delta['mean']:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
