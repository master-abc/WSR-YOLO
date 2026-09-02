from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

try:
    from .common import atomic_json_dump, sha256_file
except ImportError:
    from common import atomic_json_dump, sha256_file


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate(paths: list[Path], output: Path) -> dict[str, Any]:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not records:
        raise ValueError("No mitigation summaries supplied")
    thresholds = list(records[0]["operating_points"])
    if any(list(record["operating_points"]) != thresholds for record in records):
        raise ValueError("Mitigation summaries use different thresholds")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "seeds": [],
        "coco_map50_95": {},
        "operating_points": {},
        "inputs": {str(path.resolve()): sha256_file(path) for path in paths},
    }
    for path, record in zip(paths, records):
        seed = int(path.stem.rsplit("seed", 1)[-1])
        payload["seeds"].append(seed)
    order = sorted(range(len(paths)), key=lambda index: payload["seeds"][index])
    payload["seeds"] = [payload["seeds"][index] for index in order]
    records = [records[index] for index in order]

    for condition in ("original", "mitigated"):
        values = [record["coco"][condition]["metrics"]["map50_95"] for record in records]
        payload["coco_map50_95"][condition] = _summary(values)
    payload["coco_map50_95"]["paired_difference"] = _summary(
        [
            record["coco"]["mitigated"]["metrics"]["map50_95"]
            - record["coco"]["original"]["metrics"]["map50_95"]
            for record in records
        ]
    )

    metric_paths = {
        "board_false_positive_rate": ("negative", "board_false_positive_rate"),
        "false_positives_per_image": ("negative", "false_positives_per_image"),
        "precision": ("positive", "overall", "precision"),
        "recall": ("positive", "overall", "recall"),
        "f1": ("positive", "overall", "f1"),
    }
    for threshold in thresholds:
        payload["operating_points"][threshold] = {}
        for metric, keys in metric_paths.items():
            entry = {}
            condition_values = {}
            for condition in ("original", "mitigated"):
                values = []
                for record in records:
                    value: Any = record["operating_points"][threshold][condition]
                    for key in keys:
                        value = value[key]
                    values.append(float(value))
                condition_values[condition] = values
                entry[condition] = _summary(values)
            entry["paired_difference"] = _summary(
                [
                    new - old
                    for old, new in zip(
                        condition_values["original"], condition_values["mitigated"]
                    )
                ]
            )
            payload["operating_points"][threshold][metric] = entry
    atomic_json_dump(payload, output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate paired mitigation results by seed")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate([path.resolve() for path in args.inputs], args.output.resolve())
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
