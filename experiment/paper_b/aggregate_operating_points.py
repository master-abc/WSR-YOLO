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


def aggregate(paths: list[Path], seeds: list[int], output: Path) -> dict[str, Any]:
    if not paths or len(paths) != len(seeds):
        raise ValueError("Provide one seed for every operating-point record")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    order = sorted(range(len(seeds)), key=seeds.__getitem__)
    records = [records[index] for index in order]
    paths = [paths[index] for index in order]
    seeds = [seeds[index] for index in order]

    rows = []
    for seed, record in zip(seeds, records):
        positive = record["positive_test"]["calibrated"]["overall"]
        negative = record["negative_holdout"]["calibrated"]
        rows.append(
            {
                "seed": seed,
                "threshold": record["selection"]["selected_threshold"],
                "calibration_board_fpr": record["negative_calibration"][
                    "board_false_positive_rate"
                ],
                "holdout_board_fpr": negative["board_false_positive_rate"],
                "holdout_fppi": negative["false_positives_per_image"],
                "precision": positive["precision"],
                "recall": positive["recall"],
                "f1": positive["f1"],
            }
        )

    metric_names = [name for name in rows[0] if name != "seed"]
    payload = {
        "schema_version": 1,
        "seeds": seeds,
        "records": rows,
        "summary": {
            name: _summary([float(row[name]) for row in rows])
            for name in metric_names
        },
        "inputs": {str(path.resolve()): sha256_file(path) for path in paths},
    }
    atomic_json_dump(payload, output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate validation-selected operating points across seeds"
    )
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate(
        [path.resolve() for path in args.inputs],
        args.seeds,
        args.output.resolve(),
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
