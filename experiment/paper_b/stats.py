from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .common import PAPER_B_DIR, atomic_json_dump
except ImportError:
    from common import PAPER_B_DIR, atomic_json_dump


METRICS = (
    "map50_95",
    "map50",
    "map75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar_1",
    "ar_10",
    "ar_100",
    "ar_small",
    "ar_medium",
    "ar_large",
)


def t_critical_95(df: int) -> float:
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df))
    except ImportError:
        lookup = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
        return lookup.get(df, 1.96)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    n = len(values)
    mean = statistics.fmean(values) if values else None
    std = statistics.stdev(values) if n >= 2 else None
    half = t_critical_95(n - 1) * std / math.sqrt(n) if n >= 2 and std is not None else None
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half if half is not None else None,
        "ci95_high": mean + half if half is not None else None,
    }


def load_results(root: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(root.rglob("standardized_result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path.resolve())
        required = ("track", "dataset", "model", "seed", "metrics")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"{path} is missing fields: {missing}")
        results.append(payload)
    return results


def aggregate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[(result["track"], result["dataset"], result["model"])].append(result)
    rows = []
    for (track, dataset, model), runs in sorted(groups.items()):
        row: dict[str, Any] = {
            "track": track,
            "dataset": dataset,
            "model": model,
            "seeds": sorted(int(run["seed"]) for run in runs),
        }
        for metric in METRICS:
            values = [float(run["metrics"][metric]) for run in runs if run["metrics"].get(metric) is not None]
            row[metric] = summarize(values)
        rows.append(row)
    return rows


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_significance(
    results: list[dict[str, Any]], baseline: str, metric: str
) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for result in results:
        value = result["metrics"].get(metric)
        if value is not None:
            by_group[(result["track"], result["dataset"], result["model"])][int(result["seed"])] = float(value)

    comparisons = []
    for (track, dataset, model), candidate in sorted(by_group.items()):
        if model == baseline:
            continue
        reference = by_group.get((track, dataset, baseline))
        if not reference:
            continue
        seeds = sorted(set(reference) & set(candidate))
        if len(seeds) < 3:
            continue
        left = [reference[seed] for seed in seeds]
        right = [candidate[seed] for seed in seeds]
        differences = [b - a for a, b in zip(left, right)]
        try:
            from scipy.stats import wilcoxon

            test = wilcoxon(right, left, alternative="two-sided", zero_method="wilcox")
            p_value = float(test.pvalue)
            statistic = float(test.statistic)
        except (ImportError, ValueError):
            p_value = 1.0
            statistic = None
        diff_std = statistics.stdev(differences) if len(differences) >= 2 else 0.0
        effect = statistics.fmean(differences) / diff_std if diff_std > 0 else None
        comparisons.append(
            {
                "track": track,
                "dataset": dataset,
                "baseline": baseline,
                "candidate": model,
                "metric": metric,
                "seeds": seeds,
                "mean_difference": statistics.fmean(differences),
                "paired_cohen_dz": effect,
                "wilcoxon_statistic": statistic,
                "p_raw": p_value,
                "note": (
                    "Descriptive only: an exact two-sided Wilcoxon test cannot reach p<0.05 "
                    "with fewer than six non-zero pairs"
                    if len(seeds) < 6
                    else "Primary paired test"
                ),
            }
        )
    adjusted = holm_adjust([row["p_raw"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["p_holm"] = value
        row["significant_0.05"] = value < 0.05
    return comparisons


def format_cell(summary: dict[str, Any]) -> str:
    if summary["mean"] is None:
        return "--"
    if summary["std"] is None:
        return f"{100 * summary['mean']:.2f}"
    return f"{100 * summary['mean']:.2f} ± {100 * summary['std']:.2f}"


def write_outputs(rows: list[dict[str, Any]], significance: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for row in rows:
        flat: dict[str, Any] = {key: row[key] for key in ("track", "dataset", "model")}
        flat["seeds"] = ";".join(str(value) for value in row["seeds"])
        for metric in METRICS:
            for field, value in row[metric].items():
                flat[f"{metric}_{field}"] = value
        flat_rows.append(flat)
    if flat_rows:
        with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)

    header = "| Track | Dataset | Model | n | AP50:95 | AP50 | AP75 | APs | APm | APl |\n"
    divider = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    lines = [header, divider]
    latex = [
        "\\begin{tabular}{lllrrrrrrr}",
        "\\toprule",
        "Track & Dataset & Model & $n$ & AP & AP$_{50}$ & AP$_{75}$ & AP$_S$ & AP$_M$ & AP$_L$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [
            format_cell(row[metric])
            for metric in ("map50_95", "map50", "map75", "ap_small", "ap_medium", "ap_large")
        ]
        n = row["map50_95"]["n"]
        lines.append(
            f"| {row['track']} | {row['dataset']} | {row['model']} | {n} | " + " | ".join(cells) + " |\n"
        )
        latex.append(
            " & ".join([row["track"], row["dataset"], row["model"], str(n), *cells]) + " \\\\"
        )
    latex.extend(["\\bottomrule", "\\end{tabular}"])
    (output / "summary.md").write_text("".join(lines), encoding="utf-8")
    (output / "summary.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    atomic_json_dump(rows, output / "summary.json")
    atomic_json_dump(significance, output / "significance.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate seed-level results with paired statistics")
    parser.add_argument("--root", type=Path, default=PAPER_B_DIR / "generated" / "runs")
    parser.add_argument("--output", type=Path, default=PAPER_B_DIR / "generated" / "tables")
    parser.add_argument("--baseline", default="yolo11s")
    parser.add_argument("--metric", default="map50_95", choices=METRICS)
    args = parser.parse_args()
    results = load_results(args.root.resolve())
    if not results:
        print(f"No standardized results found under {args.root.resolve()}")
        return 2
    rows = aggregate(results)
    significance = paired_significance(results, args.baseline, args.metric)
    write_outputs(rows, significance, args.output.resolve())
    print(f"Aggregated {len(results)} runs into {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
