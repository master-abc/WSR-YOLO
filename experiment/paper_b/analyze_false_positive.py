from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t, wilcoxon

try:
    from .common import atomic_json_dump, sha256_file
except ImportError:
    from common import atomic_json_dump, sha256_file


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def analyze(baseline_path: Path, candidate_path: Path, output: Path) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if baseline["image_manifest_sha256"] != candidate["image_manifest_sha256"]:
        raise ValueError("Baseline and candidate use different negative-image manifests")
    baseline_rows = {row["image"]: row["scores"] for row in baseline["per_image"]}
    candidate_rows = {row["image"]: row["scores"] for row in candidate["per_image"]}
    if baseline_rows.keys() != candidate_rows.keys():
        raise ValueError("Baseline and candidate per-image records do not align")
    images = sorted(baseline_rows)
    rows = []
    for raw_threshold in baseline["metrics"]:
        threshold = float(raw_threshold)
        baseline_counts = np.asarray(
            [sum(score >= threshold for score in baseline_rows[image]) for image in images]
        )
        candidate_counts = np.asarray(
            [sum(score >= threshold for score in candidate_rows[image]) for image in images]
        )
        difference = candidate_counts - baseline_counts
        baseline_positive = baseline_counts > 0
        candidate_positive = candidate_counts > 0
        candidate_only = int(np.logical_and(candidate_positive, ~baseline_positive).sum())
        baseline_only = int(np.logical_and(baseline_positive, ~candidate_positive).sum())
        discordant = candidate_only + baseline_only
        mcnemar_p = float(binomtest(min(candidate_only, baseline_only), discordant, 0.5).pvalue) if discordant else 1.0
        count_test = wilcoxon(candidate_counts, baseline_counts, zero_method="wilcox", method="auto")
        mean_difference = float(difference.mean())
        std = float(difference.std(ddof=1))
        half_width = float(t.ppf(0.975, len(difference) - 1)) * std / np.sqrt(len(difference))
        rows.append(
            {
                "threshold": threshold,
                "images": len(images),
                "candidate_only_positive_boards": candidate_only,
                "baseline_only_positive_boards": baseline_only,
                "exact_mcnemar_p": mcnemar_p,
                "mean_paired_fp_difference": mean_difference,
                "mean_paired_fp_difference_ci95": [mean_difference - half_width, mean_difference + half_width],
                "count_wilcoxon_statistic": float(count_test.statistic),
                "count_wilcoxon_p": float(count_test.pvalue),
            }
        )
    mcnemar_adjusted = holm([row["exact_mcnemar_p"] for row in rows])
    count_adjusted = holm([row["count_wilcoxon_p"] for row in rows])
    for row, mcnemar_value, count_value in zip(rows, mcnemar_adjusted, count_adjusted):
        row["exact_mcnemar_holm_p"] = mcnemar_value
        row["count_wilcoxon_holm_p"] = count_value
    payload = {
        "schema_version": 1,
        "baseline_json_sha256": sha256_file(baseline_path),
        "candidate_json_sha256": sha256_file(candidate_path),
        "image_manifest_sha256": baseline["image_manifest_sha256"],
        "tests": {
            "board_rate": "two-sided exact McNemar/binomial test",
            "counts": "two-sided paired Wilcoxon",
            "multiplicity": "Holm within four declared thresholds and test family",
        },
        "rows": rows,
    }
    atomic_json_dump(payload, output.resolve())
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired analysis of negative-image false positives")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.baseline, args.candidate, args.output)
    for row in payload["rows"]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
