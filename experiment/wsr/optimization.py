from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

try:
    from .ablation import validation_run_directory
    from .common import atomic_json_dump, environment_snapshot, sha256_file
    from .run import load_protocol
except ImportError:
    from ablation import validation_run_directory
    from common import atomic_json_dump, environment_snapshot, sha256_file
    from run import load_protocol


DEFAULT_CONFIG = Path(__file__).resolve().parent / "protocol.yaml"


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _result_directory(protocol: dict[str, Any], model: str, seed: int) -> Path:
    return validation_run_directory(protocol, model, seed, False, "ablation")


def evaluate(protocol: dict[str, Any]) -> dict[str, Any]:
    config = protocol["optimization_round"]
    if config.get("selection_split") != "val" or config.get("test_evaluated") is not False:
        raise ValueError("Optimization selection must remain validation-only")

    baseline = str(config["baseline"])
    candidates = [str(value) for value in config["candidates"]]
    seeds = [int(value) for value in config["seeds"]]
    protocol_hash = sha256_file(protocol["_path"])
    values: dict[str, list[float]] = {name: [] for name in [baseline, *candidates]}
    enrichments: dict[str, list[float]] = {name: [] for name in candidates}
    latencies: dict[str, float] = {}
    architecture_hashes: dict[str, str] = {}
    result_hashes: dict[str, str] = {}
    commits: set[str] = set()

    for model in [baseline, *candidates]:
        for seed in seeds:
            directory = _result_directory(protocol, model, seed)
            path = directory / "ablation_result.json"
            payload = _read(path)
            if payload.get("model") != model or int(payload.get("seed", -1)) != seed:
                raise ValueError(f"Result identity mismatch: {path}")
            if payload.get("selection_split") != "val" or payload.get("test_evaluated") is not False:
                raise ValueError(f"Optimization result touched a non-validation split: {path}")
            if payload.get("budget_profile") != "ablation":
                raise ValueError(f"Optimization result used a non-ablation budget: {path}")
            if payload.get("protocol_sha256") != protocol_hash:
                raise ValueError(f"Optimization result used another protocol revision: {path}")
            environment = payload.get("environment_at_start", {})
            if environment.get("git_dirty") or not environment.get("git_commit"):
                raise ValueError(f"Optimization result lacks clean Git provenance: {path}")
            commits.add(str(environment["git_commit"]))
            values[model].append(float(payload["metrics"]["map50_95"]))
            result_hashes[str(path.resolve())] = sha256_file(path)
            if model in candidates:
                observed = str(payload["architecture_definition_sha256"])
                if model in architecture_hashes and architecture_hashes[model] != observed:
                    raise ValueError(f"Architecture changed across seeds for {model}")
                architecture_hashes[model] = observed

                diagnostic_path = directory / "route_diagnostics_val.json"
                diagnostic = _read(diagnostic_path)
                if diagnostic.get("selection_split") != "val":
                    raise ValueError(f"Diagnostic used a non-validation split: {diagnostic_path}")
                if diagnostic.get("weights_sha256") != payload.get("weights_sha256"):
                    raise ValueError(f"Diagnostic checkpoint mismatch: {diagnostic_path}")
                enrichment = diagnostic.get("summary", {}).get("route_enrichment", {}).get("mean")
                if enrichment is None:
                    raise ValueError(f"Missing route enrichment: {diagnostic_path}")
                enrichments[model].append(float(enrichment))
                result_hashes[str(diagnostic_path.resolve())] = sha256_file(diagnostic_path)

    latency_seed = seeds[0]
    for model in [baseline, *candidates]:
        directory = _result_directory(protocol, model, latency_seed)
        benchmark_path = directory / "latency_val_fp32.json"
        benchmark = _read(benchmark_path)
        result = _read(directory / "ablation_result.json")
        if benchmark.get("selection_split") != "val":
            raise ValueError(f"Latency used a non-validation split: {benchmark_path}")
        if benchmark.get("weights_sha256") != result.get("weights_sha256"):
            raise ValueError(f"Latency checkpoint mismatch: {benchmark_path}")
        latencies[model] = float(benchmark["model_only"]["mean_ms"])
        result_hashes[str(benchmark_path.resolve())] = sha256_file(benchmark_path)

    baseline_mean = statistics.fmean(values[baseline])
    rows = []
    for candidate in candidates:
        candidate_mean = statistics.fmean(values[candidate])
        gain = candidate_mean - baseline_mean
        enrichment = statistics.fmean(enrichments[candidate])
        latency_ratio = latencies[candidate] / latencies[baseline]
        checks = {
            "accuracy": gain >= float(config["minimum_mean_ap50_95_gain"]),
            "mechanism": enrichment >= float(config["minimum_route_enrichment"]),
            "latency": latency_ratio <= float(config["maximum_latency_ratio"]),
        }
        rows.append(
            {
                "candidate": candidate,
                "values": values[candidate],
                "mean_map50_95": candidate_mean,
                "mean_map50_95_gain": gain,
                "mean_route_enrichment": enrichment,
                "model_only_mean_ms": latencies[candidate],
                "latency_ratio": latency_ratio,
                "checks": checks,
                "pass": all(checks.values()),
                "architecture_definition_sha256": architecture_hashes[candidate],
            }
        )

    passing = [row for row in rows if row["pass"]]
    passing.sort(
        key=lambda row: (
            -float(row["mean_map50_95"]),
            float(row["model_only_mean_ms"]),
            str(row["candidate"]),
        )
    )
    selected = passing[0] if passing else None
    decision = {
        "schema_version": 1,
        "status": "PASS" if selected else "FAIL",
        "selection_split": "val",
        "test_evaluated": False,
        "optimization_round": str(config["name"]),
        "dataset": str(config["dataset"]),
        "seeds": seeds,
        "baseline": baseline,
        "baseline_values": values[baseline],
        "baseline_mean_map50_95": baseline_mean,
        "baseline_model_only_mean_ms": latencies[baseline],
        "candidates": rows,
        "selected_candidate": selected["candidate"] if selected else None,
        "selected_architecture_definition_sha256": (
            selected["architecture_definition_sha256"] if selected else None
        ),
        "protocol_sha256": protocol_hash,
        "git_commits": sorted(commits),
        "thresholds": {
            "minimum_mean_ap50_95_gain": float(config["minimum_mean_ap50_95_gain"]),
            "minimum_route_enrichment": float(config["minimum_route_enrichment"]),
            "maximum_latency_ratio": float(config["maximum_latency_ratio"]),
        },
        "result_sha256": result_hashes,
        "environment": environment_snapshot(),
    }
    output = (
        protocol["_output_root"]
        / "optimization"
        / str(config["name"])
        / "screening_decision.json"
    )
    atomic_json_dump(decision, output)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Validation-only stable WSR candidate gate")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("command", choices=("evaluate",))
    args = parser.parse_args()
    protocol = load_protocol(args.config)
    print(json.dumps(evaluate(protocol), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
