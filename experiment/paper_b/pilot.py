from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

try:
    from .ablation import train_validation_only
    from .benchmark import benchmark_weights
    from .common import PAPER_B_DIR, atomic_json_dump, environment_snapshot, file_hashes, sha256_file
    from .mechanism_diagnostics import diagnose_routes
    from .run import generated_dataset_yaml, load_protocol
except ImportError:
    from ablation import train_validation_only
    from benchmark import benchmark_weights
    from common import PAPER_B_DIR, atomic_json_dump, environment_snapshot, file_hashes, sha256_file
    from mechanism_diagnostics import diagnose_routes
    from run import generated_dataset_yaml, load_protocol


DEFAULT_CONFIG = PAPER_B_DIR / "paper_b.yaml"


def pilot_config(protocol: dict[str, Any]) -> dict[str, Any]:
    config = protocol["pilot_gate"]
    if not config.get("validation_only", False):
        raise ValueError("Pilot selection must be validation-only")
    if config.get("mechanism_split") != "val" or config.get("latency_split") != "val":
        raise ValueError("Pilot diagnostics and latency must use the validation split")
    return config


def pilot_models(protocol: dict[str, Any]) -> tuple[str, str]:
    config = pilot_config(protocol)
    return str(config["baseline"]), str(config["candidate"])


def run_directory(protocol: dict[str, Any], model: str, seed: int, smoke: bool = False) -> Path:
    config = pilot_config(protocol)
    scope = "smoke" if smoke else "runs"
    return (
        protocol["_output_root"]
        / scope
        / "pilot"
        / str(config["dataset"])
        / model
        / f"seed_{seed}"
    )


def result_path(protocol: dict[str, Any], model: str, seed: int, smoke: bool = False) -> Path:
    return run_directory(protocol, model, seed, smoke) / "ablation_result.json"


def diagnostics_path(protocol: dict[str, Any], seed: int) -> Path:
    _, candidate = pilot_models(protocol)
    return run_directory(protocol, candidate, seed) / "route_diagnostics_val.json"


def benchmark_path(protocol: dict[str, Any], model: str) -> Path:
    seed = int(pilot_config(protocol)["latency_seed"])
    return run_directory(protocol, model, seed) / "latency_val_fp32.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def plan(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    baseline, candidate = pilot_models(protocol)
    dataset = str(pilot_config(protocol)["dataset"])
    return [
        {"dataset": dataset, "model": model, "seed": int(seed), "split": "val"}
        for seed in pilot_config(protocol)["seeds"]
        for model in (baseline, candidate)
    ]


def train_pilot(
    protocol: dict[str, Any],
    device: str,
    selected_seed: int | None = None,
    smoke: bool = False,
    force: bool = False,
    resume: bool = False,
) -> list[Path]:
    outputs = []
    for item in plan(protocol):
        if selected_seed is not None and int(item["seed"]) != selected_seed:
            continue
        outputs.append(
            train_validation_only(
                protocol,
                str(item["model"]),
                int(item["seed"]),
                device,
                smoke,
                force,
                "pilot",
                resume,
            )
        )
    return outputs


def run_diagnostics(protocol: dict[str, Any], device: str) -> list[Path]:
    config = pilot_config(protocol)
    _, candidate = pilot_models(protocol)
    data = generated_dataset_yaml(protocol, str(config["dataset"]))
    outputs = []
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        result = read_json(result_path(protocol, candidate, seed))
        if result.get("selection_split") != "val" or result.get("test_evaluated") is not False:
            raise RuntimeError(f"Pilot result touched the wrong split: {result_path(protocol, candidate, seed)}")
        weights = Path(result["weights"])
        output = diagnostics_path(protocol, seed)
        diagnose_routes(
            weights,
            data,
            output,
            str(config["mechanism_split"]),
            int(result["train_args"]["imgsz"]),
            device,
            int(config["mechanism_limit"]),
        )
        outputs.append(output)
    return outputs


def run_benchmarks(protocol: dict[str, Any], device: str) -> list[Path]:
    config = pilot_config(protocol)
    data = generated_dataset_yaml(protocol, str(config["dataset"]))
    seed = int(config["latency_seed"])
    outputs = []
    for model in pilot_models(protocol):
        result = read_json(result_path(protocol, model, seed))
        weights = Path(result["weights"])
        output = benchmark_path(protocol, model)
        benchmark_weights(
            weights,
            data,
            output,
            device,
            int(result["train_args"]["imgsz"]),
            int(config["latency_warmup"]),
            int(config["latency_repetitions"]),
            False,
            str(config["latency_split"]),
        )
        outputs.append(output)
    return outputs


def validate_result(payload: dict[str, Any], expected_model: str, seed: int) -> None:
    if payload.get("schema_version") != 2 or payload.get("track") != "ablation_validation_only":
        raise ValueError(f"Invalid pilot result schema for {expected_model}/seed={seed}")
    if payload.get("smoke"):
        raise ValueError("Smoke results cannot determine the pilot gate")
    if payload.get("model") != expected_model or int(payload.get("seed", -1)) != seed:
        raise ValueError(f"Pilot result identity mismatch for {expected_model}/seed={seed}")
    if payload.get("selection_split") != "val" or payload.get("test_evaluated") is not False:
        raise ValueError("Pilot selection must never evaluate test")
    if payload.get("budget_profile") != "pilot":
        raise ValueError("Pilot selection must use the registered pilot training budget")
    environment = payload.get("environment_at_start", {})
    if environment.get("git_dirty") or not environment.get("git_commit"):
        raise ValueError("Pilot result must originate from a clean Git commit")
    transfer = payload.get("pretrained_transfer", {})
    if float(transfer.get("loaded_parameter_fraction", 0.0)) < float(
        transfer.get("minimum_parameter_fraction", 1.0)
    ):
        raise ValueError("Pilot pretrained-transfer threshold failed")


def evaluate_gate(protocol: dict[str, Any]) -> dict[str, Any]:
    config = pilot_config(protocol)
    baseline, candidate = pilot_models(protocol)
    seeds = [int(value) for value in config["seeds"]]
    baseline_values = []
    candidate_values = []
    enrichments = []
    commits = set()
    result_hashes: dict[str, str] = {}
    candidate_architecture_hash = None
    protocol_hash = sha256_file(protocol["_path"])

    for seed in seeds:
        for model, values in ((baseline, baseline_values), (candidate, candidate_values)):
            path = result_path(protocol, model, seed)
            payload = read_json(path)
            validate_result(payload, model, seed)
            if payload.get("protocol_sha256") != protocol_hash:
                raise ValueError(f"Pilot result was produced by a different protocol: {path}")
            values.append(float(payload["metrics"]["map50_95"]))
            commits.add(payload["environment_at_start"]["git_commit"])
            result_hashes[str(path.resolve())] = sha256_file(path)
            if model == candidate:
                observed_hash = payload["architecture_definition_sha256"]
                if candidate_architecture_hash not in {None, observed_hash}:
                    raise ValueError("Candidate architecture changed across pilot seeds")
                candidate_architecture_hash = observed_hash

        diagnostic_file = diagnostics_path(protocol, seed)
        diagnostic = read_json(diagnostic_file)
        result_hashes[str(diagnostic_file.resolve())] = sha256_file(diagnostic_file)
        if diagnostic.get("selection_split") != "val":
            raise ValueError(f"Mechanism diagnostic used a non-validation split: {diagnostic_file}")
        candidate_result = read_json(result_path(protocol, candidate, seed))
        if diagnostic.get("weights_sha256") != candidate_result.get("weights_sha256"):
            raise ValueError(f"Mechanism diagnostic checkpoint mismatch: {diagnostic_file}")
        enrichment = diagnostic.get("summary", {}).get("route_enrichment", {}).get("mean")
        if enrichment is None:
            raise ValueError(f"No route enrichment was measured: {diagnostic_file}")
        enrichments.append(float(enrichment))

    if len(commits) != 1:
        raise ValueError(f"Pilot results span multiple commits: {sorted(commits)}")

    latency = {}
    latency_seed = int(config["latency_seed"])
    for model in (baseline, candidate):
        path = benchmark_path(protocol, model)
        report = read_json(path)
        result_hashes[str(path.resolve())] = sha256_file(path)
        model_result = read_json(result_path(protocol, model, latency_seed))
        if report.get("selection_split") != "val":
            raise ValueError(f"Latency benchmark used a non-validation split: {path}")
        if report.get("weights_sha256") != model_result.get("weights_sha256"):
            raise ValueError(f"Latency checkpoint mismatch: {path}")
        latency[model] = float(report["model_only"]["mean_ms"])

    baseline_mean = statistics.fmean(baseline_values)
    candidate_mean = statistics.fmean(candidate_values)
    gain = candidate_mean - baseline_mean
    enrichment_mean = statistics.fmean(enrichments)
    latency_ratio = latency[candidate] / latency[baseline]
    checks = {
        "accuracy": gain >= float(config["minimum_mean_ap50_95_gain"]),
        "mechanism": enrichment_mean >= float(config["minimum_route_enrichment"]),
        "latency": latency_ratio <= float(config["maximum_latency_ratio"]),
    }
    decision = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "selection_split": "val",
        "test_evaluated": False,
        "dataset": config["dataset"],
        "baseline": baseline,
        "candidate": candidate,
        "formal_model": config["formal_model"],
        "seeds": seeds,
        "git_commit": next(iter(commits)),
        "protocol_sha256": sha256_file(protocol["_path"]),
        "candidate_architecture_definition_sha256": candidate_architecture_hash,
        "measurements": {
            "baseline_mean_map50_95": baseline_mean,
            "candidate_mean_map50_95": candidate_mean,
            "mean_map50_95_gain": gain,
            "mean_route_enrichment": enrichment_mean,
            "baseline_model_only_mean_ms": latency[baseline],
            "candidate_model_only_mean_ms": latency[candidate],
            "latency_ratio": latency_ratio,
        },
        "thresholds": {
            "minimum_mean_ap50_95_gain": float(config["minimum_mean_ap50_95_gain"]),
            "minimum_route_enrichment": float(config["minimum_route_enrichment"]),
            "maximum_latency_ratio": float(config["maximum_latency_ratio"]),
        },
        "checks": checks,
        "result_sha256": result_hashes,
        "source_files_sha256": file_hashes(
            [Path(__file__), PAPER_B_DIR / "ablation.py", PAPER_B_DIR / "mechanism_diagnostics.py", PAPER_B_DIR / "benchmark.py"]
        ),
        "environment": environment_snapshot(),
    }
    output = protocol["_output_root"] / "pilot" / "pilot_decision.json"
    atomic_json_dump(decision, output)
    return decision


def freeze_decision(protocol: dict[str, Any]) -> Path:
    current = environment_snapshot()
    if current.get("git_dirty"):
        raise RuntimeError("Freeze the pilot decision only from a clean Git worktree")
    decision = evaluate_gate(protocol)
    if decision["status"] != "PASS":
        raise RuntimeError("Pilot gate failed; the proposed model must not enter formal test runs")
    target = PAPER_B_DIR / "selection" / "pilot_decision.json"
    if target.exists():
        existing = read_json(target)
        if existing != decision:
            raise RuntimeError(f"Refusing to overwrite a different frozen pilot decision: {target}")
    else:
        atomic_json_dump(decision, target)
    return target


def status(protocol: dict[str, Any]) -> dict[str, Any]:
    baseline, candidate = pilot_models(protocol)
    missing = []
    for seed_value in pilot_config(protocol)["seeds"]:
        seed = int(seed_value)
        for model in (baseline, candidate):
            path = result_path(protocol, model, seed)
            if not path.exists():
                missing.append(str(path.resolve()))
        path = diagnostics_path(protocol, seed)
        if not path.exists():
            missing.append(str(path.resolve()))
    for model in (baseline, candidate):
        path = benchmark_path(protocol, model)
        if not path.exists():
            missing.append(str(path.resolve()))
    return {"ready_to_evaluate": not missing, "missing": missing}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validation-only three-seed pilot gate")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    train = subparsers.add_parser("train")
    train.add_argument("--device", default="0")
    train.add_argument("--seed", type=int)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--force", action="store_true")
    train.add_argument("--resume", action="store_true")
    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument("--device", default="0")
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--device", default="0")
    subparsers.add_parser("evaluate")
    subparsers.add_parser("freeze")
    subparsers.add_parser("status")
    args = parser.parse_args()
    protocol = load_protocol(args.config.resolve())

    if args.command == "plan":
        for item in plan(protocol):
            print(f"{item['dataset']:<16} {item['model']:<24} seed={item['seed']} split=val")
        print(f"Total pilot runs: {len(plan(protocol))}")
        return 0
    if args.command == "train":
        train_pilot(protocol, args.device, args.seed, args.smoke, args.force, args.resume)
        return 0
    if args.command == "diagnose":
        run_diagnostics(protocol, args.device)
        return 0
    if args.command == "benchmark":
        run_benchmarks(protocol, args.device)
        return 0
    if args.command == "evaluate":
        decision = evaluate_gate(protocol)
        print(json.dumps(decision, indent=2, ensure_ascii=False))
        return 0 if decision["status"] == "PASS" else 2
    if args.command == "freeze":
        print(freeze_decision(protocol))
        return 0
    if args.command == "status":
        report = status(protocol)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ready_to_evaluate"] else 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
