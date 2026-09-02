from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

try:
    from .common import REPRODUCIBILITY_DIR, atomic_json_dump, environment_snapshot, sha256_file
except ImportError:
    from common import REPRODUCIBILITY_DIR, atomic_json_dump, environment_snapshot, sha256_file


def validate_result(payload: dict[str, Any], path: Path) -> None:
    if int(payload.get("schema_version", 0)) < 2:
        raise ValueError(f"Legacy result cannot be frozen: {path}")
    if payload.get("track") == "smoke" or payload.get("smoke"):
        raise ValueError(f"Smoke result cannot enter the paper registry: {path}")
    if not payload.get("unified_evaluation"):
        raise ValueError(f"Result was not produced by unified COCOeval: {path}")
    start = payload.get("environment_at_start", payload.get("environment", {}))
    if not start.get("git_commit"):
        raise ValueError(f"Result has no source Git commit: {path}")
    if start.get("git_dirty"):
        raise ValueError(f"Result started from a dirty worktree: {path}")
    transfer = payload.get("pretrained_transfer") or {}
    minimum = float(transfer.get("minimum_parameter_fraction", 0.0))
    loaded = float(transfer.get("loaded_parameter_fraction", 1.0))
    if minimum and loaded < minimum:
        raise ValueError(f"Pretrained fairness threshold failed: {path}")


def freeze(root: Path, output: Path) -> Path:
    current = environment_snapshot()
    if current.get("git_dirty"):
        raise RuntimeError("Commit the current code before freezing formal paper results")
    result_paths = sorted(root.resolve().rglob("standardized_result.json"))
    if not result_paths:
        raise FileNotFoundError(f"No standardized results under {root.resolve()}")

    entries = []
    for source in result_paths:
        payload = json.loads(source.read_text(encoding="utf-8"))
        validate_result(payload, source)
        protocol_hash = str(payload.get("protocol_sha256", "unknown"))
        relative = source.relative_to(root.resolve())
        target = output.resolve() / protocol_hash[:12] / relative
        source_hash = sha256_file(source)
        if target.exists() and sha256_file(target) != source_hash:
            raise RuntimeError(f"Refusing to overwrite a different frozen result: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        entries.append(
            {
                "track": payload["track"],
                "dataset": payload["dataset"],
                "model": payload["model"],
                "seed": int(payload["seed"]),
                "protocol_sha256": protocol_hash,
                "source": str(source),
                "frozen": str(target),
                "sha256": source_hash,
                "weights_sha256": payload.get("weights_sha256"),
                "predictions_sha256": payload.get("unified_evaluation", {}).get(
                    "predictions_sha256"
                ),
            }
        )
    manifest = output.resolve() / "manifest.json"
    atomic_json_dump(
        {
            "schema_version": 1,
            "git_commit": current.get("git_commit"),
            "results": entries,
        },
        manifest,
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze validated formal result JSONs for Git")
    parser.add_argument("--root", type=Path, default=REPRODUCIBILITY_DIR / "generated" / "runs")
    parser.add_argument("--output", type=Path, default=REPRODUCIBILITY_DIR / "frozen_results")
    args = parser.parse_args()
    print(freeze(args.root, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
