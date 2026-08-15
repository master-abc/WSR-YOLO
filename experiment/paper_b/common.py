from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import yaml


PAPER_B_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PAPER_B_DIR.parent
PROJECT_DIR = EXPERIMENT_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def resolve_path(value: str | Path, base: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        temp_path = Path(stream.name)
    temp_path.replace(path)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_version(command: list[str], cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
            cwd=str(cwd) if cwd is not None else None,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = (result.stdout or result.stderr).strip()
    return value if value else None


def environment_snapshot() -> dict[str, Any]:
    git_status = run_version(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=PROJECT_DIR
    ) or ""
    git_diff = run_version(["git", "diff", "--binary", "HEAD"], cwd=PROJECT_DIR) or ""
    snapshot: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git_commit": run_version(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR),
        "git_dirty": bool(git_status),
        "git_status": git_status.splitlines(),
        "git_diff_sha256": sha256_text(git_diff),
        "nvidia_smi": run_version(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }
    for module_name in ("torch", "torchvision", "ultralytics", "numpy", "scipy"):
        try:
            module = __import__(module_name)
            snapshot[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            snapshot[module_name] = None
    return snapshot


def file_hashes(paths: Iterable[str | Path]) -> dict[str, str]:
    """Hash existing research source files using project-relative keys."""

    hashes: dict[str, str] = {}
    for value in paths:
        path = Path(value).resolve()
        if not path.is_file():
            continue
        try:
            key = path.relative_to(PROJECT_DIR).as_posix()
        except ValueError:
            key = path.as_posix()
        hashes[key] = sha256_file(path)
    return hashes


def image_to_label_path(image: Path) -> Path:
    parts = list(image.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        index = len(parts) - 1 - lowered[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    sibling = image.parent.parent / "labels" / image.parent.name / image.with_suffix(".txt").name
    return sibling


def read_image_source(source: str | Path, dataset_root: str | Path) -> list[Path]:
    source_path = resolve_path(source, dataset_root)
    if source_path.is_file() and source_path.suffix.lower() == ".txt":
        images: list[Path] = []
        for raw in source_path.read_text(encoding="utf-8-sig").splitlines():
            value = raw.strip()
            if not value:
                continue
            images.append(resolve_path(value, source_path.parent))
        return sorted(dict.fromkeys(images))
    if source_path.is_dir():
        return sorted(
            path.resolve()
            for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if source_path.is_file() and source_path.suffix.lower() in IMAGE_SUFFIXES:
        return [source_path]
    raise FileNotFoundError(f"Image source does not exist: {source_path}")


def dataset_sources(dataset_yaml: str | Path) -> tuple[dict[str, Any], Path, dict[str, list[Path]]]:
    dataset_yaml = Path(dataset_yaml).resolve()
    data = load_yaml(dataset_yaml)
    root = resolve_path(data.get("path", dataset_yaml.parent), dataset_yaml.parent)
    sources: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        value = data.get(split)
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        merged: list[Path] = []
        for item in values:
            merged.extend(read_image_source(item, root))
        sources[split] = sorted(dict.fromkeys(merged))
    return data, root, sources


def class_names(data: dict[str, Any]) -> list[str]:
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    return [str(value) for value in names]


def write_lines(lines: Iterable[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(f"{line}\n" for line in lines)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".txt", dir=path.parent, delete=False
    ) as stream:
        stream.write(content)
        temp_path = Path(stream.name)
    temp_path.replace(path)
