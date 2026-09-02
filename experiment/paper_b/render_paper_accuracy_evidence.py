from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiment" / "paper_b" / "frozen_results"
OUTPUT = ROOT / "paper" / "figures" / "accuracy_evidence.pdf"

BASELINE = "yolo11s"
CANDIDATE = "wsr_yolo11s_p3_r25"
BLUE = "#0072B2"
ORANGE = "#D55E00"
INK = "#2B2B2B"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def controlled_root() -> Path:
    matches = [
        path
        for path in RESULTS.glob("*/controlled")
        if (path / "dspcbsd_plus" / BASELINE).is_dir()
        and (path / "dspcbsd_plus" / CANDIDATE).is_dir()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one frozen controlled-result root, found {matches}")
    return matches[0]


def model_records(root: Path, dataset: str, model: str) -> dict[int, dict]:
    records = {
        int(record["seed"]): record
        for path in sorted((root / dataset / model).glob("seed_*/standardized_result.json"))
        for record in [read_json(path)]
    }
    if not records:
        raise RuntimeError(f"No frozen results found for {dataset}/{model}")
    return records


def paired_records(root: Path, dataset: str) -> list[tuple[dict, dict]]:
    baseline = model_records(root, dataset, BASELINE)
    candidate = model_records(root, dataset, CANDIDATE)
    if baseline.keys() != candidate.keys():
        raise RuntimeError(
            f"Unpaired seeds for {dataset}: baseline={sorted(baseline)}, "
            f"candidate={sorted(candidate)}"
        )
    return [(baseline[seed], candidate[seed]) for seed in sorted(baseline)]


def paired_ap_differences(pairs: list[tuple[dict, dict]]) -> np.ndarray:
    return np.asarray(
        [
            100.0 * (candidate["metrics"]["map50_95"] - baseline["metrics"]["map50_95"])
            for baseline, candidate in pairs
        ],
        dtype=float,
    )


def class_mean_differences(
    pairs: list[tuple[dict, dict]], class_names: list[str]
) -> np.ndarray:
    differences = []
    for class_name in class_names:
        values = [
            100.0
            * (
                candidate["metrics"]["per_class_ap50_95"][class_name]
                - baseline["metrics"]["per_class_ap50_95"][class_name]
            )
            for baseline, candidate in pairs
        ]
        differences.append(float(np.mean(values)))
    return np.asarray(differences, dtype=float)


def seed_offsets(count: int) -> np.ndarray:
    # Keep the paired observations visibly separate from the categorical mean row.
    return np.linspace(-0.12, 0.12, count)


def render_paired_effects(axis: plt.Axes, values_by_dataset: list[tuple[str, np.ndarray, str]]) -> None:
    centers = np.arange(len(values_by_dataset) - 1, -1, -1, dtype=float)
    for center, (_, values, color) in zip(centers, values_by_dataset, strict=True):
        y = center + seed_offsets(len(values))
        axis.hlines(y, 0.0, values, color=color, lw=0.55, alpha=0.55, zorder=1)

        mean = float(values.mean())
        # A short mean tick avoids covering observations or their connecting lines.
        axis.vlines(
            mean,
            center - 0.055,
            center + 0.055,
            color=INK,
            linewidth=1.15,
            zorder=2,
        )
        axis.scatter(
            values,
            y,
            marker="o",
            s=18,
            color=color,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
        axis.annotate(
            f"mean {mean:+.2f}",
            (mean, center),
            xytext=(4, 9),
            textcoords="offset points",
            fontsize=6.4,
            color=INK,
            va="bottom",
        )

    axis.axvline(0.0, color=INK, lw=0.7, zorder=0)
    axis.set_yticks(centers, [label for label, _, _ in values_by_dataset])
    axis.set_ylim(centers[-1] - 0.47, centers[0] + 0.47)
    axis.set_xlim(-1.5, 8.0)
    axis.set_xticks([0, 2, 4, 6, 8])
    axis.set_title("(a) Seed-paired effects", loc="left")
    axis.set_xlabel(r"Paired $\Delta$AP$_{50:95}$ (points)")


def render_class_differences(
    axis: plt.Axes,
    values: np.ndarray,
    labels: list[str],
    title: str,
    xlim: tuple[float, float],
    xticks: list[float],
) -> None:
    positions = np.arange(len(values))
    colors = [BLUE if value >= 0.0 else ORANGE for value in values]
    axis.axvline(0.0, color=INK, lw=0.7, zorder=0)
    axis.barh(positions, values, height=0.62, color=colors, edgecolor="none", zorder=2)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(*xlim)
    axis.set_xticks(xticks)
    axis.set_title(title, loc="left")
    axis.set_xlabel(r"WSR $-$ YOLO11s AP (points)")


def render() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.2,
            "axes.titlesize": 7.6,
            "axes.titleweight": "semibold",
            "axes.labelsize": 7.1,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 0.0,
            "xtick.major.width": 0.6,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    root = controlled_root()
    dspcbsd_pairs = paired_records(root, "dspcbsd_plus")
    deeppcb_pairs = paired_records(root, "deeppcb")

    dspcbsd_classes = [
        "base_material_foreign_object",
        "conductor_foreign_object",
        "conductor_scratch",
        "hole_breakout",
        "mouse_bite",
        "open",
        "short",
        "spur",
        "spurious_copper",
    ]
    dspcbsd_labels = [
        "base FO",
        "cond. FO",
        "scratch",
        "breakout",
        "mouse bite",
        "open",
        "short",
        "spur",
        "spurious Cu",
    ]
    deeppcb_classes = ["copper", "mousebite", "open", "pin-hole", "short", "spur"]
    deeppcb_labels = ["copper", "mouse bite", "open", "pin-hole", "short", "spur"]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.0166667, 2.2666667),
        gridspec_kw={"width_ratios": [1.16, 1.0, 0.93]},
    )
    render_paired_effects(
        axes[0],
        [
            (r"DsPCBSD+ ($n$=7)", paired_ap_differences(dspcbsd_pairs), BLUE),
            (r"DeepPCB ($n$=3)", paired_ap_differences(deeppcb_pairs), ORANGE),
        ],
    )
    render_class_differences(
        axes[1],
        class_mean_differences(dspcbsd_pairs, dspcbsd_classes),
        dspcbsd_labels,
        "(b) DsPCBSD+ classes",
        (-0.65, 1.42),
        [-0.5, 0.0, 0.5, 1.0],
    )
    render_class_differences(
        axes[2],
        class_mean_differences(deeppcb_pairs, deeppcb_classes),
        deeppcb_labels,
        "(c) DeepPCB classes",
        (0.0, 6.5),
        [0, 2, 4, 6],
    )

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    figure.subplots_adjust(left=0.14, right=0.995, bottom=0.22, top=0.88, wspace=0.48)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the original figure's physical dimensions so this visual-only
    # change cannot alter the paper's float pagination.
    figure.savefig(OUTPUT)
    plt.close(figure)


if __name__ == "__main__":
    render()
