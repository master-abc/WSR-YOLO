from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiment" / "wsr"
OUTPUT = ROOT / "paper" / "figures" / "context_route_evidence.pdf"
BLUE = "#0072B2"
ORANGE = "#D55E00"
INK = "#2B2B2B"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def unique_result_root(track: str, required_model: str) -> Path:
    matches = [
        path
        for path in (RESULTS / "frozen_results").glob(f"*/{track}/dspcbsd_plus")
        if (path / required_model).is_dir()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {track} result root for {required_model}, found {matches}")
    return matches[0]


def model_summary(root: Path, model: str) -> tuple[float, float, float]:
    records = [read_json(path) for path in sorted((root / model).glob("seed_*/standardized_result.json"))]
    if not records:
        raise RuntimeError(f"No frozen results found for {model}")
    ap = np.asarray([record["metrics"]["map50_95"] * 100.0 for record in records], dtype=float)
    params = {int(record["complexity"]["parameters"]) for record in records}
    if len(params) != 1:
        raise RuntimeError(f"Inconsistent parameter counts for {model}: {params}")
    return next(iter(params)) / 1e6, float(ap.mean()), float(ap.std(ddof=1))


def route_enrichment(dataset: str) -> dict[str, float]:
    payload = read_json(RESULTS / "generated" / "revision_results" / f"{dataset}_mechanism_controls.json")
    controls = payload["control_summary"]
    return {
        "Uniform random": controls["uniform_random"]["route_enrichment"]["mean"],
        "Center prior": controls["center_prior"]["route_enrichment"]["mean"],
        "Train occupancy": controls["train_occupancy_prior"]["route_enrichment"]["mean"],
        "Activation energy": controls["activation_energy"]["route_enrichment"]["mean"],
        "Fixed Haar": controls["fixed_haar"]["route_enrichment"]["mean"],
        "Learned WSR": controls["actual"]["route_enrichment"]["mean"],
        "Cross-image shuffled": payload["shuffled_route_control"]["0"]
        ["mean_route_enrichment_by_permutation"]["mean"],
    }


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
            "legend.fontsize": 6.3,
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
    sota_root = unique_result_root("sota", "yolo26_m")
    controlled_root = unique_result_root("controlled", "wsr_yolo11s_p3_r25")
    models = [
        ("yolov10_m", "V10-M", (-29, -3)),
        ("yolo11_m", "V11-M", (-24, -10)),
        ("yolov12_m", "V12-M", (5, 6)),
        ("yolo26_m", "V26-M", (5, -7)),
        ("rtdetrv2_s", "RTv2-S", (5, -3)),
        ("dfine_m", "D-FINE", (-29, 8)),
        ("deim_dfine_m", "DEIM", (6, 1)),
        ("rf_detr_m", "RF-DETR", (-34, 6)),
    ]

    figure, (left, right) = plt.subplots(1, 2, figsize=(7.02, 2.32), gridspec_kw={"width_ratios": [1.05, 1.35]})
    for model, label, label_offset in models:
        x, y, error = model_summary(sota_root, model)
        left.errorbar(x, y, yerr=error, fmt="o", ms=3.7, color=BLUE, mec="white", mew=0.35, capsize=1.8, lw=0.65, zorder=3)
        left.annotate(
            label,
            (x, y),
            xytext=label_offset,
            textcoords="offset points",
            fontsize=5.7,
            va="center",
        )

    x, y, error = model_summary(controlled_root, "wsr_yolo11s_p3_r25")
    left.errorbar(x, y, yerr=error, fmt="*", ms=7.5, color=ORANGE, mec="white", mew=0.3, capsize=1.8, lw=0.7, zorder=4)
    left.annotate("WSR-11s", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=6.2, va="center")
    left.set_title("(a) Accuracy–size context", loc="left")
    left.set_xlabel("Parameters (M)")
    left.set_ylabel(r"DsPCBSD+ AP$_{50:95}$")
    left.set_xlim(7.5, 35.5)
    left.set_ylim(45.6, 54.1)

    labels = [
        "Uniform random",
        "Center prior",
        "Train occupancy",
        "Activation energy",
        "Fixed Haar",
        "Learned WSR",
        "Cross-image shuffled",
    ]
    ds = route_enrichment("dspcbsd_plus")
    deep = route_enrichment("deeppcb")
    positions = np.arange(len(labels))
    ds_values = [ds[label] for label in labels]
    deep_values = [deep[label] for label in labels]
    right.axvline(1.0, color=INK, ls=(0, (3, 2)), lw=0.7, zorder=1)
    right.scatter(ds_values, positions, s=18, color=BLUE, edgecolors="white", linewidths=0.35, label="DsPCBSD+", zorder=3)
    right.scatter(
        deep_values,
        positions,
        s=24,
        facecolors="white",
        edgecolors=ORANGE,
        linewidths=1.0,
        label="DeepPCB",
        zorder=3,
    )
    right.set_yticks(positions, labels)
    right.invert_yaxis()
    right.set_title("(b) Same-budget route controls", loc="left")
    right.set_xlabel(r"Same-budget route enrichment ($\times$)")
    right.set_xlim(0.55, 3.78)
    right.legend(loc="lower right", frameon=False, ncol=2, handletextpad=0.25, columnspacing=0.8)

    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.21, top=0.88, wspace=0.48)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT)
    plt.close(figure)


if __name__ == "__main__":
    render()
