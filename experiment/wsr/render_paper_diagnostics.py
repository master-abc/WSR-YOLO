from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "experiment" / "wsr" / "generated"
FIGURES = ROOT / "paper" / "figures"

BLUE = "#0072B2"
ORANGE = "#D55E00"
RED = "#B2182B"
INK = "#2B2B2B"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.4,
            "axes.titlesize": 7.8,
            "axes.titleweight": "semibold",
            "axes.labelsize": 7.2,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.6,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "xtick.major.size": 2.3,
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


def finish_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)


def evaluation_map(path: Path, condition_key: str) -> dict[str, float]:
    payload = read_json(path)
    return {
        record["condition"][condition_key]: 100.0 * record["metrics"]["map50_95"]
        for record in payload["evaluations"]
    }


def corruption_means(path: Path) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for record in read_json(path)["evaluations"]:
        key = record["condition"]["corruption"]
        grouped.setdefault(key, []).append(100.0 * record["metrics"]["map50_95"])
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def render_robustness() -> None:
    auxiliary = GENERATED / "r" / "auxiliary"
    baseline_corruptions = corruption_means(auxiliary / "deeppcb_yolo11s_corruptions.json")
    wsr_corruptions = corruption_means(auxiliary / "deeppcb_wsr_yolo11s_p3_r25_corruptions.json")
    differences = {
        key: wsr_corruptions[key] - baseline_corruptions[key]
        for key in baseline_corruptions
    }
    display = {
        "brightness": "Brightness",
        "contrast": "Contrast",
        "defocus_blur": "Defocus blur",
        "gaussian": "Gaussian noise",
        "jpeg": "JPEG",
        "motion_blur": "Motion blur",
        "poisson": "Poisson noise",
        "speckle": "Speckle noise",
    }
    ordered = sorted(differences, key=differences.get, reverse=True)

    baseline_frequency = evaluation_map(
        auxiliary / "deeppcb_yolo11s_frequency.json", "frequency_intervention"
    )
    wsr_frequency = evaluation_map(
        auxiliary / "deeppcb_wsr_yolo11s_p3_r25_frequency.json", "frequency_intervention"
    )
    frequency_order = ["low_only", "remove_lh", "remove_hl", "remove_hh", "high_only"]
    frequency_labels = ["Low only", "Remove LH", "Remove HL", "Remove HH", "High only"]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(3.48, 1.78), gridspec_kw={"width_ratios": [1.02, 1.0]}
    )

    positions = np.arange(len(ordered))
    values = np.asarray([differences[key] for key in ordered])
    colors = [BLUE if value >= 0 else ORANGE for value in values]
    left.axvline(0.0, color=INK, lw=0.65, zorder=1)
    left.barh(positions, values, height=0.62, color=colors, edgecolor="none", zorder=2)
    left.set_yticks(positions, [display[key] for key in ordered])
    left.invert_yaxis()
    left.set_title("(a) Corruption effect", loc="left")
    left.set_xlabel(r"Mean $\Delta$AP$_{50:95}$ (points)")
    finish_axis(left)

    positions = np.arange(len(frequency_order))
    base_values = np.asarray([baseline_frequency[key] for key in frequency_order])
    wsr_values = np.asarray([wsr_frequency[key] for key in frequency_order])
    bar_height = 0.32
    right.barh(positions - 0.18, base_values, height=bar_height, color=BLUE, edgecolor="none", label="YOLO11s", zorder=2)
    right.barh(positions + 0.18, wsr_values, height=bar_height, color=ORANGE, edgecolor="none", label="WSR-YOLO11s", zorder=2)
    right.set_yticks(positions, frequency_labels)
    right.invert_yaxis()
    right.set_xlim(0, 72)
    right.set_xticks([0, 20, 40, 60])
    right.set_title("(b) Haar ablation", loc="left")
    right.set_xlabel(r"AP$_{50:95}$ (\%)")
    right.legend(loc="lower right", frameon=False, handletextpad=0.25, borderaxespad=0.15)
    finish_axis(right)

    figure.subplots_adjust(left=0.225, right=0.995, bottom=0.23, top=0.88, wspace=0.72)
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "robustness_frequency_compact.pdf")
    plt.close(figure)


def false_positive_rates(path: Path) -> tuple[np.ndarray, np.ndarray]:
    metrics = read_json(path)["metrics"]
    thresholds = np.asarray(sorted(float(key) for key in metrics), dtype=float)
    rates = np.asarray(
        [100.0 * metrics[f"{threshold:g}"]["board_false_positive_rate"] for threshold in thresholds]
    )
    return thresholds, rates


def nested(payload: dict, *keys: str) -> float:
    value: object = payload
    for key in keys:
        if not isinstance(value, dict):
            raise TypeError(f"Expected a mapping before key {key!r}")
        value = value[key]
    return float(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def intersection_over_union(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def match_predictions(predictions: list[dict], ground_truth: list[dict]) -> tuple[list[bool], int]:
    matched_ground_truth: set[int] = set()
    true_positive = []
    for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        candidates = [
            (intersection_over_union(prediction["xyxy"], target["xyxy"]), index)
            for index, target in enumerate(ground_truth)
            if index not in matched_ground_truth and prediction["class_id"] == target["class_id"]
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        is_true_positive = best_iou >= 0.5
        true_positive.append(is_true_positive)
        if is_true_positive:
            matched_ground_truth.add(best_index)
    return true_positive, len(ground_truth) - len(matched_ground_truth)


def compact_class_name(name: str) -> str:
    return {
        "conductor_foreign_object": "FO",
        "hole_breakout": "BR",
        "conductor_scratch": "SC",
    }.get(name, name.replace("_", " "))


def draw_box(axis: plt.Axes, box: list[float], color: str, dashed: bool = False) -> None:
    x1, y1, x2, y2 = box
    axis.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=0.8,
            linestyle=(0, (3, 2)) if dashed else "solid",
        )
    )


def draw_tag(axis: plt.Axes, box: list[float], text_value: str, color: str) -> None:
    x1, y1, x2, _ = box
    align_right = x1 > 190
    near_bottom = y1 > 205
    axis.text(
        x2 - 1 if align_right else x1 + 1,
        y1 - 1 if near_bottom else y1 + 1,
        text_value,
        color="white",
        fontsize=5.0,
        ha="right" if align_right else "left",
        va="bottom" if near_bottom else "top",
        bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.92, "pad": 0.55},
        clip_on=True,
    )


def render_qualitative(source_dir: Path) -> None:
    qualitative = read_json(GENERATED / "revision_results" / "qualitative.json")
    operating = read_json(GENERATED / "revision_results" / "qualitative_operating_point.json")
    source_by_hash = {
        sha256_file(path): path
        for path in source_dir.glob("*.jpg")
    }
    models = [("YOLO11s", BLUE), ("WSR-YOLO11s", ORANGE)]
    figure, axes = plt.subplots(len(qualitative["rows"]), 3, figsize=(3.48, 2.10), squeeze=False)

    for row_index, row in enumerate(qualitative["rows"]):
        image_path = source_by_hash.get(row["image_sha256"])
        if image_path is None:
            raise FileNotFoundError(f"No source image with SHA-256 {row['image_sha256']} in {source_dir}")
        source_image = Image.open(image_path).convert("RGB")
        for axis in axes[row_index]:
            axis.imshow(source_image, interpolation="none")
            axis.set_axis_off()

        for target in row["ground_truth"]:
            draw_box(axes[row_index, 0], target["xyxy"], "#009E73")
            draw_tag(
                axes[row_index, 0],
                target["xyxy"],
                compact_class_name(target["class_name"]),
                "#009E73",
            )

        operating_row = operating["rows"][row_index]
        for column_index, (model_name, model_color) in enumerate(models, start=1):
            threshold = operating["inputs"]["models"][model_name]["selected_policy"]["thresholds"]
            predictions = sorted(
                [
                    prediction
                    for prediction in row["predictions"][model_name]
                    if prediction["confidence"] >= threshold[prediction["class_name"]]
                ],
                key=lambda item: item["confidence"],
                reverse=True,
            )
            true_positive, false_negative = match_predictions(predictions, row["ground_truth"])
            counts = operating_row["models"][model_name]
            if (
                sum(true_positive) != counts["tp"]
                or len(predictions) - sum(true_positive) != counts["fp"]
                or false_negative != counts["fn"]
            ):
                raise RuntimeError(f"Rendered counts do not match the frozen manifest for row {row_index}/{model_name}")
            for prediction, is_true_positive in zip(predictions, true_positive, strict=True):
                if is_true_positive:
                    draw_box(axes[row_index, column_index], prediction["xyxy"], model_color)
                    draw_tag(
                        axes[row_index, column_index],
                        prediction["xyxy"],
                        f"{compact_class_name(prediction['class_name'])} {prediction['confidence']:.2f}".replace(" 0.", " ."),
                        model_color,
                    )
                else:
                    draw_box(axes[row_index, column_index], prediction["xyxy"], RED, dashed=True)
            axes[row_index, column_index].text(
                0.985,
                0.985,
                f"TP {counts['tp']}  FP {counts['fp']}  FN {counts['fn']}",
                transform=axes[row_index, column_index].transAxes,
                ha="right",
                va="top",
                color="white",
                fontsize=5.1,
                bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.78, "pad": 0.7},
            )

    for column, title in enumerate(("Ground truth", "YOLO11s", "WSR-YOLO11s")):
        axes[0, column].set_title(title, fontsize=7.2, fontweight="semibold", pad=2.5)
    figure.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.91, wspace=0.035, hspace=0.035)
    figure.savefig(FIGURES / "qualitative_operating_point.png", dpi=300)
    plt.close(figure)


def render_false_alarm() -> None:
    revision = GENERATED / "revision_results"
    baseline_thresholds, baseline_rates = false_positive_rates(
        revision / "deeppcb_test_templates_yolo11s_false_positive.json"
    )
    wsr_thresholds, wsr_rates = false_positive_rates(
        revision / "deeppcb_test_templates_wsr_yolo11s_p3_r25_false_positive.json"
    )
    if not np.array_equal(baseline_thresholds, wsr_thresholds):
        raise RuntimeError("Baseline and WSR confidence sweeps use different thresholds")

    local = GENERATED / "negative_aware_local"
    stage3 = read_json(local / "stage3_hard25_r3_test_aggregate.json")
    calibrated = read_json(local / "stage3_hard25_r3_calibrated_01_aggregate.json")
    consensus = read_json(local / "stage3_consensus_zero_test.json")
    paired = read_json(GENERATED / "evidence" / "final_r2_result.json")
    tradeoff = {
        "S3": (
            100.0 * nested(stage3, "operating_points", "0.25", "board_false_positive_rate", "mitigated", "mean"),
            100.0 * nested(stage3, "operating_points", "0.25", "recall", "mitigated", "mean"),
        ),
        "Cal.": (
            100.0 * nested(calibrated, "summary", "holdout_board_fpr", "mean"),
            100.0 * nested(calibrated, "summary", "recall", "mean"),
        ),
        "Ens.": (
            100.0 * nested(consensus, "results", "0.0", "test", "negative", "board_false_positive_rate"),
            100.0 * nested(consensus, "results", "0.0", "test", "positive", "overall", "recall"),
        ),
        "Pair": (
            100.0 * nested(paired, "negative_test", "board_false_positive_rate"),
            100.0 * nested(paired, "positive_test", "overall", "recall"),
        ),
    }

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(3.48, 1.72), gridspec_kw={"width_ratios": [1.0, 1.08]}
    )
    sweep_positions = np.arange(len(baseline_thresholds))
    left.plot(sweep_positions, baseline_rates, color=BLUE, marker="o", ms=3.2, lw=1.0, label="YOLO11s")
    left.plot(sweep_positions, wsr_rates, color=ORANGE, marker="s", ms=3.2, lw=1.0, label="WSR-YOLO11s")
    left.set_xticks(sweep_positions, [f"{value:.2f}" for value in baseline_thresholds])
    left.set_ylim(20, 92)
    left.set_title("(a) Confidence sweep", loc="left")
    left.set_xlabel("Confidence threshold")
    left.set_ylabel("Board-FPR (%)")
    left.legend(loc="lower left", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    finish_axis(left)

    marker_by_name = {"S3": "o", "Cal.": "s", "Ens.": "^", "Pair": "*"}
    color_by_name = {"S3": BLUE, "Cal.": BLUE, "Ens.": BLUE, "Pair": ORANGE}
    offset_by_name = {"S3": (-15, 5), "Cal.": (4, -1), "Ens.": (4, -1), "Pair": (4, 4)}
    for name, (fpr, recall) in tradeoff.items():
        right.scatter(
            fpr,
            recall,
            s=36 if name == "Pair" else 22,
            marker=marker_by_name[name],
            color=color_by_name[name],
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        right.annotate(name, (fpr, recall), xytext=offset_by_name[name], textcoords="offset points", fontsize=6.8)
    right.set_xscale("log")
    right.set_xlim(0.45, 50)
    right.set_ylim(76.5, 97.5)
    right.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 40]))
    right.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    right.set_title("(b) Post-hoc trade-off", loc="left")
    right.set_xlabel("Board-FPR (%, log scale)")
    right.set_ylabel("Recall (%)")
    finish_axis(right)

    figure.subplots_adjust(left=0.14, right=0.995, bottom=0.235, top=0.87, wspace=0.50)
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "false_alarm_tradeoff.pdf")
    plt.close(figure)


def render(qualitative_source_dir: Path | None = None) -> None:
    apply_style()
    render_robustness()
    render_false_alarm()
    if qualitative_source_dir is not None:
        render_qualitative(qualitative_source_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render publication figures from frozen paper evidence")
    parser.add_argument(
        "--qualitative-source-dir",
        type=Path,
        help="Directory containing the two original DsPCBSD+ JPGs referenced by the frozen manifest",
    )
    arguments = parser.parse_args()
    render(arguments.qualitative_source_dir)
