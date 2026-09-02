# WSR-YOLO: Wavelet-Conditioned Sparse Routing for PCB Defect Detection

This repository contains the paper, implementation, experiment protocol, and frozen audit artifacts for **Wavelet Conditioned Sparse Routing for PCB Defect Detection**.

> [!IMPORTANT]
> The formal paper evidence is produced exclusively by the audited WSR track in [`experiment/wsr/`](experiment/wsr/). This directory is the current paper's reproducibility package, not a legacy experiment.

## Overview

Wavelet-Conditioned Top-k Sparse Routing (WSR) ranks P3/8 feature locations using fixed Haar responses and applies a gather--refine--scatter channel transform to an exact top-k token budget. The study separates four questions that are often conflated in conditional-computation work:

1. Are selected feature cells enriched inside defect boxes?
2. Does routing improve held-out detection accuracy?
3. Does lower arithmetic cost translate into measured acceleration?
4. Are any gains specific to the proposed routing cues rather than added capacity?

The repository deliberately reports null and adverse findings. On DsPCBSD+, the paired accuracy difference is not statistically significant, and WSR is slower than the baseline despite adding only 13.3K parameters and 0.038 GFLOPs.

## Audited Findings

| Evaluation | YOLO11s | WSR-YOLO11s | Interpretation |
|---|---:|---:|---|
| DsPCBSD+ AP50:95, seven paired seeds | 46.54 +/- 0.36 | 46.69 +/- 0.72 | +0.15 AP; two-sided Wilcoxon `p=0.8125` |
| DeepPCB AP50:95, three descriptive seeds | 64.61 +/- 1.91 | 68.81 +/- 4.12 | +4.20 AP, but more alarms on defect-free templates |
| Forward latency ratio | 1.00x | 1.18--1.21x | Nominal sparsity did not produce acceleration |
| Route enrichment inside boxes | -- | 3.16x / 3.55x | Selection is image-dependent, but spatial priors explain much of the enrichment |

These results support WSR as an auditable routing probe, not as a reliable accuracy or systems advantage.

## Paper and Reproducibility Artifacts

- English paper source: [`paper/main.tex`](paper/main.tex)
- Chinese translation source: [`paper/main_zh.tex`](paper/main_zh.tex)
- Audited experiment guide: [`experiment/wsr/README.md`](experiment/wsr/README.md)
- Current experiment status: [`experiment/wsr/STATUS.md`](experiment/wsr/STATUS.md)
- Frozen formal results: [`experiment/wsr/frozen_results/`](experiment/wsr/frozen_results/)
- Tests: [`experiment/wsr/tests/`](experiment/wsr/tests/)

## Repository Structure

```text
.
|-- algorithm/                     # WSR and false-alarm refinement modules
|-- experiment/
|   |-- wsr/       # Audited protocol used by the current paper
|   `-- configs/                   # Current WSR model and dataset configurations
|-- paper/                         # IEEE LaTeX sources, figures, and generated documents
`-- requirements.txt
```

## Installation

The controlled paper environment uses Python 3.10, PyTorch 2.5.1 with CUDA 12.1, and Ultralytics 8.4.50. Install the appropriate PyTorch build for your hardware if CUDA 12.1 is unavailable.

```bash
conda create -n wsr-yolo python=3.10 -y
conda activate wsr-yolo

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify the custom-module registration:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "from algorithm.register import register_custom_modules; register_custom_modules()"
```

## Reproducing the Paper Experiments

There are two reproduction levels. Rebuilding the reported statistics uses the committed frozen results and does not require a GPU or dataset:

```bash
python -m experiment.wsr.stats \
  --root experiment/wsr/frozen_results \
  --output experiment/wsr/generated/tables
```

Full retraining requires the official datasets and a CUDA GPU. Run the following steps from the repository root.

### 1. Download, Convert, and Audit the Datasets

`prepare_dspcbsd` downloads and converts the official DsPCBSD+ release. Place DeepPCB under `datasets/DeepPCB` before preparing its fixed split.

```bash
python -m experiment.wsr.prepare_dspcbsd
python -m experiment.wsr.run prepare --dataset dspcbsd_plus --coco
python -m experiment.wsr.run prepare --dataset deeppcb --coco
python -m experiment.wsr.run audit
```

The audit must complete without fatal annotation, exact-duplicate, or split-integrity findings.

### 2. Freeze Validation-Only Model Selection

```bash
python -m experiment.wsr.pilot plan
python -m experiment.wsr.pilot train --device 0
python -m experiment.wsr.pilot diagnose --device 0
python -m experiment.wsr.pilot benchmark --device 0
python -m experiment.wsr.pilot evaluate
python -m experiment.wsr.pilot freeze
```

This stage uses only validation data. Commit the resulting selection decision before starting formal runs.

### 3. Inspect and Run the Registered Controlled Matrix

```bash
python -m experiment.wsr.run plan

# Primary DsPCBSD+ experiment: seven paired seeds and two architectures.
for seed in 13 42 3407 4703 8391 9475 10501; do
  python -m experiment.wsr.run train --dataset dspcbsd_plus --model yolo11s --seed "$seed" --device 0
  python -m experiment.wsr.run train --dataset dspcbsd_plus --model wsr_yolo11s_p3_r25 --seed "$seed" --device 0
done

# Descriptive DeepPCB experiment: three paired seeds and two architectures.
for seed in 13 42 3407; do
  python -m experiment.wsr.run train --dataset deeppcb --model yolo11s --seed "$seed" --device 0
  python -m experiment.wsr.run train --dataset deeppcb --model wsr_yolo11s_p3_r25 --seed "$seed" --device 0
done
```

Each `train` command trains one model, selects `best.pt` from validation behavior, evaluates the locked test split once with the shared COCOeval implementation, and writes `standardized_result.json`. Use `--resume` only for an interrupted run with matching provenance.

Formal runs require a clean, resolvable Git commit, a successful full-data audit, a frozen validation-only pilot decision, and at least 99% pretrained-parameter coverage after WSR insertion.

### 4. Freeze Results and Rebuild Statistical Tables

```bash
python -m experiment.wsr.freeze_results \
  --root experiment/wsr/generated/runs \
  --output experiment/wsr/frozen_results

python -m experiment.wsr.stats \
  --root experiment/wsr/frozen_results \
  --output experiment/wsr/generated/tables
```

Controlled run outputs are stored under `experiment/wsr/generated/runs/controlled/<dataset>/<model>/seed_<seed>/`. The [complete reproduction guide](experiment/wsr/README.md) documents validation-only ablations, official recent-detector environments, latency measurement, mechanism diagnostics, corruptions, cross-domain evaluation, negative-template mitigation, and operating-point analysis.

## Evaluation Principles

- Use one standardized `pycocotools.COCOeval` implementation for all formal AP values.
- Select architectures, checkpoints, and operating points on validation data only.
- Keep the formal test set locked until the protocol is frozen.
- Preserve per-image prediction hashes and checkpoint hashes.
- Report paired seeds, uncertainty, latency, memory, and negative-board false alarms.
- Do not combine values from different splits, resolutions, evaluators, or input contracts in one ranking.

## Tests

```bash
python -m pytest experiment/wsr/tests -q
```

The cleaned repository passes all 43 protocol and inference tests.

## Citation

```bibtex
@misc{liang2026wsr,
  title        = {Wavelet Conditioned Sparse Routing for PCB Defect Detection},
  author       = {Liang, Jiefeng and Luo, Lihua and Tan, Sijin and Lin, Yanni and Zhao, Zhizhuo and Cai, Zhaofeng},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/master-abc/WSR-YOLO}
}
```
