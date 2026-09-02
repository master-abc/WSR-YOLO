# WSR-YOLO: Wavelet-Conditioned Sparse Routing for PCB Defect Detection

This repository contains the paper, implementation, experiment protocol, and frozen audit artifacts for **Wavelet Conditioned Sparse Routing for PCB Defect Detection**.

> [!IMPORTANT]
> The formal paper evidence is produced exclusively by the audited track in [`experiment/paper_b/`](experiment/paper_b/). The older `experiment/exp1`--`experiment/exp5` DWGSA/YOLO11m experiments are retained only for historical traceability and must not be used to support the current paper's conclusions.

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
- Audited experiment guide: [`experiment/paper_b/README.md`](experiment/paper_b/README.md)
- Current experiment status: [`experiment/paper_b/STATUS.md`](experiment/paper_b/STATUS.md)
- Frozen formal results: [`experiment/paper_b/frozen_results/`](experiment/paper_b/frozen_results/)
- Tests: [`experiment/paper_b/tests/`](experiment/paper_b/tests/)

## Repository Structure

```text
.
|-- algorithm/                     # WSR, legacy DWGSA, and comparison modules
|-- experiment/
|   |-- paper_b/                   # Audited protocol used by the current paper
|   |-- configs/                   # Model and dataset configurations
|   |-- exp1/ ... exp5/            # Legacy DWGSA experiments
|   `-- scripts/                   # Supporting evaluation and visualization tools
|-- paper/                         # IEEE LaTeX sources, figures, and generated documents
|-- tests/                         # Repository-level tests
|-- requirements.txt
`-- run_all.py
```

## Installation

The controlled paper environment uses Python 3.10, PyTorch 2.5.1 with CUDA 12.1, and Ultralytics 8.4.50. Install the appropriate PyTorch build for your hardware if CUDA 12.1 is unavailable.

```bash
conda create -n wsr-yolo python=3.10 -y
conda activate wsr-yolo

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r experiment/paper_b/requirements-paper-b.txt
```

Verify the custom-module registration:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "from algorithm.register import register_custom_modules; register_custom_modules()"
```

## Audited Experiment Workflow

Prepare datasets and run the leakage audit before training:

```bash
python -m experiment.paper_b.prepare_dspcbsd
python -m experiment.paper_b.run prepare --dataset dspcbsd_plus --coco
python -m experiment.paper_b.run prepare --dataset deeppcb --coco
python -m experiment.paper_b.run audit
```

Materialize and run the controlled same-architecture experiment:

```bash
python -m experiment.paper_b.run plan
python -m experiment.paper_b.run train --dataset dspcbsd_plus --model yolo11s --seed 13 --device 0
python -m experiment.paper_b.run train --dataset dspcbsd_plus --model wsr_yolo11s_p3_r25 --seed 13 --device 0
```

Formal runs require a clean, resolvable Git commit, a completed full-data audit, and at least 99% pretrained-parameter coverage after WSR insertion. See the [full experiment guide](experiment/paper_b/README.md) for paired seeds, comparator environments, evaluation, negative-template mitigation, mechanism diagnostics, and result freezing.

## Evaluation Principles

- Use one standardized `pycocotools.COCOeval` implementation for all formal AP values.
- Select architectures, checkpoints, and operating points on validation data only.
- Keep the formal test set locked until the protocol is frozen.
- Preserve per-image prediction hashes and checkpoint hashes.
- Report paired seeds, uncertainty, latency, memory, and negative-board false alarms.
- Do not combine values from different splits, resolutions, evaluators, or input contracts in one ranking.

## Tests

```bash
python -m pytest experiment/paper_b/tests -q
```

The last verified test run completed with 43 passed tests and one skipped test.

## Legacy DWGSA Experiments

The original YOLO11m/DWGSA experiments remain under `experiment/exp1`--`experiment/exp5`, with an English overview in [`experiment/README_experiments.md`](experiment/README_experiments.md). They are useful for implementation history only. Their datasets, model scale, evaluation assumptions, and claims differ from the frozen WSR paper protocol.

## Citation

```bibtex
@misc{liang2026wsr,
  title        = {Wavelet Conditioned Sparse Routing for PCB Defect Detection},
  author       = {Liang, Jiefeng and Luo, Lihua and Tan, Sijin and Lin, Yanni and Zhao, Zhizhuo and Cai, Zhaofeng},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/master-abc/DWGSA-YOLO}
}
```
