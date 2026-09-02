# Archived Preliminary Experiments

> [!WARNING]
> This directory documents preliminary experiments that predate the final WSR protocol. It is retained for historical traceability and does not provide evidence for the current paper. Use [`paper_b/`](paper_b/) for the audited formal protocol.

The archived track uses an earlier YOLO11m attention prototype, different datasets, and different evaluation assumptions. Its results must not be mixed with the current paper tables.

## Experiment Overview

| Experiment | Research Question | Original Objective |
|---|---|---|
| `exp1` | How does the preliminary module compare with generic attention modules? | Compare domain-specific and general-purpose attention |
| `exp2` | How much does each preliminary component contribute? | Perform component ablations |
| `exp3` | Does the model transfer across datasets? | Explore cross-dataset generalization |
| `exp4` | Does attention focus on defects? | Visualize the original design motivation |
| `exp5` | Is the model robust to synthetic noise? | Explore the effect of wavelet features under corruption |

## Quick Start

```bash
# 1. Generate miniature synthetic datasets for smoke tests.
python experiment/create_mini_dataset.py

# 2. Run short smoke tests.
python experiment/exp1/run.py --smoke
python experiment/exp2/run.py --smoke
python experiment/exp3/run.py --smoke
python experiment/exp4/run.py --smoke

# 3. Run the archived full experiments with real datasets and a GPU.
python experiment/exp1/run.py --full
python experiment/exp2/run.py --full
python experiment/exp3/run.py --full
python experiment/exp4/run.py --full --dwgsa-weights experiment/exp1/runs/dwgsa_yolo11m/weights/best.pt
```

## Datasets

- **DeepPCB:** 1,500 images and six classes; used by the original Exp1 and Exp2. Download with `python experiment/data/download_deeppcb.py`.
- **DefectDet:** 268 images and five classes; used by the original transfer experiment. Convert with `python experiment/data/convert_defectdet.py`.
- **PKU_PCB:** Image-level class annotations with placeholder-like bounding boxes. The audited WSR protocol rejects it as a detection benchmark.
- **Mini datasets:** Synthetic data generated only for smoke testing.

## Archived Experiment Details

### Exp1: Attention-Module Comparison

Compares the preliminary attention module with CBAM, EMA, SimAM, CoordAtt, and FDSA. The original hypothesis was that high-frequency anomalies and PCB geometry could benefit from domain-specific attention.

### Exp2: Component Ablation

Removes the original components one at a time:

- Wavelet branch for high-frequency feature extraction
- Geometry-guided sparse attention
- Adaptive fusion driven by the physical signal
- Two-level DWT for multiresolution decomposition

### Exp3: Cross-Dataset Exploration

Explores transfer to DefectDet and PKU_PCB. These runs do not satisfy the data-independence and annotation-quality requirements of the current paper protocol.

### Exp4: Visualization

Produces Grad-CAM, wavelet-subband, and geometry-prior visualizations for the archived module.

### Exp5: Synthetic-Noise Robustness

Evaluates clean-trained models under Gaussian noise with `sigma` values of 0, 3, 6, 9, and 15. This exploratory setup must not be confused with the eight-corruption audited suite in `paper_b`.
