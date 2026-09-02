# Audited Paper Experiment Track

This track replaces the original "single smoke test plus empty tables" workflow with an auditable and reproducible paper protocol. It does not guarantee paper acceptance or manufacture state-of-the-art claims before training is complete. The versioned paper source is [`../../paper/main.tex`](../../paper/main.tex).

See [`STATUS.md`](STATUS.md) for completion status, blocking items, and submission gates. Install the track-specific dependencies before running the workflow:

```powershell
pip install -r experiment\paper_b\requirements-paper-b.txt
```

## Core Research Questions

The proposed module is Wavelet-Conditioned Top-k Sparse Routing (WSR). Unlike the earlier soft mask, WSR gathers only the top `rho * H * W` P3 tokens, applies LayerNorm plus an MLP, and scatters them back to the original feature map. The refinement operator therefore scales linearly with `rho`; the default routing budget is `rho=12.5%`.

The paper addresses three primary questions:

1. Does WSR consistently outperform the baseline under the same YOLO11s training budget?
2. Is it competitive in accuracy, latency, and model size against runnable detectors released from 2024 to 2026?
3. Does routing concentrate on defects rather than merely produce an attractive attention map?

## Data Protocol

| Dataset | Train | Validation | Test | Role |
|---|---:|---:|---:|---|
| DsPCBSD+ | 7,387 | 821 | 2,051 | Primary industrial dataset; the official validation split is locked as the final test set |
| DeepPCB | 850 | 150 | 500 | Classical benchmark, robustness analysis, and cross-domain experiments |
| DefectDet | 188 | 40 | 40 | Exploratory only; disabled by default because sequence/template groups are unavailable |
| PCB-IND | Pending official data | Pending board-level grouping | Pending lock | External 2026 industrial validation; disabled by default |

The data-split seed is fixed at 2026 and is separate from the model seeds. Perceptual auditing identifies visually similar board candidates in public PCB datasets but no exact cross-split SHA-256 duplicates. Because the original releases do not provide complete board or lot identifiers, this repository claims that the risk was audited and disclosed, not that board-level independence was proven. The local `PKU_PCB` boxes are almost all placeholder-like `0.5 0.5 0.8 0.8` boxes, so the audit rejects that dataset. PCB-IND must not be enabled until the official data and license are available and a board- or lot-grouped split can be established.

```powershell
cd DWGSA-YOLO
python -m experiment.paper_b.prepare_dspcbsd
python -m experiment.paper_b.run prepare --dataset dspcbsd_plus --coco
python -m experiment.paper_b.run prepare --dataset deeppcb --coco
python -m experiment.paper_b.run audit
```

## Required Execution Order

### 1. Complete Pilot Selection and Ablation on Validation Data Only

```powershell
python -m experiment.paper_b.pilot plan
python -m experiment.paper_b.pilot train --device 0
python -m experiment.paper_b.pilot diagnose --device 0
python -m experiment.paper_b.pilot benchmark --device 0
python -m experiment.paper_b.pilot evaluate
python -m experiment.paper_b.pilot freeze

# The full paper ablation remains validation-only.
python -m experiment.paper_b.ablation materialize
python -m experiment.paper_b.ablation train --variant route_p3_12p5 --seed 13 --device 0
```

The pilot runs the YOLO11s validation baseline and the P3/12.5% candidate with seeds 13, 42, and 3407. It checks mean AP50:95 gain, route enrichment, and same-hardware latency on the validation set. Both models use a fixed 35% training subset, a 30-epoch limit, and the complete validation split. `freeze` records a selection decision only if all three gates pass. The full component ablation also evaluates validation data only and never reads the test set.

### 2. Run the Same-Architecture Controlled Experiment

```powershell
python -m experiment.paper_b.run plan
python -m experiment.paper_b.run train --dataset dspcbsd_plus --model yolo11s --seed 13 --device 0
python -m experiment.paper_b.run train --dataset dspcbsd_plus --model wsr_yolo11s_p3_r25 --seed 13 --device 0
```

The baseline and proposed model use seven paired seeds: 13, 42, 3407, 4703, 8391, 9475, and 10501. Seven pairs allow an exact two-sided Wilcoxon test to reach `p<0.05`; with five pairs, the theoretical minimum p-value is 0.0625. Add `--resume` after an interruption. A run with an existing standardized result is skipped by default.

Before a formal run, commit the current code, keep the Git worktree clean, and complete the full data audit with SHA-256 and perceptual near-duplicate checks. Inserting WSR at P3 changes later Ultralytics layer indices, so the scripts explicitly remap pretrained YOLO11s parameters. Training is rejected when target-parameter coverage is below 99%. The formal budget is at most 80 epochs with `batch=8`, `nbs=64`, and patience 15. WSR must not receive a uniquely reduced resolution or training budget. See [`COMPUTE_BUDGET.md`](COMPUTE_BUDGET.md) for measured cost estimates.

### 3. Run the Recent-Detector Comparison Track

The main table accepts only runnable official implementations: YOLOv10-M, YOLO11-M, YOLOv12-M, YOLO26-M, RF-DETR-M, RT-DETRv2-S, D-FINE-M, and DEIM-D-FINE-M. RT-DETRv4 depends on a DINOv3 teacher, and DEIMv2 has a high training cost, so both remain deferred runnable entries. PCB-MMF, PCB-FS, Structure-Guided PCB Detection, UniPCB, SCP-DETR, and LSDM-PCB appear only in a separate domain-method table with their dataset, split, and `reported only` status clearly identified.

```powershell
python -m experiment.paper_b.run prepare --dataset dspcbsd_plus --coco
python -m experiment.paper_b.external bootstrap --repos-root D:\paper_b_repos
python -m experiment.paper_b.external materialize --dataset dspcbsd_plus --repos-root D:\paper_b_repos --device 0
```

Use a separate environment for each official repository. The generated `commands.ps1` never selects a test checkpoint automatically. Select a checkpoint on validation data, replace `<BEST_VALIDATION_CHECKPOINT>`, and then perform exactly one test-only evaluation. Never select weights by test AP.

Every framework must export raw COCO detection JSON before the shared `pycocotools.COCOeval` evaluator computes AP. Do not transcribe AP manually from detector logs. Official COCO summaries use at most 100 detections per image; the export stage may retain 300 candidates. Single-point precision, recall, and F1 are excluded from the main table because no validation threshold was preregistered.

### 4. Run Mechanism, Robustness, and Transfer Experiments

```powershell
# Measure whether routes are enriched inside ground-truth boxes.
python -m experiment.paper_b.mechanism_diagnostics --weights PATH\best.pt --data experiment\paper_b\generated\datasets\deeppcb\dataset.yaml --output route.json

# Generate eight corruption families at five severity levels.
python -m experiment.paper_b.corruptions experiment\paper_b\generated\datasets\deeppcb\dataset.yaml experiment\paper_b\generated\robustness\deeppcb

# Generate pixel-level LL, high-frequency, and directional-subband interventions.
python -m experiment.paper_b.frequency_interventions experiment\paper_b\generated\datasets\deeppcb\dataset.yaml experiment\paper_b\generated\frequency\deeppcb

# Evaluate generated suites with one frozen checkpoint.
python -m experiment.paper_b.evaluate_suite --weights PATH\best.pt --suite-root experiment\paper_b\generated\frequency\deeppcb --output frequency_result.json --model wsr_yolo11s_p3_r25 --dataset deeppcb --seed 13

# Generate few-shot subsets.
python -m experiment.paper_b.few_shot experiment\paper_b\generated\datasets\dspcbsd_plus\dataset.yaml experiment\paper_b\generated\few_shot\dspcbsd_plus

# Build a genuine zero-shot cross-domain pair over five shared classes.
python -m experiment.paper_b.cross_domain remap experiment\paper_b\generated\datasets\deeppcb\dataset.yaml experiment\paper_b\generated\cross_domain\deeppcb_common5
python -m experiment.paper_b.cross_domain remap experiment\paper_b\generated\datasets\dspcbsd_plus\dataset.yaml experiment\paper_b\generated\cross_domain\dspcbsd_common5
python -m experiment.paper_b.cross_domain pair experiment\paper_b\generated\cross_domain\deeppcb_common5\dataset.yaml experiment\paper_b\generated\cross_domain\dspcbsd_common5\dataset.yaml experiment\paper_b\generated\cross_domain\deep_to_dsp
```

If the official DeepPCB template images are restored, `false_positive.py` can report defect-free board FPR and FPPI. These metrics are valid only after manual confirmation that every input is defect-free.

#### Negative-Aware Mitigation

False-alarm mitigation must isolate official test templates from training and threshold selection. `restore_deeppcb_split.py` restores the frozen 850/150/500 split from the official release. `negative_aware.py prepare` adds corresponding defect-free templates with empty annotations to training and validation while exporting, but never training on, the 500 test templates.

Stage 1 uses seed 13 validation data to select a 25% fraction of training negatives and freezes that choice for the other seeds. Stage 2 starts from each Stage-1 checkpoint, adds all 850 training templates, and fine-tunes for three epochs. Stage 3 uses seed 13 training-template scores to freeze the hardest 25% and repeats those examples three times, then applies the same three-epoch `lr0=1e-4` recipe to all three Stage-2 checkpoints. Every stage selects checkpoints on a combined positive-plus-defect-free validation set. These are post-hoc mitigation experiments with additional data and compute; they do not replace the original formal results.

```powershell
python -m experiment.paper_b.negative_aware prepare `
  --base-data experiment\paper_b\generated\datasets\deeppcb\dataset.yaml `
  --template-list PATH\deeppcb_official_templates.txt `
  --output experiment\paper_b\generated\negative_aware\dataset

python -m experiment.paper_b.negative_aware train `
  --data experiment\paper_b\generated\negative_aware\dataset\dataset.yaml `
  --architecture experiment\configs\wsr_yolo11s_p3_r25.yaml `
  --pretrained yolo11s.pt --seed 13 --device 0 `
  --output experiment\paper_b\generated\negative_aware\runs\wsr_seed13 `
  --positive-test-annotations experiment\paper_b\generated\coco\deeppcb\annotations\instances_test.json `
  --positive-test-images experiment\paper_b\generated\coco\deeppcb\test

# Stage 2: choose the recipe on seed 13 validation data, then reproduce it unchanged.
python experiment\paper_b\negative_aware.py prepare `
  --base-data experiment\paper_b\generated\negative_aware_local\base\dataset.yaml `
  --template-list experiment\paper_b\generated\negative_aware_local\base\templates.txt `
  --train-negative-fraction 1.0 `
  --output experiment\paper_b\generated\negative_aware_local\frac100

python experiment\paper_b\negative_aware.py finetune `
  --data experiment\paper_b\generated\negative_aware_local\frac100\dataset.yaml `
  --initial-weights PATH\stage1_seed13_best.pt `
  --output experiment\paper_b\generated\negative_aware_local\runs\stage2_frac100_seed13 `
  --seed 13 --epochs 3 --lr0 0.0003 --batch 8 --workers 4 --device 0

# Stage 3: rank training templates only and reuse one frozen hard-negative dataset.
python experiment\paper_b\prepare_hard_negatives.py `
  --base-data experiment\paper_b\generated\negative_aware_local\frac100\dataset.yaml `
  --negative-audit experiment\paper_b\generated\negative_aware_local\stage2_seed13_negative_train.json `
  --hard-fraction 0.25 --repeat 3 `
  --output experiment\paper_b\generated\negative_aware_local\hard25_r3

python experiment\paper_b\negative_aware.py finetune `
  --data experiment\paper_b\generated\negative_aware_local\hard25_r3\dataset.yaml `
  --initial-weights PATH\stage2_seed13_best.pt `
  --output experiment\paper_b\generated\negative_aware_local\runs\stage3_hard25_r3_seed13 `
  --seed 13 --epochs 3 --lr0 0.0001 --batch 8 --workers 4 --device 0
```

#### Operating Points and Consensus

For deployment operating points, `operating_point.py` selects a global threshold on an explicitly separate pool of 150 validation templates and evaluates it once on the isolated 500 test templates and positive test images. The script rejects overlapping calibration and holdout pools. `positive_operating_point.py` selects per-class F1 or F-beta thresholds from positive validation data only. `aggregate_mitigation.py` and `aggregate_operating_points.py` summarize multi-seed fixed-threshold and calibrated-threshold results. These results must be labeled as operating-point or post-hoc mitigation evidence and kept separate from threshold-independent COCO AP.

`consensus_ensemble.py` strictly separates `select` and `evaluate`. Selection reads validation predictions and freezes a policy JSON; evaluation may process test predictions only after that policy exists. The current exploratory policy requires three models to emit same-class boxes, each with IoU at least 0.3 to a common anchor and confidence at least 0.65. It produces 0.6% board FPR and 0.813 recall on 500 test templates. Because this method family was introduced after observing single-model test behavior, its JSON retains a protocol-adaptation warning and the result is not confirmatory or production-ready without a new holdout.

```powershell
python experiment\paper_b\consensus_ensemble.py select `
  --positive-annotations PATH\instances_val.json `
  --positive-predictions PATH\seed13_positive_val.json PATH\seed42_positive_val.json PATH\seed3407_positive_val.json `
  --negative-predictions PATH\seed13_negative_val.json PATH\seed42_negative_val.json PATH\seed3407_negative_val.json `
  --target-board-fprs 0 --output PATH\consensus_policy.json

python experiment\paper_b\consensus_ensemble.py evaluate `
  --policy PATH\consensus_policy.json --positive-annotations PATH\instances_test.json `
  --positive-predictions PATH\seed13_positive_test.json PATH\seed42_positive_test.json PATH\seed3407_positive_test.json `
  --negative-predictions PATH\seed13_negative_test.json PATH\seed42_negative_test.json PATH\seed3407_negative_test.json `
  --output PATH\consensus_test.json
```

DsPCBSD+ does not contain independent defect-free boards, so only false detections within positive images can be post-processed. `validation_postprocess.py` jointly selects per-class F-beta thresholds and same-/cross-class overlap suppression from a validation grid, then freezes the policy for test predictions. It reports maximum-F1 and recall-constrained policies, the complete candidate grid, and input hashes. Report these results separately from the common-threshold formal comparison.

```powershell
python experiment\paper_b\validation_postprocess.py `
  --validation-annotations PATH\dspcbsd_instances_val.json `
  --validation-predictions PATH\dspcbsd_wsr_seed13_val_predictions.json `
  --test-annotations PATH\dspcbsd_instances_test.json `
  --test-predictions PATH\dspcbsd_wsr_seed13_test_predictions.json `
  --output PATH\dspcbsd_wsr_seed13_postprocess.json
```

### 5. Measure Efficiency and Aggregate Statistics

```powershell
python -m experiment.paper_b.benchmark --weights PATH\best.pt --data experiment\paper_b\generated\datasets\deeppcb\dataset.yaml --output speed_fp16.json --half
python -m experiment.paper_b.freeze_results --root experiment\paper_b\generated\runs
python -m experiment.paper_b.stats --root experiment\paper_b\frozen_results --output experiment\paper_b\generated\tables
```

The statistics tool writes CSV, Markdown, and LaTeX outputs with 95% t intervals, paired Wilcoxon tests, Holm correction, and paired Cohen's dz. `paper/main.tex` reads generated LaTeX tables to avoid manual transcription errors.

## Mandatory Pre-Submission Checks

- Complete at least seven paired controlled DsPCBSD+ runs and three repeats for each recent-detector comparison.
- Never mix different splits, resolutions, evaluators, or paper-reported values in one ranking.
- Do not select ablation structures on the test set.
- Report AP50:95, AP75, per-class AP, variance, intervals, latency, and peak memory.
- Generate every main-table result with the same COCOeval implementation, preserve prediction JSON hashes, and require at least 99% pretrained-parameter coverage.
- Run formal experiments only from a clean Git commit and freeze results into a traceable directory with `freeze_results`.
- Reduce the paper's mechanism claims if frequency counterfactuals and route enrichment do not support them.
- Generate every visualization from a trained, frozen checkpoint.
- Leave unfinished values blank; never write "state of the art" before the evidence exists.
