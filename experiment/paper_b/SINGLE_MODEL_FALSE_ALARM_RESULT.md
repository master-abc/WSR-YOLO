# Single-model false-alarm refinement result

## Status

The paired-reference detector materially reduces all three operational error
types with one YOLO model and one model forward pass. It does **not** yet meet a
strict 1% defect-free board-FPR target on the final fresh synthetic holdout:
the observed result is 1.4% (7/500 boards).

## Frozen operating contract

- Input: one inspected PCB and its pixel-aligned golden reference.
- Encoding (BGR): candidate, candidate plus signed change, candidate minus
  signed change; noise floor 3 and gain 4.
- Detector: one `epoch4.pt` checkpoint, one forward pass.
- Class score thresholds: open 0.20, short 0.48, mousebite 0.42, spur 0.28,
  copper 0.76, pin-hole 0.46.
- Reference evidence gate: the largest connected change component after a 3x3
  opening must occupy at least 6% of the predicted box.
- Policy and weight hashes are recorded in
  `generated/evidence/frozen_structural_safe_policy.json`.

## Results

The comparison baseline is the existing Stage-3 hard-negative seed-13 model at
confidence 0.25. Positive-image matching uses IoU 0.5.

| Metric | Stage-3 baseline | Frozen paired-reference model |
|---|---:|---:|
| Defect-free board FPR | 28.0% (140/500) | 1.4% (7/500) |
| Defect-free false boxes | 270 | 12 |
| Positive-image TP | 2970 | 3004 |
| Positive-image FP | 313 | 124 |
| Positive-image FN | 170 | 136 |
| Precision | 0.9047 | 0.9604 |
| Recall | 0.9459 | 0.9567 |
| F1 | 0.9248 | 0.9585 |
| COCO AP50-95 | 0.7161 | 0.7263 |

Validation-only selection produced 0/600 defect-free board alarms and positive
TP/FP/FN of 968/19/35 (precision 0.9807, recall 0.9651, F1 0.9729). The final
negative audit used separately generated perturbation replicate r2, which was
absent from training and policy selection.

Residual r2 alarms are concentrated in seven fixed board/template cases:
pin-hole contributes five boxes, mousebite five, and spur two. One template
contributes all five mousebite boxes.

## Deployment command

```powershell
python -m experiment.paper_b.single_model_inference `
  --candidate path/to/inspected.jpg `
  --reference path/to/aligned_golden.jpg `
  --policy experiment/paper_b/generated/evidence/frozen_structural_safe_policy.json `
  --output prediction.json `
  --annotated-image prediction.png `
  --device 0
```

The output records `single_model: true` and `forward_passes: 1`. The inference
entry point validates the frozen weight hash and applies exactly the frozen
encoding, class thresholds, and reference-evidence statistic.

## Limitations

- Candidate/reference registration is mandatory. The benchmark does not cover
  geometric misregistration.
- The final negative holdout is a deterministic synthetic acquisition-noise
  realization over DeepPCB test templates, not an independently collected
  production no-defect dataset.
- The 1.4% result is a 95% relative board-FPR reduction, but it must not be
  reported as zero false alarms or as satisfying a strict <=1% target.
- Production sign-off still requires a locked set of real defect-free boards,
  ideally stratified by template, camera, lighting, and lot.

## Verification

`python -m pytest experiment/paper_b/tests -q` completed with 43 passed and one
skipped test. An end-to-end deployment smoke test is recorded at
`generated/evidence/inference_smoke.json`.
