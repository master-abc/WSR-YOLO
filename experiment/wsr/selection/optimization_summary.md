# Validation-Locked WSRStable Optimization Audit

## Decision

The three pre-registered engineering rounds are complete. No candidate passed the joint validation gate, so **no optimized candidate was selected and the locked test split was not evaluated**.

This is a negative selection result, not evidence for the originally targeted claim of a stable accuracy gain of at least 1.00 AP point with no more than 20% model-only latency overhead.

## Locked protocol

- Dataset/split: DsPCBSD+ training subset (35%) and the full validation split.
- Seeds: 13, 42, and 3407; 30 epochs per run.
- Accuracy gate: mean validation AP50:95 gain >= 0.0100 (1.00 AP point).
- Mechanism gate: mean route enrichment >= 2.0x.
- Efficiency gate: same-GPU model-only latency ratio <= 1.20x.
- Selection data: validation results only. `test_evaluated=false` in every frozen decision.
- Repeated baseline AP50:95 values: 0.329394, 0.325491, and 0.325480 (mean 0.326789) in all three rounds.

The audit contains 42 completed train/evaluate jobs (12 + 15 + 15), 33 seed-level route diagnostics, and 14 same-GPU latency measurements. No training job failed.

## Candidate results

AP gain is reported in percentage points relative to the repeated validation baseline. A check mark denotes passage of the individual pre-registered gate.

| Round | Candidate | Mean AP50:95 | AP gain (pp) | Seed wins | Route enrichment | Latency ratio | Accuracy | Mechanism | Latency |
|---|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 1 | `stable_p3_12p5` | 0.332108 | +0.532 | 2/3 | 2.832x | 1.125x | no | yes | yes |
| 1 | `stable_p3_25` | 0.332656 | +0.587 | 2/3 | 2.239x | 1.200x | no | yes | no |
| 1 | `stable_p3p4_12p5` | 0.339187 | +1.240 | 3/3 | 3.380x | 1.400x | yes | yes | no |
| 2 | `stable_p4_12p5_s10` | 0.330382 | +0.359 | 2/3 | 3.497x | 1.199x | no | yes | yes |
| 2 | `stable_p4_25_s10` | 0.329443 | +0.265 | 2/3 | 2.844x | 1.153x | no | yes | yes |
| 2 | `stable_p4_12p5_s20` | 0.333311 | +0.652 | 2/3 | 3.071x | 1.153x | no | yes | yes |
| 2 | `stable_p4_25_s20` | 0.326900 | +0.011 | 2/3 | 2.773x | 1.195x | no | yes | yes |
| 3 | `stable_p3_r12p5_w35_s10` | 0.327659 | +0.087 | 1/3 | 2.966x | 1.190x | no | yes | yes |
| 3 | `stable_p3_r25_w35_s10` | 0.332452 | +0.566 | 2/3 | 2.355x | 1.217x | no | yes | no |
| 3 | `stable_p3_r12p5_w45_s10` | 0.326995 | +0.021 | 2/3 | 2.968x | 1.181x | no | yes | yes |
| 3 | `stable_p3_r25_w45_s10` | 0.331266 | +0.448 | 2/3 | 2.433x | 1.251x | no | yes | no |

Round decisions and protocol hashes:

- Round 1, `stable_residual_v1`: `FAIL`; protocol `9a483a9beac79d76ae04de687c3bf29bd62bcc7c8037dfd7e3649fb8669d9eab`.
- Round 2, `stable_p4_factorial_v2`: `FAIL`; protocol `a13dcd5d8646a45478ea0875b3490f4c55291716c54d44b058e21778700df02b`.
- Round 3, `stable_p3_asymmetric_v3`: `FAIL`; protocol `c8f9ff4735d9585433a197d0bac3fdeed4525779e8768f27c314044cc1a6fcaa`.

## Interpretation for the paper

The previously frozen seven-seed formal result provides the necessary context: original WSR scores 0.466909 versus 0.465363 for YOLO11s on DsPCBSD+ (gain +0.155 AP points; 4/7 wins; two-sided Wilcoxon p=0.8125; paired standardized effect dz=0.207; 95% paired-t interval -0.535 to +0.844 AP points). Its mean also remains below the frozen recent M-scale detector means (0.494444 to 0.530779). The new validation-only optimization does not overturn that formal-test conclusion.

### Supported by these experiments

- The stable router consistently concentrates computation on higher-error locations: every candidate passes the mechanism gate, with route enrichment from 2.239x to 3.497x.
- A measurable accuracy/latency trade-off exists. The highest validation gain is +1.240 AP points with 3/3 seed wins, but it costs 1.400x baseline latency.
- Among candidates that meet the 1.20x latency limit, the best validation result is `stable_p4_12p5_s20`: +0.652 AP points, 3.071x route enrichment, and 1.153x latency.
- The implementation-level invariants for scale-stable routing and unchanged unrouted sparse tokens are covered by automated tests.

### Not supported

- A repeatable >=1.00 AP-point improvement while remaining within 1.20x latency.
- A new formal-test improvement from the optimized architecture; the test split was intentionally not opened after gate failure.
- State-of-the-art superiority or a statistically reliable primary-dataset improvement.
- Treating mechanism enrichment alone as proof of detection-quality improvement.

## Defensible manuscript position

The results can support a mechanism and accuracy/efficiency trade-off study, including a transparent negative selection result. They cannot support the manuscript as a state-of-the-art accuracy claim. If the primary thesis requires the original joint claim, the current evidence is insufficient and the claim must be weakened or the method materially redesigned before a new independently registered experiment.

Frozen machine-readable decisions:

- `optimization_round1_decision.json`
- `optimization_round2_decision.json`
- `optimization_round3_decision.json`
