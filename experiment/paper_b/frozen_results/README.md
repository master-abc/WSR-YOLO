# Frozen formal results

Only validated, non-smoke `standardized_result.json` files belong here. Generate
them with `python -m experiment.paper_b.freeze_results` from a clean Git commit.
Large checkpoints and raw prediction files remain outside Git; their SHA-256
digests are recorded in the frozen result and manifest.

The preregistered full-budget component study is validation-only evidence, not
a formal test result. Its 27 compact records and audited summary are tracked
separately under `../generated/revision_results/confirmatory/` and
`../generated/revision_results/confirmatory_summary.json`. Every one of those
records sets `selection_split=val` and `test_evaluated=false`; they must not be
used to reopen model selection or replace the frozen controlled test results.

