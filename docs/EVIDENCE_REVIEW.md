# Current evidence: fixed-split refit/model-selection sensitivity

Updated 2026-10-10. The registered extension repeats model selection and refitting in 256 paired draws. At the original 30% budget, visit-gain 2.5th/97.5th percentiles are [-0.252, +1.262] percentage points; 16 draws show no gain. Spend improvement is unvalidated. The original fixed-model +0.946 pp estimate remains historical, not a model-selection-robust promise.

Read [the complete comparison and limitations](REFIT_UNCERTAINTY_REVIEW.md), [protocol](../reference/refit_uncertainty/PROTOCOL.json) and [all results](../reference/refit_uncertainty/summary.json). These empirical ranges do not establish nominal coverage or a new untouched experiment. Treatment-specific policies and confirmation on genuinely new locked data remain open.

The note below is preserved historical evidence; its refit task is completed only within this newly defined fixed-split scope.

---

# Evidence review: Causal Decision Science

Independent portfolio research. Updated 2026-10-09. Development and documentation include AI assistance; the committed executable code, data provenance and test outputs establish the work products. No institutional endorsement or third-party authorship review is claimed.

## Research purpose

Experimentation, targeting and statistical decision-making.

## Published evidence

64,000 randomized customers; reconciled 12,800-row evaluation; same-budget targeting gain 0.946 percentage points, conditional 95% CI [0.333, 1.561].

## Interpretation boundary

Public retrospective evaluation; uncertainty is conditional on the fitted ranking. No profit or individual-treatment-effect claim.

## Review standard

Check the source data and split before interpreting a score. Compare the strongest result with a simple baseline. Inspect failed diagnostics and uncertainty. Reproduce the calculations from the documented environment; distinguish measured findings from simulated or assumed scenarios. The tests cover specific documented invariants and do not establish complete production correctness.

## Next research extension

Evaluate a preregistered policy on a new experiment and propagate model-selection/refit uncertainty. Expand to treatment-specific policies rather than collapsing both email variants. This release does not pretend to have collected a new experiment.
