# Causal Decision Science — Randomized Experiments & Targeting

Independent research using Hillstrom’s public 64,000-customer randomized email experiment. This release reproduces all outputs together, records data/environment identity and reconciles policy summaries to decile aggregates.

**Review:** [Policy audit](results/same_budget_policy_audit.json) · [Bootstrap effects](results/final_test_policy_bootstrap.csv) · [Source/environment identity](results/run_metadata.json) · [Full CI reproduction](https://github.com/Prasanna-K-123/causal-marketing-experimentation-uplift/actions/runs/37962514536) · [Evidence review](docs/EVIDENCE_REVIEW.md)

## Registered refit and selection challenge (October 10)

The earlier fixed-model gain is reproduced exactly, but repeating model selection and refitting in 256 paired customer resamples gives visit-gain 2.5th/97.5th percentiles **[-0.252, +1.262] percentage points** at the same 30% budget. **16/256** draws have no gain. Logistic wins selection in 143 draws and boosting in 113. Spend improvement remains unvalidated. These are empirical sensitivity ranges conditional on this inspected experiment/split, not coverage-certified confidence intervals or a new untouched trial.

[Full methods, comparisons and adverse findings](docs/REFIT_UNCERTAINTY_REVIEW.md) · [Registered protocol](reference/refit_uncertainty/PROTOCOL.json) · [All 256 draws](reference/refit_uncertainty/replicates.csv) · [Summary and Monte Carlo precision](reference/refit_uncertainty/summary.json) · [Pinned replay workflow](.github/workflows/refit-uncertainty.yml)

The extension preserves the three actual randomized arms, refits the original chosen family, repeats the original two-family validation choice, and pairs evaluation draws across those modes. A separate known-design 50:50 email-mixture check also crosses zero. Policy membership/frequency, all 4,096 fixed-ranking draws and exact source/split hashes are retained. No new policy is selected from these outcomes.

## Experimental design

Customers were randomized between men’s email, women’s email and no-email control. Visit, conversion and spend outcomes are analysed separately. Exact duplicate rows remain because there is no customer identifier that justifies deduplication. Pre-treatment balance has maximum absolute SMD 0.0086.

All six email-versus-control average-effect tests remain significant after Holm correction. Each average effect uses 5,000 bootstrap resamples. The pooled-email targeting stage compares logistic and gradient-boosted T-learners using a fixed 60/20/20 train/validation/evaluation split. The 30% budget and visit outcome are fixed; final evaluation does not choose another budget or model.

## Preserved fixed-model evaluation (October 9)

| Result | Estimate | Conditional bootstrap 95% interval |
|---|---:|---:|
| Top-30% visit effect | 8.284 pp | [5.715, 10.887] pp |
| Remaining-70% visit effect | 3.780 pp | [2.369, 5.173] pp |
| Top-minus-bottom visit effect | 4.503 pp | [1.585, 7.433] pp |
| Same-budget targeting gain versus random 30% targeting | 0.946 pp | [0.333, 1.561] pp |

The last row measures additional visits across the entire eligible population, not only the contacted group: q(1-q)(tau_top - tau_bottom), with q=0.30. The audit verifies that the saved policy and decile artifacts agree.

Spend heterogeneity does not validate: top-minus-bottom is approximately -$0.485/customer and its 95% interval [-$1.455, $0.461] spans zero. No revenue, profit or ROI improvement is claimed. Bootstrap fractions above zero are descriptive resampling frequencies, not posterior probabilities.

## Reproduce

```bash
pip install -r requirements.txt pytest
python -m pytest -q
python src/causal_experimentation_uplift.py
python audit_policy.py
```

The loader tries the author-hosted URL before the recorded public mirror. The successful 2026-10-09 run used the mirror; payload and parsed-row hashes are retained. CI stores the full scored evaluation records as an artifact and exports all compact result tables together. The published holdout has been inspected and reproduced; subsequent research requires a genuinely new locked evaluation, and this release does not rebrand the rerun as a new independent experiment.

## Scope

The historical policy interval is conditional on the learned ranking and fixed segments; the separate registered extension propagates fixed-split refit/selection variation with the explicit limits above. Pooled email combines two treatment variants. Segment-average causal evidence does not reveal an individual customer’s treatment effect. This public experiment is not client work or production deployment.
