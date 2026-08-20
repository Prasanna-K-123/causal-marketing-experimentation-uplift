# Causal Marketing Experimentation, Bootstrapping & Uplift Modeling

Independent portfolio project using the **Hillstrom 64,000-customer randomized email experiment**.

## Objective

Separate three questions that are often incorrectly mixed together:

1. **Average causal effect:** did the email campaigns cause incremental visits, conversions, and spend?
2. **Uncertainty:** how precise are those effect estimates?
3. **Heterogeneity:** can pre-treatment customer features identify a segment with stronger incremental visit response?

## Dataset

The experiment contains **64,000 customers** randomized approximately equally across:

- Men's Email
- Women's Email
- No Email control

Observed post-treatment outcomes include:

- visit
- conversion
- spend

Pre-treatment features include recency, purchase history, prior men's/women's merchandise behavior, new-customer status, ZIP category, and channel.

The notebook attempts the original author-hosted data URL first and uses a mirror only if that source is unavailable.

### Data-integrity choice

The file contains exact duplicate rows but no unique customer identifier. These rows are **not deduplicated**, because identical observed rows can represent different randomized customers. Removing them without an identifier would alter the experiment without evidence.

## 1. Randomization balance

Pre-treatment numeric/binary covariates were checked using standardized mean differences.

**Maximum absolute SMD: 0.0086**

This is far below the common 0.10 practical-imbalance threshold and supports the integrity of the randomized comparison on the observed covariates.

## 2. Average treatment effects

Each email arm is compared with the no-email control on visit, conversion, and spend.

| Treatment | Outcome | Absolute effect |
|---|---|---:|
| Men's Email | Visit | **+7.66 pp** |
| Men's Email | Conversion | **+0.681 pp** |
| Men's Email | Spend | **+$0.770/customer** |
| Women's Email | Visit | **+4.52 pp** |
| Women's Email | Conversion | **+0.311 pp** |
| Women's Email | Spend | **+$0.424/customer** |

All **6/6** treatment-vs-control effects remain statistically significant after **Holm family-wise error correction**.

These comparisons support causal interpretation because treatment assignment was randomized.

The project does **not** claim that Men's Email is statistically better than Women's Email; that would require a direct treatment-vs-treatment contrast.

## 3. Bootstrap uncertainty

Each of the six average treatment effects is evaluated with **5,000 bootstrap resamples**.

Examples:

- Men's Email visit effect: **+7.66 pp**, bootstrap 95% CI approximately **[+7.02, +8.33] pp**
- Men's Email spend effect: **+$0.770/customer**, bootstrap 95% CI approximately **[$0.492, $1.059]**
- Women's Email visit effect: **+4.52 pp**, bootstrap 95% CI approximately **[+3.88, +5.15] pp**

All six bootstrap intervals remain above zero.

## 4. Uplift modeling design

For the targeting stage, the two email arms are collapsed into:

**any email vs no email**

The pre-selected uplift target is **visit**.

An honest **60 / 20 / 20** train-validation-final-test split is used:

- Train: 38,400
- Validation: 12,800
- Untouched final test: 12,800

Two T-learners are compared on validation data:

- logistic T-learner
- gradient-boosted T-learner

The model-selection criterion is the empirical **top-30% minus bottom-70% visit uplift** on validation.

The logistic T-learner wins the validation comparison and is then frozen.

## 5. Final untouched-test result

After retraining the selected logistic T-learner on train + validation data, the model is evaluated once on the final test set.

| Final-test segment | Randomized visit uplift |
|---|---:|
| Top 30% predicted uplift | **+8.29 pp** |
| Remaining 70% | **+3.76 pp** |
| Difference | **+4.53 pp** |

Bootstrap validation of the difference:

- **95% CI: [+1.50, +7.52] pp**
- **Probability top-30% > bottom-70%: 99.9%**

This is evidence of **segment-level treatment-effect heterogeneity** on the untouched final test.

It is not a claim that the model knows each customer's individual causal effect.

## 6. Negative result retained

Spend heterogeneity did **not** validate:

- top-30% minus bottom-70% spend effect: about **-$0.47/customer**
- bootstrap 95% CI spans zero
- probability top-30% > bottom-70%: **17.2%**

No spend-targeting, profit, or ROI claim is made.

Keeping this failure is deliberate: the project separates a successful visit-uplift result from an unsupported spend-uplift story.

## Tech stack

- Python
- Pandas / NumPy
- SciPy
- Statsmodels
- scikit-learn
- Logistic regression
- Gradient boosting
- T-learner uplift modeling
- bootstrap resampling
- Holm multiple-testing correction
- Matplotlib

## Repository structure

```text
.
├── README.md
├── causal_marketing_experimentation_uplift.ipynb
├── requirements.txt
└── results/
    ├── experiment_arm_outcomes.csv
    ├── randomization_balance.csv
    ├── average_treatment_effects.csv
    ├── bootstrap_treatment_effects.csv
    ├── uplift_model_validation.csv
    ├── final_test_uplift_deciles.csv
    ├── final_test_policy_bootstrap.csv
    └── final_project6_metrics.csv
```

## What this project demonstrates

- randomized experimentation and causal interpretation;
- practical randomization-balance diagnostics;
- hypothesis testing;
- multiple-testing correction;
- bootstrap confidence intervals;
- distinction between prediction and uplift;
- honest train/validation/final-test model selection;
- heterogeneous treatment-effect analysis;
- preservation of negative findings;
- business decision-making under uncertainty.

## Limitations

- The binary uplift stage collapses the two email variants and therefore does not model treatment-specific heterogeneity between Men's and Women's Email.
- T-learner uplift estimates are model-based and should not be interpreted as observed individual causal effects.
- The 30% targeting threshold is a modeling policy choice, not a proven economically optimal contact rate.
- No contact cost or customer lifetime value is available, so ROI is not calculated.
- Bootstrap intervals capture sampling uncertainty, not every source of model-selection uncertainty.
