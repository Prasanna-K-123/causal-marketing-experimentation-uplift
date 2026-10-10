# Causal targeting challenged by refitting and model selection

The original fixed-model estimate is reproduced, but its apparent stability does not survive the full registered sensitivity calculation. Repeating selection and refitting in **256 paired customer resamples** produces visit-gain 2.5th/97.5th percentiles **[-0.252, +1.262] percentage points**; **16/256** draws have no gain. The median is **+0.630 pp**. Spend improvement remains unvalidated. This is a substantive adverse result, not a new winning model or a live campaign outcome.

Independent portfolio research with AI assistance. No institutional endorsement, client use or independent human review is claimed. The original public experiment and evaluation were already inspected. The registration precedes this extension's fits, not the 2008 experiment or all knowledge of outcomes. Empirical sensitivity ranges here are **not coverage-certified confidence intervals after model selection**.

## Why this method

| Alternative | What it answers | Decision |
|---|---|---|
| Keep the fitted ranking fixed and bootstrap outcomes | Evaluation variation conditional on a chosen fitted ranking | Preserve original evidence and add the paired conditional reference; insufficient alone for the refit question. |
| Refit only the original selected family | Sensitivity to development observations, holding family choice fixed | Execute on the same draws to isolate this assumption. |
| Repeat both candidate fits, validation choice and final refit | Development, candidate-choice and evaluation variation within the actual two-family procedure | Primary registered extension; no new configurations, features or budget. |
| Repeatedly change train/test memberships | Split sensitivity and potentially a different evaluation scheme | Not mixed into this fixed-split study. Would need a separately specified design; repeated use of inspected outcomes is not new independent evidence. |
| Learn treatment-specific policies, new estimators or a larger model menu | A different learning question | Remains future work. No post-result policy, family or majority-vote ensemble is promoted from this study. |
| Confirm on a genuinely new locked experiment | Independent population/campaign evidence | Stronger confirmation still needed; no such experiment was collected here. |

## Frozen source, learning rule and unit

- Public registration commit **1ae042393950d130d81020c20e94f0da89859b7e**, tree `77e54263def4bc34b456332de5dcddbd3224132b`; protocol SHA256 `ddcbeecdb1ac9ffb693df73ea7cf2970e3744e4c10d355b8e9e849cfc192c55d`. Nine new paths were remote-blob verified before source collection/real study fits. The original builder source is unchanged at `96e8b7c497c82dc6c2f6c458c20e7060c47f696e`. Frozen method guards reject later edits.
- Same pinned public mirror as the earlier run: [source](https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz), **443,070 bytes**, payload SHA256 `bab6578f60db5d792f1c2372c502f029152a5249cf5ea84390f3b7f885d7234f`, parsed-row SHA256 `f57d0ec295edfe413cc5ffe7d32fe7443d839cbfaefb4babacc7f80febe96bae`. No full raw source is redistributed. Collector refuses a changed payload or silently substituted mirror.
- **64,000 customer records**, arms: men email **21,307**, no email **21,306**, women email **21,387**. **6,562 identical records remain**; absent a customer ID, equality of published attributes does not justify deduplication. Source-order row ID is a reconstruction key, not observed identity. Record independence is assumed; household/clustering information is unavailable.
- Reconstruct original binary-email-stratified seed-2026 **38,400 training / 12,800 validation / 12,800 evaluation** roles and order. Original source-row assignments are disjoint; bootstrap copies remain inside their original role. Split uncertainty is excluded.
- Original logistic and 150-tree gradient-boosted T-learners, identical seven pre-treatment inputs and original fixed parameters. Preprocessing fits only within the relevant training treatment branch. Selection uses validation visit effect, not spend or evaluation outcomes; exact selection ties favour original insertion-order logistic. The chosen family is refitted on **51,200 resampled development records**.
- Budget fixed at **30%**, or **3,840 of 12,800 evaluation records**. Score-only rank uses descending predicted uplift then ascending source-order key (draw position last for repeated copies). Original final pandas order and this rule select identical customers; baseline cutoff has one equal score. Both original validation scores and **+0.009457130314735352** visit gain reproduce exactly.
- Each of 256 draws resamples whole records independently inside the three original randomized arms within each role, preserving each role/arm count. SeedSequence `[20261010, replicate, role]`; no outcome stratification. Same evaluation draw pairs all three fitted-policy modes. Model random-state is fixed, so algorithmic-seed variation is excluded.
- Original fixed family is refitted every draw. Repeated selection chooses logistic **143/256** and boosting **113/256**. Every draw is retained; zero failed/convergence-warning draws. The study executes **1,762 branch-classifier fits** for these 256 comparisons, plus six baseline fits; these are not 1,768 independent experiments.

## Policy value and full adverse evidence

Primary gain per eligible customer versus random selection at the same email budget is the preserved pooled-email estimator: `q*(1-q)*(tau_top-tau_bottom)`, `q=0.30`. It describes the randomized email mixture; it does not decide which email variant an individual should receive. The baseline +0.946 pp is an observed fixed-fit estimate, not the median across refitted learners.

All values below are percentage points in additional visits across the **entire eligible population**. Each range is the 2.5th/50th/97.5th empirical percentile, not the full minimum/maximum and not certified confidence coverage.

| Registered mode | 2.5th | Median | 97.5th | Nonpositive draws |
|---|---:|---:|---:|---:|
| Fixed original ranking; evaluation resampled | 0.309 | 0.952 | 1.537 | 1/256 |
| Original selected family refitted; evaluation resampled | 0.077 | 0.680 | 1.394 | 5/256 |
| Selection repeated + winner refitted; evaluation resampled | -0.252 | 0.630 | 1.262 | 16/256 |
| Refit family; original evaluation held fixed | 0.278 | 0.678 | 1.060 | 0/256 |
| Selection + refit; original evaluation held fixed | 0.114 | 0.623 | 1.030 | 3/256 |

Holding observed evaluation outcomes fixed isolates fit/selection variation and excludes evaluation sampling. Its positive lower percentile must not be substituted for the fuller range that includes evaluation variation. Likewise, the fixed-family range's lower tail has Monte Carlo uncertainty that crosses zero; it is not a reason to declare that only family selection matters.

Original conditional interval **[+0.333, +1.561] pp** remains historical. The additional **4,096** fixed-ranking draws have percentiles **[+0.386, +0.968, +1.592] pp**. This additional scheme preserves actual three-arm totals and recomputes the capacity cutoff; the original scheme fixed top/bottom memberships and resampled treatment/outcome groups. They answer different conditional sampling questions and are not supposed to have identical endpoints.

Spend gain is dollars per whole eligible customer at the same 30% budget. The original top-minus-bottom spend contrast **-$0.485** remains an inconclusive adverse finding; multiplying it by 0.21 gives baseline same-budget spend gain approximately **-$0.102/customer**. No profit, revenue, ROI or monetary advantage is validated.

| Registered mode | 2.5th dollars | Median dollars | 97.5th dollars | Nonpositive draws |
|---|---:|---:|---:|---:|
| Fixed original ranking; evaluation resampled | -0.288 | -0.100 | 0.073 | 208/256 |
| Original selected family refitted; evaluation resampled | -0.266 | -0.041 | 0.213 | 172/256 |
| Selection repeated + winner refitted; evaluation resampled | -0.279 | -0.054 | 0.172 | 173/256 |

## Randomization-weighted check and policy stability

The author's [experiment description](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) reports three equally probable randomized arms and two-week outcomes. A separate Horvitz-Thompson policy-minus-random estimate uses `mean((selected-q)*(1.5*Y*I(email)-3*Y*I(control)))`. It defines contacted customers as a **50:50 randomized men/women email mixture**, using design probabilities 1/3. It does not use observed subgroup email composition to redefine the policy. This is a registered secondary estimator; empirical ratios and design weighting need not match exactly in a finite trial.

Baseline visit gain is **+0.874 pp** under this estimator. Paired visit sensitivities, in percentage points:

| Registered mode | 2.5th | Median | 97.5th | Nonpositive draws |
|---|---:|---:|---:|---:|
| Fixed original ranking; evaluation resampled | 0.227 | 0.892 | 1.591 | 1/256 |
| Original selected family refitted; evaluation resampled | -0.110 | 0.605 | 1.365 | 10/256 |
| Selection repeated + winner refitted; evaluation resampled | -0.233 | 0.593 | 1.326 | 16/256 |

Reselection/refit design-weighted spend percentiles are **[-$0.280, -$0.054, +$0.179]** per eligible customer. Both estimators retain losses and an unvalidated spend advantage.

Against the original selected set, fixed-family refit Jaccard median is **0.641**, percentile range **[0.493, 0.781]**. Reselection/refit median is **0.527**, range **[0.347, 0.752]**. Jaccard is intersection divided by union, not percentage of the original contacted group. All original-evaluation selection frequencies remain available; they are descriptive stability evidence, not a newly validated targeting policy.

## Monte Carlo precision, reproducibility and limits

There are only 256 full refits. The summary reports binomial order-statistic brackets for both tail quantiles. For the primary lower 2.5th percentile, the Monte Carlo bracket is **[-0.467, -0.033] pp** (order indices 2–13); for the upper tail, **[1.181, 1.417] pp** (indices 244–255). These brackets concern simulation precision, not a second population confidence interval. Full sample minima/maxima and all descriptive nonpositive fractions are retained; these fractions are not posterior probabilities or p-values.

[Austern and Syrgkanis (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/58b7483ba899e0ce4d97ac5eecf6fa99-Abstract.html) establish bootstrap results under stability conditions. This project has not established those conditions for its discontinuous model choice and capacity cutoff. Adding refits does not by itself prove nominal coverage. These results condition on this empirical experiment, split, observed arm counts, independence assumption, fixed feature/candidate menu and fixed algorithm seed. They omit new campaign/population/calendar, split-choice, researcher-choice and unobserved-cluster uncertainty.

```bash
pip install -r requirements-refit.txt
python -m pytest -q
python collect_refit_source.py
python verify_refit_uncertainty.py --refit-check
```

The collector verifies pinned bytes and parsed rows. Verification reconstructs all role assignments, all 256 draw identities, every stored policy/metric, all 4,096 conditional draws, frequencies and summary hashes. The refit option recomputes baseline and registered replicate IDs **0, 85, 170, 255**, checking exact memberships and bounded numerical equality; it does **not** refit all 256 again. `--all-refits` explicitly requests full refit replay. The original unpinned historical workflow remains a separate historical reproduction; the extension pins NumPy/pandas/SciPy/sklearn/statsmodels/requests.

For a clean new execution, use a separate copy of the registered source before completed outputs and run `benchmark_refit_uncertainty.py --workers 8` after hash-checked collection. The benchmark refuses to overwrite a completed manifest and rejects mismatching resume identity or retained failure events. Raw source, mutable work files and fitted weights are omitted. Full replicate metrics and bit-packed policy memberships are committed for audit.

This closes the defined fixed-split refit/model-selection sensitivity task. Treatment-specific learning and confirmation on a genuinely new locked experiment remain open. No new budget/model/feature choice is justified by these inspected outcomes.
