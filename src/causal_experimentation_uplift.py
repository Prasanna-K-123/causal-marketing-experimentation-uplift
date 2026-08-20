#!/usr/bin/env python3
"""Causal marketing experimentation + uplift modeling on the Hillstrom dataset."""

import io
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


PRIMARY_URL = (
    "https://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
FALLBACK_URL = (
    "https://hillstorm1.s3.us-east-2.amazonaws.com/"
    "hillstorm_no_indices.csv.gz"
)

CONTROL = "No E-Mail"
TREATMENTS = ["Mens E-Mail", "Womens E-Mail"]
OUTCOMES = ["visit", "conversion", "spend"]
FEATURES = [
    "recency", "history", "mens", "womens", "newbie", "zip_code", "channel"
]
NUMERIC = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL = ["zip_code", "channel"]
TARGET = "visit"
N_BOOT = 5000


def load_hillstrom():
    errors = []
    for label, url in [("primary", PRIMARY_URL), ("fallback", FALLBACK_URL)]:
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            if url.endswith(".gz"):
                df = pd.read_csv(io.BytesIO(r.content), compression="gzip")
            else:
                df = pd.read_csv(io.BytesIO(r.content))
            if len(df) >= 60000:
                print(f"Loaded {len(df):,} rows from {label} source.")
                return df
        except Exception as exc:
            errors.append((label, repr(exc)))
    raise RuntimeError(f"Dataset load failed: {errors}")


def standardize_columns(df):
    out = df.copy()
    out.columns = (
        out.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    return out


def standardized_mean_difference(a, b):
    a = pd.Series(a).dropna().astype(float)
    b = pd.Series(b).dropna().astype(float)
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return 0.0 if pooled_sd == 0 else (a.mean() - b.mean()) / pooled_sd


def audit_experiment(df):
    expected = {
        "recency", "history_segment", "history", "mens", "womens", "zip_code",
        "newbie", "channel", "segment", "visit", "conversion", "spend"
    }
    missing = expected - set(df.columns)
    assert len(df) == 64000, f"Expected 64,000 rows, got {len(df):,}"
    assert not missing, f"Missing columns: {sorted(missing)}"
    assert set(df["segment"].unique()) == {
        "Mens E-Mail", "Womens E-Mail", "No E-Mail"
    }
    print("\nTreatment-arm counts:")
    print(df["segment"].value_counts().sort_index().to_string())
    return (
        df.groupby("segment")
          .agg(
              customers=("segment", "size"),
              visit_rate=("visit", "mean"),
              conversion_rate=("conversion", "mean"),
              avg_spend=("spend", "mean"),
          )
          .reset_index()
    )


def randomization_balance(df):
    balance_vars = ["recency", "history", "mens", "womens", "newbie"]
    rows = []
    arms = sorted(df["segment"].unique())
    for arm_a, arm_b in combinations(arms, 2):
        for variable in balance_vars:
            smd = standardized_mean_difference(
                df.loc[df["segment"] == arm_a, variable],
                df.loc[df["segment"] == arm_b, variable],
            )
            rows.append({
                "arm_a": arm_a,
                "arm_b": arm_b,
                "variable": variable,
                "smd": smd,
                "abs_smd": abs(smd),
            })
    return pd.DataFrame(rows)


def average_treatment_effects(df):
    rows = []
    control_df = df[df["segment"] == CONTROL]
    for treatment in TREATMENTS:
        treatment_df = df[df["segment"] == treatment]
        for outcome in OUTCOMES:
            y_t = treatment_df[outcome].astype(float)
            y_c = control_df[outcome].astype(float)
            diff = y_t.mean() - y_c.mean()
            se = np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c))
            test = stats.ttest_ind(y_t, y_c, equal_var=False)
            rows.append({
                "treatment": treatment,
                "outcome": outcome,
                "treated_mean": y_t.mean(),
                "control_mean": y_c.mean(),
                "absolute_effect": diff,
                "relative_lift_vs_control": (
                    diff / y_c.mean() if y_c.mean() != 0 else np.nan
                ),
                "normal_95ci_low": diff - 1.96 * se,
                "normal_95ci_high": diff + 1.96 * se,
                "p_value": test.pvalue,
            })
    out = pd.DataFrame(rows)
    reject, p_holm, _, _ = multipletests(
        out["p_value"], alpha=0.05, method="holm"
    )
    out["holm_p_value"] = p_holm
    out["significant_after_holm"] = reject
    return out


def bootstrap_ates(df, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    control_df = df[df["segment"] == CONTROL]
    for treatment in TREATMENTS:
        treatment_df = df[df["segment"] == treatment]
        for outcome in OUTCOMES:
            y_t = treatment_df[outcome].to_numpy(dtype=float)
            y_c = control_df[outcome].to_numpy(dtype=float)
            effects = np.empty(n_boot)
            for b in range(n_boot):
                effects[b] = (
                    rng.choice(y_t, len(y_t), replace=True).mean()
                    - rng.choice(y_c, len(y_c), replace=True).mean()
                )
            observed = y_t.mean() - y_c.mean()
            lo, hi = np.quantile(effects, [0.025, 0.975])
            rows.append({
                "treatment": treatment,
                "outcome": outcome,
                "observed_effect": observed,
                "bootstrap_95ci_low": lo,
                "bootstrap_95ci_high": hi,
                "bootstrap_probability_effect_positive": (effects > 0).mean(),
            })
    return pd.DataFrame(rows)


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def make_logistic():
    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", LogisticRegression(max_iter=2000)),
    ])


def make_gradient_boosting():
    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.04, max_depth=3,
            min_samples_leaf=50, random_state=2026
        )),
    ])


def train_tlearner(training_df, scoring_df, builder):
    treated = training_df[training_df["treatment"] == 1]
    control = training_df[training_df["treatment"] == 0]

    model_t = builder()
    model_c = builder()
    model_t.fit(treated[FEATURES], treated[TARGET])
    model_c.fit(control[FEATURES], control[TARGET])

    scored = scoring_df.copy()
    scored["p_if_email"] = model_t.predict_proba(scored[FEATURES])[:, 1]
    scored["p_if_control"] = model_c.predict_proba(scored[FEATURES])[:, 1]
    scored["predicted_uplift"] = scored["p_if_email"] - scored["p_if_control"]
    return scored, model_t, model_c


def empirical_effect(group, outcome="visit"):
    treated = group.loc[group["treatment"] == 1, outcome]
    control = group.loc[group["treatment"] == 0, outcome]
    return treated.mean() - control.mean()


def targeting_metrics(scored, fraction=0.30):
    ranked = scored.sort_values("predicted_uplift", ascending=False).reset_index(drop=True)
    n_top = int(len(ranked) * fraction)
    top = ranked.iloc[:n_top]
    bottom = ranked.iloc[n_top:]
    return {
        "top30_visit_lift": empirical_effect(top, "visit"),
        "bottom70_visit_lift": empirical_effect(bottom, "visit"),
        "top_minus_bottom_visit": (
            empirical_effect(top, "visit") - empirical_effect(bottom, "visit")
        ),
        "top30_spend_lift": empirical_effect(top, "spend"),
        "bottom70_spend_lift": empirical_effect(bottom, "spend"),
        "top_minus_bottom_spend": (
            empirical_effect(top, "spend") - empirical_effect(bottom, "spend")
        ),
    }


def bootstrap_group_effect(group, outcome, n_boot=N_BOOT, seed=2026):
    rng = np.random.default_rng(seed)
    treated = group.loc[group["treatment"] == 1, outcome].to_numpy(dtype=float)
    control = group.loc[group["treatment"] == 0, outcome].to_numpy(dtype=float)
    effects = np.empty(n_boot)
    for b in range(n_boot):
        effects[b] = (
            rng.choice(treated, len(treated), replace=True).mean()
            - rng.choice(control, len(control), replace=True).mean()
        )
    return effects


def main():
    df = standardize_columns(load_hillstrom())
    arm_outcomes = audit_experiment(df)
    balance = randomization_balance(df)
    ate = average_treatment_effects(df)
    bootstrap = bootstrap_ates(df)

    uplift_df = df.copy()
    uplift_df["treatment"] = (uplift_df["segment"] != CONTROL).astype(int)

    train_val, final_test = train_test_split(
        uplift_df, test_size=0.20, random_state=2026,
        stratify=uplift_df["treatment"]
    )
    train_data, validation = train_test_split(
        train_val, test_size=0.25, random_state=2026,
        stratify=train_val["treatment"]
    )

    candidate_builders = {
        "Logistic T-learner": make_logistic,
        "Gradient-boosted T-learner": make_gradient_boosting,
    }

    validation_rows = []
    for model_name, builder in candidate_builders.items():
        scored_val, _, _ = train_tlearner(train_data, validation, builder)
        validation_rows.append({"model": model_name, **targeting_metrics(scored_val)})
    validation_comparison = pd.DataFrame(validation_rows)

    winner = (
        validation_comparison
        .sort_values("top_minus_bottom_visit", ascending=False)
        .iloc[0]["model"]
    )
    winner_builder = candidate_builders[winner]
    print(f"\nSelected model on validation: {winner}")

    final_training = pd.concat([train_data, validation], ignore_index=True)
    final_scored, _, _ = train_tlearner(
        final_training, final_test, winner_builder
    )
    final_scored = final_scored.sort_values(
        "predicted_uplift", ascending=False
    ).reset_index(drop=True)

    top_n = int(len(final_scored) * 0.30)
    top30 = final_scored.iloc[:top_n]
    bottom70 = final_scored.iloc[top_n:]

    top_visit = bootstrap_group_effect(top30, "visit", seed=2026)
    bottom_visit = bootstrap_group_effect(bottom70, "visit", seed=2027)
    visit_diff = top_visit - bottom_visit

    top_spend = bootstrap_group_effect(top30, "spend", seed=2028)
    bottom_spend = bootstrap_group_effect(bottom70, "spend", seed=2029)
    spend_diff = top_spend - bottom_spend

    final_policy = pd.DataFrame([
        {
            "metric": "top30_visit_lift",
            "estimate": empirical_effect(top30, "visit"),
            "bootstrap_ci_low": np.quantile(top_visit, 0.025),
            "bootstrap_ci_high": np.quantile(top_visit, 0.975),
        },
        {
            "metric": "bottom70_visit_lift",
            "estimate": empirical_effect(bottom70, "visit"),
            "bootstrap_ci_low": np.quantile(bottom_visit, 0.025),
            "bootstrap_ci_high": np.quantile(bottom_visit, 0.975),
        },
        {
            "metric": "top30_minus_bottom70_visit",
            "estimate": empirical_effect(top30, "visit") - empirical_effect(bottom70, "visit"),
            "bootstrap_ci_low": np.quantile(visit_diff, 0.025),
            "bootstrap_ci_high": np.quantile(visit_diff, 0.975),
        },
        {
            "metric": "top30_minus_bottom70_spend",
            "estimate": empirical_effect(top30, "spend") - empirical_effect(bottom70, "spend"),
            "bootstrap_ci_low": np.quantile(spend_diff, 0.025),
            "bootstrap_ci_high": np.quantile(spend_diff, 0.975),
        },
    ])

    output_dir = Path("project6_outputs")
    output_dir.mkdir(exist_ok=True)
    arm_outcomes.to_csv(output_dir / "experiment_arm_outcomes.csv", index=False)
    balance.to_csv(output_dir / "randomization_balance.csv", index=False)
    ate.to_csv(output_dir / "average_treatment_effects.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_treatment_effects.csv", index=False)
    validation_comparison.to_csv(output_dir / "uplift_model_validation.csv", index=False)
    final_policy.to_csv(output_dir / "final_test_policy_bootstrap.csv", index=False)

    print("\nFinal policy metrics:")
    print(final_policy.to_string(index=False))
    print(
        "\nProbability top-30% visit uplift > bottom-70%:",
        round((visit_diff > 0).mean(), 4),
    )
    print(
        "Probability top-30% spend uplift > bottom-70%:",
        round((spend_diff > 0).mean(), 4),
    )


if __name__ == "__main__":
    main()
