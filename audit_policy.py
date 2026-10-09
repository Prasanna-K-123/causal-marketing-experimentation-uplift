"""Reconcile frozen policy effects and same-budget comparisons from saved data.

This is a retrospective audit of an already-published test set. The 30% policy
was previously selected; no new budget/model is optimized on this audit.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def pooled_effect(frame):
    treatment_visits=np.sum(frame.treated_visit_rate*frame.treated_customers)
    control_visits=np.sum(frame.control_visit_rate*frame.control_customers)
    return float(treatment_visits/frame.treated_customers.sum()-control_visits/frame.control_customers.sum())


def same_budget_gain(top_effect,bottom_effect,fraction):
    if not 0<=fraction<=1: raise ValueError('Budget fraction must be within [0,1]')
    return fraction*(1-fraction)*(top_effect-bottom_effect)


def main():
    path=ROOT/'results/final_test_uplift_deciles.csv'
    frame=pd.read_csv(path)
    saved=pd.read_csv(ROOT/'results/final_test_policy_bootstrap.csv').set_index('metric')
    top,bottom=pooled_effect(frame.iloc[:3]),pooled_effect(frame.iloc[3:])
    stored=saved.loc[['top30_visit_lift','bottom70_visit_lift'],'estimate'].to_numpy()
    reconciles=bool(np.allclose([top,bottom],stored,atol=1e-10,rtol=0))
    q=.30
    difference=saved.loc['top30_minus_bottom70_visit']
    summary=dict(test_deciles_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 test_rows=int(frame.customers.sum()),fixed_budget=q,top_visit_effect=top,bottom_visit_effect=bottom,
                 targeted_incremental_visits_per_customer=q*top,
                 random_same_budget_incremental_visits_per_customer=q*(q*top+(1-q)*bottom),
                 targeted_minus_random_same_budget=same_budget_gain(top,bottom,q),
                 artifacts_reconcile=reconciles,
                 saved_policy_effects=stored.tolist(),
                 artifact_point_differences=(np.array([top,bottom])-stored).tolist(),
                 conditional_bootstrap_ci95=(q*(1-q)*difference[['bootstrap_ci_low','bootstrap_ci_high']]).tolist() if reconciles else None,
                 interpretation='Per entire eligible customer population, versus randomized targeting at the same 30% email budget.',
                 caveats=['Conditional bootstrap interval from frozen ranked segments; no model-refit uncertainty.',
                          'No validated spend heterogeneity, revenue, profit or ROI improvement.',
                          'Published holdout: future experiments need a fresh locked test set.',
                          'If artifacts do not reconcile, withhold a confidence interval and regenerate all outputs together.'])
    (ROOT/'results/same_budget_policy_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
