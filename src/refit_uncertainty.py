"""Registered sensitivity analysis. Selection never receives evaluation outcomes."""
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from src.causal_experimentation_uplift import (
    CONTROL, FEATURES, make_logistic, make_gradient_boosting, train_tlearner,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reference/refit_uncertainty'
SOURCE = ROOT / 'data/hillstrom_no_indices.csv.gz'
BUILDERS = {'Logistic T-learner': make_logistic,
            'Gradient-boosted T-learner': make_gradient_boosting}
ARMS = ('Mens E-Mail', 'No E-Mail', 'Womens E-Mail')
SEED = 20261010
Q = .30
_WORKER = None


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def protocol():
    p = OUT / 'PROTOCOL.json'
    return json.loads(p.read_text()), sha_bytes(p.read_bytes())


def require_registration():
    _, digest = protocol()
    receipt = json.loads((OUT/'registration_receipt.json').read_text())
    if receipt['protocol_sha256'] != digest:
        raise RuntimeError('Remote registration protocol mismatch')
    for item in receipt['method_files']:
        if sha_bytes((ROOT/item['path']).read_bytes()) != item['sha256']:
            raise RuntimeError('Registered method changed: '+item['path'])
    return receipt


def load_source(path=SOURCE):
    spec, _ = protocol()
    b = Path(path).read_bytes()
    if sha_bytes(b) != spec['source_payload_sha256']:
        raise ValueError('Source payload mismatch')
    df = pd.read_csv(path, compression='gzip')
    row_sha = sha_bytes(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    if len(df) != spec['source_rows'] or row_sha != spec['source_parsed_rows_sha256']:
        raise ValueError('Parsed source identity mismatch')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)
    if set(df.segment.unique()) != set(ARMS):
        raise ValueError('Unexpected randomized arms')
    if df[FEATURES + ['segment', 'visit', 'spend']].isna().any().any():
        raise ValueError('Missing required source values')
    if not set(df.visit.unique()) <= {0, 1}:
        raise ValueError('Visit is not binary')
    df['_row_id'] = np.arange(len(df), dtype=np.int64)
    df['treatment'] = (df.segment != CONTROL).astype(int)
    return df


def split_source(df):
    tv, ev = train_test_split(df, test_size=.20, random_state=2026,
                              stratify=df.treatment)
    tr, va = train_test_split(tv, test_size=.25, random_state=2026,
                              stratify=tv.treatment)
    parts = tuple(x.reset_index(drop=True) for x in (tr, va, ev))
    sets = [set(x._row_id) for x in parts]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Original roles overlap')
    if sum(map(len, sets)) != len(df):
        raise ValueError('Incomplete role assignment')
    return parts


def role_draw(frame, replicate, role):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, replicate, role]))
    draws = []
    segments = frame.segment.to_numpy()
    for arm in ARMS:
        locations = np.flatnonzero(segments == arm)
        if len(locations) == 0:
            raise ValueError('Missing original randomized arm')
        draws.append(rng.choice(locations, len(locations), replace=True))
    return np.concatenate(draws)


def policy_mask(scores, ids, fraction=Q):
    scores, ids = np.asarray(scores), np.asarray(ids)
    if len(scores) != len(ids) or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError('Invalid policy scores')
    if not 0 < fraction < 1:
        raise ValueError('Invalid capacity')
    n = int(len(scores) * fraction)
    if n == 0 or n == len(scores):
        raise ValueError('Empty capacity segment')
    order = np.lexsort((np.arange(len(scores)), ids, -scores))
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[:n]] = True
    return selected


def effect(y, treatment, mask):
    y, treatment, mask = np.asarray(y), np.asarray(treatment), np.asarray(mask)
    t, c = mask & (treatment == 1), mask & (treatment == 0)
    if not t.any() or not c.any():
        raise ValueError('Missing treatment in policy segment')
    return float(y[t].mean() - y[c].mean())


def policy_metrics(frame, selected):
    selected = np.asarray(selected, dtype=bool)
    if len(selected) != len(frame) or selected.sum() != int(Q * len(frame)):
        raise ValueError('Policy capacity/length mismatch')
    for mask in (selected, ~selected):
        if set(frame.loc[mask, 'segment'].unique()) != set(ARMS):
            raise ValueError('Missing actual randomized arm in policy segment')
    result = {}
    t = frame.treatment.to_numpy()
    for outcome in ('visit', 'spend'):
        y = frame[outcome].to_numpy(dtype=float)
        top, bottom = effect(y, t, selected), effect(y, t, ~selected)
        result[f'{outcome}_top'] = top
        result[f'{outcome}_bottom'] = bottom
        result[f'{outcome}_gain'] = Q * (1-Q) * (top-bottom)
        pseudo = np.where(t == 1, 1.5*y, -3*y)
        result[f'{outcome}_ipw_gain'] = float(np.mean((selected.astype(float)-Q)*pseudo))
    return result


def validation_score(frame, scores):
    mask = policy_mask(scores, frame._row_id)
    t, y = frame.treatment.to_numpy(), frame.visit.to_numpy()
    return effect(y, t, mask) - effect(y, t, ~mask)


def choose_candidate(scores):
    # Stable candidate order is an explicit exact-tie rule, not an evaluation choice.
    return max(BUILDERS, key=lambda name: scores[name])


def fit_scores(training, scoring, name):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        scored, _, _ = train_tlearner(training, scoring, BUILDERS[name])
    values = scored.predicted_uplift.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite fitted score')
    return values


def baseline(parts):
    tr, va, ev = parts
    comparisons = {}
    for name in BUILDERS:
        comparisons[name] = validation_score(va, fit_scores(tr, va, name))
    winner = choose_candidate(comparisons)
    final = pd.concat([tr, va], ignore_index=True)
    scores = fit_scores(final, ev, winner)
    selected = policy_mask(scores, ev._row_id)
    # Reconstruct the original pandas ranking separately; don't overwrite its results.
    legacy_order = pd.DataFrame({'score': scores}).sort_values('score', ascending=False).index.to_numpy()
    legacy_mask = np.zeros(len(ev), dtype=bool)
    legacy_mask[legacy_order[:int(Q*len(ev))]] = True
    cutoff = np.sort(scores)[len(scores)-int(Q*len(scores))]
    report = {'validation': comparisons, 'selected': winner,
              'evaluation': policy_metrics(ev, selected),
              'legacy_order_evaluation': policy_metrics(ev, legacy_mask),
              'policy_disagreements_with_legacy_order': int(np.sum(selected != legacy_mask)),
              'boundary_equal_scores': int(np.sum(scores == cutoff))}
    return scores, selected, report


def initialize_worker(parts, base_scores, base_selected, base_winner):
    global _WORKER
    _WORKER = (parts, base_scores, base_selected, base_winner)


def replicate(r):
    parts, base_scores, base_selected, base_winner = _WORKER
    tr, va, ev = parts
    it, iv, ie = role_draw(tr, r, 1), role_draw(va, r, 2), role_draw(ev, r, 3)
    bt, bv, be = tr.iloc[it].reset_index(drop=True), va.iloc[iv].reset_index(drop=True), ev.iloc[ie].reset_index(drop=True)
    selection = {name: validation_score(bv, fit_scores(bt, bv, name)) for name in BUILDERS}
    winner = choose_candidate(selection)
    final = pd.concat([bt, bv], ignore_index=True)
    fixed_scores = fit_scores(final, ev, base_winner)
    adaptive_scores = fixed_scores if winner == base_winner else fit_scores(final, ev, winner)
    fixed_original = policy_mask(fixed_scores, ev._row_id)
    adaptive_original = policy_mask(adaptive_scores, ev._row_id)
    fixed_boot = policy_mask(fixed_scores[ie], be._row_id)
    adaptive_boot = policy_mask(adaptive_scores[ie], be._row_id)
    conditional_boot = policy_mask(base_scores[ie], be._row_id)
    row = {'replicate': int(r), 'selected': winner,
           'train_draw_sha256': sha_bytes(it.astype('<i8').tobytes()),
           'validation_draw_sha256': sha_bytes(iv.astype('<i8').tobytes()),
           'evaluation_draw_sha256': sha_bytes(ie.astype('<i8').tobytes()),
           'logistic_validation': selection['Logistic T-learner'],
           'boosting_validation': selection['Gradient-boosted T-learner']}
    for prefix, frame, mask in (
        ('conditional', be, conditional_boot),
        ('fixed_refit', be, fixed_boot), ('reselect_refit', be, adaptive_boot),
        ('fixed_refit_original', ev, fixed_original),
        ('reselect_refit_original', ev, adaptive_original)):
        row.update({f'{prefix}_{k}': v for k, v in policy_metrics(frame, mask).items()})
    for prefix, mask, values in (('fixed_refit', fixed_original, fixed_scores),
                                  ('reselect_refit', adaptive_original, adaptive_scores)):
        row[prefix+'_jaccard'] = float(np.sum(mask & base_selected) / np.sum(mask | base_selected))
        row[prefix+'_scores_sha256'] = sha_bytes(values.astype('<f8').tobytes())
    masks = np.stack([np.packbits(m, bitorder='little') for m in
                      (fixed_original, adaptive_original, fixed_boot, adaptive_boot)])
    return row, masks


def quantile_range(values):
    a = np.sort(np.asarray(values, dtype=float))
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite replicate metric')
    n = len(a)
    result = {'replicates': n, 'percentiles_2_5_50_97_5': np.quantile(a, [.025, .5, .975]).tolist(),
              'min': float(a[0]), 'max': float(a[-1]),
              'fraction_nonpositive': float(np.mean(a <= 0))}
    brackets = {}
    for p in (.025, .975):
        lower = int(binom.ppf(.025, n, p))
        upper = int(binom.ppf(.975, n, p)) + 1
        lo, hi = max(1, lower), min(n, upper)
        brackets[str(p)] = {'one_based_order_indices': [lo, hi],
                            'metric_range': [float(a[lo-1]), float(a[hi-1])],
                            'interpretation': 'Binomial order-statistic Monte Carlo bracket; not a second population confidence interval.'}
    result['tail_monte_carlo_brackets_95'] = brackets
    return result


def summarize(frame, conditional, base_report):
    spec, digest = protocol()
    metrics = [c for c in frame if c.endswith(('_gain', '_ipw_gain'))]
    return {'protocol_sha256': digest, 'base_commit': spec['base_commit'],
            'outer_replicates': len(frame), 'conditional_replicates': len(conditional),
            'baseline': base_report,
            'selection_counts': {str(k): int(v) for k,v in frame.selected.value_counts().items()},
            'ranges': {c: quantile_range(frame[c]) for c in metrics},
            'fixed_ranking_4096': {c: quantile_range(conditional[c]) for c in conditional if c.endswith(('_gain','_ipw_gain'))},
            'policy_jaccard': {c: quantile_range(frame[c]) for c in frame if c.endswith('_jaccard')},
            'failed_replicates': 0,
            'interval_type': 'Empirical percentile sensitivity ranges, conditional on registered assumptions. Nominal frequentist coverage after selection is not established.',
            'caveats': spec['interpretation_limits']}
