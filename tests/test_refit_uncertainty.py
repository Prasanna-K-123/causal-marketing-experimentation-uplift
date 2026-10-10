import numpy as np
import pandas as pd
import pytest
from src.refit_uncertainty import (
    ARMS, policy_mask, policy_metrics, effect, role_draw, split_source,
    choose_candidate, validation_score, quantile_range,
)
from verify_refit_uncertainty import independent_metrics


def example():
    arms=['Mens E-Mail','No E-Mail','Womens E-Mail']*3+['No E-Mail']
    return pd.DataFrame({'segment':arms,'treatment':[int(a!='No E-Mail') for a in arms],
        'visit':[1,0,1,0,1,1,1,0,0,1],
        'spend':[10,0,20,0,5,12,3,0,0,7], '_row_id':np.arange(10)})


def test_hand_counted_policy_value_and_known_design_weights():
    f=example();mask=np.zeros(10,dtype=bool);mask[:3]=True
    result=policy_metrics(f,mask)
    # Top email rate 1, control 0; remaining email 2/4, control 2/3.
    assert result['visit_top']==1
    assert result['visit_bottom']==pytest.approx(-1/6)
    assert result['visit_gain']==pytest.approx(.245)
    # Direct potential policy-minus-random weights applied to all observed records.
    manual=sum((int(mask[i])-.3)*(1.5 if f.treatment[i] else -3)*f.visit[i] for i in range(10))/10
    assert result['visit_ipw_gain']==pytest.approx(manual)
    for k,v in independent_metrics(f,mask).items(): assert result[k]==pytest.approx(v)


def test_capacity_and_ties_use_only_score_and_source_key():
    scores=np.ones(10);ids=np.arange(10)[::-1]
    mask=policy_mask(scores,ids)
    assert ids[mask].tolist()==[2,1,0]
    assert mask.sum()==3
    assert np.array_equal(mask,policy_mask(scores,ids))


def test_duplicate_bootstrap_keys_break_tie_by_draw_position():
    mask=policy_mask(np.ones(10),np.zeros(10,dtype=int))
    assert np.flatnonzero(mask).tolist()==[0,1,2]


def test_evaluation_outcome_change_cannot_change_ranking():
    f=example();scores=np.arange(10,dtype=float)
    before=policy_mask(scores,f._row_id)
    f['visit']=1-f.visit;f['spend']=1e9
    after=policy_mask(scores,f._row_id)
    assert np.array_equal(before,after)


@pytest.mark.parametrize('scores,ids',[(np.array([np.nan,1]),[0,1]),([1,2],[0]),([[1,2]],[0])])
def test_invalid_score_vector_rejected(scores,ids):
    with pytest.raises(ValueError): policy_mask(scores,ids)


@pytest.mark.parametrize('q',[0,1,-.1,1.1])
def test_invalid_capacity_rejected(q):
    with pytest.raises(ValueError): policy_mask(np.ones(10),np.arange(10),q)


def test_missing_randomized_arm_blocks_metrics():
    f=example();f.loc[:2,'segment']='No E-Mail'
    mask=np.zeros(10,dtype=bool);mask[:3]=True
    with pytest.raises(ValueError): policy_metrics(f,mask)


def test_missing_pooled_treatment_blocks_effect():
    with pytest.raises(ValueError): effect(np.zeros(10),np.zeros(10),np.ones(10,dtype=bool))


def test_stratified_draw_keeps_whole_customer_and_actual_arm_counts():
    f=example();draw=role_draw(f,7,1);b=f.iloc[draw]
    assert b.segment.value_counts().to_dict()==f.segment.value_counts().to_dict()
    assert np.array_equal(draw,role_draw(f,7,1))
    assert not np.array_equal(draw,role_draw(f,7,2))
    assert set(b._row_id)<=set(f._row_id)
    # Outcome perturbations do not choose the resampling indices.
    changed=f.copy();changed['visit']=0;changed['spend']=100
    assert np.array_equal(draw,role_draw(changed,7,1))


def test_original_roles_remain_disjoint_with_duplicates_inside_role():
    f=pd.concat([example() for _ in range(6)],ignore_index=True)
    f['_row_id']=np.arange(len(f))
    parts=split_source(f)
    assert tuple(map(len,parts))==(36,12,12)
    ids=[set(p._row_id) for p in parts]
    assert not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    for s,p in enumerate(parts,1): assert set(p.iloc[role_draw(p,0,s)]._row_id)<=set(p._row_id)


def test_selection_exact_tie_preserves_original_candidate_order():
    assert choose_candidate({'Logistic T-learner':.1,'Gradient-boosted T-learner':.1})=='Logistic T-learner'
    assert choose_candidate({'Logistic T-learner':.1,'Gradient-boosted T-learner':.2})=='Gradient-boosted T-learner'


def test_validation_selection_uses_visit_not_spend():
    f=example();scores=np.arange(10,dtype=float)
    before=validation_score(f,scores)
    changed=f.copy();changed['spend']=1e12
    assert validation_score(changed,scores)==before


def test_quantile_ranges_preserve_losses_and_label_monte_carlo():
    a=np.arange(256,dtype=float)-100
    summary=quantile_range(a)
    assert summary['min']==-100 and summary['max']==155
    assert summary['fraction_nonpositive']==101/256
    assert summary['percentiles_2_5_50_97_5']==np.quantile(a,[.025,.5,.975]).tolist()
    assert 'not a second population confidence interval' in summary['tail_monte_carlo_brackets_95']['0.025']['interpretation']


def test_packed_policy_round_trip_keeps_exact_capacity():
    mask=policy_mask(np.arange(10),np.arange(10))
    packed=np.packbits(mask,bitorder='little')
    assert np.array_equal(np.unpackbits(packed,bitorder='little')[:len(mask)].astype(bool),mask)
