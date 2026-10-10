"""Source reconstruction, independent metric audit, optional registered refit replay."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.refit_uncertainty import (
    OUT, ROOT, ARMS, Q, protocol, load_source, split_source, sha_bytes,
    policy_mask, role_draw, baseline, initialize_worker, replicate, summarize, require_registration,
)


def independent_metrics(frame, mask):
    # Separate explicit arm totals; do not call the benchmark's effect/metrics routine.
    output={}
    for outcome in ('visit','spend'):
        y=frame[outcome].to_numpy(dtype=float)
        treatment=frame.treatment.to_numpy()
        subgroup=[]
        for group in (mask,~mask):
            t=(treatment==1)&group;c=(treatment==0)&group
            if not t.any() or not c.any(): raise AssertionError('Missing pooled arm')
            subgroup.append(float(y[t].sum()/t.sum()-y[c].sum()/c.sum()))
        output[outcome+'_top'],output[outcome+'_bottom']=subgroup
        output[outcome+'_gain']=float(.21*(subgroup[0]-subgroup[1]))
        value=0.
        for arm,weight in (('Mens E-Mail',1.5),('Womens E-Mail',1.5),('No E-Mail',-3.)):
            is_arm=frame.segment.to_numpy()==arm
            value+=weight*(float(y[is_arm&mask].sum())-Q*float(y[is_arm].sum()))
        output[outcome+'_ipw_gain']=value/len(frame)
    return output


def close_dict(actual,expected,atol=2e-11):
    for key,value in actual.items():
        if isinstance(value,(float,int)):
            if not np.isclose(value,expected[key],atol=atol,rtol=0):
                raise AssertionError(f'{key}: {value} != {expected[key]}')
        elif value!=expected[key]: raise AssertionError(key)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--refit-check',action='store_true')
    parser.add_argument('--all-refits',action='store_true')
    args=parser.parse_args()
    spec,digest=protocol()
    require_registration()
    manifest=json.loads((OUT/'artifact_manifest.json').read_text())
    assert manifest['protocol_sha256']==digest
    for entry in manifest['files']:
        b=(ROOT/entry['path']).read_bytes()
        assert len(b)==entry['bytes'] and sha_bytes(b)==entry['sha256'],entry['path']
    source=load_source()
    parts=split_source(source);tr,va,ev=parts
    assert tuple(map(len,parts))==(38400,12800,12800)
    assignments=pd.read_csv(OUT/'split_assignments.csv.gz')
    for role,part in zip(('train','validation','evaluation'),parts):
        stored=assignments[assignments.role==role].sort_values('role_position')
        assert np.array_equal(part._row_id,stored._row_id)
        assert np.array_equal(part.segment,stored.segment)
    prediction=pd.read_csv(OUT/'baseline_evaluation_predictions.csv.gz',float_precision='round_trip')
    for c in ('_row_id','segment','treatment','visit','spend'):
        assert np.array_equal(prediction[c].to_numpy(),ev[c].to_numpy()),c
    scores=prediction.predicted_uplift.to_numpy()
    selected=policy_mask(scores,ev._row_id)
    assert np.array_equal(selected,prediction.selected.to_numpy().astype(bool))
    report=json.loads((OUT/'baseline.json').read_text())
    close_dict(independent_metrics(ev,selected),report['evaluation'])
    legacy=json.loads((ROOT/'results/same_budget_policy_audit.json').read_text())
    # Software differences or a real tie change are disclosed, not hidden by editing history.
    legacy_delta=report['legacy_order_evaluation']['visit_gain']-legacy['targeted_minus_random_same_budget']
    frame=pd.read_csv(OUT/'replicates.csv',float_precision='round_trip')
    assert frame.replicate.tolist()==list(range(spec['replicates']))
    with np.load(OUT/'policy_membership.npz',allow_pickle=False) as f:
        packed=f['masks'];assert np.array_equal(f['row_ids'],ev._row_id)
    assert packed.shape==(spec['replicates'],4,(len(ev)+7)//8)
    masks=np.unpackbits(packed,axis=2,bitorder='little')[:,:,:len(ev)].astype(bool)
    assert np.all(masks.sum(axis=2)==int(Q*len(ev)))
    for r,row in frame.iterrows():
        ie=role_draw(ev,r,3);boot=ev.iloc[ie].reset_index(drop=True)
        close_dict({
            'train_draw_sha256':sha_bytes(role_draw(tr,r,1).astype('<i8').tobytes()),
            'validation_draw_sha256':sha_bytes(role_draw(va,r,2).astype('<i8').tobytes()),
            'evaluation_draw_sha256':sha_bytes(ie.astype('<i8').tobytes())},row)
        for prefix,view,mask in (
            ('conditional',boot,policy_mask(scores[ie],boot._row_id)),
            ('fixed_refit_original',ev,masks[r,0]),('reselect_refit_original',ev,masks[r,1]),
            ('fixed_refit',boot,masks[r,2]),('reselect_refit',boot,masks[r,3])):
            values={prefix+'_'+k:v for k,v in independent_metrics(view,mask).items()}
            close_dict(values,row)
        for idx,prefix in ((0,'fixed_refit'),(1,'reselect_refit')):
            mask=masks[r,idx]
            close_dict({prefix+'_jaccard':float(np.sum(mask&selected)/np.sum(mask|selected))},row)
        expected='Logistic T-learner' if row.logistic_validation>=row.boosting_validation else 'Gradient-boosted T-learner'
        assert row.selected==expected
    conditional=pd.read_csv(OUT/'conditional_replicates.csv',float_precision='round_trip')
    assert conditional.replicate.tolist()==list(range(spec['conditional_replicates']))
    for r,row in conditional.iterrows():
        draw=role_draw(ev,r,spec['conditional_seed_role'])
        boot=ev.iloc[draw].reset_index(drop=True)
        mask=policy_mask(scores[draw],boot._row_id)
        close_dict(independent_metrics(boot,mask),row)
    frequency=pd.read_csv(OUT/'selection_frequency.csv.gz',float_precision='round_trip')
    assert np.array_equal(frequency._row_id,ev._row_id)
    for idx,name in ((0,'fixed_refit'),(1,'reselect_refit')):
        assert np.allclose(frequency[name+'_selection_frequency'],masks[:,idx,:].mean(axis=0),atol=1e-15,rtol=0)
    expected=summarize(frame,conditional,report)
    stored=json.loads((OUT/'summary.json').read_text())
    assert json.dumps(expected,sort_keys=True)==json.dumps(stored,sort_keys=True),'Summary identity'
    replayed=[];different_byte_hashes=[]
    if args.refit_check or args.all_refits:
        current_scores,current_selected,current_report=baseline(parts)
        assert current_report['selected']==report['selected']
        assert np.allclose(current_scores,scores,atol=2e-10,rtol=0)
        assert np.array_equal(current_selected,selected)
        initialize_worker(parts,current_scores,current_selected,report['selected'])
        ids=range(spec['replicates']) if args.all_refits else spec['replay_refit_ids']
        for r in ids:
            actual,actual_packed=replicate(r)
            expected_row=frame.iloc[r].to_dict()
            for k in list(actual):
                if k.endswith('_scores_sha256'):
                    if actual[k]!=expected_row[k]: different_byte_hashes.append({'replicate':r,'key':k})
                    actual.pop(k)
            close_dict(actual,expected_row,atol=2e-10)
            assert np.array_equal(actual_packed,packed[r]),f'Policy replay {r}'
            replayed.append(int(r))
            print('Refit replay passed:',r,flush=True)
    print(json.dumps({'verified':True,'source_rows':len(source),'outer_replicates':len(frame),
        'conditional_replicates':len(conditional),'manifest_files':len(manifest['files']),
        'baseline_legacy_gain_delta':legacy_delta,'refit_replayed_ids':replayed,
        'portable_score_byte_hash_differences':different_byte_hashes}),flush=True)

if __name__=='__main__': main()
