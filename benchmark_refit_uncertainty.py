"""Run the exact registered budget; restart only matching completed work."""
import argparse
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from src.refit_uncertainty import (
    OUT, ROOT, protocol, load_source, split_source, baseline, initialize_worker,
    replicate, role_draw, policy_mask, policy_metrics, summarize, sha_bytes, write_json, require_registration,
)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    spec,digest=protocol()
    if (OUT/'artifact_manifest.json').exists():
        raise RuntimeError('Completed release exists; refuse overwrite')
    require_registration()
    parts=split_source(load_source())
    tr,va,ev=parts
    if tuple(map(len,parts))!=(38400,12800,12800): raise RuntimeError('Wrong role sizes')
    work=ROOT/'.refit_work'
    work.mkdir(exist_ok=True)
    if list(work.glob('failure_*.json')):
        raise RuntimeError('Earlier replicate failure retained; explicit amendment required')
    guard=work/'protocol_sha256.txt'
    if guard.exists() and guard.read_text()!=digest: raise RuntimeError('Resume protocol changed')
    guard.write_text(digest)
    scores,selected,report=baseline(parts)
    write_json(OUT/'baseline.json',report)
    assignments=[]
    for role,frame in zip(('train','validation','evaluation'),parts):
        a=frame[['_row_id','segment']].copy()
        a['role']=role; a['role_position']=np.arange(len(a))
        assignments.append(a)
    split=pd.concat(assignments,ignore_index=True)
    raw=split.to_csv(index=False).encode()
    (OUT/'split_assignments.csv.gz').write_bytes(gzip.compress(raw,mtime=0))
    prediction=ev[['_row_id','segment','treatment','visit','spend']].copy()
    prediction['predicted_uplift']=scores;prediction['selected']=selected.astype(int)
    (OUT/'baseline_evaluation_predictions.csv.gz').write_bytes(gzip.compress(prediction.to_csv(index=False,float_format='%.17g').encode(),mtime=0))
    pending=[]
    for r in range(spec['replicates']):
        jp,npz=work/f'{r:04d}.json',work/f'{r:04d}.npz'
        if jp.exists() and npz.exists():
            saved=json.loads(jp.read_text())
            if saved['protocol_sha256']!=digest or saved['row']['replicate']!=r: raise RuntimeError('Resume identity mismatch')
        else: pending.append(r)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_worker,
                             initargs=(parts,scores,selected,report['selected'])) as pool:
        futures={pool.submit(replicate,r):r for r in pending}
        completed=spec['replicates']-len(pending)
        for future in as_completed(futures):
            r=futures[future]
            try: row,masks=future.result()
            except Exception as exc:
                write_json(work/f'failure_{r:04d}.json',{'replicate':r,'error':repr(exc),'protocol_sha256':digest})
                raise
            np.savez_compressed(work/f'{r:04d}.npz',masks=masks)
            write_json(work/f'{r:04d}.json',{'protocol_sha256':digest,'row':row})
            completed+=1
            print(f'Completed refit {completed}/{spec["replicates"]}; replicate {r}',flush=True)
    rows=[];all_masks=[]
    for r in range(spec['replicates']):
        rows.append(json.loads((work/f'{r:04d}.json').read_text())['row'])
        with np.load(work/f'{r:04d}.npz',allow_pickle=False) as f: all_masks.append(f['masks'])
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'replicates.csv',index=False,float_format='%.17g')
    np.savez_compressed(OUT/'policy_membership.npz',masks=np.stack(all_masks),
                        row_ids=ev._row_id.to_numpy(dtype=np.int64))
    conditional=[]
    for r in range(spec['conditional_replicates']):
        draw=role_draw(ev,r,spec['conditional_seed_role'])
        boot=ev.iloc[draw].reset_index(drop=True)
        mask=policy_mask(scores[draw],boot._row_id)
        conditional.append({'replicate':r,**policy_metrics(boot,mask)})
    conditional=pd.DataFrame(conditional)
    conditional.to_csv(OUT/'conditional_replicates.csv',index=False,float_format='%.17g')
    write_json(OUT/'summary.json',summarize(frame,conditional,report))
    frequency=ev[['_row_id']].copy()
    packed=np.stack(all_masks)
    for idx,name in ((0,'fixed_refit'),(1,'reselect_refit')):
        masks=np.unpackbits(packed[:,idx,:],axis=1,bitorder='little')[:,:len(ev)]
        frequency[name+'_selection_frequency']=masks.mean(axis=0)
    (OUT/'selection_frequency.csv.gz').write_bytes(gzip.compress(frequency.to_csv(index=False).encode(),mtime=0))
    write_json(OUT/'environment.json',{'python':os.sys.version,
        'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','statsmodels','requests']},
        'thread_environment':{n:os.environ.get(n) for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
        'workers':args.workers})
    frozen=['src/refit_uncertainty.py','collect_refit_source.py','benchmark_refit_uncertainty.py',
            'verify_refit_uncertainty.py','tests/test_refit_uncertainty.py','requirements-refit.txt',
            '.github/workflows/refit-uncertainty.yml','reference/refit_uncertainty/PROTOCOL.json']
    paths=[ROOT/p for p in frozen]
    paths+=[p for p in OUT.iterdir() if p.is_file() and p.name!='PROTOCOL.json']
    files=[]
    for p in sorted(set(paths)):
        b=p.read_bytes();files.append({'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':sha_bytes(b)})
    write_json(OUT/'artifact_manifest.json',{'protocol_sha256':digest,'files':files})
    print(json.dumps({'study_complete':True,'replicates':len(frame),'selected':report['selected'],
                      'selection_counts':frame.selected.value_counts().to_dict()}),flush=True)

if __name__=='__main__': main()
