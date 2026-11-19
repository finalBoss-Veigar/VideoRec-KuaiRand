"""Verify real split isolation and saved predictions; no test metrics are computed."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from rec_lab.data import FeatureEncoder, write_json
from rec_lab.runner import load_parts, read_json, saved_probabilities, targets_for

def verify_suite(suite, data):
    suite, data=Path(suite),Path(data)
    parts, manifest=load_parts(data,include_test=True)
    ids=[set(f.source_row_id) for f in parts.values()]
    assert all(not ids[i]&ids[j] for i in range(3) for j in range(i+1,3))
    enc=FeatureEncoder().fit(parts['train'])
    row=parts['valid'].head(20).copy()
    cat,num=enc.encode(row)
    row['is_like']=1-row.is_like; row['is_click']=1-row.is_click
    c2,n2=enc.encode(row)
    np.testing.assert_array_equal(cat,c2); np.testing.assert_array_equal(num,n2)
    row['user_id']=int(max(f.user_id.max() for f in parts.values()))+1
    row['video_id']=int(max(f.video_id.max() for f in parts.values()))+1
    cat,_=enc.encode(row); assert (cat[:,:2]==0).all()
    records=[]
    for run in sorted(suite.iterdir()):
        if not (run/'config.json').is_file(): continue
        cfg=read_json(run/'config.json'); targets=targets_for(cfg['task'])
        fresh=saved_probabilities(run,parts['valid'].drop(columns=['is_like','is_click']))
        old=pd.read_csv(run/'valid_predictions.csv')
        stored=old[['p_'+t for t in targets]].to_numpy()
        np.testing.assert_allclose(fresh,stored,rtol=1e-6,atol=1e-7)
        assert np.isfinite(saved_probabilities(run,row)).all()
        records.append(dict(run=run.name,max_reload_difference=float(np.abs(fresh-stored).max()),unseen_prediction_finite=True))
    result=dict(data_id=manifest['data_id'],checks=['strict chronological boundaries','disjoint source rows',
                'same-row labels excluded','unknown IDs map to zero','saved model reproduces validation predictions'],runs=records)
    write_json(suite/'verification.json',result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True)
    p.add_argument('--data',default='data/processed/real_stage_20261107');a=p.parse_args()
    print(verify_suite(a.suite,a.data))
if __name__=='__main__':main()
