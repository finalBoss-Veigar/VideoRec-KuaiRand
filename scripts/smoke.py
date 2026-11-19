"""End-to-end acceptance on synthetic data only; never a research result.
Run: python scripts/smoke.py --out runs/smoke_check
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='runs/smoke_check');a=p.parse_args()
    root=Path(a.out)
    if root.exists():
        raise ValueError('Smoke output exists; choose a new --out directory.')
    root.mkdir(parents=True)
    def run(*args):
        result=subprocess.run([sys.executable,'-m','rec_lab',*map(str,args)],capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(result.stdout+'\n'+result.stderr)
    raw=root/'synthetic.csv';data=root/'processed'
    run('demo','--out',raw,'--rows','6000')
    run('prepare','--logs',raw,'--out',data,'--kind','synthetic')
    specs=[('constant','engage'),('constant','like'),('lr','engage'),('lr','like'),
           ('deepfm','engage'),('deepfm','like'),('single','engage'),('single','like'),('shared','both'),('mmoe','both')]
    dirs=[]
    for model,task in specs:
        target=root/f'{model}_{task}'
        run('train','--data',data,'--run',target,'--model',model,'--task',task,'--epochs','3')
        run('evaluate','--run',target)
        restored=root/f'{model}_{task}_reload.csv'
        run('predict','--run',target,'--input',data/'test.csv','--out',restored)
        expected=pd.read_csv(target/'test_predictions.csv')
        actual=pd.read_csv(restored)
        columns=[c for c in actual if c.startswith('p_')]
        for c in columns:
            np.testing.assert_allclose(expected[c],actual[c],rtol=1e-6,atol=1e-7)
        dirs.append(target)
        print(f'PASS {model}/{task}: train, validation, saved test evaluation, reload prediction')
    run('compare','--runs',*dirs,'--out',root/'synthetic_comparison.csv')
    print('PASS same-data comparison; results are SYNTHETIC plumbing checks only.')
    (root/'verification.json').write_text(json.dumps({'synthetic_only':True,'runs':len(specs),
        'checks':['training','validation','saved_checkpoint_test_evaluation','reload_prediction_equality','comparison_same_data']},indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
