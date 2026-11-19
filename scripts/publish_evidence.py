"""Export aggregate evidence only. No weights, source rows or predictions are copied."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]; run=root/a.suite
    status=json.loads((run/'status.json').read_text(encoding='utf-8'))
    if status['state']!='completed': raise ValueError('Only completed suites can be published')
    dest=root/'docs/evidence'/run.name;dest.mkdir(parents=True,exist_ok=False)
    for name in ['summary.csv','comparison.csv','selection.json','protocol.json','verification.json','status.json','source_hashes.json']:
        shutil.copy2(run/name,dest/name)
    frame=pd.read_csv(run/'comparison.csv');frame['experiment']=frame.run.str.replace(r'_s\d+$','',regex=True)
    summary=frame.groupby(['experiment','split','target']).agg(seeds=('seed','count'),auc=('auc','mean'),auc_sd=('auc','std'),
        logloss=('logloss','mean'),logloss_sd=('logloss','std'),AP=('pr_auc_average_precision','mean'),AP_sd=('pr_auc_average_precision','std'))
    summary.to_csv(dest/'all_metrics.csv')
    groups=[]
    for name in status['runs']:
        data=json.loads((run/name/'test_groups.json').read_text(encoding='utf-8'))
        for target,values in data.items():
            for group,m in values.items():
                groups.append(dict(run=name,target=target,group=group,**{k:v for k,v in m.items() if k not in ['calibration']}))
    pd.DataFrame(groups).to_csv(dest/'group_metrics.csv',index=False)
    env=json.loads((run/'environment.json').read_text(encoding='utf-8'))
    env={k:v for k,v in env.items() if k not in ['root','interpreter']}
    (dest/'environment.json').write_text(json.dumps(env,ensure_ascii=False,indent=2),encoding='utf-8')
    source=json.loads((run/status['runs'][0]/'data_manifest.json').read_text(encoding='utf-8'))
    public={k:source[k] for k in ['data_id','dataset_kind','source_rows','selected_audit','split_audit','split_hashes','protocol','feature_availability','limitations']}
    (dest/'data_summary.json').write_text(json.dumps(public,ensure_ascii=False,indent=2),encoding='utf-8')
    checks={str(x.relative_to(dest)):hashlib.sha256(x.read_bytes()).hexdigest() for x in dest.iterdir() if x.is_file()}
    (dest/'checksums.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(dest)

if __name__=='__main__':main()
