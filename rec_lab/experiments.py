"""Small YAML experiment suites; no automatic final-test selection."""
import hashlib
import inspect
import json
import re
from pathlib import Path
import pandas as pd
import yaml
from .data import write_json,sha256
from .runner import train,read_json,compare,load_parts

ROOT=Path(__file__).resolve().parent.parent

def suite(config_path,data_override=None,out_override=None):
    config=yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    if not isinstance(config,dict) or set(config)-{'data','output','seeds','defaults','experiments'}:
        raise ValueError('Suite YAML has unknown top-level keys.')
    data=Path(data_override or config['data']);data=data if data.is_absolute() else ROOT/data
    output=Path(out_override or config['output']);output=output if output.is_absolute() else ROOT/output
    _,manifest=load_parts(data,include_test=False)
    seeds=config.get('seeds',[42]);specs=config['experiments'];defaults=config.get('defaults',{})
    if not isinstance(seeds,list) or not seeds or any(type(s)!=int or s<0 for s in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('Seeds must be unique non-negative integers.')
    allowed=set(inspect.signature(train).parameters)-{'data_dir','run_dir','evaluate_test','seed'}
    planned=[];names=set()
    for specification in specs:
        item=dict(specification);name=item.pop('name')
        if not re.fullmatch('[A-Za-z0-9_-]+',name) or name in names:raise ValueError('Use unique simple experiment names.')
        names.add(name);options={**defaults,**item}
        if set(options)-allowed:raise ValueError('Unknown experiment options: '+str(set(options)-allowed))
        for seed in seeds:
            options_for_run={**options,'seed':seed}
            identity={'options':options_for_run,'data_id':manifest['data_id'],
                      'source_hashes':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}}
            fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            planned.append((output/f'{name}_s{seed}',options_for_run,fingerprint))
    # Check every existing run before starting new mutations.
    for path,options,fingerprint in planned:
        if path.exists():
            request=path/'request.json';status=path/'status.json'
            if not request.exists() or read_json(request).get('fingerprint')!=fingerprint:
                raise ValueError(f'{path}: existing run has a different request.')
            if not status.exists() or read_json(status).get('state')!='completed':
                raise ValueError(f'{path}: incomplete run; preserve it and choose a new output directory.')
    output.mkdir(parents=True,exist_ok=True)
    run_dirs=[]
    for path,options,fingerprint in planned:
        if path.exists():print('SKIP completed:',path.name,flush=True)
        else:
            train(data,path,**options)
            write_json(path/'request.json',{'fingerprint':fingerprint,'options':options,'data_id':manifest['data_id']})
        run_dirs.append(path)
    write_json(output/'suite.json',{'config':config,'data_id':manifest['data_id'],
                                   'runs':[p.name for p in run_dirs],'final_test_evaluated':False})
    comparison=output/'comparison.csv'
    if not comparison.exists():compare(run_dirs,comparison)
    summary=seed_summary(pd.read_csv(comparison))
    summary.to_csv(output/'seed_summary.csv',index=False)
    return {'output':str(output),'runs':len(run_dirs),'comparison':str(comparison),'summary':str(output/'seed_summary.csv')}

def seed_summary(frame):
    frame=frame.copy()
    for key in ['aux_weight','selection_target']:
        if key not in frame:frame[key]=None
    keys=['data_id','dataset_kind','split','target','model','task','features','like_weight',
          'embedding_dim','hidden','experts','batch_size','learning_rate','weight_decay','patience','epoch_budget','device',
          'aux_weight','selection_target']
    values=['auc','logloss','pr_auc_average_precision','brier_score','training_seconds','parameters']
    duplicate=frame.duplicated(keys+['seed'])
    if duplicate.any():raise ValueError('Duplicate seed/settings would bias the summary. Compare each run once.')
    result=frame.groupby(keys,dropna=False)[values].agg(['count','mean','std']).reset_index()
    result.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in result.columns]
    return result
