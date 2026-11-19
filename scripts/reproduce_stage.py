"""Frozen CPU experiment; run as python -m scripts.reproduce_stage from project root."""
import argparse
import contextlib
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from rec_lab.data import write_json, sha256
from rec_lab.runner import train, evaluate_saved, compare, read_json, now

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]
SPECS = [dict(name='constant', model='constant', task='like'),
         dict(name='lr', model='lr', task='like'),
         dict(name='single', model='single', task='like'),
         *[dict(name='shared_aux'+str(w).replace('.', 'p'), model='shared', task='both', aux_weight=w)
           for w in [0.0, 0.2, 1.0]]]
DEFAULTS = dict(epochs=12, batch_size=512, lr=.001, device='cpu', embedding_dim=8,
                hidden=(32, 16), patience=3, weight_decay=1e-5, clip_grad=5., selection_target='is_like')

def git_state():
    def run(*args):
        p = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, encoding='utf-8')
        return p.stdout.strip() if p.returncode == 0 else None
    return {'commit': run('rev-parse', 'HEAD'), 'branch': run('branch', '--show-current'),
            'dirty': run('status', '--porcelain'), 'identity_fallback': 'source_hashes.json'}

EXTENDED_SPECS = [*SPECS,
    dict(name='deepfm', model='deepfm', task='like'),
    dict(name='mmoe_aux0p2', model='mmoe', task='both', aux_weight=.2),
    dict(name='shared_no_user', model='shared', task='both', aux_weight=.2, variant='no_user'),
    dict(name='shared_small', model='shared', task='both', aux_weight=.2, hidden=(16,8))]

def environment():
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True) if shutil.which('nvidia-smi') else None
    cpu = platform.processor()
    if sys.platform == 'win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0]
    return dict(at=now(), root=str(ROOT), interpreter=sys.executable, platform=platform.platform(),
                cpu=cpu, physical_cores=psutil.cpu_count(logical=False), logical_cores=psutil.cpu_count(),
                ram_bytes=psutil.virtual_memory().total, available_ram_bytes=psutil.virtual_memory().available,
                disk_free_bytes=shutil.disk_usage(ROOT).free, torch=torch.__version__, cuda_build=torch.version.cuda,
                cuda_available=torch.cuda.is_available(), gpu_inventory=gpu.stdout.strip() if gpu else None,
                git=git_state())

def summarize(out, runs):
    frame = compare(runs, out/'comparison.csv')
    frame['experiment'] = frame.run.str.replace(r'_s\d+$', '', regex=True)
    summary = frame.groupby(['experiment', 'split', 'target'], dropna=False).agg(
        seeds=('seed', 'count'), auc_mean=('auc', 'mean'), auc_std=('auc', 'std'),
        logloss_mean=('logloss', 'mean'), logloss_std=('logloss', 'std'),
        parameters=('parameters', 'first'), training_seconds_mean=('training_seconds', 'mean'))
    summary.to_csv(out/'summary.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    test = summary.reset_index().query("split == 'test' and target == 'is_like'")
    if len(test):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric in zip(axes, ['auc', 'logloss']):
            ax.bar(test.experiment, test[metric+'_mean'], yerr=test[metric+'_std'].fillna(0), capsize=3)
            ax.tick_params(axis='x', labelrotation=35)
            ax.set_title('Test like '+metric+' (seed mean +/- sample SD)')
        fig.tight_layout(); fig.savefig(out/'test_metrics.png', dpi=150); plt.close(fig)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    p.add_argument('--suite', choices=['legacy','resume'], default='legacy')
    p.add_argument('--data', default='data/processed/real_stage_20261107')
    p.add_argument('--mode', choices=['all','smoke','one'], default='all')
    a=p.parse_args()
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.run_id): p.error('Use a simple unique run-id')
    data=(ROOT/a.data).resolve(); out=ROOT/'runs'/a.run_id
    if out.exists(): raise ValueError('Run already exists. Choose a new run-id; never overwrite results.')
    manifest=read_json(data/'manifest.json')
    specs=(EXTENDED_SPECS if a.suite=='resume' else SPECS) if a.mode=='all' else [SPECS[2] if a.mode=='smoke' else SPECS[1]]
    seeds=SEEDS if a.mode=='all' else [42]
    defaults={**DEFAULTS, **({'epochs':2, 'patience':0} if a.mode=='smoke' else {})}
    out.mkdir(parents=True)
    write_json(out/'status.json', dict(state='running', started_at=now()))
    write_json(out/'environment.json', environment())
    (out/'requirements-lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    snapshot=out/'source_snapshot'
    for folder in ['rec_lab','scripts','configs','tests']:
        for file in (ROOT/folder).glob('*'):
            if file.is_file() and file.suffix in ['.py','.yaml','.json']:
                dest=snapshot/folder/file.name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(file,dest)
    write_json(out/'source_hashes.json',{str(f.relative_to(snapshot)):sha256(f) for f in snapshot.rglob('*') if f.is_file()})
    protocol=dict(created_at=now(), data_id=manifest['data_id'], split_hashes=manifest['split_hashes'],
                  specs=specs,seeds=seeds,defaults=defaults,mode=a.mode,suite=a.suite,
                  hypothesis='Test auxiliary transfer, feature interaction, task-specific gates, user-ID and capacity controls under one fixed data protocol.',
                  primary_change='Weight: shared 0/0.2/1; user-ID: shared0.2 vs no_user; capacity: shared0.2 32/16 vs 16/8. MMoE/DeepFM are architecture comparisons, not isolated causal attributions.',
                  constants='Same data, seeds, epoch cap, optimizer and like validation stopping; architecture/feature differences are explicitly recorded; LR uses its own fixed solver budget.',
                  judgment='Choose auxiliary weight by three-seed mean validation like LogLoss; AUC is secondary. Final test all predeclared models once.',
                  limitations='Old examples use overlapping source periods. This is reproducibility and a controlled pilot, not independent confirmation. Seed SD is not a confidence interval.')
    write_json(out/'protocol.json',protocol)
    runs=[]; start=time.perf_counter()
    try:
        for spec in specs:
            name=spec['name']; options={k:v for k,v in spec.items() if k!='name'}
            for seed in seeds:
                run=out/f'{name}_s{seed}'
                log=out/f'{run.name}.log'
                print('START',run.name,flush=True)
                with log.open('w',encoding='utf-8') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    try: train(data,run,seed=seed,**{**defaults,**options})
                    except BaseException: traceback.print_exc(); raise
                shutil.copy2(log,run/'train.log')
                runs.append(run)
                print('DONE',run.name,round(read_json(run/'config.json')['training_seconds'],2),'seconds',flush=True)
        validation=[]
        for run in runs:
            cfg=read_json(run/'config.json'); m=read_json(run/'metrics.json')['valid']['is_like']
            validation.append(dict(run=run.name,experiment=run.name.rsplit('_s',1)[0],logloss=m['logloss'],auc=m['auc'],seed=cfg['seed']))
        vf=pd.DataFrame(validation); vf.to_csv(out/'validation_selection.csv',index=False)
        averages=vf.groupby('experiment').logloss.mean().sort_values()
        shared=averages[averages.index.str.startswith('shared')]
        write_json(out/'selection.json',dict(frozen_at=now(),rule='minimum mean validation like LogLoss',
                    selected_model=averages.index[0],selected_shared=shared.index[0] if len(shared) else None,
                    mean_validation_logloss=averages.to_dict(),test_not_evaluated_yet=True))
        from scripts.verify_stage import verify_suite
        verify_suite(out, data)
        if a.mode!='smoke':
            for run in runs:
                print('FINAL TEST',run.name,flush=True); evaluate_saved(run,data_dir=data)
        summarize(out,runs)
        write_json(out/'status.json',dict(state='completed',finished_at=now(),wall_seconds=time.perf_counter()-start,
                                        runs=[r.name for r in runs],final_test_evaluated=a.mode!='smoke'))
        print('RESULT',out,flush=True)
    except BaseException as exc:
        write_json(out/'status.json',dict(state='failed',finished_at=now(),error=str(exc),completed_runs=[r.name for r in runs]))
        raise

if __name__=='__main__': main()
