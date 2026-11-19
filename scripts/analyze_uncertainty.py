"""Reanalyze frozen predictions; never train, tune, or overwrite source experiments."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import sklearn
from rec_lab.data import sha256, write_json
from rec_lab.uncertainty import paired_user_bootstrap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True); p.add_argument('--output', required=True)
    p.add_argument('--resamples', type=int, default=1000); p.add_argument('--seed', type=int, default=20261119)
    args = p.parse_args(); suite = Path(args.suite); out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    evidence = {'analysis_id': out.name, 'started_at': datetime.now(timezone.utc).isoformat(),
                'command': [sys.executable, *sys.argv], 'code_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'workspace_status': subprocess.check_output(['git','status','--porcelain'],text=True).strip(),
                'source_sha256': {name: sha256(name) for name in ['rec_lab/uncertainty.py','scripts/analyze_uncertainty.py']},
                'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'platform': platform.platform()},
                'status': 'running', 'files': {}}
    write_json(out/'run.json', evidence)
    try:
        selection = json.loads((suite/'selection.json').read_text(encoding='utf-8'))
        assert selection['selected_model'] == 'shared_small'
        evidence['selection_sha256'] = sha256(suite/'selection.json')
        identity = None; arrays = {}; data_ids = set(); test_hashes = set()
        keys = ['source_row_id','user_id','video_id','time_ms','tab','is_like']
        for model in ('lr','shared_small'):
            arrays[model] = []
            for seed in (42,43,44):
                directory = suite/f'{model}_s{seed}'
                assert json.loads((directory/'status.json').read_text(encoding='utf-8'))['state'] == 'completed'
                config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
                assert config['seed'] == seed and config['dataset_kind'] == 'real'
                data_ids.add(config['data_id']); test_hashes.add(config['data_hashes']['test'])
                file = directory/'test_predictions.csv'; frame = pd.read_csv(file).sort_values('source_row_id').reset_index(drop=True)
                if frame.source_row_id.duplicated().any():raise ValueError('Duplicate prediction identities')
                if identity is None:identity = frame[keys].copy()
                elif not identity.equals(frame[keys]):raise ValueError('Prediction event/label identities differ')
                arrays[model].append(frame.p_is_like.to_numpy())
                evidence['files'][str(file.relative_to(suite))] = sha256(file)
        if len(data_ids)!=1 or len(test_hashes)!=1:raise ValueError('Training/test data versions differ')
        result, draws = paired_user_bootstrap(identity.is_like.to_numpy(),identity.user_id.to_numpy(),
                                               arrays['lr'],arrays['shared_small'],resamples=args.resamples,seed=args.seed)
        result.update(source_suite=suite.name, selected_model='shared_small', training_seeds=[42,43,44],
                      data_id=next(iter(data_ids)), test_split_sha256=next(iter(test_hashes)))
        pd.DataFrame(draws,columns=['auc_difference','logloss_difference','ap_difference']).to_csv(out/'bootstrap_draws.csv',index=False)
        write_json(out/'summary.json', result)
        evidence.update(status='completed',ended_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
                        output_sha256={f:sha256(out/f) for f in ['summary.json','bootstrap_draws.csv']})
        write_json(out/'run.json',evidence); print(json.dumps(result,ensure_ascii=False,indent=2))
    except BaseException as error:
        evidence.update(status='failed',error=str(error),elapsed_seconds=time.perf_counter()-started);write_json(out/'run.json',evidence);raise


if __name__=='__main__':main()
