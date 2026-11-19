"""Safe minimal features and a global chronological split.

No same-event watch time, feedback, or month-wide aggregates enter features.
User/video vocabularies and numerical scaling are fitted on TRAIN only.
"""
import hashlib
import json
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ['user_id', 'video_id', 'time_ms', 'is_click', 'is_like', 'tab', 'duration_ms', 'is_rand']
SAFE_CATEGORIES = ['user_id', 'video_id', 'tab']
SAFE_NUMBERS = ['duration_log', 'hour_sin', 'hour_cos']

def write_json(path, obj):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False,suffix='.tmp') as f:
        temporary=Path(f.name)
        try:
            json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary,path)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def audit(df):
    return {
        'rows': len(df), 'users': int(df.user_id.nunique()), 'videos': int(df.video_id.nunique()),
        'exact_duplicate_rows': int(df.duplicated().sum()),
        'missing': {k: int(v) for k, v in df.isna().sum().items()},
        'tabs': {str(k): int(v) for k, v in df.tab.value_counts().items()},
        'feedback_rate': float(df.is_click.mean()), 'like_rate': float(df.is_like.mean()),
        'time_min_ms': int(df.time_ms.min()), 'time_max_ms': int(df.time_ms.max())
    }

def validate(df):
    missing = set(REQUIRED) - set(df)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if df.empty or df[REQUIRED].isna().any().any():
        raise ValueError('Empty data or missing values in required columns; inspect and document first.')
    for c in ['is_click', 'is_like', 'is_rand']:
        if not set(df[c].unique()).issubset({0, 1}):
            raise ValueError(f'{c} must be 0/1.')
    for c in ['time_ms', 'duration_ms']:
        if not np.isfinite(pd.to_numeric(df[c], errors='coerce')).all():
            raise ValueError(f'{c} must contain finite numbers.')
    if (df.duration_ms <= 0).any():
        raise ValueError('duration_ms must be positive; inspect invalid records before dropping them.')
    for c in ['user_id','video_id','tab','time_ms']:
        x=pd.to_numeric(df[c],errors='coerce')
        if not np.isfinite(x).all() or (x<0).any() or (x%1!=0).any():
            raise ValueError(f'{c} needs finite non-negative integers.')

def chronological_split(df, fractions=(0.7, 0.15, 0.15), cutoffs=None):
    if len(df) < 30 or not np.isclose(sum(fractions), 1) or min(fractions) <= 0:
        raise ValueError('Need at least 30 events and valid positive split fractions.')
    df = df.sort_values('time_ms', kind='stable').reset_index(drop=True)
    n = len(df)
    if cutoffs is None:
        t1 = int(df.time_ms.iloc[int(n * fractions[0])])
        t2 = int(df.time_ms.iloc[int(n * (fractions[0] + fractions[1]))])
    else:
        def milliseconds(value):
            t=pd.Timestamp(value)
            t=t.tz_localize('Asia/Shanghai') if t.tzinfo is None else t
            return int(t.timestamp()*1000)
        t1,t2=map(milliseconds,cutoffs)
        if t1>=t2:
            raise ValueError('train-before must be earlier than valid-before.')
    parts = {
        'train': df[df.time_ms < t1].copy(),
        'valid': df[(df.time_ms >= t1) & (df.time_ms < t2)].copy(),
        'test': df[df.time_ms >= t2].copy()
    }
    if any(p.empty for p in parts.values()):
        raise ValueError('Timestamp ties created an empty split. Choose explicit wider time windows.')
    assert parts['train'].time_ms.max() < parts['valid'].time_ms.min()
    assert parts['valid'].time_ms.max() < parts['test'].time_ms.min()
    return parts

def data_identity(manifest):
    contract={k:manifest.get(k) for k in ['dataset_kind','scene_tab','policy','signal_note','protocol','split_hashes','feature_contract']}
    return hashlib.sha256(json.dumps(contract,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def prepare(logs, out, kind, tab=None, policy='standard', signal_note='', max_rows=0,
            invalid_rows='error', train_before=None, valid_before=None):
    out=Path(out)
    if out.exists():
        raise ValueError('Processed output already exists. Choose a new data version directory.')
    if kind not in ['real','synthetic'] or policy not in ['standard','random']:
        raise ValueError('Invalid data kind or policy.')
    if invalid_rows not in ['error','drop'] or max_rows<0:
        raise ValueError('invalid_rows must be error/drop; max_rows must be non-negative.')
    if bool(train_before)!=bool(valid_before):
        raise ValueError('Provide both train-before and valid-before, or neither.')
    if kind == 'real' and (tab is None or not signal_note.strip()):
        raise ValueError('Real data require --tab and --signal-note; is_click has UI-dependent meaning.')
    hashes=[sha256(p) for p in logs]
    if len(set(hashes))!=len(hashes):
        raise ValueError('Duplicate input files detected. Select each raw file once.')
    frames=[];raw_rows=0;selected_rows=0;invalid_count=0
    for path in logs:
        columns=pd.read_csv(path,nrows=0).columns
        missing=set(REQUIRED)-set(columns)
        if missing:
            raise ValueError(f'{Path(path).name}: missing required fields {sorted(missing)}')
        with pd.read_csv(path,usecols=REQUIRED,chunksize=100000) as reader:
            for chunk in reader:
                raw_rows+=len(chunk)
                for c in REQUIRED:
                    chunk[c]=pd.to_numeric(chunk[c],errors='coerce')
                chunk=chunk[chunk.is_rand==(1 if policy=='random' else 0)]
                if tab is not None:
                    chunk=chunk[chunk.tab==tab]
                selected_rows+=len(chunk)
                valid=np.isfinite(chunk[REQUIRED]).all(axis=1)&(chunk.duration_ms>0)
                for c in ['is_click','is_like','is_rand']:
                    valid &= chunk[c].isin([0,1])
                for c in ['user_id','video_id','tab','time_ms']:
                    valid &= (chunk[c]>=0)&(chunk[c]%1==0)
                invalid_count+=int((~valid).sum())
                if invalid_rows=='error' and not valid.all():
                    raise ValueError('Invalid rows in selected cohort. Audit first; explicitly choose invalid_rows=drop to exclude and record them.')
                chunk=chunk[valid].copy()
                for c in REQUIRED:
                    if c!='duration_ms':
                        chunk[c]=chunk[c].astype('int64')
                frames.append(chunk)
    if not frames:
        raise ValueError('No data read.')
    df=pd.concat(frames,ignore_index=True)
    if df.empty:
        raise ValueError('No rows after scene/policy selection; inspect tab values with inspect command.')
    df = df.sort_values('time_ms', kind='stable')
    if max_rows and len(df) > max_rows:
        # Evenly spaced mini subset preserves time coverage; no label-based selection.
        df = df.iloc[np.linspace(0, len(df)-1, max_rows, dtype=int)].copy()
    validate(df)
    selected_audit=audit(df)
    parts = chronological_split(df,cutoffs=(train_before,valid_before) if train_before else None)
    if any(parts['train'][label].nunique() < 2 for label in ['is_click', 'is_like']):
        raise ValueError('Both targets need positive and negative training examples. Inspect cohort or enlarge data.')
    manifest = {
        'schema_version':2,
        'dataset_kind': kind, 'inputs': [{'name': Path(p).name, 'sha256': h} for p,h in zip(logs, hashes)],
        'scene_tab': tab, 'policy': policy, 'signal_note': signal_note,
        'subset_max_rows': max_rows,'sampling':'evenly spaced chronological rows after cohort filtering',
        'source_rows':raw_rows,'cohort_rows_before_cleaning':selected_rows,
        'invalid_rows_policy':invalid_rows,'dropped_invalid_rows':invalid_count,'selected_audit':selected_audit,
        'split_audit': {k: audit(v) for k,v in parts.items()},
        'feature_contract': {'categories': SAFE_CATEGORIES, 'numbers': SAFE_NUMBERS},
        'protocol': {'split':'global chronological','ratio':[.7,.15,.15] if not train_before else None,
                     'train_before':train_before,'valid_before':valid_before,'timezone':'Asia/Shanghai',
                     'ties':'same timestamp stays in same split','target_order':['is_click','is_like']}
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=out.name+'.staging-',dir=out.parent))
    try:
        for name,part in parts.items():
            part.to_csv(staging/f'{name}.csv',index=False)
        manifest['split_hashes']={k:sha256(staging/f'{k}.csv') for k in parts}
        manifest['data_id']=data_identity(manifest)
        write_json(staging/'manifest.json',manifest)
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise
    return manifest

def make_demo(path, rows=6000, seed=42):
    """Artificial correlated tasks for plumbing checks, not portfolio evidence."""
    if Path(path).exists() or rows<100:
        raise ValueError('Choose a new synthetic file path and at least 100 rows.')
    rng = np.random.default_rng(seed)
    users = rng.integers(0, 80, rows)
    videos = rng.integers(0, 120, rows)
    affinity = rng.normal(size=80)[users] + .7*rng.normal(size=120)[videos]
    duration = rng.integers(5000, 100000, rows)
    p_click = 1/(1+np.exp(-(affinity-.3)))
    click = rng.binomial(1, p_click)
    p_like = 1/(1+np.exp(-(affinity-2)))
    like = rng.binomial(1, p_like) * click
    df = pd.DataFrame({'user_id':users, 'video_id':videos, 'time_ms':1649376000000+np.arange(rows)*60000,
        'is_click':click, 'is_like':like, 'tab':np.zeros(rows, dtype=int), 'duration_ms':duration,
        'is_rand':np.zeros(rows, dtype=int)})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path,index=False)
    return audit(df)

def engineered(df):
    out = df.copy()
    for c in SAFE_CATEGORIES:
        out[c]=pd.to_numeric(out[c],errors='raise').astype('int64')
    out['duration_log'] = np.log1p(out.duration_ms.astype(float))
    hour = pd.to_datetime(out.time_ms, unit='ms', utc=True).dt.tz_convert('Asia/Shanghai').dt.hour
    out['hour_sin'] = np.sin(2*np.pi*hour/24)
    out['hour_cos'] = np.cos(2*np.pi*hour/24)
    return out

class FeatureEncoder:
    def __init__(self, variant='all'):
        variants = {'all':SAFE_CATEGORIES, 'no_user': ['video_id','tab'],
                    'no_video':['user_id','tab'], 'context':['tab']}
        self.categories = list(variants[variant])
        self.numbers = list(SAFE_NUMBERS)
        self.variant = variant

    def fit(self, train):
        train = engineered(train)
        self.maps = {c:{v:i+1 for i,v in enumerate(sorted(train[c].astype(str).unique()))} for c in self.categories}
        x = train[self.numbers].to_numpy(dtype=np.float32)
        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0),1e-6)
        return self

    @property
    def sizes(self):
        return [len(self.maps[c])+1 for c in self.categories]

    def encode(self, df):
        required=['user_id','video_id','time_ms','tab','duration_ms']
        if df.empty or set(required)-set(df):
            raise ValueError('Inference needs non-empty user_id/video_id/time_ms/tab/duration_ms data.')
        for c in required:
            values=pd.to_numeric(df[c],errors='coerce')
            if not np.isfinite(values).all() or (values<0).any() or (c!='duration_ms' and (values%1!=0).any()):
                raise ValueError(f'Invalid inference field {c}')
        if (df.duration_ms<=0).any():
            raise ValueError('Inference duration must be positive.')
        df = engineered(df)
        cat = np.column_stack([df[c].astype(str).map(self.maps[c]).fillna(0).to_numpy(dtype=np.int64) for c in self.categories])
        num = (df[self.numbers].to_numpy(dtype=np.float32)-self.mean)/self.std
        return cat,num

    def transform(self,df):
        cat,num=self.encode(df)
        y=df[['is_click','is_like']].to_numpy(dtype=np.float32)
        return cat, num, y

    def to_dict(self):
        return {'variant':self.variant, 'categories':self.categories, 'numbers':self.numbers,
                'maps':self.maps, 'mean':self.mean.tolist(), 'std':self.std.tolist()}

    @classmethod
    def from_dict(cls, obj):
        self=cls(obj['variant'])
        self.maps=obj['maps']
        self.mean=np.asarray(obj['mean'],dtype=np.float32)
        self.std=np.asarray(obj['std'],dtype=np.float32)
        return self
