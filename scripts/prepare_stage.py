"""Reproducible KuaiRand-Pure standard-exposure cohort, with row-level provenance.

Run from the project root: python -m scripts.prepare_stage
This preparation uses no feedback as an input feature and no feedback to sample rows.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from rec_lab.data import (REQUIRED, SAFE_CATEGORIES, SAFE_NUMBERS, audit,
                          chronological_split, data_identity, sha256, validate,
                          write_json)

FILES = ['log_standard_4_08_to_4_21_pure.csv',
         'log_standard_4_22_to_5_08_pure.csv']
TRAIN_BEFORE = '2022-04-18T00:00:00+08:00'
VALID_BEFORE = '2022-04-28T00:00:00+08:00'
TIME_MIN = int(pd.Timestamp('2022-04-08T00:00:00+08:00').timestamp() * 1000)
TIME_MAX = int(pd.Timestamp('2022-05-09T00:00:00+08:00').timestamp() * 1000)


def labels_by(frame, column):
    """Descriptive counts; invalid labels never contribute to a positive rate."""
    result = []
    for key, group in frame.groupby(column, dropna=False, sort=True):
        row = {column: str(key), 'rows': len(group)}
        for target in ['is_click', 'is_like']:
            values = pd.to_numeric(group[target], errors='coerce')
            valid = values.isin([0, 1])
            row[target + '_valid'] = int(valid.sum())
            row[target + '_positive'] = int((values == 1).sum())
            row[target + '_rate'] = float(values[valid].mean()) if valid.any() else None
        result.append(row)
    return pd.DataFrame(result)


def quality_masks(frame):
    """Overlapping reasons are reported separately; union gives removed rows."""
    masks = {}
    for column in REQUIRED:
        masks['missing_or_nonfinite_' + column] = ~np.isfinite(frame[column])
    for column in ['is_like', 'is_click', 'is_rand']:
        masks['invalid_binary_' + column] = ~frame[column].isin([0, 1])
    for column in ['user_id', 'video_id', 'tab', 'time_ms']:
        masks['invalid_integer_' + column] = ((frame[column] < 0) |
                                              (frame[column] % 1 != 0))
    masks['tab_outside_official_range'] = ~frame.tab.between(0, 14)
    masks['timestamp_outside_collection_window'] = ~frame.time_ms.between(TIME_MIN, TIME_MAX - 1)
    masks['duration_zero'] = frame.duration_ms == 0
    masks['duration_negative'] = frame.duration_ms < 0
    return masks


def label_blind_sample(frame, rows=50000, seed=20261107):
    if rows < 30:
        raise ValueError('Request at least 30 rows.')
    # Select positions only. Neither labels nor positive rates enter this operation.
    positions = np.random.default_rng(seed).choice(len(frame), min(rows, len(frame)), replace=False)
    return frame.iloc[np.sort(positions)].sort_values(['time_ms', 'source_row_id'], kind='stable').copy()


def detailed_audit(frame, train=None):
    info = audit(frame)
    for target in ['is_click', 'is_like']:
        info[target + '_positive_count'] = int(frame[target].sum())
    for edge in ['min', 'max']:
        info['time_' + edge + '_shanghai'] = str(pd.to_datetime(info['time_' + edge + '_ms'], unit='ms', utc=True).tz_convert('Asia/Shanghai'))
    if train is not None:
        for entity in ['user', 'video']:
            key = entity + '_id'
            unseen = ~frame[key].isin(train[key])
            info['unseen_' + entity + '_row_count'] = int(unseen.sum())
            info['unseen_' + entity + '_row_rate'] = float(unseen.mean())
            info['unseen_' + entity + '_unique_count'] = int(frame.loc[unseen, key].nunique())
            info['unseen_' + entity + '_unique_rate'] = float(frame.loc[unseen, key].nunique() / frame[key].nunique())
    return info


def prepare_stage(raw, out, rows=50000, seed=20261107,
                  train_before=TRAIN_BEFORE, valid_before=VALID_BEFORE):
    raw, out = Path(raw), Path(out)
    if out.exists():
        raise ValueError('Output already exists; choose a new data version to preserve results.')
    def normalize_cutoff(value):
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize('Asia/Shanghai')
        return stamp.isoformat()
    train_before, valid_before = map(normalize_cutoff, [train_before, valid_before])
    paths = [raw / name for name in FILES]
    inputs = [{'name': p.name, 'path': str(p.resolve()), 'bytes': p.stat().st_size,
               'sha256': sha256(p)} for p in paths]
    frames, missing_raw, nonnumeric = [], {}, {}
    for path in paths:
        frame = pd.read_csv(path)
        missing = set(REQUIRED) - set(frame.columns)
        if missing:
            raise ValueError(f'{path.name}: missing fields {sorted(missing)}')
        missing_raw[path.name] = {c: int(n) for c, n in frame.isna().sum().items()}
        # Provenance is assigned before filtering, deduplication or sampling.
        frame['source_row_id'] = [f'{path.name}:{i}' for i in range(1, len(frame) + 1)]
        frame['source_file'] = path.name
        frames.append(frame)
    full = pd.concat(frames, ignore_index=True)
    raw_rows = len(full)
    original_columns = [c for c in full if c not in ['source_row_id', 'source_file']]
    # Equality covers every parsed raw field, not only a user/video/time key.
    duplicate = full.duplicated(subset=original_columns, keep='first')
    duplicate_ids = full.loc[duplicate, ['source_row_id']].copy()
    for c in REQUIRED:
        old = full[c]
        full[c] = pd.to_numeric(old, errors='coerce')
        nonnumeric[c] = int((old.notna() & full[c].isna()).sum())
    global_quality = {k: int(v.sum()) for k, v in quality_masks(full).items()}
    raw_tabs = labels_by(full, 'tab')
    time = pd.to_datetime(full.time_ms, unit='ms', utc=True, errors='coerce').dt.tz_convert('Asia/Shanghai')
    full['day'] = time.dt.strftime('%Y-%m-%d').fillna('invalid_time')
    raw_days = labels_by(full, 'day')
    standard = full[(~duplicate) & (full.is_rand == 0)].copy()
    valid_tab = standard.tab.between(0, 14) & (standard.tab % 1 == 0)
    counts = standard.loc[valid_tab, 'tab'].value_counts().sort_index()
    if counts.empty:
        raise ValueError('No valid standard-exposure scene.')
    tab = int(counts.idxmax())  # count only, with smallest tab as deterministic tie-break.
    cohort = standard[standard.tab == tab].copy()
    masks = quality_masks(cohort)
    bad = pd.Series(False, index=cohort.index)
    reasons = {}
    for reason, mask in masks.items():
        reasons[reason] = int(mask.sum())
        bad |= mask
    rejected = cohort.loc[bad, ['source_row_id']].copy()
    rejected['reasons'] = ['|'.join(k for k, v in masks.items() if bool(v.loc[i])) for i in rejected.index]
    clean = cohort.loc[~bad].copy()
    if clean.empty:
        raise ValueError('No valid rows after cleaning.')
    for c in REQUIRED:
        if c != 'duration_ms':
            clean[c] = clean[c].astype('int64')
    selected = label_blind_sample(clean, rows, seed)
    preserved_feedback = {'zero_play_time_clean_rows': int((clean.play_time_ms == 0).sum()) if 'play_time_ms' in clean else None,
                          'zero_play_time_selected_rows': int((selected.play_time_ms == 0).sum()) if 'play_time_ms' in selected else None,
                          'is_click_zero_selected_rows': int((selected.is_click == 0).sum()),
                          'is_like_zero_selected_rows': int((selected.is_like == 0).sum())}
    # No full-period user/video table or same-row feedback-derived feature is joined.
    selected = selected[REQUIRED + ['source_row_id', 'source_file']].copy()
    validate(selected)
    parts = chronological_split(selected, cutoffs=(train_before, valid_before))
    split_audit = {name: detailed_audit(frame, parts['train']) for name, frame in parts.items()}
    for split, frame in parts.items():
        for target in ['is_like', 'is_click']:
            if frame[target].nunique() < 2:
                raise ValueError(f'{split}/{target} has one class; increase time coverage/rows explicitly. Counts: {split_audit}')
    manifest = {
        'schema_version': 2, 'dataset_kind': 'real', 'inputs': inputs,
        'scene_tab': tab, 'scene_selection': 'highest standard-exposure row count after exact-raw-row deduplication; smallest tab breaks ties; labels not used',
        'policy': 'standard',
        'signal_note': 'is_like = hit like button (primary); is_click = official UI-dependent raw feedback (auxiliary), not a universal CTR label. Exact tab-to-UI mapping is not asserted.',
        'source_rows': raw_rows, 'cohort_rows_before_cleaning': len(cohort),
        'subset_max_rows': rows, 'sampling_seed': seed,
        'sampling': 'uniform without replacement by row positions over complete cleaned cohort, then chronological ordering; no label stratification',
        'invalid_rows_policy': 'drop union of documented required-field validity failures; never filter by is_click=0 or play_time_ms=0',
        'dropped_invalid_rows': int(bad.sum()), 'invalid_reasons_overlap': reasons,
        'raw_invalid_reasons_overlap': global_quality,
        'missing_raw': missing_raw, 'numeric_parse_failures': nonnumeric,
        'duplicate_policy': 'drop exact duplicates across every parsed raw column; retain first source row; no user/video/time-only deduplication',
        'exact_duplicate_rows_removed': int(duplicate.sum()),
        'nonstandard_rows_excluded': int((full.is_rand != 0).sum()),
        'preserved_negatives': preserved_feedback,
        'clean_cohort_audit': detailed_audit(clean[REQUIRED]),
        'selected_audit': detailed_audit(selected), 'split_audit': split_audit,
        'feature_contract': {'categories': SAFE_CATEGORIES, 'numbers': SAFE_NUMBERS},
        'feature_availability': {
            'prediction_time': 'immediately before exposure',
            'user_id/video_id/tab': 'request user, candidate video, scenario known before exposure',
            'duration_ms': 'intrinsic video duration; assume the logged duration equals metadata available before exposure; no mutable full-period table used',
            'hour_sin/hour_cos': 'exposure request clock; timezone Asia/Shanghai',
            'excluded': 'all same-row feedback, play_time_ms, full-period user table, video statistics and all historical feedback features',
            'history': 'no historical label features; validation/test labels never update inputs',
            'missing': 'required raw fields with missing/nonfinite values are excluded and counted; no fitted imputation; scaling and vocabularies fitted on train only'},
        'protocol': {'split': 'global chronological', 'ratio': None,
                     'train_before': train_before, 'valid_before': valid_before,
                     'train_before_ms': int(pd.Timestamp(train_before).timestamp() * 1000),
                     'valid_before_ms': int(pd.Timestamp(valid_before).timestamp() * 1000),
                     'timezone': 'Asia/Shanghai', 'ties': 'same timestamp stays in same split',
                     'target_order': ['is_click', 'is_like'], 'primary_target': 'is_like'},
        'limitations': ['Uniform sample is only descriptive of this filtered scene within the Pure candidate pool; no online effect or all-platform representativeness claim.',
                       'Pure logs omit videos outside the candidate pool and are not complete user histories.',
                       'New sample and explicit time boundaries still overlap the source period of previously published example results; not an independent new dataset.',
                       'Excluding invalid/nonpositive intrinsic duration can introduce selection bias; zero viewing time and unclicked exposures are retained.']}
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=out.name + '.staging-', dir=out.parent))
    if stage.resolve().parent != out.parent.resolve():
        raise ValueError('Unexpected staging directory outside requested output parent.')
    try:
        membership = []
        for name, frame in parts.items():
            frame.to_csv(stage / f'{name}.csv', index=False)
            member = frame[['source_row_id', 'source_file', 'time_ms']].copy()
            member['split'] = name
            membership.append(member)
        pd.concat(membership).to_csv(stage / 'split_membership.csv', index=False)
        raw_tabs.to_csv(stage / 'raw_by_tab.csv', index=False)
        raw_days.to_csv(stage / 'raw_by_day.csv', index=False)
        labels_by(clean, 'day').to_csv(stage / 'cohort_by_day.csv', index=False)
        sample_days = selected.copy()
        sample_days['day'] = pd.to_datetime(sample_days.time_ms, unit='ms', utc=True).dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d')
        labels_by(sample_days, 'day').to_csv(stage / 'sample_by_day.csv', index=False)
        rejected.to_csv(stage / 'rejected_rows.csv', index=False)
        duplicate_ids.to_csv(stage / 'duplicate_rows.csv', index=False)
        manifest['split_hashes'] = {name: sha256(stage / f'{name}.csv') for name in parts}
        manifest['membership_sha256'] = sha256(stage / 'split_membership.csv')
        manifest['data_id'] = data_identity(manifest)
        write_json(stage / 'manifest.json', manifest)
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', default='data/raw/kuairand_extracted/KuaiRand-Pure/data')
    parser.add_argument('--out', default='data/processed/real_stage_20261107')
    parser.add_argument('--rows', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--train-before', default=TRAIN_BEFORE)
    parser.add_argument('--valid-before', default=VALID_BEFORE)
    args = parser.parse_args()
    m = prepare_stage(args.raw, args.out, args.rows, args.seed, args.train_before, args.valid_before)
    print(json.dumps({'data_id': m['data_id'], 'out': str(Path(args.out).resolve()),
                      'scene_tab': m['scene_tab'], 'splits': m['split_audit']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
