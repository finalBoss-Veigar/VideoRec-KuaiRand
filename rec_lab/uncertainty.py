"""Paired user-cluster resampling conditional on already trained predictors."""
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score

METRICS = ('auc', 'logloss', 'ap')


def metrics(y, predictions, weights=None):
    p = np.clip(np.asarray(predictions, dtype=float), 1e-7, 1-1e-7)
    return np.array([roc_auc_score(y, p, sample_weight=weights),
                     log_loss(y, p, sample_weight=weights, labels=[0, 1]),
                     average_precision_score(y, p, sample_weight=weights)])


def paired_user_bootstrap(y, users, baseline, candidate, *, resamples=1000, seed=20261119):
    y = np.asarray(y); users = np.asarray(users)
    a = np.asarray(baseline, dtype=float); b = np.asarray(candidate, dtype=float)
    if (y.ndim != 1 or users.shape != y.shape or not len(y) or
        a.ndim != 2 or a.shape != b.shape or a.shape[1] != len(y) or not a.shape[0] or
        set(np.unique(y)) != {0, 1} or not np.isfinite(a).all() or not np.isfinite(b).all() or
        np.any((a < 0) | (a > 1) | (b < 0) | (b > 1)) or resamples < 1):
        raise ValueError('Need aligned binary labels, users, and [seed,event] probabilities')
    unique, group = np.unique(users, return_inverse=True)
    if len(unique) < 2:
        raise ValueError('Need multiple user clusters')
    before = np.mean([metrics(y, p) for p in a], axis=0)
    after = np.mean([metrics(y, p) for p in b], axis=0)
    rng = np.random.default_rng(seed); draws = []; invalid = 0
    for _ in range(resamples):
        # Sampling users retains every event of a sampled user and pairs models/seeds.
        sampled = rng.integers(0, len(unique), size=len(unique))
        weights = np.bincount(sampled, minlength=len(unique))[group]
        if len(np.unique(y[weights > 0])) != 2:
            invalid += 1
            draws.append([None, None, None])
            continue
        difference = np.mean([metrics(y, pb, weights) - metrics(y, pa, weights)
                              for pa, pb in zip(a, b)], axis=0)
        draws.append(difference.tolist())
    valid = np.array([r for r in draws if r[0] is not None], dtype=float)
    if len(valid) < max(1, .95*resamples):
        raise ValueError('Too many one-class resamples; do not report a misleading interval')
    low, high = np.quantile(valid, [.025, .975], axis=0)
    result = {'rows': len(y), 'positive_events': int(y.sum()), 'user_clusters': len(unique),
              'training_seeds_averaged': a.shape[0], 'resamples': resamples,
              'valid_resamples': len(valid), 'invalid_one_class_resamples': invalid, 'bootstrap_seed': seed,
              'unit': 'user cluster; all within-user events retained; paired across both predictors and training seeds',
              'estimand': 'mean of per-training-seed metrics; NOT metric of ensemble-averaged probabilities',
              'difference_direction': 'candidate minus LR; AUC/AP higher is better, LogLoss lower is better',
              'interval_method': 'two-sided percentile, nominal 95%, conditional on frozen trained models',
              'metrics': {name: {'baseline': float(before[i]), 'candidate': float(after[i]),
                                 'difference': float(after[i]-before[i]),
                                 'ci95': [float(low[i]), float(high[i])]} for i, name in enumerate(METRICS)},
              'limitations': ['Post-hoc uncertainty analysis of an already used test period; not a new holdout.',
                             'Assumes independent user clusters; cross-user item/time dependence is not modeled.',
                             'Does not retrain models or estimate uncertainty from training data and seed selection.',
                             'Approximate percentile coverage; no online causal or population-wide significance claim.']}
    return result, draws
