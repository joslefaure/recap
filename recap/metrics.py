"""Task metrics and uncertainty conditional on saved predictions."""
import numpy as np
from sklearn.metrics import roc_auc_score


def balanced_accuracy(y, scores):
    y = np.asarray(y)
    pred = np.asarray(scores).argmax(1) if np.ndim(scores) == 2 else np.asarray(scores)
    if y.ndim != 1 or pred.shape != y.shape or not len(y):
        raise ValueError('Expected aligned, nonempty single-label predictions')
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def macro_auroc(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    if y.ndim != 2 or y.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError('Expected aligned multilabel scores')
    valid = [j for j in range(y.shape[1]) if len(np.unique(y[:, j])) == 2]
    if not valid:
        raise ValueError('No label has both positive and negative examples')
    return float(np.mean([roc_auc_score(y[:, j], scores[:, j]) for j in valid]))


def evaluate(y, scores):
    return macro_auroc(y, scores) if np.ndim(y) == 2 else balanced_accuracy(y, scores)


def within_count_auroc(y, scores, counts):
    values = []
    for count in np.unique(counts):
        mask = np.asarray(counts) == count
        for j in range(y.shape[1]):
            if len(np.unique(y[mask, j])) == 2:
                values.append(roc_auc_score(y[mask, j], scores[mask, j]))
    if not values:
        raise ValueError('No valid label/count strata')
    return {'macro_auroc': float(np.mean(values)), 'strata': len(values)}


def paired_bootstrap(y, first, second, groups, *, metric=evaluate, draws=5000, seed=20261011,
                     stratified=False, resamples=None):
    """Resample source groups jointly; return first minus second (fraction units)."""
    y, first, second, groups = map(np.asarray, (y, first, second, groups))
    if not (len(y) == len(first) == len(second) == len(groups)):
        raise ValueError('Unaligned bootstrap inputs')
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    if stratified:
        if y.ndim != 1 or any(len(np.unique(y[ix])) != 1 for ix in members.values()):
            raise ValueError('Stratification requires one class per source group')
        strata = [np.array([g for g in unique if y[members[g][0]] == c]) for c in np.unique(y)]
    else:
        strata = [unique]
    if resamples is not None:
        resamples = np.asarray(resamples)
        if len(unique) != len(groups) or resamples.shape[0] != draws or resamples.min() < 0 or resamples.max() >= len(y):
            raise ValueError('Explicit resamples require valid indices of unique source scenes')
    rng = np.random.default_rng(seed)
    values = []
    for draw in range(draws):
        if resamples is None:
            sampled = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
            ix = np.concatenate([members[g] for g in sampled])
        else:
            ix = resamples[draw]
        values.append(metric(y[ix], first[ix]) - metric(y[ix], second[ix]))
    return {'estimate': metric(y, first) - metric(y, second),
            'ci_low': float(np.quantile(values, .025)), 'ci_high': float(np.quantile(values, .975)),
            'draws': draws, 'seed': seed}
