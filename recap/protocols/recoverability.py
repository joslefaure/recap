"""Nested held-out readers, count-conditioned CLEVR, and temporal controls."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from ..data import SEEDS, folds
from ..metrics import evaluate, balanced_accuracy, within_count_auroc, paired_bootstrap
from ..models import truncate
from ..readers import RIDGE_GRID, LOGISTIC_GRID, fit_scores


def nested(bank, manifest, width, *, seeds=SEEDS, tune=True):
    x = truncate(bank['x'], bank['model'], width)
    prompts = truncate(bank['prompts'], bank['model'], width)
    y, groups = manifest['y'], manifest['groups']
    k, n = len(manifest['labels']), len(y)
    counts = np.asarray([r['count'] for r in manifest['items']]) if manifest['task'] == 'CLEVR' else None
    static = np.einsum('nd,tkd->tnk', x, prompts)
    readers = ['prompt', 'prototype', 'ridge10'] + (['ridge', 'logistic'] if tune and y.ndim == 1 else ['ridge'] if tune else [])
    rows, arrays, choices = [], {}, []
    for seed in seeds:
        oof = {r: np.full((n, k), np.nan, np.float32) for r in readers}
        assignment = np.full(n, -1)
        for fi, (train, test) in enumerate(folds(y, groups, seed, count=counts)):
            inner = folds(y[train], groups[train], seed+1000+fi,
                          count=None if counts is None else counts[train])
            prompt_scores = [np.mean([evaluate(y[train[v]], static[t, train[v]]) for _, v in inner])
                             for t in range(4)]
            template = int(np.argmax(prompt_scores))
            alpha, c = 10., 1.
            if tune:
                a_scores = [np.mean([evaluate(y[train[v]], fit_scores('ridge', x[train[t]], y[train[t]],
                                   x[train[v]], k, alpha=a)) for t, v in inner]) for a in RIDGE_GRID]
                alpha = RIDGE_GRID[int(np.argmax(a_scores))]
                if y.ndim == 1:
                    c_scores = [np.mean([evaluate(y[train[v]], fit_scores('logistic', x[train[t]], y[train[t]],
                                       x[train[v]], k, c=c0)) for t, v in inner]) for c0 in LOGISTIC_GRID]
                    c = LOGISTIC_GRID[int(np.argmax(c_scores))]
            # CLEVR's reported native baseline is the fixed present template.
            oof['prompt'][test] = static[0 if counts is not None else template, test]
            for reader in readers[1:]:
                oof[reader][test] = fit_scores('ridge' if reader == 'ridge10' else reader,
                    x[train], y[train], x[test], k, alpha=10. if reader == 'ridge10' else alpha, c=c)
            assignment[test] = fi
            choices.append({'seed': int(seed), 'fold': fi, 'template': template, 'alpha': alpha, 'c': c})
        for reader, score in oof.items():
            if not np.isfinite(score).all():
                raise ValueError('Incomplete held-out predictions')
            row = {'model': bank['model'], 'task': manifest['task'], 'width': width, 'modality': str(bank.get('modality', manifest['modality'])),
                   'seed': int(seed), 'reader': reader, 'score': evaluate(y, score)}
            if counts is not None:
                row['within_count'] = within_count_auroc(y, score, counts)
            rows.append(row)
            arrays[f'{reader}_{seed}'] = score
        arrays[f'fold_{seed}'] = assignment
    return {'protocol': 'recoverability', 'metric': 'macro_auroc' if y.ndim == 2 else 'balanced_accuracy',
            'rows': rows, 'choices': choices}, arrays


def fresh_scenes(train_bank, train_manifest, test_bank, test_manifest, schedule, width):
    """Refit original-scene supports and evaluate only on freshly rendered scenes."""
    if train_bank['model'] != test_bank['model']:
        raise ValueError('Fresh-scene banks must use the same checkpoint')
    x = truncate(train_bank['x'], train_bank['model'], width)
    test = truncate(test_bank['x'], test_bank['model'], width)
    queries = truncate(test_bank['prompts'][0], test_bank['model'], width)
    y, yt = train_manifest['y'], test_manifest['y']
    if train_manifest['labels'] != test_manifest['labels']:
        raise ValueError('CLEVR label order differs')
    if set(train_manifest['groups']) & set(test_manifest['groups']):
        raise ValueError('Fresh scenes overlap training scenes')
    counts = np.array([r['count'] for r in test_manifest['items']])
    rng = np.random.default_rng(202610120003)
    sampled = {}
    for count in range(3, 11):
        ix = np.flatnonzero(counts == count)
        if len(ix) != 250:
            raise ValueError('Fresh scenes require 250 scenes at each object count')
        sampled[count] = ix[rng.integers(0, len(ix), size=(5000, len(ix)))].astype(np.int32)
    rows, arrays = [], {}
    for target in ('low', 'high'):
        mask = np.array([(r['count'] <= 6) == (target == 'low') for r in test_manifest['items']])
        local = np.full(len(counts), -1, dtype=int)
        local[mask] = np.arange(mask.sum())
        draws = local[np.concatenate([sampled[c] for c in range(3, 11) if (c <= 6) == (target == 'low')], axis=1)]
        for domain in ('low', 'high'):
            supports = [r for r in schedule if r['domain'] == domain and r['target_domain'] == target]
            if len(supports) != 45:
                raise ValueError(f'Expected 45 original-scene supports for {domain} to {target}')
            for reader in ('ridge', 'positive_centroid'):
                predictions = []
                for support in supports:
                    ix = np.asarray(support['indices'])
                    if len(ix) != 5000 or len(np.unique(ix)) != 5000 or ix.min() < 0 or ix.max() >= len(y):
                        raise ValueError('Invalid original-scene support')
                    if any((train_manifest['items'][i]['count'] <= 6) != (domain == 'low') for i in ix):
                        raise ValueError('Support crosses count domains')
                    predictions.append(fit_scores(reader, x[ix], y[ix], test[mask], y.shape[1]))
                fitted = np.moveaxis(np.stack(predictions), 0, 1)
                native = np.repeat((test[mask] @ queries.T)[:, None], len(supports), axis=1)
                # Average the metric over fits, not score scales across fits.
                def metric(labels, scores):
                    return float(np.mean([evaluate(labels, scores[:, j]) for j in range(scores.shape[1])]))
                interval = paired_bootstrap(yt[mask], fitted, native, test_manifest['groups'][mask], metric=metric,
                                            seed=202610120003, resamples=draws)
                rows.append({'train_domain': domain, 'test_domain': target, 'reader': reader,
                             'score': metric(yt[mask], fitted), 'native': metric(yt[mask], native), **interval})
                arrays[f'{target}_{domain}_{reader}'] = fitted
    return {'protocol': 'fresh_scenes', 'model': train_bank['model'], 'width': width, 'rows': rows}, arrays


def threshold(y, scores):
    unique = np.unique(np.asarray(scores, dtype=np.float64))
    if len(unique) > 2000:
        unique = np.quantile(unique, np.linspace(0., 1., 2000))
    candidates = np.r_[-np.inf, (unique[:-1]+unique[1:])/2, np.inf]
    values = [(balanced_accuracy(y, (scores > t).astype(int)), abs(t), t) for t in candidates]
    best = max(v[0] for v in values)
    return float(min((v for v in values if v[0] >= best-1e-12), key=lambda v: v[1])[2])


def temporal(bank, manifest, width, *, seeds=SEEDS):
    """Paired forward/reverse training; all views of a source share a fold."""
    x = truncate(bank['x'], bank['model'], width)
    reverse = truncate(bank['reverse'], bank['model'], width)
    rows, arrays = [], {}
    predictions, task_targets, task_groups = [], [], []
    for task in manifest['order_tasks']:
        ix = np.asarray(task['indices'])
        y = np.asarray(task['targets'])
        groups = manifest['groups'][ix]
        if len(set(groups)) != len(ix):
            raise ValueError('Temporal task must contain one pair per source')
        for seed in seeds:
            pred = np.full((len(ix), 2), -1, dtype=int)
            for fi, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(ix, y)):
                scores = np.full((len(tr), 2), np.nan)
                for it, iv in StratifiedKFold(3, shuffle=True, random_state=seed+5000+fi).split(tr, y[tr]):
                    a, v = tr[it], tr[iv]
                    fit = Ridge(alpha=10., solver='lsqr').fit(np.r_[x[ix[a]], reverse[ix[a]]], np.r_[y[a], 1-y[a]])
                    scores[iv] = np.column_stack((fit.predict(x[ix[v]]), fit.predict(reverse[ix[v]])))
                cutoff = threshold(np.r_[y[tr], 1-y[tr]], np.r_[scores[:, 0].astype(np.float32), scores[:, 1].astype(np.float32)])
                fit = Ridge(alpha=10., solver='lsqr').fit(np.r_[x[ix[tr]], reverse[ix[tr]]], np.r_[y[tr], 1-y[tr]])
                pred[te] = np.column_stack((fit.predict(x[ix[te]]), fit.predict(reverse[ix[te]]))) > cutoff
            correct = (pred[:, 0] == y) & (pred[:, 1] == 1-y)
            rows.append({'task': task['name'], 'seed': int(seed), 'both_correct': float(correct.mean())})
            arrays[f"{task['name']}_{seed}"] = pred
            predictions.append(correct); task_targets.append(np.full(len(ix), len(rows)-1)); task_groups.append(groups)
    # Equal task/seed weighting; one source draw is shared across all tasks and seeds.
    values = np.concatenate(predictions).astype(float)
    cells, groups = np.concatenate(task_targets), np.concatenate(task_groups)
    cell_ids = np.unique(cells)
    def cell_mean(labels, score):
        if not np.array_equal(np.unique(labels), cell_ids):
            raise ValueError('A bootstrap draw omitted a task/seed cell')
        return float(np.mean([score[labels == c].mean() for c in cell_ids]))
    interval = paired_bootstrap(cells, values, np.zeros_like(values), groups, metric=cell_mean, draws=1000, seed=20261019)
    return {'protocol': 'temporal', 'model': bank['model'], 'width': width, 'rows': rows, 'joint': interval}, arrays
