"""Portable input manifests and frozen vector banks."""
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from .readers import normalize

SEEDS = (20261011, 20261012, 20261013)
DRAW_SEEDS = (20261021, 20261022, 20261023)


def load_manifest(path):
    doc = json.loads(Path(path).read_text())
    rows = doc['items']
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Duplicate IDs or empty manifest')
    labels = doc['labels']
    y = np.asarray([r['target'] for r in rows])
    if len(set(labels)) != len(labels):
        raise ValueError('Duplicate class names')
    if y.ndim == 1:
        if not np.issubdtype(y.dtype, np.integer) or set(y) != set(range(len(labels))):
            raise ValueError('Targets must cover the fixed class vocabulary')
    elif y.shape != (len(rows), len(labels)) or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid binary targets')
    doc['y'] = y
    doc['ids'] = np.asarray(ids)
    doc['groups'] = np.asarray([r['group'] for r in rows])
    return doc


def load_bank(path, manifest):
    with np.load(path, allow_pickle=False) as z:
        bank = {k: z[k] for k in z.files}
    if not np.array_equal(bank['ids'], manifest['ids']):
        raise ValueError('Bank IDs differ from manifest order')
    if len(bank['x']) != len(manifest['ids']):
        raise ValueError('Vector count differs from manifest')
    bank['x'] = normalize(bank['x'])
    if 'prompts' in bank:
        bank['prompts'] = normalize(bank['prompts'])
        if bank['prompts'].shape != (4, len(manifest['labels']), bank['x'].shape[1]):
            raise ValueError('Expected four prompt templates in class order')
    if 'model' not in bank:
        raise ValueError('Bank must record its checkpoint key')
    bank['model'] = str(bank['model'])
    from .models import MODELS
    if bank['model'] not in MODELS or bank['x'].ndim != 2 or bank['x'].shape[1] not in MODELS[bank['model']]['widths']:
        raise ValueError('Unknown checkpoint or unsupported bank width')
    if 'reverse' in bank:
        if bank['reverse'].shape != bank['x'].shape:
            raise ValueError('Forward/reverse bank shapes differ')
        bank['reverse'] = normalize(bank['reverse'])
    return bank


def folds(y, groups, seed, *, count=None, n_splits=5):
    if count is not None:
        if len(set(groups)) != len(groups):
            raise ValueError('Count-stratified CLEVR expects unique scenes')
        split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        result = list(split.split(np.zeros(len(y)), count))
    elif y.ndim == 2:
        split = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        result = list(split.split(np.zeros(len(y)), groups=groups))
    else:
        split = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        result = list(split.split(np.zeros(len(y)), y, groups))
    for train, test in result:
        if set(groups[train]) & set(groups[test]):
            raise ValueError('Source group crosses a partition')
    return result


def save_result(output, result, arrays=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if arrays:
        np.savez_compressed(output/'predictions.npz', **arrays)
