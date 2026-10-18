"""AVE: compare source-trained readers on identical held-out target inputs."""
import numpy as np
from ..metrics import balanced_accuracy, paired_bootstrap
from ..readers import fit_scores
from ..models import MODELS, truncate


def run(train_banks, train, test_banks, test, width):
    if train['labels'] != test['labels']:
        raise ValueError('Train/test class orders differ')
    if set(train['groups']) & set(test['groups']):
        raise ValueError('AVE train/test source overlap')
    modalities = ('audio', 'video')
    models = {bank['model'] for bank in [*train_banks.values(), *test_banks.values()]}
    if len(models) != 1:
        raise ValueError('Transfer requires one checkpoint across both modalities')
    if 'audio' not in MODELS[next(iter(models))]['modalities']:
        raise ValueError('This checkpoint has no native audio pathway')
    for modality in modalities:
        for bank, manifest in ((train_banks[modality], train), (test_banks[modality], test)):
            if len(bank['x']) != len(manifest['y']):
                raise ValueError('AVE bank and labels differ in length')
            if 'modality' in bank and str(bank['modality']) != modality:
                raise ValueError('Supply audio then video banks in the declared order')
    k = len(train['labels'])
    matrix, arrays = np.zeros((2, 2)), {}
    for i, source in enumerate(modalities):
        x = truncate(train_banks[source]['x'], train_banks[source]['model'], width)
        for j, target in enumerate(modalities):
            z = truncate(test_banks[target]['x'], test_banks[target]['model'], width)
            score = fit_scores('logistic', x, train['y'], z, k, c=1.)
            matrix[i, j] = balanced_accuracy(test['y'], score)
            arrays[f'{source}_to_{target}'] = score
    penalties = {}
    for target, source in [('audio', 'video'), ('video', 'audio')]:
        penalties[target] = paired_bootstrap(test['y'], arrays[f'{target}_to_{target}'],
                                            arrays[f'{source}_to_{target}'], test['groups'], stratified=True)
    return {'protocol': 'transfer', 'model': next(iter(models)), 'width': width,
            'modalities': list(modalities), 'matrix': matrix.tolist(), 'penalties': penalties}, arrays
