"""Refit fixed Ridge at supported width endpoints on identical folds."""
from ..models import MODELS
from .recoverability import nested


def run(bank, manifest):
    widths = MODELS[bank['model']]['widths']
    if len(widths) == 1:
        raise ValueError('This checkpoint has no compression endpoint')
    rows, arrays = [], {}
    for width in (max(widths), min(widths)):
        result, predictions = nested(bank, manifest, width, tune=False)
        rows.extend(result['rows'])
        arrays.update({f'd{width}_{k}': v for k, v in predictions.items()})
    effects = []
    for seed in sorted({r['seed'] for r in rows}):
        def gap(w):
            values = {r['reader']: r['score'] for r in rows if r['seed'] == seed and r['width'] == w}
            return values['ridge10'] - values['prompt']
        full, minimum = gap(max(widths)), gap(min(widths))
        effects.append({'seed': seed, 'full_gap': full, 'minimum_gap': minimum,
                        'swing': minimum-full, 'reversal': bool(full*minimum < 0)})
    return {'protocol': 'compression', 'rows': rows, 'effects': effects}, arrays
