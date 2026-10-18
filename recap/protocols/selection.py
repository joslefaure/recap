"""Joint model/width/reader selection with capped source-group support."""
import numpy as np
from ..data import SEEDS, DRAW_SEEDS, folds
from ..models import MODELS, truncate
from ..metrics import balanced_accuracy
from ..readers import fit_scores

MODEL_ORDER = ('gemma2', 'qwen2b', 'jina', 'ovis', 'omni23', 'omni09', 'qwen8b')
READER_ORDER = ('prompt', 'prototype', 'ridge', 'logistic')


def choose_support(train, y, groups, classes, budget, draw, seed, fold):
    rng = np.random.default_rng(draw+seed*10+fold)
    selected = set()
    for label in classes:
        candidates = sorted(set(groups[train][y[train] == label]))
        if len(candidates) < budget:
            raise ValueError('Insufficient source groups for fixed support budget')
        selected.update(rng.permutation(candidates)[:budget].tolist())
    return train[np.isin(y[train], classes) & np.isin(groups[train], list(selected))]


def choose(candidates, scores):
    best = max(scores)
    tied = [c for c, s in zip(candidates, scores) if best-s <= 1e-12]
    return min(tied, key=lambda c: (c['width'], READER_ORDER.index(c['reader']),
                                   MODEL_ORDER.index(c['model']), c['template']))


def run(banks, manifest, config, *, seeds=SEEDS, draws=DRAW_SEEDS, budgets=(1, 2, 4, 8), caps=(512, 2048)):
    classes = config['classes']
    mapping = {c: j for j, c in enumerate(classes)}
    original_y, groups = manifest['y'], manifest['groups']
    if original_y.ndim != 1:
        raise ValueError('Selection requires a single-label task')
    y = np.array([mapping.get(int(c), -1) for c in original_y])
    eligible = config['widths']
    for model, widths in eligible.items():
        if widths and model not in banks:
            raise ValueError(f'Missing candidate vector bank: {model}')
        if any(w not in MODELS[model]['widths'] for w in widths):
            raise ValueError('Selection includes an unsupported width')
    vectors, prompts = {}, {}
    for model in MODEL_ORDER:
        for width in eligible.get(model, []):
            key = model, width
            vectors[key] = truncate(banks[model]['x'], model, width)
            prompts[key] = truncate(banks[model]['prompts'][:, classes], model, width)
    def predict(candidate, train, test):
        key = candidate['model'], candidate['width']
        if candidate['reader'] == 'prompt':
            return vectors[key][test] @ prompts[key][candidate['template']].T
        return fit_scores(candidate['reader'], vectors[key][train], y[train], vectors[key][test], len(classes))
    rows, arrays = [], {}
    for seed in seeds:
        # Outer folds use the original complete vocabulary, before the fixed class restriction.
        for fi, (train, outer_test) in enumerate(folds(original_y, groups, seed)):
            test = outer_test[np.isin(original_y[outer_test], classes)]
            for draw in draws:
                for budget in budgets:
                    support = choose_support(train, original_y, groups, classes, budget, draw, seed, fi)
                    inner = inner_splits(support, y, groups, budget, seed, fi, draw)
                    for cap in caps:
                        candidates = []
                        for model in MODEL_ORDER:
                            for width in eligible.get(model, []):
                                if width*4 > cap:
                                    continue
                                candidates.extend({'model': model, 'width': width, 'reader': 'prompt', 'template': t}
                                                  for t in range(4))
                                if budget > 1:
                                    candidates.extend({'model': model, 'width': width, 'reader': r, 'template': -1}
                                                      for r in READER_ORDER[1:])
                        if not candidates:
                            raise ValueError('No candidate fits the byte cap')
                        scores = []
                        for candidate in candidates:
                            if not inner:
                                value = balanced_accuracy(y[support], predict(candidate, support, support))
                            else:
                                value = np.mean([balanced_accuracy(y[support[v]], predict(candidate, support[t], support[v]))
                                                 for t, v in inner])
                            scores.append(float(value))
                        for strategy in ('native', 'joint'):
                            ix = [i for i, c in enumerate(candidates) if strategy == 'joint' or c['reader'] == 'prompt']
                            selected = choose([candidates[i] for i in ix], [scores[i] for i in ix])
                            score = predict(selected, support, test)
                            row = {'task': manifest['task'], 'seed': int(seed), 'fold': fi, 'draw': int(draw),
                                   'groups_per_class': budget, 'cap': cap, 'strategy': strategy,
                                   'selected': selected, 'score': balanced_accuracy(y[test], score),
                                   'support': support.tolist(), 'test': test.tolist()}
                            rows.append(row)
                            key = f'{seed}_{fi}_{draw}_{budget}_{cap}_{strategy}'
                            arrays[key] = score.argmax(1)
    return {'protocol': 'selection', 'task': manifest['task'], 'classes': classes, 'candidate_widths': eligible, 'rows': rows}, arrays


def inner_splits(support: np.ndarray, y_local: np.ndarray, groups: np.ndarray,
                 budget: int, seed: int, fold: int, draw: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if budget == 1:
        return []
    n_splits = min(5, budget)
    support_y = y_local[support]
    support_groups = groups[support]
    group_counts = [len(set(support_groups[support_y == label].tolist()))
                    for label in range(int(np.max(support_y)) + 1)]
    if min(group_counts) < n_splits:
        raise ValueError(f"Support cannot sustain class-covered {n_splits}-fold CV: {group_counts}")
    # Greedy grouped stratification can put both groups of a class in one fold.
    # Assign groups explicitly so each validation fold covers every class.
    class_group_lists = {
        label: sorted(set(support_groups[support_y == label].tolist()))
        for label in range(len(group_counts))
    }
    splits = []
    base_seed = int(draw + seed * 10 + fold)
    for attempt in range(256):
        assignment: dict[str, int] = {}
        fold_class_counts = np.zeros((n_splits, len(group_counts)), dtype=np.int32)
        fold_sizes = np.zeros(n_splits, dtype=np.int32)
        rng = np.random.default_rng(base_seed + attempt)
        for label in sorted(class_group_lists, key=lambda value: (len(class_group_lists[value]), value)):
            label_groups = class_group_lists[label]
            shuffled = np.asarray(label_groups, dtype=object)[rng.permutation(len(label_groups))]
            for group in shuffled:
                key = str(group)
                if key in assignment:
                    fold_class_counts[assignment[key], label] += 1
                    continue
                least_class = int(fold_class_counts[:, label].min())
                choices = np.flatnonzero(fold_class_counts[:, label] == least_class)
                least_total = int(fold_sizes[choices].min())
                choices = choices[fold_sizes[choices] == least_total]
                chosen_fold = int(rng.choice(choices))
                assignment[key] = chosen_fold
                fold_class_counts[chosen_fold, label] += 1
                fold_sizes[chosen_fold] += 1
        if np.all(fold_class_counts > 0):
            splits = []
            for heldout in range(n_splits):
                valid = np.flatnonzero(np.asarray([assignment[str(g)] == heldout for g in support_groups]))
                train = np.flatnonzero(np.asarray([assignment[str(g)] != heldout for g in support_groups]))
                splits.append((train, valid))
            break
    if not splits:
        raise ValueError(f"Could not create class-covered stratified group folds in 256 deterministic attempts")
    for train, valid in splits:
        if (set(np.unique(support_y[train]).tolist()) != set(range(len(group_counts)))
                or set(np.unique(support_y[valid]).tolist()) != set(range(len(group_counts)))):
            missing_train = sorted(set(range(len(group_counts))) - set(np.unique(support_y[train]).tolist()))
            missing_valid = sorted(set(range(len(group_counts))) - set(np.unique(support_y[valid]).tolist()))
            raise ValueError(f"Stratified grouped validation fold lacks class coverage; "
                             f"missing_train={missing_train}; missing_valid={missing_valid}")
        if set(support_groups[train].tolist()) & set(support_groups[valid].tolist()):
            raise ValueError("inner split leaks a source group")
    return splits
