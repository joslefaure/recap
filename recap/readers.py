"""Readers fit on cached vectors; encoders never enter this module."""
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge

RIDGE_GRID = (0.1, 1., 10., 100., 1000.)
LOGISTIC_GRID = (0.01, 0.1, 1., 10., 100.)


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norm <= 1e-12):
        raise ValueError('Nonfinite or zero-norm vector')
    return x / norm


def fit_scores(reader, x, y, test, classes, *, alpha=10., c=1.):
    multi = y.ndim == 2
    targets = y.astype(np.float32) if multi else np.eye(classes, dtype=np.float32)[y]
    if reader == 'ridge':
        return Ridge(alpha=alpha, solver='lsqr').fit(x, targets).predict(test).astype(np.float32)
    if reader in ('prototype', 'positive_centroid'):
        if multi:
            if np.any(targets.sum(0) == 0) or np.any((1-targets).sum(0) == 0):
                raise ValueError('Prototype training lacks a binary class')
            positive = normalize(targets.T @ x / targets.sum(0)[:, None])
            negative = normalize((1-targets).T @ x / (1-targets).sum(0)[:, None])
            return test @ (positive if reader == 'positive_centroid' else positive-negative).T
        if set(np.unique(y)) != set(range(classes)):
            raise ValueError('Prototype training lacks a class')
        prototypes = normalize(np.stack([x[y == j].mean(0) for j in range(classes)]))
        return test @ prototypes.T
    if reader != 'logistic':
        raise ValueError(reader)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        if multi:
            if any(len(np.unique(y[:, j])) != 2 for j in range(classes)):
                raise ValueError('Logistic training lacks a binary class')
            return np.column_stack([LogisticRegression(C=c, solver='liblinear', max_iter=500)
                                    .fit(x, y[:, j]).decision_function(test) for j in range(classes)])
        clf = LogisticRegression(C=c, solver='lbfgs', max_iter=500).fit(x, y)
    score = clf.decision_function(test)
    if len(clf.classes_) == 2:
        score = np.column_stack((-score, score))
    result = np.full((len(test), classes), -1e6, dtype=np.float32)
    result[:, clf.classes_.astype(int)] = score
    return result


def class_bias(scores, y, strength=1.):
    """Fit offsets using training labels only; preserve the encoder geometry."""
    def objective(b):
        z = scores + b
        p = np.exp(z-logsumexp(z, axis=1, keepdims=True))
        value = (logsumexp(z, axis=1)-z[np.arange(len(y)), y]).mean()+.5*strength*(b*b).sum()
        gradient = (p.sum(0)-np.bincount(y, minlength=len(b)))/len(y)+strength*b
        return value, gradient
    result = minimize(objective, np.zeros(scores.shape[1]), jac=True, method='L-BFGS-B')
    if not result.success and np.linalg.norm(result.jac) > 1e-4:
        raise RuntimeError(result.message)
    return result.x.astype(np.float32)
