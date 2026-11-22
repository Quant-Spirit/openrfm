"""Frozen Kumo-Tabular predictions on exported features: support draws, fit/predict, ensemble, metric and SE."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from openrfm.models.consumers import KumoPredictor

DRAW_SEED, CONSUMER_SEED, BOOTSTRAP_SEED, BOOTSTRAPS = 101212, 1008, 101212, 256


def nested_order(labels, draw, task_type):
    """Seeded support order; for classification the classes alternate, each in its own random order."""
    rng = np.random.default_rng(1008+draw)
    if task_type != 'classification':
        return rng.permutation(len(labels))
    labels = np.asarray(labels)
    parts = [rng.permutation(np.flatnonzero(labels == c)).tolist() for c in np.unique(labels)]
    return np.asarray([part[i] for i in range(max(map(len, parts), default=0)) for part in parts if i < len(part)],
                      dtype=int)


def support_draw(labels, k, draw, selection, task_type, disjoint=False):
    """Positions of the `k` support rows of draw `draw`; disjoint draws are consecutive `k`-row blocks (cyclic) of one seeded permutation."""
    if disjoint:
        return np.random.default_rng(DRAW_SEED).permutation(len(labels))[(draw*k+np.arange(k)) % len(labels)]
    if selection == 'random':
        return np.random.default_rng(DRAW_SEED+draw).permutation(len(labels))[:k]
    return nested_order(labels, draw, task_type)[:k]


def standardize(sx, qx):
    """Support-fitted standardization; a constant column keeps scale 1."""
    mean, std = sx.mean(0), sx.std(0, unbiased=False)
    std = torch.where(std > 1e-8, std, torch.ones_like(std))
    return (sx-mean)/std, (qx-mean)/std


def fit_predict(kumo, sx, sy, qx, task_type, chunk):
    """Positive-class probabilities or median predictions of the queries, in query chunks of `chunk` rows."""
    torch.manual_seed(CONSUMER_SEED)
    np.random.seed(CONSUMER_SEED)
    sx, qx = standardize(torch.as_tensor(sx, device='cuda'), torch.as_tensor(qx, device='cuda'))
    dtype = torch.long if task_type == 'classification' else torch.float32
    kumo.fit(sx, torch.as_tensor(sy, device='cuda', dtype=dtype))
    out = []
    try:
        for start in range(0, len(qx), chunk):
            p = kumo.predict(qx[start:start+chunk])
            out.append((p.softmax(-1)[:, 1] if task_type == 'classification' else p.median(-1).values).cpu().numpy())
    finally:
        kumo.clear()
    return np.concatenate(out)


def draw_predictions(export, cell, task_type):
    """[E, Q] predictions of the E support draws of a recipe cell on `export` (sx, sy, qx)."""
    sy, k = export['sy'], min(cell['K'], len(export['sy']))
    kumo = KumoPredictor(task_type, 'small')
    transform = cell['transform']
    preds = []
    for draw in range(cell['E']):
        keep = support_draw(sy, k, draw, cell['selection'], task_type, cell.get('disjoint_draws', False))
        if transform is None:
            chunk = 4096 if k <= 16384 else 500
            preds.append(fit_predict(kumo, export['sx'][keep], sy[keep], export['qx'], task_type, chunk))
            continue
        # Regression targets are transformed, then centred on the support median and scaled by its std.
        y = np.log1p(sy) if transform == 'log1p' else np.sqrt(sy) if transform == 'sqrt' else sy
        center, scale = float(np.median(y[keep])), max(float(np.std(y[keep])), 1e-6)
        p = fit_predict(kumo, export['sx'][keep], ((y-center)/scale)[keep], export['qx'], task_type, 500)*scale+center
        preds.append(np.expm1(p) if transform == 'log1p' else np.square(np.maximum(p, 0)) if transform == 'sqrt' else p)
    return np.asarray(preds)


def ensemble(preds, task_type):
    """Mean of the draws for classification, median for regression."""
    return np.median(preds, axis=0) if task_type == 'regression' else preds.mean(0)


def metric(y, p, task_type):
    """AUROC for classification, mean absolute error for regression."""
    if task_type == 'classification':
        return float(roc_auc_score(y, p))
    return float(np.mean(np.abs(np.asarray(y)-p)))


def metric_se(y, p, entity, task_type):
    """Entity-bootstrap standard error of the metric."""
    y, entity = np.asarray(y), np.asarray(entity)
    order = np.argsort(entity, kind='stable')
    index = np.split(order, np.flatnonzero(entity[order][1:] != entity[order][:-1])+1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAPS):
        take = np.concatenate([index[i] for i in rng.integers(len(index), size=len(index))])
        if task_type == 'classification' and len(np.unique(y[take])) < 2:
            continue
        values.append(metric(y[take], p[take], task_type))
    return float(np.std(values, ddof=1))
