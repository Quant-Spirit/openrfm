"""Sample positive parameters on a log scale and score regression predictions."""

import numpy as np


def _log(rng, bounds):
    return float(np.exp(rng.uniform(*np.log(bounds))))


def _r2(pred: np.ndarray, y: np.ndarray) -> float:
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
