"""Shared constants, configuration and latent/table helpers of the fact families."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from openrfm.synth.stats import _log, _r2


RECIPE = "fact_families_v1"
ID_OFFSET = {"F1": 61_000_000, "F2": 62_000_000, "F3": 63_000_000, "F4": 64_000_000}
TAG = {"F1": 61001, "F2": 62001, "F3": 63001, "F4": 64001}
EPOCH_S = 946_684_800
MAX_EVENT_S = 2**31 - 1
LABEL_TABLE, TASK_TABLE, TARGET = "label_history", "a5_task_0", "a5_y_0"
BIG = np.int64(1) << 41  # pub/cutoff seconds < 2**41; ids < 2**21 keep ids*BIG inside int64
KINDS = ("count", "rate_change", "recency", "burst", "value_sum")
TYPED_KINDS = ("typed_count", "typed_sum", "typed_mean", "typed_max", "type_ratio")


@dataclass(frozen=True)
class FamilyConfig:
    span_range: tuple = (32.0, 1024.0)
    seconds_per_unit_range: tuple = (60.0, 86400.0 * 30)
    parents_range: tuple = (60, 500)  # F1 parents, F2 actors, F3 actors, F4 entities
    items_range: tuple = (40, 400)  # F2 / F3 items
    rows_per_parent_range: tuple = (1.5, 60.0)
    pair_rows_range: tuple = (1500, 20000)  # F2 fact rows
    row_cap: int = 30000
    covariates_range: tuple = (1, 5)
    distractor_range: tuple = (0, 4)
    context_tables_range: tuple = (0, 2)
    signal_share_range: tuple = (0.35, 0.95)
    parent_share_range: tuple = (0.25, 0.8)  # share of the signal that is parent state
    drift_max: float = 0.6
    late_start_max: float = (
        0.6  # per world: share of parents/items that first appear after the cases begin
    )
    low_rate_share: float = 0.5
    low_rate_range: tuple = (0.01, 0.10)
    high_rate_range: tuple = (0.10, 0.5)
    regression_share: float = 0.4
    label_delay_share: float = 0.5
    label_delay_fraction_range: tuple = (0.002, 0.05)
    case_start_fraction: float = 0.35
    max_cases: int = 1500
    slate_size_range: tuple = (2, 12)
    slates_range: tuple = (150, 1500)
    outside_option_share: float = 0.5
    mean_events_range: tuple = (3.0, 120.0)
    window_fraction_range: tuple = (0.01, 0.15)
    window_target_kinds: tuple = ("count", "rate_change", "recency", "burst", "value_sum")
    history_quota_choices: tuple = (0, 8, 32, 128)
    cutoffs_per_entity_range: tuple = (1, 3)
    label_noise_max: float = 0.1
    min_cases: int = 100
    min_class_count: int = 10
    oracle_floor_cls: float = 0.6
    oracle_floor_reg: float = 0.05


def _std(x):
    s = np.std(x)
    return (x - np.mean(x)) / (s if s > 0 else 1.0)


def _tags(rng, n, prefix):
    out = set()
    while len(out) < n:
        out.add(f"{prefix}_{int(rng.integers(16**6)):06x}")
    return sorted(out, key=lambda _: rng.random())


def _unique_ticks(T):
    """Strictly increasing ticks from sorted ticks, moving each tie forward by the minimum."""
    i = np.arange(len(T))
    return np.maximum.accumulate(T - i) + i


def _rate(rng, cfg):
    lo, hi = cfg.low_rate_range if rng.random() < cfg.low_rate_share else cfg.high_rate_range
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _auc(score, y):
    from openrfm.synth.worlds.history_oracle import _auc_fast

    return _auc_fast(np.asarray(score, float), np.asarray(y).astype(int))


def metric(kind, pred, y):
    return _auc(pred, y) if kind == "binary" else _r2(pred, y)


def _label_delay(rng, cfg, n, span_s):
    """Publication delay in seconds; at least one second, so a label never shares its row's tick."""
    if rng.random() >= cfg.label_delay_share:
        return np.ones(n, np.int64), 0.0
    frac = _log(rng, cfg.label_delay_fraction_range)
    delay = 1 + np.floor(rng.exponential(frac * span_s, n)).astype(np.int64)
    return np.minimum(delay, MAX_EVENT_S - EPOCH_S - span_s - cfg.row_cap - 2), frac


def evidence(ids, pub, g, h, case_ids, case_t, prior_var):
    """Shrunk one-step parent effect from labels published strictly before each cutoff; returns (effect, n_visible)."""
    keys = ids.astype(np.int64) * BIG + pub.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    cg = np.r_[0.0, np.cumsum(g[order])]
    ch = np.r_[0.0, np.cumsum(h[order])]
    lo = np.searchsorted(keys, case_ids.astype(np.int64) * BIG, "left")
    hi = np.searchsorted(
        keys, case_ids.astype(np.int64) * BIG + case_t.astype(np.int64), "left"
    )
    return (cg[hi] - cg[lo]) / (ch[hi] - ch[lo] + 1.0 / prior_var), hi - lo


def _probit_gh(offset, meas, tau, sigma):
    z = (offset - tau) / sigma
    p = np.clip(ndtr(z), 1e-6, 1 - 1e-6)
    phi = np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
    return phi / sigma * (meas - p) / (p * (1 - p)), phi**2 / sigma**2 / (p * (1 - p))


def _gauss_gh(offset, meas, sigma):
    return (meas - offset) / sigma**2, np.full(len(offset), 1.0 / sigma**2)


def _to_y(kind, pred, tau, sigma, transform):
    """Map a latent-scale prediction to a score on the label scale."""
    if kind == "binary":
        return ndtr((pred - tau) / sigma)
    return np.exp(transform * pred + transform**2 * sigma**2 / 2) if transform else pred


def census_visible(n):
    n = np.asarray(n)
    return dict(
        share_cold=float((n == 0).mean()),
        share_ge16=float((n >= 16).mean()),
        share_ge64=float((n >= 64).mean()),
        mean=float(n.mean()),
        median=float(np.median(n)),
        p90=float(np.quantile(n, 0.9)),
        max=int(n.max()),
    )


def _row_model(rng, cfg, N):
    """Row covariates (x, categorical, distractors) and their raw contribution to the signal."""
    d = int(rng.integers(cfg.covariates_range[0], cfg.covariates_range[1] + 1))
    x = rng.normal(size=(N, d))
    K = int(rng.integers(2, 9))
    cat = rng.integers(K, size=N)
    raw = x @ rng.normal(size=d) + rng.normal(size=K)[cat]
    if rng.random() < 0.5 and d >= 2:
        raw = raw + rng.normal() * np.tanh(x[:, 0] * x[:, 1])
    extra = rng.normal(
        size=(N, int(rng.integers(cfg.distractor_range[0], cfg.distractor_range[1] + 1)))
    )
    return x, cat, extra, raw


def _finish_latent(rng, cfg, s, kind):
    """Binary threshold or regression value from standardised signal s; returns y, u, aux."""
    share = float(rng.uniform(*cfg.signal_share_range))
    sig = np.sqrt(share) * s
    u = sig + np.sqrt(1 - share) * rng.normal(size=len(s))
    if kind == "binary":
        r = _rate(rng, cfg)
        tau = float(np.quantile(u, 1 - r))
        return (u > tau).astype(float), dict(
            share=share,
            tau=tau,
            sigma=float(np.sqrt(1 - share)),
            transform=0.0,
            target_rate=r,
            sig=sig,
        )
    c = float(rng.uniform(0.4, 1.0)) if rng.random() < 0.25 else 0.0
    y = np.exp(c * u) if c else u
    return y, dict(
        share=share,
        tau=0.0,
        sigma=float(np.sqrt(1 - share)),
        transform=c,
        target_rate=None,
        sig=sig,
    )


def _walk(rng, n):
    K = 6
    return np.cumsum(rng.normal(size=(n, K)), 1) / np.sqrt(np.arange(1, K + 1))


def _drift(walk, rho, ids, frac):
    K = walk.shape[1]
    pos = np.clip(frac, 0, 1) * (K - 1)
    i0 = np.minimum(pos.astype(int), K - 2)
    f = pos - i0
    return rho * (walk[ids, i0] * (1 - f) + walk[ids, i0 + 1] * f)


def _clock(rng, cfg):
    span = _log(rng, cfg.span_range)
    unit = int(round(_log(rng, cfg.seconds_per_unit_range)))
    # Native row clocks are signed int32 seconds. Reserve room for publication
    # delays and tie-breaking; preserve the sampled unit as public metadata.
    return min(int(span * unit), (MAX_EVENT_S - EPOCH_S - 2 * cfg.row_cap) // 2), unit


def _context_tables(rng, cfg, parent_cols, tables):
    """Distractor lookup tables reached by a foreign key from the parent table."""
    n = int(rng.integers(cfg.context_tables_range[0], cfg.context_tables_range[1] + 1))
    names = _tags(rng, 2 * n, "ctx")
    fks = {}
    for j in range(n):
        size = int(rng.integers(4, 33))
        table, key = names[2 * j], names[2 * j + 1]
        tables[table] = (
            dict(**{key: np.arange(size, dtype=np.int64), "feat": rng.normal(size=size)}),
            key,
            {},
            None,
        )
        parent_cols[key] = rng.integers(
            size, size=len(next(iter(parent_cols.values())))
        ).astype(np.int64)
        fks[key] = table
    return fks


def _fact_columns(names, x, cat, extra):
    cols = {names["x"][j]: x[:, j] for j in range(x.shape[1])}
    cols[names["cat"]] = cat.astype(np.int64)
    for j in range(extra.shape[1]):
        cols[names["extra"][j]] = extra[:, j]
    return cols


def _names(rng, x, extra, nfk):
    pool = _tags(rng, 4 + x.shape[1] + extra.shape[1] + nfk, "c")
    return dict(
        fact=pool[0],
        parents=pool[1:3],
        cat=pool[3],
        x=pool[4 : 4 + x.shape[1]],
        extra=pool[4 + x.shape[1] : 4 + x.shape[1] + extra.shape[1]],
        fk=pool[4 + x.shape[1] + extra.shape[1] :],
    )

