"""F1/F2 fact-row and F3 same-time slate worlds."""

from __future__ import annotations

import numpy as np

from openrfm.synth.stats import _log
from openrfm.synth.worlds.fact.base import (
    FamilyConfig, TAG, _clock, _drift, _finish_latent, _label_delay, _row_model, _std, _unique_ticks, _walk,
)


def _autocomplete_world(
    rng,
    cfg,
    family,
    kind,
    T,
    pids,
    nparents,
    parent_size,
    row_part,
    parent_parts,
    x,
    cat,
    extra,
    delay,
    info,
):
    """Assemble tables, private arrays and census for F1/F2 once the latent signal is built."""
    ws = np.array([w for w, _ in parent_parts] + [1 - sum(w for w, _ in parent_parts)])
    s = np.sqrt(ws[-1]) * row_part
    for w, part in parent_parts:
        s = s + np.sqrt(w) * part
    y, aux = _finish_latent(rng, cfg, s, kind)
    share = aux["share"]
    offset = np.sqrt(share) * np.sqrt(ws[-1]) * row_part
    sigma_ref = float(np.sqrt(1 - share + share * info.get("unseen", 0.0)))
    pv = [share * w for w, _ in parent_parts]
    pub = T + delay
    start = int(cfg.case_start_fraction * info["span_s"])
    pool = np.flatnonzero(T >= start)
    cases = (
        np.sort(rng.choice(pool, min(cfg.max_cases, len(pool)), replace=False))
        if len(pool)
        else pool
    )
    return dict(
        family=family,
        kind=kind,
        T=T,
        pids=pids,
        pub=pub,
        y=y,
        aux=aux,
        offset=offset,
        sigma_ref=sigma_ref,
        pv=pv,
        cases=cases,
        state=aux["sig"],
        x=x,
        cat=cat,
        extra=extra,
        nparents=nparents,
        parent_size=parent_size,
        info=info,
    )




def f1_world(seed, slot, cfg=FamilyConfig(), kind=None):
    rng = np.random.default_rng([seed, slot, TAG["F1"]])
    kind = kind or ("regression" if rng.random() < cfg.regression_share else "binary")
    span_s, unit = _clock(rng, cfg)
    P = int(_log(rng, cfg.parents_range))
    mean_rows = _log(rng, cfg.rows_per_parent_range)
    disp = rng.uniform(0.3, 1.5)
    rate = mean_rows * np.exp(disp * rng.normal(size=P) - disp**2 / 2)
    first = np.where(
        rng.random(P) < rng.uniform(0, cfg.late_start_max),
        rng.uniform(0.4, 0.9, P),
        rng.uniform(0, 0.3, P),
    )
    counts = rng.poisson(rate * (1 - first))
    if counts.sum() > cfg.row_cap:
        counts = rng.poisson(counts * cfg.row_cap / counts.sum())
    pid = np.repeat(np.arange(P), counts)
    N = len(pid)
    fr = first[pid]
    tf = fr + (1 - fr) * rng.random(N)
    burst = rng.uniform(0, 0.6)
    prev = np.r_[0.0, tf[:-1]]
    use = (rng.random(N) < burst) & np.r_[False, pid[1:] == pid[:-1]]
    tf = np.where(use, np.clip(prev + rng.exponential(0.01, N), fr, 1.0), tf)
    T = np.floor(tf * span_s).astype(np.int64)
    order = np.argsort(T, kind="stable")
    pid, T = pid[order], _unique_ticks(T[order])
    frac = T / span_s
    z = rng.normal(size=P)
    walk, rho = _walk(rng, P), rng.uniform(0, cfg.drift_max)
    parent_raw = z[pid] + _drift(walk, rho, pid, frac)
    x, cat, extra, row_raw = _row_model(rng, cfg, N)
    slope = rng.random() < 0.35
    if slope:
        parent_raw = parent_raw + rng.uniform(0.3, 1.0) * rng.normal(size=P)[pid] * x[:, 0]
    ps = float(rng.uniform(*cfg.parent_share_range))
    delay, delay_frac = _label_delay(rng, cfg, N, span_s)
    info = dict(
        span_s=span_s,
        unit=unit,
        drift=None,
        rho=float(rho),
        slope=bool(slope),
        burst=float(burst),
        label_delay_fraction=delay_frac,
        unseen=0.0,
        parent_share=ps,
        mean_rows=float(mean_rows),
        static_leak=float(rng.uniform(0, 0.5)),
    )
    W = _autocomplete_world(
        rng,
        cfg,
        "F1",
        kind,
        T,
        [pid],
        P,
        [P],
        _std(row_raw),
        [(ps, _std(parent_raw))],
        x,
        cat,
        extra,
        delay,
        info,
    )
    eta = info["static_leak"]
    W["parent_static"] = [eta * z + np.sqrt(1 - eta**2) * rng.normal(size=P)]
    W["rng_tail"] = rng
    return W


def f2_world(seed, slot, cfg=FamilyConfig(), kind=None):
    rng = np.random.default_rng([seed, slot, TAG["F2"]])
    kind = kind or ("regression" if rng.random() < cfg.regression_share else "binary")
    span_s, unit = _clock(rng, cfg)
    A, I = int(_log(rng, cfg.parents_range)), int(_log(rng, cfg.items_range))
    N = int(min(cfg.row_cap, _log(rng, cfg.pair_rows_range)))
    disp_a, disp_i = (
        rng.uniform(0.3, 1.2),
        rng.uniform(0.5, 2.0),
    )
    wa, wi = np.exp(disp_a * rng.normal(size=A)), np.exp(disp_i * rng.normal(size=I))
    first_a = np.where(
        rng.random(A) < rng.uniform(0, cfg.late_start_max),
        rng.uniform(0.4, 0.9, A),
        rng.uniform(0, 0.3, A),
    )
    first_i = np.where(
        rng.random(I) < rng.uniform(0, cfg.late_start_max),
        rng.uniform(0.4, 0.9, I),
        rng.uniform(0, 0.3, I),
    )
    a = rng.choice(A, N, p=wa / wa.sum())
    i = rng.choice(I, N, p=wi / wi.sum())
    start = np.maximum(first_a[a], first_i[i])
    tf = start + (1 - start) * rng.random(N)
    T = np.floor(tf * span_s).astype(np.int64)
    order = np.argsort(T, kind="stable")
    a, i, T = a[order], i[order], _unique_ticks(T[order])
    frac = T / span_s
    walk_a, walk_i, rho = _walk(rng, A), _walk(rng, I), rng.uniform(0, cfg.drift_max)
    ua = rng.normal(size=A)[a] + _drift(walk_a, rho, a, frac)
    vi = rng.normal(size=I)[i] + _drift(walk_i, rho, i, frac)
    rank = int(rng.integers(0, 3))
    inter = (
        np.zeros(N)
        if rank == 0
        else (rng.normal(size=(A, rank))[a] * rng.normal(size=(I, rank))[i]).sum(1)
    )
    x, cat, extra, row_raw = _row_model(rng, cfg, N)
    weights = rng.dirichlet([1.2, 1.2, 0.8 if rank else 1e-3, 1.0])
    delay, delay_frac = _label_delay(rng, cfg, N, span_s)
    info = dict(
        span_s=span_s,
        unit=unit,
        drift=None,
        rho=float(rho),
        interaction_rank=rank,
        interaction_share=float(weights[2]),
        label_delay_fraction=delay_frac,
        unseen=float(weights[2]),
        weights=weights.tolist(),
    )
    parts = [(float(weights[0]), _std(ua)), (float(weights[1]), _std(vi))]
    if rank:
        parts.append((float(weights[2]), _std(inter)))
    W = _autocomplete_world(
        rng,
        cfg,
        "F2",
        kind,
        T,
        [a, i],
        A,
        [A, I],
        _std(row_raw),
        parts,
        x,
        cat,
        extra,
        delay,
        info,
    )
    W["pv"] = W["pv"][:2]
    W["pids"] = [a, i]
    W["parent_static"] = [rng.normal(size=A), rng.normal(size=I)]
    W["rng_tail"] = rng
    return W


def f3_world(seed, slot, cfg=FamilyConfig(), kind=None):
    rng = np.random.default_rng([seed, slot, TAG["F3"]])
    kind = kind or ("contrast" if rng.random() < cfg.regression_share else "choice")
    span_s, unit = _clock(rng, cfg)
    A, I = int(_log(rng, cfg.parents_range)), int(_log(rng, cfg.items_range))
    S = int(_log(rng, cfg.slates_range))
    mlo, mhi = cfg.slate_size_range
    mhi = min(mhi, I)
    typical = int(rng.integers(mlo, mhi + 1))
    m = np.clip(typical + rng.integers(-2, 3, S), mlo, mhi)
    wi = np.exp(rng.uniform(0.3, 1.5) * rng.normal(size=I))
    first_i = np.where(
        rng.random(I) < rng.uniform(0, cfg.late_start_max),
        rng.uniform(0.4, 0.9, I),
        rng.uniform(0, 0.3, I),
    )
    wa = np.exp(rng.uniform(0.3, 1.2) * rng.normal(size=A))
    t_sl = _unique_ticks(np.sort(np.floor(rng.random(S) * span_s).astype(np.int64)))
    actor = rng.choice(A, S, p=wa / wa.sum())
    rows_slate, rows_item = [], []
    for s in range(S):
        p = wi * (first_i <= t_sl[s] / span_s)
        if (p > 0).sum() < m[s]:
            p = wi
        pick = rng.choice(I, int(m[s]), replace=False, p=p / p.sum())
        rows_slate.append(np.full(len(pick), s))
        rows_item.append(pick)
    sl, it = np.concatenate(rows_slate), np.concatenate(rows_item)
    N = len(sl)
    pos = np.concatenate([np.arange(k) for k in m])
    ac, T = actor[sl], t_sl[sl]
    walk, rho = _walk(rng, I), rng.uniform(0, cfg.drift_max)
    q = rng.normal(size=I)[it] + _drift(walk, rho, it, T / span_s)
    rank = int(rng.integers(0, 3))
    taste = (
        np.zeros(N)
        if rank == 0
        else (rng.normal(size=(A, rank))[ac] * rng.normal(size=(I, rank))[it]).sum(1)
    )
    d = int(rng.integers(cfg.covariates_range[0], min(cfg.covariates_range[1], 3) + 1))
    x = rng.normal(size=(N, d))
    cov_raw = x @ rng.normal(size=d) + rng.normal() * np.log1p(pos)
    w = rng.dirichlet([1.2, 0.8 if rank else 1e-3, 1.0])
    beta = float(rng.uniform(0.8, 2.5))
    U = beta * (
        np.sqrt(w[0]) * _std(q) + np.sqrt(w[1]) * _std(taste) + np.sqrt(w[2]) * _std(cov_raw)
    )
    cov_u = beta * np.sqrt(w[2]) * _std(cov_raw)
    count = np.bincount(sl)
    gmean = lambda v: (np.bincount(sl, v) / count)[sl]
    outside = bool(rng.random() < cfg.outside_option_share and kind == "choice")
    o_base = float(rng.uniform(-1.5, 6.0))
    o_slate = (
        o_base
        + 0.6 * rng.normal(size=A)[actor]
        if outside
        else np.full(S, -np.inf)
    )
    extra = rng.normal(
        size=(N, int(rng.integers(cfg.distractor_range[0], cfg.distractor_range[1] + 1)))
    )
    if kind == "choice":
        score = U + rng.gumbel(size=N)
        best = np.full(S, -np.inf)
        np.maximum.at(best, sl, score)
        out_win = (o_slate + rng.gumbel(size=S)) > best
        y = ((score == best[sl]) & ~out_win[sl]).astype(float)
        e = np.exp(U - U.max())
        state = (
            e / (np.bincount(sl, e) + np.exp(o_slate - U.max()))[sl]
        )
        aux = dict(beta=beta, outside=outside, o_base=o_base if outside else None)
        pv = beta**2 * float(w[0])
    else:
        share = float(rng.uniform(*cfg.signal_share_range))
        uc_scale = float(np.std(U - gmean(U)))
        Uc = (U - gmean(U)) / uc_scale
        y = np.sqrt(share) * Uc + np.sqrt(1 - share) * rng.normal(size=N)
        state = np.sqrt(share) * Uc
        aux = dict(share=share, beta=beta, outside=False, o_base=None, uc_scale=uc_scale)
        pv = share * beta**2 * float(w[0]) / uc_scale**2
    delay, delay_frac = _label_delay(rng, cfg, N, span_s)
    late = np.flatnonzero(t_sl >= int(cfg.case_start_fraction * span_s))
    perm = rng.permutation(late)
    chosen = (
        perm[: max(1, int(np.searchsorted(np.cumsum(m[perm]), cfg.max_cases)))]
        if len(perm)
        else perm
    )
    cases = np.flatnonzero(np.isin(sl, chosen))
    info = dict(
        span_s=span_s,
        unit=unit,
        rho=float(rho),
        taste_rank=rank,
        weights=w.tolist(),
        label_delay_fraction=delay_frac,
        slates=S,
        mean_slate_size=float(m.mean()),
        outside_option=outside,
        drift=None,
    )
    return dict(
        family="F3",
        kind=kind,
        T=T,
        slate=sl,
        item=it,
        actor=ac,
        pos=pos,
        x=x,
        extra=extra,
        pub=T + delay,
        y=y,
        aux=aux,
        cov_u=cov_u,
        state=state,
        cases=cases,
        nparents=[A, I],
        slates=S,
        actor_of_slate=actor,
        t_slate=t_sl,
        gmean_cov=gmean(cov_u),
        pv=pv,
        info=info,
        rng_tail=rng,
    )

