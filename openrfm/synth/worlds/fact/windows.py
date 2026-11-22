"""Event streams, window statistics and the F4 event-window world."""

from __future__ import annotations

import numpy as np

from openrfm.synth.stats import _log
from openrfm.synth.worlds.fact.base import FamilyConfig, KINDS, TAG, TYPED_KINDS, _clock, _label_delay, _rate


def _stream(rng, mean_events, span, start, branching, decay, season_amp, period):
    G = 64
    dt = span / G
    centers = (np.arange(G) + 0.5) * dt
    jump = rng.random(G) < 0.12
    regime = np.exp(0.9 * np.cumsum(rng.normal(size=G) * jump))
    active, act = True, np.empty(G)
    for g in range(G):
        if rng.random() < (0.055 if active else 0.13):
            active = not active
        act[g] = 1.0 if active else 0.02
    inten = (
        regime
        * act
        * (1 + season_amp * np.sin(2 * np.pi * centers / period + rng.uniform(0, 2 * np.pi)))
    )
    inten = inten / max(inten.mean(), 1e-9) * mean_events / span
    n = rng.poisson(inten * dt * (1 - branching))
    t = (np.repeat(np.arange(G), n) + rng.random(n.sum())) * dt
    kids, frontier = [t], t
    while len(frontier):
        c = rng.poisson(branching, len(frontier))
        frontier = np.repeat(frontier, c) + rng.exponential(decay, c.sum())
        frontier = frontier[frontier < span]
        kids.append(frontier)
    t = np.sort(np.concatenate(kids))
    return t[t >= start]


def window_stats(t, v, c, w, with_burst=False):
    """Window statistics at cutoffs c over A=[c-w1,c) and B=[c-w1-w2,c-w1) from sorted ticks t and marks v."""
    hi = np.searchsorted(t, c, "left")
    lo1 = np.searchsorted(t, c - w["w1"], "left")
    lo2 = np.searchsorted(t, c - w["w1"] - w["w2"], "left")
    cv = np.r_[0.0, np.cumsum(v)]
    last = (
        np.where(hi > 0, t[np.maximum(hi - 1, 0)], -w["L"] - 1)
        if len(t)
        else np.full(len(c), -w["L"] - 1)
    )
    gap = np.minimum(c - last, w["L"])
    burst = np.zeros(len(c))
    for k in range(len(c) if with_burst else 0):
        if hi[k] > lo1[k]:
            inside = t[lo1[k] : hi[k]]
            end = np.minimum(np.searchsorted(t, inside + w["b"], "left"), hi[k])
            burst[k] = (end - np.arange(lo1[k], hi[k])).max()
    return dict(
        nA=(hi - lo1).astype(float),
        nB=(lo1 - lo2).astype(float),
        gap=gap.astype(float),
        burst=burst,
        vsum=(cv[hi] - cv[lo1]) if len(t) else np.zeros(len(c)),
        nHalf=(hi - np.searchsorted(t, c - w["w1"] // 2, "left")).astype(float),
        nDouble=(hi - np.searchsorted(t, c - 2 * w["w1"], "left")).astype(float),
    )


def target_stat(stats, kind, w):
    if kind in TYPED_KINDS:
        return stats[kind]
    if kind == "count":
        return stats["nA"]
    if kind == "rate_change":
        return np.log(stats["nA"] + 1) - np.log(stats["nB"] + 1) + np.log(w["w2"] / w["w1"])
    if kind == "recency":
        return stats["gap"]
    if kind == "burst":
        return stats["burst"]
    if kind == "value_sum":
        return stats["vsum"]
    raise ValueError(f"unknown window target {kind}")


def typed_window_stats(t, marks, types, cutoffs, w, query_types):
    """Strict [cutoff-H, cutoff) typed aggregates (-1 selects all events); empty windows give zero."""
    lo = np.searchsorted(t, cutoffs - w["w1"], "left")
    hi = np.searchsorted(t, cutoffs, "left")
    out = {kind: np.zeros(len(cutoffs)) for kind in TYPED_KINDS}
    for i, (a, b, selected) in enumerate(zip(lo, hi, query_types, strict=False)):
        values = marks[a:b]
        if selected >= 0:
            keep = types[a:b] == selected
            values = values[keep]
        n = len(values)
        out["typed_count"][i] = n
        out["typed_sum"][i] = values.sum()
        out["typed_mean"][i] = values.mean() if n else 0.0
        out["typed_max"][i] = values.max() if n else 0.0
        out["type_ratio"][i] = n / (b - a) if b > a else 0.0
    return out


def _window_stats_for(d, entity, cutoffs, query_types=None):
    stats = window_stats(
        d["ent_t"][entity], d["ent_v"][entity], cutoffs, d["w"], d["target_kind"] == "burst"
    )
    if d["target_kind"] in TYPED_KINDS:
        stats.update(
            typed_window_stats(
                d["ent_t"][entity],
                d["ent_v"][entity],
                d["ent_c"][entity],
                cutoffs,
                d["w"],
                query_types,
            )
        )
    return stats


def _reg_value(stat, target_kind, w):
    return (
        np.log1p(stat)
        if target_kind in ("count", "typed_count")
        else np.log1p(stat / w["w1"])
        if target_kind == "recency"
        else stat
    )


def _label(rng, d, stat):
    if d["kind"] == "binary":
        y = (d["direction"] * stat > d["threshold"]).astype(float)
        flip = rng.random(len(y)) < d["noise"]
        return np.where(flip, 1 - y, y)
    return (
        _reg_value(stat, d["target_kind"], d["w"])
        + d["noise"] * d["scale"] * rng.normal(size=len(stat))
    )


def summary_rows(d, c_by_entity):
    """Materialised summary group for every (entity, cutoff) in c_by_entity: name -> array per entity order."""
    out = {k: [] for k in ("nHalf", "nA", "nDouble", "nB", "gap")}
    if d["target_kind"] in TYPED_KINDS:
        out.update({k: [] for k in TYPED_KINDS})
    for entry in c_by_entity:
        e, c = entry[:2]
        s = _window_stats_for(d, e, c, entry[2] if len(entry) == 3 else None)
        for k in out:
            out[k].append(s[k])
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}



def f4_world(seed, slot, cfg=FamilyConfig(), kind=None, target_kind=None):
    rng = np.random.default_rng([seed, slot, TAG["F4"]])
    kind = kind or ("regression" if rng.random() < cfg.regression_share else "binary")
    kinds = cfg.window_target_kinds
    if (
        not kinds
        or len(set(kinds)) != len(kinds)
        or set(kinds) - set(KINDS + TYPED_KINDS)
    ):
        raise ValueError("nonempty distinct supported window target kinds required")
    target_kind = target_kind or kinds[int(rng.integers(len(kinds)))]
    if target_kind not in KINDS + TYPED_KINDS:
        raise ValueError(target_kind)
    typed = target_kind in TYPED_KINDS
    span_s, unit = _clock(rng, cfg)
    E = int(_log(rng, cfg.parents_range))
    mean_events = _log(rng, cfg.mean_events_range)
    cap_mean = cfg.row_cap / E
    mean_events = min(mean_events, cap_mean)
    disp = rng.uniform(0.3, 1.2)
    branching, decay_frac = rng.uniform(0, 0.55), _log(rng, (0.002, 0.05))
    season_amp, period_frac = rng.uniform(0, 0.9), _log(rng, (0.04, 0.8))
    late = rng.random(E) < rng.uniform(0, cfg.late_start_max)
    start = np.where(late, rng.uniform(0.3, 0.8, E), rng.uniform(0, 0.15, E)) * span_s
    level = rng.normal(size=E)
    cat_k = int(rng.integers(2, 9))
    ent_t, ent_v, ent_c = [], [], []
    for e in range(E):
        tt = _stream(
            rng,
            mean_events * np.exp(disp * rng.normal() - disp**2 / 2),
            span_s,
            start[e],
            branching,
            decay_frac * span_s,
            season_amp,
            period_frac * span_s,
        )
        tk = np.unique(np.floor(tt).astype(np.int64))
        ent_t.append(tk)
        if typed:
            # Per-entity type intensity regimes, independent of its total
            # stream size; numeric marks vary by type as well as entity.
            logits = rng.normal(size=(64, cat_k))
            logits = np.cumsum(logits * (rng.random((64, 1)) < 0.12), axis=0)
            logits -= logits.max(axis=1, keepdims=True)
            probability = np.exp(logits)
            probability /= probability.sum(axis=1, keepdims=True)
            segment = np.minimum(63, tk * 64 // span_s)
            cats = (rng.random(len(tk))[:, None] > probability[segment].cumsum(axis=1)).sum(
                axis=1
            )
            cats = np.minimum(cats, cat_k - 1)
            ent_c.append(cats)
            type_level = rng.normal(size=cat_k)
            ent_v.append(level[e] + type_level[cats] + rng.normal(0, 0.5, len(tk)))
        else:
            ent_v.append(level[e] + rng.normal(0, 0.5, len(tk)))
            ent_c.append(rng.integers(cat_k, size=len(tk)))
    w1 = max(2, int(span_s * _log(rng, cfg.window_fraction_range)))
    w = dict(
        w1=w1,
        w2=max(1, int(w1 * rng.uniform(0.5, 3.0))),
        L=int(w1 * rng.uniform(2.0, 8.0)),
        b=max(1, int(w1 * rng.uniform(0.1, 0.4))),
    )
    c_lo = int(min(0.9 * span_s, max(0.3 * span_s, w["w1"] + w["w2"])))
    per = int(
        rng.integers(cfg.cutoffs_per_entity_range[0], cfg.cutoffs_per_entity_range[1] + 1)
    )
    quota = int(rng.choice(cfg.history_quota_choices))
    case_e = np.repeat(np.arange(E), per)
    case_t = rng.integers(c_lo, span_s, len(case_e)).astype(np.int64)
    if len(case_e) > cfg.max_cases:
        pick = np.sort(rng.choice(len(case_e), cfg.max_cases, replace=False))
        case_e, case_t = case_e[pick], case_t[pick]
    stat = np.empty(len(case_e))
    # A varying public selector prevents a constant task column disappearing
    # during conversion; historical applications carry their own selector too.
    case_type = rng.integers(cat_k, size=len(case_e)) if typed else None
    if typed and target_kind != "type_ratio":
        case_type[rng.random(len(case_e)) < 0.2] = -1
    context = dict(target_kind=target_kind, ent_t=ent_t, ent_v=ent_v, ent_c=ent_c, w=w)
    for e in np.unique(case_e):
        ix = np.flatnonzero(case_e == e)
        stat[ix] = target_stat(
            _window_stats_for(context, e, case_t[ix], case_type[ix] if typed else None),
            target_kind,
            w,
        )
    order = np.argsort(case_t, kind="stable")
    case_e, case_t, stat = case_e[order], case_t[order], stat[order]
    out = dict(
        family="F4",
        kind=kind,
        target_kind=target_kind,
        w=w,
        ent_t=ent_t,
        ent_v=ent_v,
        ent_c=ent_c,
        E=E,
        case_e=case_e,
        case_t=case_t,
        stat=stat,
        quota=quota,
        level=level,
    )
    if typed:
        out["case_type"] = case_type[order]
    noise = 0.0 if rng.random() < 0.5 else float(rng.uniform(0, cfg.label_noise_max))
    r = _rate(rng, cfg)
    direction = 1.0 if (target_kind != "recency" or rng.random() < 0.5) else -1.0
    out.update(noise=noise, direction=direction, target_rate=r if kind == "binary" else None)
    if kind == "binary":
        out["threshold"] = float(
            np.quantile(direction * stat, 1 - r)
        )
    else:
        out["scale"] = float(np.std(_reg_value(stat, target_kind, w)) or 1.0)
    out["y"] = _label(rng, out, stat)
    out["oracle_score"] = (
        direction * stat
        if kind == "binary"
        else _reg_value(stat, target_kind, w)
    )
    # earlier applications of the same function, published at their own cutoff (+ delay)
    h_e = np.repeat(np.arange(E), quota)
    h_t = rng.integers(c_lo, span_s, len(h_e)).astype(np.int64)
    hs = np.empty(len(h_e))
    hist_type = rng.integers(cat_k, size=len(h_e)) if typed else None
    if typed and target_kind != "type_ratio":
        hist_type[rng.random(len(h_e)) < 0.2] = -1
    for e in range(E):
        ix = np.flatnonzero(h_e == e)
        if len(ix):
            hs[ix] = target_stat(
                _window_stats_for(context, e, h_t[ix], hist_type[ix] if typed else None),
                target_kind,
                w,
            )
    out.update(
        hist_e=h_e, hist_t=h_t, hist_y=_label(rng, out, hs) if len(hs) else hs
    )
    if typed:
        out["hist_type"] = hist_type
    out["hist_delay"], out["delay_fraction"] = _label_delay(rng, cfg, len(h_e), span_s)
    out["info"] = dict(
        span_s=span_s,
        unit=unit,
        windows=w,
        mean_events=float(mean_events),
        quota=quota,
        per_entity=per,
        branching=float(branching),
        season_amp=float(season_amp),
        target_kind=target_kind,
        noise=noise,
        direction=direction,
        late_start_share=float(late.mean()),
        drift=None,
    )
    if typed:
        out["info"].update(
            event_type_vocabulary=cat_k,
            query_type_public=True,
            type_intensity_regime="entity_local",
            empty_aggregate_value=0.0,
            all_event_selector=-1,
        )
    return out

