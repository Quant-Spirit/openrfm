"""Generate relational event streams with autocomplete and future-window targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from openrfm.synth.worlds.sql import OPERATORS, observable_facts
from openrfm.synth.stats import _log

FUNCTIONALS = (
    "exists",
    "count_gt",
    "count",
    "sum",
    "mean",
    "next_category",
    "time_to_next",
    "max",
)

FACT_NAMES = OPERATORS + (
    "own_attribute",
    "parent_attribute",
    "ancestor_attribute",
    "peer_attribute",
    "event_max",
    "event_min",
    "event_last",
)


@dataclass(frozen=True)
class PriorConfig:
    entities: int = 192
    cutoffs_per_entity: int = 4
    own_event_cap: int = 256
    peer_event_cap: int = 256
    mean_events_range: tuple[float, float] = (2.0, 256.0)
    span_range: tuple[float, float] = (32.0, 1024.0)
    horizon_fraction_range: tuple[float, float] = (0.025, 0.25)
    seconds_per_unit_range: tuple[float, float] = (60.0, 86400.0 * 30)
    width_range: tuple[int, int] = (1, 32)
    hub_count_range: tuple[int, int] = (4, 24)
    relation_count_range: tuple[int, int] = (1, 3)
    peer_count_range: tuple[int, int] = (1, 8)
    text_probability: float = 0.2
    autocomplete_probability: float = 0.5
    label_history_cap: int = 16
    label_publication_jitter_fraction_range: tuple[float, float] = (0.0, 0.0)
    version: str = "unified_stream_cells_v2"


def _mean(a):
    return float(np.mean(a)) if len(a) else 0.0


def _window(stream, cutoff, horizon, functional, threshold):
    ix = (stream["times"] > cutoff) & (stream["times"] <= cutoff + horizon)
    values, times, cats = stream["values"][ix], stream["times"][ix], stream["categories"][ix]
    if functional == "exists":
        return float(len(times) > 0)
    if functional == "count_gt":
        return float(len(times) > threshold)
    if functional == "count":
        return float(len(times))
    if functional == "sum":
        return float(values.sum())
    if functional == "mean":
        return _mean(values)  # Empty aggregate is defined as zero.
    if functional == "max":
        return float(values.max()) if len(values) else 0.0
    if functional == "next_category":
        return float(cats[0]) if len(cats) else 0.0
    if functional == "time_to_next":
        return float((times[0] - cutoff) / horizon) if len(times) else 1.0
    raise ValueError(functional)


def _history(stream, cutoff, cap, label_times, labels, label_cap, source=0):
    ix = np.flatnonzero(stream["times"] < cutoff)[-cap:] if cap else np.array([], int)
    li = np.flatnonzero(label_times < cutoff)[-label_cap:]
    return {key: stream[key][ix] for key in ("times", "values", "categories")} | {
        "label_times": label_times[li],
        "labels": labels[li],
        "source_ids": np.full(len(ix), source, dtype=int),
        "label_source_ids": np.full(len(li), source, dtype=int),
    }


def _merge(histories, cap, label_cap):
    result = {}
    for time_key, keys, limit in [
        ("times", ("times", "values", "categories", "source_ids"), cap),
        ("label_times", ("label_times", "labels", "label_source_ids"), label_cap),
    ]:
        order = np.argsort(np.concatenate([h[time_key] for h in histories]))[-limit:]
        for key in keys:
            result[key] = np.concatenate([h[key] for h in histories])[order]
    return result


def _layout(seed, slot, relations):
    """Random relational layout, independent of streams and task selection."""
    rng = np.random.default_rng([seed, slot, 84011])
    count = int(np.clip(round(_log(rng, (relations + 3, 10))), relations + 3, 10))
    compact = count < relations + 5
    return dict(
        compact=compact,
        extra_tables=count - (relations + 3 if compact else relations + 5),
        tables=count,
    )


def _sample_event_times(rng, counts, dt, branching, decay, span):
    """Sample arrivals inside each interval and their descendants within the observation span."""
    times = np.concatenate([i * dt + rng.random(n) * dt for i, n in enumerate(counts)])
    frontier, all_times = times, [times]
    while len(frontier):
        child_counts = rng.poisson(branching, len(frontier))
        frontier = np.repeat(frontier, child_counts) + rng.exponential(
            decay, child_counts.sum()
        )
        frontier = frontier[frontier < span]
        all_times.append(frontier)
    return np.sort(np.concatenate(all_times))


def generate_world(seed, slot, entities=None, config=None, task_kind=None, functional=None):
    """One world; the task and cutoff policy cannot change any underlying stream event."""
    cfg = config or PriorConfig()
    entities = entities or cfg.entities
    rng = np.random.default_rng([seed, slot, 84008])
    span, seconds = _log(rng, cfg.span_range), int(round(_log(rng, cfg.seconds_per_unit_range)))
    mean_events = _log(rng, cfg.mean_events_range)
    width = int(np.clip(round(_log(rng, cfg.width_range)), *cfg.width_range))
    hubs = int(rng.integers(cfg.hub_count_range[0], cfg.hub_count_range[1] + 1))
    relations = int(rng.integers(cfg.relation_count_range[0], cfg.relation_count_range[1] + 1))
    parent = rng.integers(hubs, size=entities)
    ancestor_ids = rng.integers(max(2, hubs // 3), size=hubs)
    ancestor_values = rng.normal(size=max(2, hubs // 3))
    parent_values, attributes = rng.normal(size=hubs), rng.normal(size=entities)
    weights = rng.dirichlet(np.ones(5) * float(_log(rng, (0.12, 2.0)))) * rng.choice(
        [-1.0, 1.0], 5
    )
    hub_share, coupling = rng.uniform(0, 1, 2)
    hub_state = rng.normal(size=hubs)
    level = np.sqrt(hub_share) * hub_state[parent] + np.sqrt(1 - hub_share) * rng.normal(
        size=entities
    )
    intensity_level = coupling * level + np.sqrt(1 - coupling**2) * rng.normal(size=entities)
    drift, hub_drift = rng.normal(0, 0.8, entities), rng.normal(0, 0.7, hubs)
    peer_attribute = np.array([_mean(attributes[parent == p]) for p in parent])
    static = (
        weights[0] * attributes
        + weights[1] * parent_values[parent]
        + weights[2] * ancestor_values[ancestor_ids[parent]]
        + weights[3] * peer_attribute
    )
    mark_noise = _log(rng, (0.15, 1.2))
    mark_scales = np.exp(rng.uniform(0, 1.1) * rng.normal(size=entities))
    parent_shortcut = bool(rng.integers(2))
    season_amp, period = rng.uniform(0, 0.95), _log(rng, (0.04 * span, 0.8 * span))
    branching, decay = rng.uniform(0, 0.65), _log(rng, (0.002 * span, 0.08 * span))
    streams_by_relation = []
    for _ in range(relations):
        streams = []
        relation_scale = _log(rng, (0.35, 2.5))
        for entity in range(entities):
            # Piecewise constant intensity on an absolute clock, with continuous
            # arrival times and subcritical Hawkes offspring.
            grid = 64
            dt = span / grid
            active, centers = True, (np.arange(grid) + 0.5) * dt
            activity = []
            for _ in centers:
                if rng.random() < (0.055 if active else 0.13):
                    active = not active
                activity.append(1.0 if active else 0.015)
            phase = rng.uniform(0, 2 * np.pi)
            state = (
                intensity_level[entity]
                + 0.6 * drift[entity] * (centers / span - 0.5)
                + 0.4 * hub_drift[parent[entity]] * (centers / span - 0.5)
            )
            intensity = (
                mean_events / span * relation_scale * np.exp(np.clip(1.3 * state, -4, 4))
            )
            intensity *= np.asarray(activity) * (
                1 + season_amp * np.sin(2 * np.pi * centers / period + phase)
            )
            counts = rng.poisson(intensity * dt * (1 - branching))
            times = _sample_event_times(rng, counts, dt, branching, decay, span)
            dynamic = (
                level[entity]
                + drift[entity] * (times / span - 0.5)
                + hub_drift[parent[entity]] * (times / span - 0.5) * hub_share
            )
            values = (
                static[entity]
                + weights[4] * dynamic
                + rng.normal(0, mark_noise * mark_scales[entity], len(times))
            )
            categories = (values + rng.normal(0, 0.3, len(times)) > 0).astype(int)
            streams.append(dict(times=times, values=values, categories=categories))
        streams_by_relation.append(streams)
    # The independent task RNG is deliberately created only after streams exist.
    trng = np.random.default_rng([seed, slot, 84009])
    task_kind = task_kind or ("autocomplete" if slot % 2 == 0 else "future_window")
    functional = functional or str(trng.choice(FUNCTIONALS))
    horizon = (
        max(1, int(round(span * _log(trng, cfg.horizon_fraction_range) * seconds))) / seconds
    )
    target_relation = int(trng.integers(relations))
    streams = streams_by_relation[target_relation]
    threshold = int(trng.integers(1, max(2, int(mean_events * horizon / span) + 1)))
    prediction_type = (
        "classification"
        if functional in ("exists", "count_gt", "next_category")
        else "regression"
    )
    if task_kind == "autocomplete":
        functional = str(trng.choice(("attribute_category", "attribute_numeric", "event_mark")))
        prediction_type = (
            "regression" if functional != "attribute_category" else "classification"
        )
    # Historical labels are earlier applications of EXACTLY this task. Future
    # labels publish at window end; autocomplete cells publish at their row time.
    origin_times = np.arange(0.08 * span, 0.96 * span - horizon, max(horizon, 0.025 * span))
    jitter_bounds = cfg.label_publication_jitter_fraction_range
    if len(jitter_bounds) != 2 or not 0 <= jitter_bounds[0] <= jitter_bounds[1] < 0.5:
        raise ValueError("publication jitter must preserve ordered clock intervals")
    # An independent clock varies historical publication times without resampling the stream.
    clock_rng = np.random.default_rng([seed, slot, 84012]) if jitter_bounds[1] else None
    clock_jitter = float(clock_rng.uniform(*jitter_bounds)) if clock_rng is not None else 0.0
    history_times, history_values = [], []
    for stream in streams:
        if task_kind == "future_window":
            starts = origin_times
            if clock_jitter:
                starts = np.clip(
                    origin_times
                    + clock_rng.uniform(-clock_jitter, clock_jitter, len(origin_times))
                    * max(horizon, 0.025 * span),
                    0.0,
                    span - horizon,
                )
            values = np.asarray(
                [_window(stream, t, horizon, functional, threshold) for t in starts]
            )
            times = starts + horizon
        else:
            times = stream["times"]
            values = (
                stream["categories"]
                if prediction_type == "classification"
                else stream["values"]
            )
        history_times.append(times)
        history_values.append(values)
    npeer = int(trng.integers(cfg.peer_count_range[0], cfg.peer_count_range[1] + 1))
    peer_cap = int(np.clip(round(_log(trng, (1, cfg.peer_event_cap))), 1, cfg.peer_event_cap))
    own_cap = int(np.clip(round(_log(trng, (1, cfg.own_event_cap))), 1, cfg.own_event_cap))
    layout = _layout(seed, slot, relations)
    peers = []
    for e in range(entities):
        pool = np.flatnonzero((parent == parent[e]) & (np.arange(entities) != e))
        peers.append(trng.choice(pool, min(npeer, len(pool)), replace=False).tolist())
    early = trng.uniform(0.30 * span, 0.40 * span)
    late = trng.uniform(0.70 * span, span - horizon)
    cutoffs = np.linspace(early, late, cfg.cutoffs_per_entity)
    cases, facts, labels, shapes, case_ids, tiers, target_covariates = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for tier, base_cutoff in enumerate(cutoffs):
        for e in range(entities):
            cutoff = float(base_cutoff)
            own = _history(
                streams[e],
                cutoff,
                own_cap,
                history_times[e],
                history_values[e],
                cfg.label_history_cap,
                e,
            )
            histories = [
                _history(
                    streams[p],
                    cutoff,
                    peer_cap,
                    history_times[p],
                    history_values[p],
                    cfg.label_history_cap,
                    p,
                )
                for p in peers[e]
            ]
            peer = (
                _merge(histories, peer_cap, cfg.label_history_cap)
                if histories
                else {k: np.asarray([], dtype=int if k == "categories" else float) for k in own}
            )
            if task_kind == "future_window":
                label = _window(streams[e], cutoff, horizon, functional, threshold)
                row_attribute = attributes[e]
            else:
                # A sampled focal observation has its own visible covariate and
                # hidden intrinsic mark, with no future time shift.
                row_attribute = attributes[e] + trng.normal(0, 0.25)
                dynamic = (
                    level[e]
                    + drift[e] * (cutoff / span - 0.5)
                    + hub_drift[parent[e]] * (cutoff / span - 0.5) * hub_share
                )
                value = (
                    static[e]
                    + weights[0] * (row_attribute - attributes[e])
                    + weights[4] * dynamic
                    + trng.normal(0, mark_noise * mark_scales[e])
                )
                label = (
                    float(value > 0) if prediction_type == "classification" else float(value)
                )
            f = observable_facts(own, peer, cutoff, horizon)
            v = own["values"]
            facts.append(
                np.r_[
                    f,
                    row_attribute,
                    parent_values[parent[e]],
                    ancestor_values[ancestor_ids[parent[e]]],
                    _mean(attributes[peers[e]]),
                    float(v.max()) if len(v) else 0.0,
                    float(v.min()) if len(v) else 0.0,
                    float(v[-1]) if len(v) else 0.0,
                ]
            )
            cases.append((e, cutoff, own, peer))
            labels.append(label)
            case_ids.append(e)
            tiers.append(tier)
            target_covariates.append(row_attribute)
            shapes.append(
                dict(
                    own_events=len(own["times"]),
                    peer_events=len(peer["times"]),
                    row_width=width + 2,
                    peer_share=len(peer["times"])
                    / max(1, len(own["times"]) + len(peer["times"])),
                    gaps_horizon=(np.diff(own["times"]) / horizon).tolist(),
                    tables=layout["tables"],
                )
            )
    return dict(
        cases=cases,
        facts=np.asarray(facts),
        labels=np.asarray(labels),
        entities=np.asarray(case_ids),
        tiers=np.asarray(tiers),
        cutoffs=np.asarray([c[1] for c in cases]),
        streams=streams,
        streams_by_relation=streams_by_relation,
        peers=peers,
        parents=parent,
        parent_values=parent_values,
        ancestor_ids=ancestor_ids,
        ancestor_values=ancestor_values,
        attributes=attributes,
        target_covariates=np.asarray(target_covariates),
        horizon=horizon,
        span=span,
        seconds_per_unit=seconds,
        task_kind=task_kind,
        functional=functional,
        prediction_type=prediction_type,
        threshold=threshold,
        target_relation=target_relation,
        shape=shapes,
        width=width,
        seed=seed,
        slot=slot,
        parent_shortcut=parent_shortcut,
        history_times=history_times,
        history_values=history_values,
        config=dict(
            asdict(cfg),
            drift_modes=("smooth", "switch", "seasonal"),
            drift_amp_range=(0.1, 0.8),
            drift_switch_range=(0.3, 0.8),
            drift_period_range=(0.1, 0.8),
            drift_coef_share=0.5,
            peer_group_size=None,
        ),
        process=dict(
            coupling=float(coupling),
            hub_share=float(hub_share),
            weights=weights.tolist(),
            label_publication_jitter_fraction=clock_jitter,
            branching=float(branching),
            mean_events=mean_events,
            period=period,
            mark_scale_spread=float(np.std(np.log(mark_scales))),
            parent_shortcut=parent_shortcut,
            drift=None,
        ),
    )

