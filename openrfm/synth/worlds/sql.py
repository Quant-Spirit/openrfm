"""Generate forecast tasks over marked event streams and their future windows."""

from __future__ import annotations

import numpy as np

OPERATORS = (
    "event_recency",
    "count_h",
    "count_2h",
    "count_4h",
    "gap_mean",
    "gap_cv",
    "label_mean",
    "label_change",
    "peer_label_mean",
    "peer_count_h",
    "value_mean",
    "value_change",
    "category_share",
    "category_diversity",
    "seasonality",
    "activity_change",
)
NAMES = [
    "recency",
    "count_h",
    "count_2h",
    "count_4h",
    "gap_mean",
    "gap_cv",
    "trend",
    "label_mean",
    "label_last",
    "peer_label_rate",
    "entity_age",
    "category_entropy",
    "category_top_share",
    "mark_mean",
    "attribute",
]
GROUPS = (
    ["activity"] * 7
    + ["outcome"] * 2
    + ["attributes_peers", "activity"]
    + ["attributes_peers"] * 4
)
TEMPLATES = (
    "exists",
    "churn",
    "count_gt_k",
    "count",
    "sum",
    "mean",
    "distinct",
    "same_category",
    "next_bucket",
)
REGRESSION = {"count", "sum", "mean", "distinct"}


def simulate_stream(rng, end=160.0):
    rate = float(np.exp(rng.normal(-1.35, 1.0)))
    period, phase, amplitude = (
        rng.uniform(6.0, 45.0),
        rng.uniform(0, 2 * np.pi),
        rng.uniform(0, 0.9),
    )
    beta, branching = rng.uniform(0.3, 2.0), rng.uniform(0.05, 0.65)
    # Churn/reactivation and observation censoring are drawn on the global clock, before query cutoffs exist.
    active = rng.random() < 0.85
    state = []
    for _ in range(int(end)):
        if rng.random() < (0.018 if active else 0.07):
            active = not active
        state.append(active)
    observed_start, observed_end = rng.uniform(0, 12), end - rng.uniform(0, 12)
    immigrants = []
    for day in range(int(end)):
        intensity = (
            rate
            * (1 if state[day] else 0.025)
            * (1 + amplitude * np.sin(2 * np.pi * day / period + phase))
        )
        immigrants.extend(day + rng.random(rng.poisson(intensity)))
    events, frontier = list(immigrants), list(immigrants)
    while frontier:
        children = []
        for parent in frontier:
            ds = parent + rng.exponential(1 / beta, rng.poisson(branching))
            children.extend(ds[ds < end])
        events.extend(children)
        frontier = children
    times = np.sort(np.asarray(events, dtype=float))
    times = times[(times >= observed_start) & (times <= observed_end)]
    # Mark dynamics are only weakly coupled to arrival intensity; independent
    # outcome state prevents all temporal operators collapsing to label history.
    level, slope = rng.normal(), rng.normal(0, 0.7)
    change = rng.uniform(30, 130)
    values = level + slope * (times >= change) + rng.normal(0, 0.5, len(times))
    cat_p = rng.dirichlet(np.ones(4) * rng.uniform(0.3, 3.0))
    categories = rng.choice(4, len(times), p=cat_p)
    label_times = np.arange(5.0, end, 7.0)
    label_prob = 1 / (1 + np.exp(-(level + 0.25 * slope * (label_times >= change))))
    labels = (rng.random(len(label_times)) < label_prob).astype(float)
    return dict(
        times=times,
        values=values,
        categories=categories,
        label_times=label_times,
        labels=labels,
        rate=rate,
        period=period,
        alpha=branching * beta,
        beta=beta,
        observed_start=observed_start,
        observed_end=observed_end,
    )


def visible_history(stream, cutoff, cap=64):
    ix = np.flatnonzero(stream["times"] < cutoff)[-cap:]
    li = np.flatnonzero(stream["label_times"] < cutoff)[-16:]
    return {k: stream[k][ix] for k in ("times", "values", "categories")} | {
        "label_times": stream["label_times"][li],
        "labels": stream["labels"][li],
    }


def observable_facts(own, peer, cutoff, horizon):
    ages = cutoff - own["times"]
    pa = cutoff - peer["times"]
    gaps = np.diff(own["times"])
    labels = own["labels"]
    values = own["values"]
    mean = lambda a: float(np.mean(a)) if len(a) else 0.0
    recent = ages <= horizon
    old = (ages > horizon) & (ages <= 2 * horizon)
    cats = own["categories"]
    probabilities = np.bincount(cats, minlength=4) / max(1, len(cats))
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))
    return np.asarray(
        [
            np.log1p((ages[-1] if len(ages) else 32 * horizon) / horizon),
            *[np.log1p(np.sum(ages <= m * horizon)) for m in (1, 2, 4)],
            np.log1p(mean(gaps) / horizon),
            float(np.std(gaps) / (mean(gaps) + 0.1)) if len(gaps) else 0.0,
            mean(labels),
            mean(labels[-4:]) - mean(labels[:-4]),
            mean(peer["labels"]),
            np.log1p(np.sum(pa <= horizon)),
            mean(values),
            mean(values[recent]) - mean(values[old]),
            mean(cats == 0),
            entropy,
            mean(np.cos(2 * np.pi * ages / horizon)),
            np.log1p(recent.sum()) - np.log1p(old.sum()),
        ],
        dtype=np.float64,
    )


def facts(
    times, cutoff, h, labels=(), cats=(), values=(), peer=(), age=np.nan, attribute=np.nan
):
    t = np.asarray(times)
    gaps = np.diff(t)
    a = cutoff - t
    x = np.full(len(NAMES), np.nan)
    if h and h > 0 and cutoff > -(2**31):
        x[1:4] = [np.log1p((a <= m * h).sum()) for m in (1, 2, 4)]
        x[6] = x[1] - np.log1p(((a > h) & (a <= 2 * h)).sum())
        if len(t):
            x[0] = np.log1p(a[-1] / h)
        if len(gaps):
            x[4:6] = [np.log1p(gaps.mean() / h), gaps.std() / max(1e-9, gaps.mean())]
    if len(labels):
        x[7:9] = [np.mean(labels), labels[-1]]
    if len(peer):
        x[9] = np.mean(peer)
    x[10] = age
    if len(cats):
        _, n = np.unique(cats, return_counts=True)
        p = n / n.sum()
        x[11:13] = [-sum(p * np.log(p)), p.max()]
    if len(values):
        x[13] = np.mean(values)
    x[14] = attribute
    return x


def target(stream, cutoff, h, template, k):
    t = stream["times"]
    ix = (t > cutoff) & (t <= cutoff + h)
    n = int(ix.sum())
    if template == "exists":
        return int(n > 0)
    if template == "count_gt_k":
        return int(n > k)
    if template == "count":
        return n
    if template == "sum":
        return float(stream["values"][ix].sum())
    if template == "mean":
        return float(stream["values"][ix].mean()) if n else 0.0
    if template == "distinct":
        return len(np.unique(stream["categories"][ix]))
    if template == "same_category":
        past = np.flatnonzero(t < cutoff)
        return int(
            bool(len(past))
            and np.any(stream["categories"][ix] == stream["categories"][past[-1]])
        )
    if template == "next_bucket":
        future = t[t > cutoff]
        dt = future[0] - cutoff if len(future) else np.inf
        return 0 if dt <= h / 2 else 1 if dt <= h else 2
    raise ValueError(template)


def history(stream, cutoff, h, template, k):
    # A fixed global grid; publication is the END of each label window.
    starts = np.arange(15.0, cutoff - h, 7.0)
    starts = starts[(starts >= stream["observed_start"]) & (starts + h < cutoff)]
    return starts + h, np.array([target(stream, s, h, template, k) for s in starts])

