"""Temporal event streams and the label signals drawn from event, entity and peer histories."""

import numpy as np


def generate_stream(seed, config, *, prior=None):
    """Generate stateful event marks over entity-specific lifetimes spanning the task cutoffs."""
    rng = np.random.default_rng(seed)
    c = config["generator"]
    E = c["entities"]
    median = np.exp(rng.uniform(*np.log(c["median_rows_per_key"])))
    sigma = rng.uniform(*c["rows_per_key_sigma"])
    counts = np.clip(
        np.rint(median * np.exp(sigma * rng.normal(size=E))), *c["rows_per_key"]
    ).astype(np.int64)
    entity_hub = rng.permutation(E)
    days = int(rng.integers(c["history_days"][0], c["history_days"][1] + 1))
    width = int(rng.integers(c["value_columns"][0], c["value_columns"][1] + 1))
    entity = np.repeat(np.arange(E), counts)
    popularity = 1 / np.arange(1, E + 1) ** rng.uniform(*c["hub_zipf"])
    event_hub = rng.choice(E, len(entity), p=popularity[rng.permutation(E)] / popularity.sum())
    start = rng.random(E) * 0.9
    stop = start + (1 - start) * rng.beta(1, 0.5, E)
    shape = rng.uniform(0.3, 3.0, E)
    fraction = start[entity] + (stop - start)[entity] * rng.random(len(entity)) ** shape[entity]
    time = np.round((-days * (1 - fraction) - 1 / 24) * 24) / 24
    order = np.lexsort((time, entity))
    time = time[order]
    # Marks are stateful: an entity effect, a slow drift within the entity and AR(1) noise.
    rho = rng.uniform(0, 0.9)
    effect, drift = rng.uniform(0.2, 1.0) * rng.normal(size=E), rng.normal(size=E) * 0.5
    noise = rng.normal(size=len(entity))
    begin = np.r_[0, np.cumsum(counts)[:-1]]
    for a, n in zip(begin, counts, strict=False):
        for i in range(a + 1, a + n):
            noise[i] = rho * noise[i - 1] + np.sqrt(1 - rho**2) * noise[i]
    position = (np.arange(len(entity)) - begin[entity]) / np.maximum(counts[entity] - 1, 1)
    values = rng.normal(size=(len(entity), width))
    values[:, 0] = effect[entity] + drift[entity] * position + noise
    values[:, 1:] = 0.5 * values[:, :1] + values[:, 1:]
    values *= np.exp(rng.uniform(-2, 3, width))
    horizon = int(rng.choice(c["horizon_days"]))
    history = int(rng.integers(c["label_history_rows"][0], c["label_history_rows"][1] + 1))
    task_entity = np.repeat(np.arange(E), history)
    # The grid sits inside the stream and leaves every label's horizon before the world ends.
    earliest = min(-days * rng.uniform(*c["cutoff_span_fraction"]), -horizon * max(2, history))
    cutoffs = np.tile(np.linspace(earliest, -horizon, history), E)
    if prior is not None:
        clock_rng = np.random.default_rng([seed, 84012])
        if clock_rng.random() < prior["irregular_clock_share"]:
            fraction = clock_rng.uniform(*prior["label_publication_jitter_fraction_range"])
            step = (-horizon - earliest) / max(1, history - 1)
            jitter = clock_rng.uniform(-fraction, fraction, (E, history)) * step
            jitter[:, [0, -1]] = 0
            cutoffs = (cutoffs.reshape(E, history) + jitter).ravel()
    future_label = rng.random() < c["future_label_share"]
    span = horizon * int(rng.choice((1, 4, 16)))
    score = np.zeros(len(task_entity))
    for i, (e, cutoff) in enumerate(zip(task_entity, cutoffs, strict=False)):
        selected = entity == e
        if future_label:
            score[i] = np.log1p((selected & (time > cutoff) & (time <= cutoff + horizon)).sum())
            continue
        past = selected & (time < cutoff)
        window = past & (time >= cutoff - span)
        recent = past & (time >= cutoff - horizon)
        v = values[window, 0] / values[:, 0].std()
        score[i] = (np.tanh(v.mean()) if len(v) else 0) + 0.5 * (
            recent.sum() / max(1, window.sum())
        )
        score[i] += 0.25 * np.tanh(values[recent, 0].sum() / values[:, 0].std())
        score[i] -= 0.25 * np.log1p(cutoff - time[past].max() if past.any() else days)
    score += rng.normal(scale=0.15, size=len(score))
    threshold = np.quantile(score, rng.uniform(0.2, 0.8))
    return dict(
        entity=entity,
        time=time,
        values=values,
        entity_hub=entity_hub,
        event_hub=event_hub,
        task_entity=task_entity,
        cutoffs=cutoffs,
        score=score,
        horizon=horizon,
        threshold=threshold,
        counts=counts,
    )


def history_signal(w, rng, cuts):
    """One lawful functional of strictly earlier rows per task row: counts, old marks, recency or hub peers."""
    time, entity, hub, values = w["time"], w["entity"], w["event_hub"], w["values"][:, 0]
    kind = str(
        rng.choice(("lifetime_count", "window_count", "old_mean", "recency", "hub_count"))
    )
    span = w["horizon"] * int(rng.choice((1, 4, 16)))
    out = np.zeros(len(cuts))
    for i, (e, cutoff) in enumerate(zip(w["task_entity"], cuts, strict=False)):
        past = (hub == w["entity_hub"][e] if kind == "hub_count" else entity == e) & (
            time < cutoff
        )
        rows = np.flatnonzero(past)
        if not len(rows):
            continue
        if kind == "window_count":
            out[i] = np.log1p((time[rows] >= cutoff - span).sum())
        elif kind == "old_mean":
            out[i] = values[rows[np.argsort(time[rows])[: max(1, len(rows) // 2)]]].mean()
        elif kind == "recency":
            out[i] = -np.log1p(cutoff - time[rows].max())
        else:
            out[i] = np.log1p(len(rows))
    return out, kind


def near_signal(w, rng):
    """Focal task-row cells and the entity's and hub's own cells; memory reads neither, a context holds all three."""
    n, E = len(w["task_entity"]), len(w["entity_hub"])
    w["task_cells"] = rng.normal(size=(n, 3))
    w["entity_cells"] = rng.normal(size=E)
    w["hub_cells"] = rng.normal(size=w["entity_hub"].max() + 1)
    weights = rng.normal(size=5)
    out = (
        w["task_cells"] @ weights[:3]
        + weights[3] * w["entity_cells"][w["task_entity"]]
        + weights[4] * w["hub_cells"][w["entity_hub"][w["task_entity"]]]
    )
    return np.tanh(out / max(np.abs(weights).sum() / 2, 1e-6))


def recent_over_baseline(w, cuts):
    """Recent rate against the entity's own long-run rate."""
    out = np.zeros(len(cuts))
    for i, (e, cutoff) in enumerate(zip(w["task_entity"], cuts, strict=False)):
        rows = (w["entity"] == e) & (w["time"] < cutoff)
        if rows.any():
            first = w["time"][rows].min()
            recent = (w["time"][rows] >= cutoff - w["horizon"]).sum() / w["horizon"]
            out[i] = np.log1p(recent) - np.log1p(rows.sum() / max(cutoff - first, 1.0))
    return out


def entity_slices(w):
    """Row range of each entity; events are sorted by (entity, time)."""
    counts = np.bincount(w["entity"], minlength=len(w["entity_hub"]))
    stops = np.cumsum(counts)
    return stops - counts, stops


def exact_targets(w, column, k, cuts):
    """Per task row from strictly earlier rows: k-th newest value, max, min, days since last event."""
    begin, stop = entity_slices(w)
    time, values = w["time"], w["values"][:, column]
    out = dict(
        kth=np.full(len(cuts), values.mean()),
        maximum=np.full(len(cuts), values.mean()),
        minimum=np.full(len(cuts), values.mean()),
        gap=np.full(len(cuts), float(time.max() - time.min())),
    )
    for i, (e, cutoff) in enumerate(zip(w["task_entity"], cuts, strict=False)):
        a = begin[e]
        b = a + np.searchsorted(time[a : stop[e]], cutoff, side="left")
        if b > a:
            out["maximum"][i], out["minimum"][i] = values[a:b].max(), values[a:b].min()
            out["gap"][i] = np.log1p(cutoff - time[b - 1])
            if b - a >= k:
                out["kth"][i] = values[b - k]
    return out


def exact_signal(w, rng, c):
    """Label from one exact read of one drawn column with fresh independent marks, among distractors."""
    column = int(rng.integers(w["values"].shape[1]))
    scale = np.exp(rng.uniform(-2, 3))
    w["values"][:, column] = rng.normal(size=len(w["entity"])) * scale
    k = int(np.exp(rng.uniform(0, np.log(c["exact_k_max"]))))
    form = str(rng.choice(("kth", "maximum", "minimum", "gap")))
    w["truth"] = dict(column=column, k=k)
    return exact_targets(w, column, k, w["cutoffs"])[form], f"{form}:{k}:x{column}"


def hawkes_events(rng, c, E, days, median, peers):
    """Marked branching process: an event calls later events on its own entity or on a peer sharing a hub."""
    ratio = rng.uniform(*c["hawkes_branching"])
    delay = rng.uniform(*c["hawkes_delay_days"])
    cross = rng.uniform(*c["hawkes_cross_share"])
    n = rng.poisson(median * (1 - ratio) * E)
    entity, time = rng.integers(E, size=n), -days * rng.random(n)
    mark = rng.normal(size=n)
    entities, times, marks = [entity], [time], [mark]
    while len(entity) and sum(map(len, times)) < c["hawkes_max_rows"]:
        children = rng.poisson(ratio * np.exp(0.5 * mark - 0.125))
        parent = np.repeat(np.arange(len(entity)), children)
        time = time[parent] + rng.exponential(delay, len(parent))
        keep = time < 0
        parent, time = parent[keep], time[keep]
        member = entity[parent]
        moved = rng.random(len(member)) < cross
        member[moved] = np.array([rng.choice(peers[i]) for i in member[moved]], dtype=np.int64)
        entity, mark = member, rng.normal(size=len(member))
        entities.append(entity)
        times.append(time)
        marks.append(mark)
    return np.concatenate(entities), np.concatenate(times), np.concatenate(marks)


def hawkes_signal(w, rng, c):
    """Peer events raise an entity's rate; the label is its next-horizon count."""
    E = len(w["entity_hub"])
    groups = max(2, int(E / rng.uniform(*c["hawkes_group_size"])))
    hub = rng.integers(groups, size=E)
    members = {g: np.flatnonzero(hub == g) for g in range(groups)}
    peers = [members[hub[e]] for e in range(E)]
    days = float(-w["time"].min()) + 1
    median = np.median(w["counts"])
    entity, time, mark = hawkes_events(rng, c, E, days, median, peers)
    order = np.lexsort((time, entity))
    entity, time, mark = entity[order], np.round(time[order] * 24) / 24 - 1 / 24, mark[order]
    width = w["values"].shape[1]
    values = rng.normal(size=(len(entity), width))
    values[:, 0] = mark
    values *= np.exp(rng.uniform(-2, 3, width))
    w.update(
        entity=entity,
        time=time,
        values=values,
        entity_hub=hub,
        event_hub=hub[entity],
        counts=np.bincount(entity, minlength=E),
    )
    w["hub_cells"] = rng.normal(size=groups)
    begin, stop = entity_slices(w)
    out = np.zeros(len(w["task_entity"]))
    for i, (e, cutoff) in enumerate(zip(w["task_entity"], w["cutoffs"], strict=False)):
        ahead = time[begin[e] : stop[e]]
        out[i] = np.log1p(
            np.searchsorted(ahead, cutoff + w["horizon"], side="right")
            - np.searchsorted(ahead, cutoff, side="right")
        )
    return out, f"hawkes:{groups}"


def regime_signal(w, rng, c):
    """Marks and rate change at a world-specific time; the label contrasts the newest rows with older ones."""
    column = int(rng.integers(w["values"].shape[1]))
    days = float(-w["time"].min())
    change = (
        -days * rng.uniform(*c["regime_change_fraction"])
        + rng.normal(size=len(w["entity_hub"])) * days * 0.05
    )
    late = w["time"] > change[w["entity"]]
    scale = w["values"][:, column].std()
    w["values"][:, column] += (
        late * rng.normal(size=len(w["entity_hub"]))[w["entity"]] * scale * rng.uniform(1, 3)
    )
    keep = late | (rng.random(len(late)) < rng.uniform(0.2, 1.0))
    w.update(
        entity=w["entity"][keep],
        time=w["time"][keep],
        values=w["values"][keep],
        event_hub=w["event_hub"][keep],
        counts=np.bincount(w["entity"][keep], minlength=len(w["entity_hub"])),
    )
    k = int(np.exp(rng.uniform(np.log(2), np.log(c["exact_k_max"]))))
    begin, stop = entity_slices(w)
    out = np.zeros(len(w["task_entity"]))
    values = w["values"][:, column]
    for i, (e, cutoff) in enumerate(zip(w["task_entity"], w["cutoffs"], strict=False)):
        a = begin[e]
        b = a + np.searchsorted(w["time"][a : stop[e]], cutoff, side="left")
        if b - a > k:
            out[i] = values[b - k : b].mean() - values[a : b - k].mean()
    return out, f"regime:{k}:x{column}"


def censored_signal(w, rng, c):
    """Time to next event, censored at the horizon, for entities that leave at a hidden mark-dependent time."""
    E = len(w["entity_hub"])
    days = float(-w["time"].min())
    begin, stop = entity_slices(w)
    state = np.array(
        [
            w["values"][a:b, 0].mean() if b > a else 0.0
            for a, b in zip(begin, stop, strict=False)
        ]
    )
    state = (state - state.mean()) / max(state.std(), 1e-9)
    death = -days * np.minimum(1, rng.beta(1, 2, E) * np.exp(-rng.uniform(0, 1) * state))
    entry = -days * rng.uniform(*c["censor_entry_fraction"], E)
    keep = (w["time"] <= death[w["entity"]]) & (w["time"] >= entry[w["entity"]])
    w.update(
        entity=w["entity"][keep],
        time=w["time"][keep],
        values=w["values"][keep],
        event_hub=w["event_hub"][keep],
        counts=np.bincount(w["entity"][keep], minlength=E),
    )
    begin, stop = entity_slices(w)
    span = w["horizon"] * int(rng.choice((4, 16)))
    risk, out = np.zeros(len(w["task_entity"]), dtype=bool), np.zeros(len(w["task_entity"]))
    for i, (e, cutoff) in enumerate(zip(w["task_entity"], w["cutoffs"], strict=False)):
        times = w["time"][begin[e] : stop[e]]
        j = np.searchsorted(times, cutoff, side="right")
        risk[i] = j > 0 and times[j - 1] >= cutoff - span
        out[i] = min(times[j] - cutoff, w["horizon"]) if j < len(times) else w["horizon"]
    if risk.sum() >= c["censor_min_task_rows"]:
        w.update(
            task_entity=w["task_entity"][risk],
            cutoffs=w["cutoffs"][risk],
            task_cells=w["task_cells"][risk],
        )
        out = out[risk]
    return out, f"censored:{span}"


def tail_signal(w, rng, c):
    """y_t = exp(s(a_t + b_t r_t)) for one history read r with drifting level and slope; heavy-tailed marks."""
    s = rng.uniform(*c["tail_sigma"])
    marks = w["values"][:, 0]
    w["values"][:, 0] = np.exp(s * (marks - marks.mean()) / max(marks.std(), 1e-9))
    read, kind = history_signal(w, rng, w["cutoffs"])
    read = (read - read.mean()) / max(read.std(), 1e-9)
    t = (w["cutoffs"] - w["cutoffs"].min()) / max(np.ptp(w["cutoffs"]), 1e-9)
    level, slope = rng.normal(size=(2, 2))
    y = np.exp(
        s * ((1 - t) * level[0] + t * level[1] + ((1 - t) * slope[0] + t * slope[1]) * read)
    )
    low, high = rng.uniform(0, c.get("tail_atom_share", 0.0), 2)
    return np.clip(y, np.quantile(y, low), np.quantile(y, 1 - high)), f"tail:{kind}"


EXACT_FORMS = dict(
    exact_signal=exact_signal,
    hawkes_signal=hawkes_signal,
    regime_signal=regime_signal,
    censored_signal=censored_signal,
    tail_signal=tail_signal,
)


def collaboration_stream(seed, config, signal, *, prior=None):
    """Generate an event stream and apply the selected label mechanism."""
    c = dict(config["generator"])
    rng = np.random.default_rng([seed, 1101])
    long_world = signal != "near_signal" and rng.random() < c["long_share"]
    if long_world:
        c.update(
            median_rows_per_key=c["long_median_rows_per_key"],
            rows_per_key=c["long_rows_per_key"],
            hub_zipf=c["long_hub_zipf"],
        )
    if signal in ("exact_signal", "regime_signal"):
        c["value_columns"] = c["distractor_value_columns"]
    w = generate_stream(seed, dict(config, generator=c), prior=prior)

    def z(x):
        return (x - x.mean()) / max(x.std(), 1e-9)

    near = z(near_signal(w, rng))
    if signal in EXACT_FORMS:
        history, form = EXACT_FORMS[signal](w, rng, c)
        score = z(history)
    elif signal == "silent_signal":
        # Silent labels are independent of the event history.
        score, form = rng.normal(size=len(near)), "silent"
    elif signal == "memory_signal":
        history, form = history_signal(w, rng, w["cutoffs"])
        score = z(history)
    elif signal == "near_signal":
        score, form = near, "near"
    elif signal == "joint_signal":
        form = str(rng.choice(("additive", "interaction", "deviation")))
        if form == "deviation":
            score = z(recent_over_baseline(w, w["cutoffs"]))
        else:
            history, kind = history_signal(w, rng, w["cutoffs"])
            score = z(history) + near if form == "additive" else z(z(history) * near)
            form = f"{form}:{kind}"
    else:
        raise ValueError(f"unknown memory signal {signal!r}")
    score = z(score) + rng.normal(scale=rng.uniform(*c["collab_noise"]), size=len(score))
    w.update(
        score=score,
        threshold=np.quantile(score, rng.uniform(0.2, 0.8)),
        signal=signal,
        signal_form=form,
        long_world=bool(long_world),
    )
    return w


def generate(seed, config, *, prior=None, signal=None):
    if signal is not None:
        return collaboration_stream(seed, config, signal, prior=prior)
    return generate_stream(seed, config, prior=prior)

