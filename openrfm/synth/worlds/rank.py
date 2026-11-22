"""Generate classification tasks from selected facts about entities and their peers."""

from __future__ import annotations

from itertools import combinations

import numpy as np

QS = (1, 2, 4, 8, 16)


def transform_facts(x, transforms):
    variants = (x, np.tanh(x), np.sign(x) * np.sqrt(np.abs(x)))
    return np.stack([variants[int(t)][:, j] for j, t in enumerate(transforms)], 1)


def draw_task(rng, facts, q):
    if q not in QS[2:]:
        raise ValueError("q must be 4, 8 or 16")
    selected = rng.choice(16, q, replace=False)
    transforms = rng.integers(0, 3, 16)
    x = transform_facts(facts, transforms)
    # Analytic symmetric transform scales, independent of generated cases.
    scales = np.array([1.0, 0.628, 0.893])[transforms]
    x = x / scales
    weights = rng.choice([-1.0, 1.0], q) * rng.uniform(0.95, 1.05, q)
    score = x[:, selected] @ weights / np.linalg.norm(weights)
    pairs = list(combinations(selected.tolist(), 2))
    pairs = [pairs[i] for i in rng.choice(len(pairs), min(3, len(pairs)), replace=False)]
    pair_weights = rng.uniform(-0.15, 0.15, len(pairs))
    for (a, b), w in zip(pairs, pair_weights, strict=False):
        score += w * x[:, a] * x[:, b]
    probability = 1 / (1 + np.exp(-(1.3 if q >= 8 else 2.1) * score))
    y = (rng.random(len(score)) < probability).astype(np.int64)
    return (
        y,
        score,
        dict(
            q=q,
            selected=selected.tolist(),
            transforms=transforms.tolist(),
            weights=weights.tolist(),
            pairs=pairs,
            pair_weights=pair_weights.tolist(),
        ),
    )


def simulate_world(seed, slot, attempt=0, entities=1800):
    """Independent entity processes with dedicated shared-hub peers; latent facts never enter features or metadata."""
    from openrfm.synth.worlds.history_process import _history_static

    rng = np.random.default_rng([seed, slot, attempt, 81008])
    facts = rng.normal(size=(entities, 16))
    labels, score, task = draw_task(rng, facts, QS[slot % 5])
    report = admission_report(facts, labels, np.arange(entities), task)
    if not report["admitted"]:
        return None, report
    E, G = 2 * entities, 80
    cutoffs = rng.integers(45, 75, entities)
    p = _history_static(rng, "drift", "classification")
    p.update(
        E=E,
        G=G,
        ng=entities,
        single_hub=True,
        grp=np.tile(np.arange(entities), 2),
        member=np.tile(np.eye(entities, dtype=np.int8), (2, 1)),
        pairs=[(e, int(cutoffs[e])) for e in rng.permutation(entities)],
        start=np.zeros(E, dtype=int),
        parent_fk=rng.integers(0, p["parents"], E),
        category=rng.integers(0, 8, E),
        static_x=rng.normal(size=(E, 2)),
        mask=np.zeros((E, G, 2), bool),
        smask=np.zeros((E, 2), bool),
    )
    nz = {"x": rng.normal(size=(E, G, 2))}
    # Past outcomes are independent of the composite target.
    t = np.arange(G)[None, :]
    own_prob = 1 / (
        1 + np.exp(-(facts[:, 6, None] + facts[:, 7, None] * (t >= cutoffs[:, None] - 4)))
    )
    peer_prob = 1 / (1 + np.exp(-facts[:, 8, None]))
    history = np.concatenate(
        [(rng.random((entities, G)) < own_prob), (rng.random((entities, G)) < peer_prob)]
    ).astype(float)
    ent, days, values, cats, misc, ptr = [], [], [], [], [], [0]
    for e in range(E):
        f = facts[e % entities]
        peer = e >= entities
        cutoff = int(cutoffs[e % entities])
        age = int(100 + 210 / (1 + np.exp(-f[9])))
        begin = 7 * cutoff - age
        p["start"][e] = max(0, begin // 7)
        ds, vs, cs, ms = [], [], [], []
        shape = np.exp(np.clip(f[3], -2, 2))
        hazard, next_arrival = 0.0, rng.gamma(shape, 1 / shape)
        for day in range(begin, 7 * (cutoff + 5)):
            lag = 7 * cutoff - day
            rate_log = f[13] if peer else (f[0] if lag <= 28 else f[2])
            rate_log += (f[1] if lag <= 7 else 0) * 0.7
            rate_log += 0.45 * f[14] * np.sin(day * 2 * np.pi / 14)
            rate = 0.38 * np.exp(np.clip(0.5 * rate_log, -2, 2))
            # Recency is independent thinning of the terminal observation interval.
            if 0 <= lag < int(1 + 6 / (1 + np.exp(-f[15]))):
                rate = 0
            hazard += rate
            k = 0
            while hazard >= next_arrival:
                hazard -= next_arrival
                next_arrival = rng.gamma(shape, 1 / shape)
                k += 1
            for _ in range(k):
                pref = 1 / (1 + np.exp(-(f[4] + (f[5] if lag <= 14 else 0))))
                cat = int(rng.random() < pref)
                if rng.random() < 0.3 / (1 + np.exp(-f[12])):
                    cat = int(rng.integers(2, 8))
                ds.append(day)
                vs.append(
                    (f[8] if peer else f[10] + (f[11] if lag <= 14 else 0))
                    + rng.normal(scale=0.3)
                )
                cs.append(cat)
                ms.append(rng.normal())
        ent.extend([e] * len(ds))
        days.extend(ds)
        values.extend(vs)
        cats.extend(cs)
        misc.extend(ms)
        ptr.append(len(ent))
    ev = dict(
        ent=np.asarray(ent),
        day=np.asarray(days),
        val=np.asarray(values),
        f1=np.asarray(cats),
        f2=np.asarray(misc),
        category=np.asarray(cats),
        ptr=np.asarray(ptr),
    )
    sim = dict(ev=ev, label=history, score=np.zeros((E, G)))
    ce = np.asarray([e for e, _ in p["pairs"]])
    return (p, nz, sim, labels[ce], score[ce], facts[ce], task), report


def write_rank_world(raw_root, db, slot, seed=1008, max_attempts=200):
    import json
    from pathlib import Path

    import pyarrow as pa

    from openrfm.synth.worlds.history_world import _history_finish, _history_tables
    from openrfm.synth.worlds.panel import WORLD_FILE

    root = Path(raw_root) / db
    if root.exists():
        raise FileExistsError(root)
    rejected = []
    for attempt in range(max_attempts):
        result, report = simulate_world(seed, slot, attempt)
        if result is None:
            rejected.append(report)
            continue
        p, nz, sim, labels, score, facts, task = result
        made = _history_tables(root, p, sim, nz)
        # Keep independent historical outcomes; replace only the case target.
        import pyarrow.parquet as pq

        path = root / "db" / "a5_task_0.parquet"
        table = pq.read_table(path)
        table = table.set_column(
            table.schema.get_field_index("a5_y_0"), "a5_y_0", pa.array(labels)
        )
        pq.write_table(table, path)
        made.update(y=labels, score=score)
        _history_finish(
            root,
            db,
            slot,
            seed,
            p,
            "drift",
            "classification",
            made,
            report["all_facts"],
            attempt,
            [],
        )
        record = json.loads((root / WORLD_FILE).read_text())
        record.update(
            family="rank",
            rank_q=task["q"],
            rank=dict(
                q=task["q"],
                admission=report,
                attempted=attempt + 1,
                rejected=rejected,
                recipe="rank_worlds_v1",
            ),
        )
        (root / WORLD_FILE).write_text(json.dumps(record, indent=1) + "\n")
        np.savez_compressed(
            root / "rank_private.npz",
            facts=facts,
            labels=labels,
            score=score,
            entities=np.asarray([e for e, _ in p["pairs"]]),
            task=json.dumps(task),
        )
        return record
    raise RuntimeError(f"{db}: failed admission after {max_attempts} attempts; last={report}")


def admission_report(facts, labels, entities, task):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    x = transform_facts(facts, task["transforms"])
    # Fixed entity split; repeated cutoffs from one entity never cross splits.
    train = np.asarray(entities) % 3 != 0
    test = ~train
    if min(len(np.unique(labels[train])), len(np.unique(labels[test]))) < 2:
        return dict(admitted=False, reason="single_class")

    def score(cols, interactions=False, exact=False):
        z = x[:, cols]
        if interactions and len(cols) > 1:
            z = np.column_stack(
                [z] + [x[:, a] * x[:, b] for a, b in task["pairs"] if a in cols and b in cols]
            )
        if exact or len(cols) > 2:
            fitted = LogisticRegression(C=1.0, max_iter=200).fit(z[train], labels[train])
            prediction = fitted.predict_proba(z[test])[:, 1]
        else:
            # Least-squares discriminant fitted only on the training entities.
            mean = z[train].mean(0)
            a = z[train] - mean
            coef = np.linalg.solve(
                a.T @ a + np.eye(a.shape[1]), a.T @ (labels[train] - labels[train].mean())
            )
            prediction = (z[test] - mean) @ coef
        auc = float(roc_auc_score(labels[test], prediction))
        return max(auc, 1 - auc)

    singles = [score([j]) for j in range(16)]
    pairs = [score([a, b], True) for a, b in combinations(range(16), 2)]
    all_facts = score(list(range(16)), True)
    q = task["q"]
    best_single, best_two = max(singles), max(pairs)
    admitted = (
        (best_single < 0.62 and best_two < 0.67 and all_facts > 0.76)
        if q >= 8
        else all_facts > 0.76
    )
    if admitted and q >= 8:
        best_two = max(
            best_two,
            max(score([a, b], True, exact=True) for a, b in combinations(range(16), 2)),
        )
        admitted = best_single < 0.62 and best_two < 0.67 and all_facts > 0.76
    return dict(
        admitted=admitted,
        best_single=best_single,
        best_two=best_two,
        all_facts=all_facts,
        train_cases=int(train.sum()),
        test_cases=int(test.sum()),
    )
