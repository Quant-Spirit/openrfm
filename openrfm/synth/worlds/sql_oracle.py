"""Draw event-stream forecast worlds admitted by a fitted oracle, and profile them."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import r2_score, roc_auc_score

from openrfm.synth.worlds.sql import (
    GROUPS, NAMES, OPERATORS, REGRESSION, TEMPLATES, facts, history, observable_facts, simulate_stream, target,
    visible_history,
)


def generate_world(seed, slot, entities=256, cutoffs_per_entity=2):
    rng = np.random.default_rng([seed, slot, 91008])
    # Separate RNG prevents cutoff policy from changing a single stream event.
    streams = [simulate_stream(rng) for _ in range(2 * entities)]
    horizon = float(rng.choice([3.0, 5.0, 7.0, 14.0]))
    group = (
        "broad_relational"
        if slot % 4 < 2
        else "useful_history"
        if slot % 4 == 2
        else "operator_sensitive"
    )
    operator = OPERATORS[(slot // 4) % len(OPERATORS)]
    residual = bool((slot // (4 * len(OPERATORS))) % 2)
    parent_values = rng.normal(size=16)
    parent_ids = rng.integers(0, 16, entities)
    # Cases are entity-disjoint for fitting/audit even with multiple cutoffs.
    cases, facts, future, valid, latent = [], [], [], [], []
    for entity in range(entities):
        own, peer = streams[entity], streams[entity + entities]
        for cutoff in rng.uniform(55.0, 120.0, cutoffs_per_entity):
            o, p = visible_history(own, cutoff), visible_history(peer, cutoff)
            fact = observable_facts(o, p, cutoff, horizon)
            cases.append((entity, cutoff, o, p))
            facts.append(fact)
            future.append(
                int(np.sum((own["times"] > cutoff) & (own["times"] <= cutoff + horizon)))
            )
            valid.append(cutoff + horizon <= own["observed_end"])
            latent.append(parent_values[parent_ids[entity]])
    facts = np.asarray(facts)
    ids = np.asarray([c[0] for c in cases])
    fit = ids % 3 != 0
    # Task calibration is world-fixed, estimated only on the fitting entities.
    col = facts[:, OPERATORS.index(operator)]
    if group == "broad_relational":
        col = np.asarray(latent)
    elif group == "useful_history":
        col = facts[:, OPERATORS.index("label_mean")]
    center, scale = np.mean(col[fit]), max(0.1, np.std(col[fit]))
    score = (col - center) / scale
    if group == "operator_sensitive" and residual:
        score += 0.4 * (facts[:, 6] - 0.5) / 0.25
    sign = float(rng.choice([-1, 1]))
    score *= sign
    labels = (rng.random(len(score)) < 1 / (1 + np.exp(-3 * score))).astype(np.int64)
    oracle_facts = np.column_stack([facts, latent])
    report = admission_report(oracle_facts, labels, score, ids)
    future = np.asarray(future)
    valid = np.asarray(valid)
    forecast_score = fitted_oracle(facts, (future > 0).astype(int), fit & valid, ~fit & valid)
    return dict(
        cases=cases,
        streams=streams,
        entities=ids,
        future=future,
        future_valid=valid,
        admission=report,
        forecast_auc=forecast_score,
        parents=parent_ids,
        parent_values=parent_values,
        seed=seed,
        slot=slot,
    )


def generate(seed, slot, entities=160, peer=False):
    data = generate_world(seed, slot, entities, cutoffs_per_entity=2)
    rng = np.random.default_rng([seed, slot, 10081008])
    h = float(rng.choice([3, 5, 7, 14]))
    template = TEMPLATES[slot % len(TEMPLATES)]
    k = int(rng.choice([1, 2, 3]))
    data["horizon"] = h
    newcases = []
    xs = []
    ys = []
    for entity, cutoff, _, _ in data["cases"]:
        own = data["streams"][entity]
        other = data["streams"][entity + entities]
        ownhist = visible_history(own, cutoff)
        peerhist = visible_history(other, cutoff)
        # Dedicated paired entities share one hub. Exclude focal entity for peer tasks.
        source = other if peer else own
        ownhist["label_times"], ownhist["labels"] = history(source, cutoff, h, template, k)
        peerhist["label_times"], peerhist["labels"] = history(
            own if peer else other, cutoff, h, template, k
        )
        y = target(source, cutoff, h, template, k)
        ys.append(y)
        newcases.append((entity, cutoff, ownhist, peerhist))
        xs.append(
            facts(
                ownhist["times"],
                cutoff,
                h,
                ownhist["labels"],
                ownhist["categories"],
                ownhist["values"],
                peerhist["labels"],
                np.log1p((cutoff - ownhist["times"][0]) / h)
                if len(ownhist["times"])
                else np.nan,
                data["parent_values"][data["parents"][entity]],
            )
        )
        assert cutoff + h < source["observed_end"]
        assert np.all(ownhist["label_times"] < cutoff) and np.all(ownhist["times"] < cutoff)
    data.update(
        cases=newcases,
        labels=np.array(ys),
        facts=np.array(xs),
        score=np.zeros(len(ys)),
        task_type="regression" if template in REGRESSION else "classification",
        group="future_peer" if peer else "future_own",
        operator=template,
        residual=False,
        rendering="timestamps",
        template=template,
        threshold=k,
        peer=peer,
    )
    return data



def fitting_orientation(y, score, fitting):
    from sklearn.metrics import roc_auc_score

    return 1.0 if roc_auc_score(y[fitting], score[fitting]) >= 0.5 else -1.0


def fitted_oracle(x, y, fitting, heldout):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    if min(len(np.unique(y[fitting])), len(np.unique(y[heldout]))) < 2:
        return None
    scale = StandardScaler().fit(x[fitting])
    model = LogisticRegression(C=1.0, max_iter=300).fit(scale.transform(x[fitting]), y[fitting])
    train_score = model.predict_proba(scale.transform(x[fitting]))[:, 1]
    orientation = 1 if roc_auc_score(y[fitting], train_score) >= 0.5 else -1
    return float(
        roc_auc_score(
            y[heldout], orientation * model.predict_proba(scale.transform(x[heldout]))[:, 1]
        )
    )


def admission_report(facts, labels, score, entities, threshold=0.65):
    from sklearn.metrics import roc_auc_score

    fitting, heldout = entities % 3 != 0, entities % 3 == 0
    if min(len(np.unique(labels[fitting])), len(np.unique(labels[heldout]))) < 2:
        return dict(latent=None, observable=None, tabicl=None, admitted=False)
    orientation = fitting_orientation(labels, score, fitting)
    latent = float(roc_auc_score(labels[heldout], orientation * score[heldout]))
    observable = fitted_oracle(facts, labels, fitting, heldout)
    # Missing level 3 is pending, NEVER implicitly admitted.
    return dict(
        latent=latent,
        observable=observable,
        tabicl=None,
        latent_oracle_kind="generating_score_oracle",
        latent_pass=latent >= threshold,
        observable_pass=observable is not None and observable >= threshold,
        tabicl_pass=None,
        admitted=False,
        orientation=orientation,
    )


def profile(x, y, entities, kind="classification"):
    x = np.asarray(x, float)
    y = np.asarray(y)
    ok = np.isfinite(y)
    x, y, entities = x[ok], y[ok], np.asarray(entities)[ok]
    fit = entities % 3 != 0
    test = ~fit
    result = dict(
        n=len(y),
        fit_n=int(fit.sum()),
        test_n=int(test.sum()),
        kind=kind,
        availability=np.isfinite(x).mean(0).tolist(),
        label_mean=float(y.mean()),
        single=[None] * len(NAMES),
        gain=[0.0] * len(NAMES),
    )
    if (
        min(fit.sum(), test.sum()) < 10
        or len(np.unique(y[fit])) < 2
        or len(np.unique(y[test])) < 2
    ):
        return result | dict(reason="insufficient_split")
    binary = len(np.unique(y)) == 2
    regression = kind == "regression"
    classes = np.unique(y[fit])
    for j in range(x.shape[1]):
        a, b = fit & np.isfinite(x[:, j]), test & np.isfinite(x[:, j])
        if min(a.sum(), b.sum()) < 10 or np.std(x[a, j]) < 1e-9:
            continue
        if regression:
            center = x[a, j].mean()
            scale = x[a, j].std()
            z = (x[a, j] - center) / scale
            slope = np.mean(z * (y[a] - y[a].mean()))
            v = r2_score(y[b], y[a].mean() + slope * (x[b, j] - center) / scale)
        else:
            scores = []
            for c in classes if not binary else classes[1:]:
                ya = y[a] == c
                yb = y[b] == c
                if len(np.unique(ya)) < 2 or len(np.unique(yb)) < 2:
                    continue
                sign = 1 if roc_auc_score(ya, x[a, j]) >= 0.5 else -1
                scores.append(roc_auc_score(yb, sign * x[b, j]))
            if not scores:
                continue
            v = np.mean(scores)
        result["single"][j] = float(v)
    cls = GradientBoostingRegressor if regression else GradientBoostingClassifier
    model = cls(
        n_estimators=80, max_depth=3, min_samples_leaf=15, learning_rate=0.04, random_state=1008
    )
    xx = np.nan_to_num(x, nan=-999.0, posinf=999.0, neginf=-999.0)
    model.fit(xx[fit], y[fit])
    pred = model.predict(xx[test]) if regression else model.predict_proba(xx[test])
    gain = model.feature_importances_.astype(float)
    gain /= max(1e-12, gain.sum())
    metric = (
        r2_score(y[test], pred)
        if regression
        else roc_auc_score(y[test], pred[:, 1])
        if binary
        else roc_auc_score(y[test], pred, multi_class="ovr", labels=classes)
    )
    result.update(gain=gain.tolist(), oracle=float(metric))
    if regression:
        threshold = float(np.median(y[fit]))
        yb = (y > threshold).astype(int)
        result["threshold_auc_profile"] = profile(x, yb, entities, "classification")
        result["threshold_auc_profile"]["threshold_fit_median"] = threshold
    result["group_gain"] = {
        g: float(gain[np.array(GROUPS) == g].sum()) for g in sorted(set(GROUPS))
    }
    result["dominant"] = (
        max(result["group_gain"], key=result["group_gain"].get) if gain.sum() else "unresolved"
    )
    result["best_single"] = max((z for z in result["single"] if z is not None), default=None)
    return result
