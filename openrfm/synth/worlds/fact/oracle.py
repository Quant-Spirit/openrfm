"""Visible-evidence references, census scores and admission of fact worlds."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openrfm.synth.worlds.fact.base import EPOCH_S, _gauss_gh, _probit_gh, _to_y, census_visible, evidence, metric


def reference_autocomplete(W, ids, pub, y, case_ids, case_t, case_offset):
    """Visible-evidence reference (history only, true row coefficients): per-slot effects and counts."""
    aux, kind = W["aux"], W["kind"]
    meas = y if kind == "binary" else (np.log(y) / aux["transform"] if aux["transform"] else y)
    if kind == "binary":
        g, h = _probit_gh(W["offset"], meas, aux["tau"], W["sigma_ref"])
    else:
        g, h = _gauss_gh(W["offset"], meas, W["sigma_ref"])
    effects, counts = [], []
    for k, pv in enumerate(W["pv"]):
        e, n = evidence(ids[k], pub, g, h, case_ids[k], case_t, pv)
        effects.append(e)
        counts.append(n)
    theta = np.sum(effects, 0)
    return _to_y(
        kind, case_offset + theta, aux["tau"], W["sigma_ref"], aux["transform"]
    ), np.stack(counts, 1)


def reference_slate(W, item, slate, pub, y, rows, case_t):
    """Item effect from earlier labelled appearances (one conditional-logit / Gaussian step)."""
    aux = W["aux"]
    if W["kind"] == "choice":
        shift = W["cov_u"].max()
        e = np.exp(W["cov_u"] - shift)
        o = np.exp(aux["o_base"] - shift) if aux["outside"] else 0.0
        pi0 = e / (np.bincount(slate, e) + o)[slate]
        g, h = y - pi0, pi0 * (1 - pi0)
    else:
        off = np.sqrt(aux["share"]) * (W["cov_u"] - W["gmean_cov"]) / aux["uc_scale"]
        g, h = _gauss_gh(off, y, np.sqrt(1 - aux["share"]))
    return evidence(item, pub, g, h, item[rows], case_t, W["pv"])


def score_slate(W, theta, rows):
    """Within-slate scores for the case rows given item effects theta."""
    inv = np.unique(W["slate"][rows], return_inverse=True)[1]
    if W["kind"] == "choice":
        u = W["cov_u"][rows] + theta
        e = np.exp(u - u.max())
        o = np.exp(W["aux"]["o_base"] - u.max()) if W["aux"]["outside"] else 0.0
        return e / (np.bincount(inv, e) + o)[inv]
    base = (
        np.sqrt(W["aux"]["share"])
        * (W["cov_u"][rows] - W["gmean_cov"][rows])
        / W["aux"]["uc_scale"]
    )
    return base + theta - (np.bincount(inv, theta) / np.bincount(inv))[inv]



def case_reference(W, c):
    """Oracle (true state), static-covariate and history-reference scores for the case rows."""
    aux, kind = W["aux"], W["kind"]
    ref, n = reference_autocomplete(
        W, W["pids"], W["pub"], W["y"], [p[c] for p in W["pids"]], W["T"][c], W["offset"][c]
    )
    static = _to_y(kind, W["offset"][c], aux["tau"], W["sigma_ref"], aux["transform"])
    state = _to_y(kind, W["state"][c], aux["tau"], aux["sigma"], aux["transform"])
    y = W["y"][c]
    m = dict(
        oracle_state=metric(kind, state, y),
        reference_visible=metric(kind, ref, y),
        reference_static=metric(kind, static, y),
    )
    m["history_gain"] = m["reference_visible"] - m["reference_static"]
    return n, dict(ref=ref, metrics=m)


def census_autocomplete(W, c, n):
    y, kind = W["y"], W["kind"]
    out = dict(
        fact_rows=int(len(y)),
        label_rows=int(len(y)),
        cases=int(len(c)),
        parents=[int(p) for p in W["parent_size"]],
        active_parents=[int(len(np.unique(p))) for p in W["pids"]],
        positive_rate_rows=float(y.mean()) if kind == "binary" else None,
        positive_rate_cases=float(y[c].mean()) if kind == "binary" else None,
    )
    for k in range(n.shape[1]):
        out[f"visible_labels_slot{k}"] = census_visible(n[:, k])
    if n.shape[1] == 2:
        out["both_cold_share"] = float(((n[:, 0] == 0) & (n[:, 1] == 0)).mean())
    return out


def top1(slate, score, y):
    """Share of slates with a positive whose highest-scored row is the positive."""
    hit = tot = 0
    for s in np.unique(slate):
        ix = np.flatnonzero(slate == s)
        if y[ix].sum() == 1:
            tot += 1
            hit += int(y[ix][np.argmax(score[ix])] == 1)
    return hit / max(tot, 1)


def oracle_window(d):
    return dict(
        oracle_state=metric(d["kind"], d["oracle_score"], d["y"]),
        noise=d["noise"],
        threshold=d.get("threshold"),
    )


def census_window(d):
    ev = np.array([len(t) for t in d["ent_t"]])
    if d["quota"]:
        pub = d["hist_t"] + d["hist_delay"]
        vis = np.array(
            [
                np.sum((d["hist_e"] == e) & (pub < t))
                for e, t in zip(d["case_e"], d["case_t"], strict=False)
            ]
        )
    else:
        vis = np.zeros(len(d["case_t"]), int)
    return dict(
        entities=int(d["E"]),
        fact_rows=int(ev.sum()),
        cases=int(len(d["case_t"])),
        label_rows=int(d["quota"] * d["E"]),
        positive_rate_cases=float(d["y"].mean()) if d["kind"] == "binary" else None,
        events_per_entity=dict(
            mean=float(ev.mean()),
            median=float(np.median(ev)),
            p90=float(np.quantile(ev, 0.9)),
            zero_share=float((ev == 0).mean()),
        ),
        visible_labels=census_visible(vis),
    )


def _read(root):
    import pyarrow.parquet as pq

    return {
        p.stem: pq.read_table(p).to_pandas()
        for p in sorted((Path(root) / "db").glob("*.parquet"))
    }


def _ns(col):
    return col.astype("int64").to_numpy() // 10**9 - EPOCH_S


def admitted(rec, cfg):
    o = rec["oracle"]
    cls = rec["prediction_type"] == "classification"
    if rec["cases"] < cfg.min_cases:
        return "too_few_cases"
    if cls:
        pr = rec["census"].get("positive_rate_cases")
        if pr is None or min(pr, 1 - pr) * rec["cases"] < cfg.min_class_count:
            return "rare_class_in_cases"
    floor = cfg.oracle_floor_cls if cls else cfg.oracle_floor_reg
    return None if o["oracle_state"] >= floor else "oracle_below_floor"

