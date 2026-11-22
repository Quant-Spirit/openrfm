"""Rebuild references from written files and compare them to the private arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from openrfm.synth.worlds.fact.base import LABEL_TABLE, TARGET, TASK_TABLE, TYPED_KINDS
from openrfm.synth.worlds.fact.oracle import _ns, _read, reference_autocomplete, reference_slate, score_slate
from openrfm.synth.worlds.fact.windows import target_stat, typed_window_stats, window_stats


def audit_autocomplete(rec, tabs, priv):
    """F1 / F2 / F3: rebuild the visible-evidence reference from the written parquet only and compare."""
    task, lab, fact = tabs[TASK_TABLE], tabs[LABEL_TABLE], tabs[rec["event_table"]]
    fk = rec["fk_columns"]
    ftime, ltime, ctime = _ns(fact["date"]), _ns(lab["date"]), _ns(task["date"])
    cases = priv["cases"]
    checks = dict(
        label_after_fact_tick=bool((ltime > ftime[lab["fact_id"].to_numpy()]).all()),
        task_rows_are_fact_rows=bool(np.array_equal(ctime, ftime[cases])),
        task_label_equals_label_history=bool(
            np.array_equal(task[TARGET].to_numpy(), lab["label_value"].to_numpy()[cases])
        ),
    )
    y = lab["label_value"].to_numpy()
    if rec["group"] == "F3":
        W = dict(
            kind=rec["slate_kind"],
            aux=rec["aux"],
            cov_u=priv["cov_u"],
            gmean_cov=priv["gmean_cov"],
            pv=float(priv["pv"]),
            slate=fact[fk[2]].to_numpy(),
        )
        theta, n = reference_slate(W, lab[fk[1]].to_numpy(), W["slate"], ltime, y, cases, ctime)
        pred = score_slate(W, theta, cases)
        key, n_ref = fk[1], n
    else:
        aux = dict(tau=float(priv["tau"]), transform=float(priv["transform"]))
        W = dict(
            kind=rec["kind"],
            aux=aux,
            offset=priv["offset"],
            pv=list(priv["pv"]),
            sigma_ref=float(priv["sigma_ref"]),
        )
        pred, n = reference_autocomplete(
            W,
            [lab[k].to_numpy() for k in fk],
            ltime,
            y,
            [task[k].to_numpy() for k in fk],
            ctime,
            priv["offset"][cases],
        )
        key, n_ref = fk[-1], n[:, -1]
    checks["reference_reproduced_from_parquet"] = bool(
        np.allclose(
            pred, priv["visible_pred" if rec["group"] != "F3" else "ref"], atol=1e-9, rtol=1e-9
        )
    )
    rng = np.random.default_rng(0)
    sample = rng.choice(len(task), min(200, len(task)), replace=False)
    lk, tk = lab[key].to_numpy(), task[key].to_numpy()
    brute = np.array([int(((lk == tk[i]) & (ltime < ctime[i])).sum()) for i in sample])
    checks["visible_counts_match_bruteforce"] = bool(np.array_equal(brute, n_ref[sample]))
    checks["no_own_label_visible"] = bool((ltime[cases] > ctime).all())
    all_count = np.array([int((lk == tk[i]).sum()) for i in sample])
    return checks, dict(future_label_power=float((all_count != brute).mean()))


def audit_window(rec, tabs, priv):
    """F4: recompute every label statistic from the written event rows with date < cutoff."""
    task, ev = tabs[TASK_TABLE], tabs[rec["event_table"]]
    fkc = rec["fk_columns"][0]
    w = dict(zip(("w1", "w2", "L", "b"), [int(v) for v in priv["w"]], strict=False))
    et, ee, ev_val = _ns(ev["date"]), ev[fkc].to_numpy(), ev[rec["value_column"]].to_numpy()
    ctime, ce = _ns(task["date"]), task[fkc].to_numpy()
    stat, leaky, nA = np.empty(len(task)), np.empty(len(task)), np.empty(len(task))
    burst = rec["window_kind"] == "burst"
    for e in np.unique(ce):
        m = ee == e
        o = np.argsort(et[m])
        t, v = et[m][o], ev_val[m][o]
        ix = np.flatnonzero(ce == e)
        stats = window_stats(t, v, ctime[ix], w, burst)
        if rec["window_kind"] in TYPED_KINDS:
            cats = ev[rec["type_column"]].to_numpy()[m][o]
            selected = task[rec["query_type_column"]].to_numpy()[ix]
            stats.update(typed_window_stats(t, v, cats, ctime[ix], w, selected))
        stat[ix], nA[ix] = target_stat(stats, rec["window_kind"], w), stats["nA"]
        later = window_stats(t, v, ctime[ix] + w["w1"], w, burst)
        if rec["window_kind"] in TYPED_KINDS:
            later.update(typed_window_stats(t, v, cats, ctime[ix] + w["w1"], w, selected))
        leaky[ix] = target_stat(later, rec["window_kind"], w)  # rows >= cutoff made visible
    checks = dict(stat_matches_oracle_from_parquet=bool(np.array_equal(stat, priv["stat"])))
    if rec["pair_role"] == "summary":
        checks["summary_count_matches_raw_rows"] = bool(
            np.array_equal(task[rec["summary_columns"][1]].to_numpy(), nA)
        )
    return checks, dict(future_row_power=float((leaky != stat).mean()))


def audit_world(raw_root, db, private_dir):
    rec = json.loads((Path(raw_root) / db / "a5_world.json").read_text())
    tabs = _read(Path(raw_root) / db)
    priv = np.load(Path(private_dir) / rec["private_file"])
    checks, power = (audit_window if rec["group"] == "F4" else audit_autocomplete)(
        rec, tabs, priv
    )
    checks["target_only_in_task_table"] = not [
        t for t, df in tabs.items() if t != TASK_TABLE and TARGET in df.columns
    ]
    return dict(
        db=db,
        family_id=rec["family_id"],
        checks=checks,
        power=power,
        oracle=rec["oracle"],
        ok=all(checks.values()),
    )


def audit_pair(raw_root, rec_raw, rec_sum):
    """The two F4 renderings carry the same process: identical events, cases and labels."""
    a, b = (
        _read(Path(raw_root) / rec_raw["db_name"]),
        _read(Path(raw_root) / rec_sum["db_name"]),
    )
    ev = rec_raw["event_table"]
    same = all(a[ev][c].equals(b[ev][c]) for c in a[ev].columns) and all(
        a[TASK_TABLE][c].equals(b[TASK_TABLE][c]) for c in a[TASK_TABLE].columns
    )
    s = set(rec_sum["summary_columns"])
    return dict(
        pair_id=rec_raw["pair_id"],
        identical_process=bool(same),
        raw_has_no_summary=not s & set(a[ev].columns) and not s & set(a[TASK_TABLE].columns),
        summary_present=s <= set(b[ev].columns) and s <= set(b[TASK_TABLE].columns),
    )
