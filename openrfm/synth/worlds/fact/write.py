"""Writers of the autocomplete, slate and window-pair task views of fact worlds."""

from __future__ import annotations

import hashlib

import numpy as np

from openrfm.synth.worlds.fact.base import (
    FamilyConfig, ID_OFFSET, LABEL_TABLE, RECIPE, TAG, TARGET, TYPED_KINDS, _context_tables, _fact_columns, _names,
    _tags, census_visible, metric,
)
from openrfm.synth.worlds.fact.io import _emit, _label_table, _save_private, _ts
from openrfm.synth.worlds.fact.oracle import (
    case_reference, census_autocomplete, census_window, oracle_window, reference_slate, score_slate, top1,
)
from openrfm.synth.worlds.fact.windows import summary_rows


def write_autocomplete(raw_root, W, slot, seed, private_dir, cfg=FamilyConfig()):
    """F1 / F2 export: the full fact log plus labels published after their tick."""
    rng = W.pop("rng_tail")
    fam, kind, T, N = W["family"], W["kind"], W["T"], len(W["T"])
    db = f"rel-synthetic-{ID_OFFSET[fam] + slot}"
    nslot = len(W["pids"])
    names = _names(rng, W["x"], W["extra"], nslot)
    pk = [f"{k}_id" for k in names["fk"][:nslot]]
    ftab, *ptabs = _tags(rng, nslot + 1, "tbl")
    tables = {}
    for k in range(nslot):
        pcols = {
            pk[k]: np.arange(W["parent_size"][k], dtype=np.int64),
            "attr": W["parent_static"][k],
        }
        tables[ptabs[k]] = (pcols, pk[k], _context_tables(rng, cfg, pcols, tables), None)
    fkeys = {pk[k]: ptabs[k] for k in range(nslot)}
    base = _fact_columns(names, W["x"], W["cat"], W["extra"])
    ids = {pk[k]: W["pids"][k].astype(np.int64) for k in range(nslot)}
    tables[ftab] = (
        {"row_idx": np.arange(N, dtype=np.int64), **ids, "date": _ts(T), **base},
        "row_idx",
        fkeys,
        "date",
    )
    tables[LABEL_TABLE] = (
        _label_table(ids, W["pub"], W["y"], np.arange(N)),
        "row_idx",
        {**fkeys, "fact_id": ftab},
        "date",
    )
    c = W["cases"]
    task = {
        "row_idx": np.arange(len(c), dtype=np.int64),
        **{k: v[c] for k, v in ids.items()},
        "date": _ts(T[c]),
        **{k: v[c] for k, v in base.items()},
        "declared_horizon_0_seconds": np.ones(len(c)),
        TARGET: W["y"][c],
    }
    n, ref = case_reference(W, c)
    record = dict(
        family=fam,
        family_id=f"NEW.fact{fam[1]}.{kind}",
        group=fam,
        kind=kind,
        seed=seed,
        slot=slot,
        cases=int(len(c)),
        prediction_type="classification" if kind == "binary" else "regression",
        event_table=ftab,
        value_column=names["x"][0],
        parents=nslot,
        rendering="raw_rows",
        process=W["info"],
        aux={k: v for k, v in W["aux"].items() if k != "sig"},
        census=census_autocomplete(W, c, n),
        fk_columns=pk,
        oracle=ref["metrics"],
        private_file=f"{db}.npz",
    )
    _save_private(
        private_dir,
        db,
        offset=W["offset"],
        pv=np.array(W["pv"]),
        sigma_ref=W["sigma_ref"],
        tau=W["aux"]["tau"],
        transform=W["aux"]["transform"],
        cases=c,
        visible_pred=ref["ref"],
        visible_n=n,
    )
    return _emit(raw_root, db, tables, task, fkeys, record, kind)


def write_slate(raw_root, W, slot, seed, private_dir, cfg=FamilyConfig()):
    rng = W.pop("rng_tail")
    kind, T, N, S = W["kind"], W["T"], len(W["T"]), W["slates"]
    mk = "binary" if kind == "choice" else "regression"
    db = f"rel-synthetic-{ID_OFFSET['F3'] + slot}"
    A, I = W["nparents"]
    pool = _tags(rng, 7 + W["x"].shape[1] + W["extra"].shape[1], "c")
    ta, ti, ts_, ftab, fa, fi, fs = pool[:7]
    xcols = pool[7 : 7 + W["x"].shape[1]]
    ecols = pool[7 + W["x"].shape[1] :]
    tables = {
        ta: ({fa: np.arange(A, dtype=np.int64), "attr": rng.normal(size=A)}, fa, {}, None),
        ti: ({fi: np.arange(I, dtype=np.int64), "attr": rng.normal(size=I)}, fi, {}, None),
        ts_: (
            {fs: np.arange(S, dtype=np.int64), fa: W["actor_of_slate"].astype(np.int64)},
            fs,
            {fa: ta},
            None,
        ),
    }
    cols = {xcols[j]: W["x"][:, j] for j in range(len(xcols))}
    cols.update({ecols[j]: W["extra"][:, j] for j in range(len(ecols))})
    cols["position"] = W["pos"].astype(np.int64)
    ids = {
        fa: W["actor"].astype(np.int64),
        fi: W["item"].astype(np.int64),
        fs: W["slate"].astype(np.int64),
    }
    fkeys = {fa: ta, fi: ti, fs: ts_}
    tables[ftab] = (
        {"row_idx": np.arange(N, dtype=np.int64), **ids, "date": _ts(T), **cols},
        "row_idx",
        fkeys,
        "date",
    )
    tables[LABEL_TABLE] = (
        _label_table({fa: ids[fa], fi: ids[fi]}, W["pub"], W["y"], np.arange(N)),
        "row_idx",
        {fa: ta, fi: ti, "fact_id": ftab},
        "date",
    )
    c = W["cases"]
    task = {
        "row_idx": np.arange(len(c), dtype=np.int64),
        **{k: v[c] for k, v in ids.items()},
        "date": _ts(T[c]),
        **{k: v[c] for k, v in cols.items()},
        "declared_horizon_0_seconds": np.ones(len(c)),
        TARGET: W["y"][c],
    }
    theta, n = reference_slate(W, W["item"], W["slate"], W["pub"], W["y"], c, T[c])
    ref, static, y = score_slate(W, theta, c), score_slate(W, np.zeros(len(c)), c), W["y"][c]
    m = dict(
        oracle_state=metric(mk, W["state"][c], y),
        reference_visible=metric(mk, ref, y),
        reference_static=metric(mk, static, y),
    )
    m["history_gain"] = m["reference_visible"] - m["reference_static"]
    if kind == "choice":
        m["top1_state"], m["top1_visible"] = (
            top1(W["slate"][c], W["state"][c], y),
            top1(W["slate"][c], ref, y),
        )
    census = dict(
        fact_rows=N,
        label_rows=N,
        cases=int(len(c)),
        slates=int(S),
        case_slates=int(len(np.unique(W["slate"][c]))),
        positive_rate_rows=float(W["y"].mean()) if kind == "choice" else None,
        positive_rate_cases=float(y.mean()) if kind == "choice" else None,
        mean_slate_size=W["info"]["mean_slate_size"],
        visible_labels_item=census_visible(n),
        cold_item_share=float((n == 0).mean()),
        parents=[A, I],
    )
    record = dict(
        family="F3",
        family_id=f"NEW.slate.{kind}",
        group="F3",
        kind=mk,
        slate_kind=kind,
        seed=seed,
        slot=slot,
        cases=int(len(c)),
        prediction_type="classification" if kind == "choice" else "regression",
        event_table=ftab,
        value_column=xcols[0],
        parents=2,
        rendering="raw_rows",
        process=W["info"],
        aux=W["aux"],
        census=census,
        oracle=m,
        fk_columns=[fa, fi, fs],
        private_file=f"{db}.npz",
    )
    _save_private(
        private_dir,
        db,
        cases=c,
        ref=ref,
        theta=theta,
        visible_n=n,
        cov_u=W["cov_u"],
        gmean_cov=W["gmean_cov"],
        pv=W["pv"],
    )
    return _emit(raw_root, db, tables, task, fkeys, record, mk)


def write_window_pair(raw_root, d, slot, seed, private_dir):
    """F4: raw and summary renderings of the same process (identical events, cases, labels)."""
    rng = np.random.default_rng([seed, slot, TAG["F4"], 7])
    E, kind, w = d["E"], d["kind"], d["w"]
    typed = d["target_kind"] in TYPED_KINDS
    ent, ev, fk, cat_c, val_c, att = _tags(rng, 6, "c")
    summary_keys = ("nHalf", "nA", "nDouble", "nB", "gap") + (TYPED_KINDS if typed else ())
    sum_names = dict(zip(summary_keys, _tags(rng, len(summary_keys), "s"), strict=False))
    pair_id = hashlib.sha256(f"{RECIPE}:F4:{seed}:{slot}".encode()).hexdigest()[:16]
    ev_e = np.concatenate([np.full(len(t), e, np.int64) for e, t in enumerate(d["ent_t"])])
    ev_t = np.concatenate(d["ent_t"]).astype(np.int64)
    order = np.argsort(ev_t, kind="stable")
    ev_e, ev_t = ev_e[order], ev_t[order]
    ev_v, ev_c = (
        np.concatenate(d["ent_v"])[order],
        np.concatenate(d["ent_c"])[order].astype(np.int64),
    )
    entity_cols = {fk: np.arange(E, dtype=np.int64), att: rng.normal(size=E)}
    roll = summary_rows(
        d,
        [
            (e, ev_t[ev_e == e], ev_c[ev_e == e]) if typed else (e, ev_t[ev_e == e])
            for e in range(E)
        ],
    )
    roll_pos = np.argsort(
        np.lexsort((ev_t, ev_e))
    )  # entity-major position of each time-ordered row
    case_roll = summary_rows(
        d,
        [
            (e, d["case_t"][d["case_e"] == e], d["case_type"][d["case_e"] == e])
            if typed
            else (e, d["case_t"][d["case_e"] == e])
            for e in np.unique(d["case_e"])
        ],
    )
    case_pos = np.argsort(np.lexsort((d["case_t"], d["case_e"])))
    hist_pub = d["hist_t"] + d["hist_delay"]
    ho = np.argsort(hist_pub, kind="stable")
    delay_cols = (
        _label_table({fk: d["hist_e"][ho].astype(np.int64)}, hist_pub[ho], d["hist_y"][ho])
        if len(ho)
        else None
    )
    if typed and delay_cols is not None:
        delay_cols[cat_c] = d["hist_type"][ho].astype(np.int64)
    census, oracle = census_window(d), oracle_window(d)
    out = []
    for rendering, offset in (("raw_events", 0), ("summary_materialised", 1)):
        db = f"rel-synthetic-{ID_OFFSET['F4'] + 2 * slot + offset}"
        events = {
            "row_idx": np.arange(len(ev_t), dtype=np.int64),
            fk: ev_e,
            "date": _ts(ev_t),
            val_c: ev_v,
            cat_c: ev_c,
        }
        task = {
            "row_idx": np.arange(len(d["case_t"]), dtype=np.int64),
            fk: d["case_e"].astype(np.int64),
            "date": _ts(d["case_t"]),
        }
        if typed:
            task[cat_c] = d["case_type"].astype(np.int64)
        task_fkeys = {fk: ent}
        if offset:
            for k, name in sum_names.items():
                events[name], task[name] = roll[k][roll_pos], case_roll[k][case_pos]
        task["declared_horizon_0_seconds"] = np.ones(len(d["case_t"]))
        task[TARGET] = d["y"]
        tables = {
            ent: (dict(entity_cols), fk, {}, None),
            ev: (events, "row_idx", {fk: ent}, "date"),
        }
        if delay_cols is not None:
            tables[LABEL_TABLE] = (delay_cols, "row_idx", {fk: ent}, "date")
        record = dict(
            family="F4",
            family_id=f"NEW.window.{d['target_kind']}.{kind}",
            group="F4",
            kind=kind,
            window_kind=d["target_kind"],
            seed=seed,
            slot=slot,
            cases=int(len(d["case_t"])),
            prediction_type="classification" if kind == "binary" else "regression",
            event_table=ev,
            value_column=val_c,
            parents=1,
            rendering=rendering,
            pair_id=pair_id,
            pair_role="summary" if offset else "raw",
            summary_columns=list(sum_names.values()) if offset else [],
            pair_db=f"rel-synthetic-{ID_OFFSET['F4'] + 2 * slot + 1 - offset}",
            process=d["info"],
            census=census,
            oracle=oracle,
            fk_columns=[fk],
            label_history_rows=int(len(ho)),
            private_file=f"{db}.npz",
        )
        if typed:
            record.update(
                type_column=cat_c, query_type_column=cat_c, generator_revision="typed_window_v1"
            )
        _save_private(
            private_dir,
            db,
            oracle_score=d["oracle_score"],
            stat=d["stat"],
            w=np.array([w["w1"], w["w2"], w["L"], w["b"]]),
        )
        out.append(_emit(raw_root, db, tables, task, task_fkeys, record, kind))
    return out


