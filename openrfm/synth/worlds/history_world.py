"""Write panel-history worlds and admit them by their oracle report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from openrfm.synth.io import _require
from openrfm.synth.worlds.history_oracle import _auc, history_admission, oracle_report
from openrfm.synth.worlds.history_process import (
    _history_noise, _history_static, history_family, history_kind, history_simulate,
)
from openrfm.synth.worlds.panel import (
    EPOCH_S, HISTORY_BASE_DAY, HISTORY_EVENT_WINDOW_DAYS, HISTORY_HORIZON_DAYS, HISTORY_OWN_EVENT_ROWS,
    HISTORY_OWN_LABEL_ROWS, HISTORY_RECIPE, HISTORY_STEP_DAYS, HUB_ROWS, MAX_ATTEMPTS, PEER_ROWS, RATE_BOUNDS, RECIPE, STAGE,
    WORLD_FILE, _write_integrity_anchor, _write_table, history_task_metadata,
)


def _history_tables(
    root: Path, p: Mapping[str, Any], sim: Mapping[str, Any], nz: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Write the panel tables and return the private identity maps and row counts."""
    import pyarrow as pa

    (root / "db").mkdir(parents=True)
    ev, lab, G = sim["ev"], sim["label"], p["G"]
    pairs = p["pairs"]
    n = len(pairs)
    ce, cg = np.asarray([a for a, _ in pairs]), np.asarray([b for _, b in pairs])
    cut = HISTORY_STEP_DAYS * cg  # relative day of each task row's cutoff
    ts = lambda days: pa.array(
        (EPOCH_S + (HISTORY_BASE_DAY + np.asarray(days, np.int64)) * 86400) * 1_000_000_000,
        pa.timestamp("ns"),
    )
    f = np.stack(
        [nz["x"][ce, cg, 0], nz["x"][ce, cg, 1], p["static_x"][ce, 0], p["static_x"][ce, 1]], 1
    )
    f[:, :2][p["mask"][ce, cg]] = np.nan
    f[:, 2:][p["smask"][ce]] = np.nan
    _write_table(
        root / "db" / "parent.parquet",
        {
            "parent_id": np.arange(p["parents"], dtype=np.int64),
            "feature_0": p["parent_x"][:, 0],
            "feature_1": p["parent_x"][:, 1],
        },
        "parent_id",
        {},
    )
    _write_table(
        root / "db" / "cases.parquet",
        {
            "case_id": np.arange(n, dtype=np.int64),
            **{
                f"feature_{j}": pa.array(f[:, j], pa.float64(), from_pandas=True)
                for j in range(4)
            },
            "category": p["category"][ce],
            "parent_id": p["parent_fk"][ce],
        },
        "case_id",
        {"parent_id": "parent"},
    )
    acc = {k: [] for k in ("own", "labels", "peers", "hub")}
    pe_own, pe_peer = [], []
    all_day = np.argsort(ev["day"], kind="stable")
    if p.get("single_hub"):
        grouped = np.lexsort((ev["day"], p["grp"][ev["ent"]]))
        boundaries = np.searchsorted(p["grp"][ev["ent"][grouped]], np.arange(p["ng"] + 1))
        grp_events = [grouped[boundaries[h] : boundaries[h + 1]] for h in range(p["ng"])]
    else:
        grp_events = [all_day[p["member"][ev["ent"][all_day], h] > 0] for h in range(p["ng"])]
    grp_days = [ev["day"][ix] for ix in grp_events]
    for cid, (e, g) in enumerate(pairs):
        c = HISTORY_STEP_DAYS * g
        ptr = slice(ev["ptr"][e], ev["ptr"][e + 1])
        hi = int(np.searchsorted(ev["day"][ptr], c))
        lo = max(0, hi - HISTORY_OWN_EVENT_ROWS)
        ix = ev["ptr"][e] + np.arange(hi - 1, lo - 1, -1)
        # Keep future events dated so readers can enforce a strict cutoff on the full stream.
        end = int(np.searchsorted(ev["day"][ptr], c + 4 * HISTORY_HORIZON_DAYS, side="right"))
        ix = np.concatenate([ix, ev["ptr"][e] + np.arange(hi, end)])
        acc["own"].append(
            (
                cid,
                ev["day"][ix],
                ev["val"][ix],
                ev["f1"][ix],
                ev["f2"][ix],
                (
                    ev["category"][ix]
                    if "category" in ev
                    else np.full(len(ix), p["grp"][e] % 16)
                ),
            )
        )
        pe_own.append(np.full(len(ix), e))
        js = np.arange(g - 1, max(p["start"][e], g - HISTORY_OWN_LABEL_ROWS) - 1, -1)
        acc["labels"].append(
            (cid, HISTORY_STEP_DAYS * js + HISTORY_HORIZON_DAYS, lab[e, js], None, None, None)
        )
        shared = (
            (p["grp"] == p["grp"][e]).astype(np.int64)
            if p.get("single_hub")
            else p["member"] @ p["member"][e]
        )
        shared[e] = 0
        peers = (
            np.flatnonzero((shared > 0) & (p["start"] <= g - 1))
            if g >= 1
            else np.zeros(0, np.int64)
        )
        peers = peers[np.lexsort((peers, -shared[peers]))][:PEER_ROWS]
        if p.get("single_hub"):
            # Independent paired-peer level needs multiple matured observations.
            pg = np.arange(max(0, g - PEER_ROWS), g)
            pe = np.repeat(peers, len(pg))
            pg = np.tile(pg, len(peers))
            alive = pg >= p["start"][pe]
            pe, pg = pe[alive], pg[alive]
            acc["peers"].append(
                (cid, HISTORY_STEP_DAYS * pg + HISTORY_HORIZON_DAYS, lab[pe, pg], shared[pe], None, None)
            )
            pe_peer.append(pe)
        else:
            acc["peers"].append(
                (
                    cid,
                    np.full(len(peers), HISTORY_STEP_DAYS * (g - 1) + HISTORY_HORIZON_DAYS),
                    lab[peers, g - 1],
                    shared[peers],
                    None,
                    None,
                )
            )
            pe_peer.append(peers)
        hub = []
        for h in np.flatnonzero(p["member"][e]):
            a, b = np.searchsorted(grp_days[h], [c - HISTORY_EVENT_WINDOW_DAYS, c])
            hub.append(grp_events[h][a:b])
        hub = np.unique(np.concatenate(hub)) if hub else np.zeros(0, np.int64)
        hub = hub[ev["ent"][hub] != e]
        hub = hub[np.argsort(-ev["day"][hub], kind="stable")][:HUB_ROWS]
        acc["hub"].append(
            (
                cid,
                ev["day"][hub],
                ev["val"][hub],
                ev["f1"][hub],
                ev["f2"][hub],
                ev["category"][hub] if "category" in ev else None,
            )
        )
    cat = lambda k, i, dt: np.concatenate(
        [np.asarray(a[i]) if a[i] is not None else np.zeros(0) for a in acc[k]]
    ).astype(dt)
    case_col = lambda k: np.concatenate([np.full(len(a[1]), a[0]) for a in acc[k]]).astype(
        np.int64
    )
    cut_of = lambda k: cut[case_col(k)]

    def write(stem: str, key: str, cols: dict[str, Any]) -> int:
        rows = len(cols["case_id"])
        _write_table(
            root / "db" / f"case_{stem}.parquet",
            {"row_idx": np.arange(rows, dtype=np.int64), **cols},
            "row_idx",
            {"case_id": "cases"},
            "date",
        )
        return rows

    day_own, day_lab, day_peer, day_hub = (
        cat(k, 1, np.int64) for k in ("own", "labels", "peers", "hub")
    )
    write(
        "own_events",
        "own",
        dict(
            case_id=case_col("own"),
            date=ts(day_own),
            age_days=(cut_of("own") - day_own).astype(np.float64),
            feature_0=cat("own", 2, np.float64),
            feature_1=cat("own", 3, np.float64),
            feature_2=cat("own", 4, np.float64),
            hub_category=cat("own", 5, np.int64),
        ),
    )
    write(
        "own_labels",
        "labels",
        dict(
            case_id=case_col("labels"),
            date=ts(day_lab),
            label_value=cat("labels", 2, np.float64),
            label_age_days=(cut_of("labels") - day_lab).astype(np.float64),
        ),
    )
    write(
        "peer_links",
        "peers",
        dict(
            case_id=case_col("peers"),
            date=ts(day_peer),
            label_value=cat("peers", 2, np.float64),
            shared_members=cat("peers", 3, np.int64),
            label_age_days=(cut_of("peers") - day_peer).astype(np.float64),
        ),
    )
    write(
        "hub_events",
        "hub",
        dict(
            case_id=case_col("hub"),
            date=ts(day_hub),
            age_days=(cut_of("hub") - day_hub).astype(np.float64),
            feature_0=cat("hub", 2, np.float64),
            feature_1=cat("hub", 3, np.float64),
            feature_2=cat("hub", 4, np.float64),
            **({"hub_category": cat("hub", 5, np.int64)} if "category" in ev else {}),
        ),
    )
    y = lab[ce, cg]
    _write_table(
        root / "db" / "a5_task_0.parquet",
        {
            "row_idx": np.arange(n, dtype=np.int64),
            "date": ts(cut),
            "entity_id": np.arange(n, dtype=np.int64),
            "a5_y_0": y.astype(np.int64) if p["kind"] == "classification" else y,
        },
        "row_idx",
        {"entity_id": "cases"},
        "date",
    )
    grid = np.where(np.arange(G)[None] >= p["start"][:, None], lab, np.nan)
    np.savez_compressed(
        root / "panel_private.npz",
        case_entity=ce,
        case_cutoff_day=HISTORY_BASE_DAY + cut,
        own_event_entity=np.concatenate(pe_own),
        peer_entity=np.concatenate(pe_peer),
        oracle_score=sim["score"][ce, cg],
        grid_cutoff_day=HISTORY_BASE_DAY + HISTORY_STEP_DAYS * np.arange(G),
        grid_labels=grid,
        entity_start=p["start"],
    )
    return dict(y=y, score=sim["score"][ce, cg], cases=n)


_log = __import__("logging").getLogger(__name__)


def write_history_world(raw_root: Path, db: str, slot: int, seed: int) -> dict[str, Any]:
    """One panel-history world; a draw failing its family's oracle admission is redrawn deterministically."""
    import shutil

    raw_root = Path(raw_root)
    root = raw_root / db
    family, kind = history_family(slot), history_kind(slot)
    _require(not root.exists(), f"refusing to overwrite world {root}")
    rejected: list[list[str]] = []
    for attempt in range(MAX_ATTEMPTS):
        tmp = raw_root / f".{db}.attempt{attempt}"
        _require(not tmp.exists(), f"stale attempt directory {tmp}")
        rng = np.random.default_rng([seed, slot, attempt, 1008])
        p = _history_static(rng, family, kind)
        nz = _history_noise(rng, p["E"], p["G"])
        sim = history_simulate(p, nz)
        made = _history_tables(tmp, p, sim, nz)
        rate = float(made["y"].mean())
        oracle = (
            _auc(made["score"], made["y"])
            if kind == "classification"
            else float(np.corrcoef(made["score"], np.arcsinh(made["y"]))[0, 1] ** 2)
        )
        if kind == "classification" and not RATE_BOUNDS[0] <= rate <= RATE_BOUNDS[1]:
            why = [f"positive rate {rate:.3f} outside {RATE_BOUNDS}"]
        else:
            _history_finish(tmp, db, slot, seed, p, family, kind, made, oracle, attempt, rejected)
            report = oracle_report(tmp)
            why = history_admission(report, family)
        if not why:
            record = json.loads((tmp / WORLD_FILE).read_text())
            record["history"]["oracle_report"] = report
            (tmp / WORLD_FILE).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
            tmp.rename(root)
            return record
        _log.warning("%s attempt %d rejected: %s", db, attempt, "; ".join(why))
        rejected.append(why)
        shutil.rmtree(tmp)
    _require(False, f"{db}: no admitted draw in {MAX_ATTEMPTS} attempts: {rejected[-1]}")


def _history_finish(
    root: Path,
    db: str,
    slot: int,
    seed: int,
    p: Mapping[str, Any],
    family: str,
    kind: str,
    made: Mapping[str, Any],
    oracle: float,
    attempt: int,
    rejected: list,
) -> None:
    import pyarrow.parquet as pq

    (root / "task_metadata.json").write_text(
        json.dumps(history_task_metadata(db, family, kind), indent=1, sort_keys=True) + "\n"
    )
    names = [
        f"{c} of {q.stem}"
        for q in sorted((root / "db").glob("*.parquet"))
        for c in pq.read_schema(q).names
    ]
    (root / "column_index.json").write_text(
        json.dumps({name: i for i, name in enumerate(names)}) + "\n"
    )
    _write_integrity_anchor(root)
    counts = {
        q.stem: pq.ParquetFile(q).metadata.num_rows
        for q in sorted((root / "db").glob("*.parquet"))
    }
    days = [
        (EPOCH_S + (HISTORY_BASE_DAY + d) * 86400) * 1_000_000_000
        for d in (0, HISTORY_STEP_DAYS * p["G"])
    ]
    record = dict(
        db_name=db,
        slot=slot,
        stage=STAGE,
        seed=seed,
        recipe=RECIPE,
        family=family,
        entities=p["E"],
        hubs=p["ng"],
        parents=p["parents"],
        cases=made["cases"],
        events=counts["case_own_events"],
        positive_rate=float(made["y"].mean()) if kind == "classification" else None,
        time_min_ns=days[0],
        time_max_ns=days[1],
        table_rows=counts,
        oracle={
            "metric": "auroc" if kind == "classification" else "r2",
            "score": "noiseless_family_latent",
            "value": oracle,
            "scale": 1.0,
        },
        attempt=attempt,
        dropped_relation=None,
        omitted_relations=[],
        history=dict(
            recipe=HISTORY_RECIPE,
            kind=kind,
            heavy_tail=bool(p["heavy"]),
            rejected=rejected,
            params={
                k: (float(v) if np.isscalar(v) else None)
                for k, v in p.items()
                if k in ("rho", "sa", "smu", "W", "kappa", "gamma", "amp", "period", "sv", "sy")
            },
        ),
    )
    (root / WORLD_FILE).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
