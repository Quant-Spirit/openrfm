"""Parquet emission, public window views and private arrays of fact worlds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from openrfm.synth.worlds.fact.base import EPOCH_S, LABEL_TABLE, RECIPE, TARGET, TASK_TABLE


def _ts(T):
    import pyarrow as pa

    seconds = EPOCH_S + np.asarray(T, np.int64)
    if np.any(seconds >= 2**31) or np.any(seconds < 0):
        raise ValueError("event clock outside native int32 seconds")
    return pa.array(seconds * np.int64(10**9), pa.timestamp("ns"))


def _json(o):
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(type(o))


def _emit(raw_root, db, tables, task_cols, task_fkeys, record, kind):
    """Write parquet tables, the task table, metadata sidecars and the manifest record of one world."""
    import pyarrow.parquet as pq

    from openrfm.synth.worlds import panel as native
    from openrfm.synth.io import column_value_digest

    root = Path(raw_root) / db
    (root / "db").mkdir(parents=True, exist_ok=False)
    for name, (cols, pkey, fkeys, tcol) in tables.items():
        native._write_table(root / "db" / f"{name}.parquet", cols, pkey, fkeys, tcol)
    native._write_table(
        root / "db" / f"{TASK_TABLE}.parquet", task_cols, "row_idx", task_fkeys, "date"
    )
    ev, val = record["event_table"], record["value_column"]
    binary = kind == "binary"
    side = lambda tables_: dict(
        tables=tables_,
        value_column=val if tables_ == [ev] else "label_value",
        center=0,
        scale=1,
    )
    schema = dict(
        version=1,
        horizon_seconds=0,
        own_event=side([ev]),
        peer_event=side([ev]),
        own_label=dict(side([LABEL_TABLE]), binary_positive=binary),
        peer_label=dict(side([LABEL_TABLE]), binary_positive=binary),
    )
    meta = native.history_task_metadata(db, "fact", "classification" if binary else "regression")
    for task in [*meta["tasks"], *meta[db].values()]:
        task.update(
            label_horizon_seconds=0,
            anchor_schema=schema,
            task_kind="autocomplete",
            task_view="autocomplete",
            unified_task_kind="autocomplete",
            functional=record["family_id"],
        )
        for key in list(task):
            if key.startswith("forecast_") or key == "label_history_rows":
                task.pop(key)
    (root / "task_metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    (root / "anchor_schema.json").write_text(json.dumps(schema, indent=1) + "\n")
    names = [
        f"{c} of {p.stem}"
        for p in sorted((root / "db").glob("*.parquet"))
        for c in pq.read_schema(p).names
    ]
    (root / "column_index.json").write_text(
        json.dumps({n: i for i, n in enumerate(names)}) + "\n"
    )
    digest = hashlib.sha256()
    for p in sorted((root / "db").glob("*.parquet")):
        t = pq.read_table(p)
        digest.update(
            p.stem.encode()
            + "".join(column_value_digest(t[c]) for c in t.column_names).encode()
        )
    record = dict(
        record,
        db_name=db,
        recipe=RECIPE,
        content_sha256=digest.hexdigest(),
        bytes=sum(p.stat().st_size for p in (root / "db").glob("*.parquet")),
    )
    (root / "a5_world.json").write_text(json.dumps(record, indent=1, default=_json) + "\n")
    write_public_window(root, record)
    native._write_integrity_anchor(root)
    write_observed_availability(root, record)
    return record


def _label_table(ids, pub, y, fact_ids=None):
    cols = {"row_idx": np.arange(len(pub), dtype=np.int64), **ids}
    if fact_ids is not None:
        cols["fact_id"] = fact_ids.astype(np.int64)
    return {**cols, "date": _ts(pub), "label_value": y}


def _save_private(private_dir, db, **arrays):
    Path(private_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(private_dir) / f"{db}.npz", **arrays)



def write_public_window(root, record):
    """Expose F4's computation window without changing outcome maturity."""
    if record.get("group") != "F4":
        return None
    horizon = record["process"]["windows"]["w1"]
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("positive public window in integer seconds required")
    path = Path(root) / "task_metadata.json"
    meta = json.loads(path.read_text())
    for task in [*meta["tasks"], *meta[record["db_name"]].values()]:
        if task["label_horizon_seconds"] != 0 or task["task_kind"] != "autocomplete":
            raise ValueError("public computation window must not alter maturity/task kind")
        task["public_task_horizon_seconds"] = horizon
    text = json.dumps(meta, indent=1) + "\n"
    if path.read_text() != text:
        pending = path.with_suffix(".tmp")
        pending.write_text(text)
        pending.replace(path)
    return horizon


def write_observed_availability(root, record):
    """Publish each tick at its latest actual label publication; Parquet values stay unchanged."""
    import pyarrow.parquet as pq

    root = Path(root)
    path = root / "observed_target_availability.json"
    task = pq.read_table(root / "db" / f"{TASK_TABLE}.parquet", columns=["date"])
    cut = task["date"].cast("int64").to_numpy() // 10**9
    if record["group"] == "F4":
        times = np.unique(cut)
        pub = times + 1
    else:
        fact = pq.read_table(root / "db" / f"{record['event_table']}.parquet", columns=["date"])
        lab = pq.read_table(root / "db" / f"{LABEL_TABLE}.parquet", columns=["date", "fact_id"])
        ftime = fact["date"].cast("int64").to_numpy() // 10**9
        ltime = lab["date"].cast("int64").to_numpy() // 10**9
        fids = lab["fact_id"].to_numpy()
        times, ix = np.unique(ftime[fids], return_inverse=True)
        pub = np.full(len(times), np.iinfo(np.int64).min, dtype=np.int64)
        np.maximum.at(pub, ix, ltime)
        needed = np.isin(times, np.unique(cut))
        times, pub = times[needed], pub[needed]
    if not np.array_equal(times, np.unique(cut)) or not (pub > times).all():
        raise ValueError("fact/support publication coverage mismatch")
    document = dict(
        schema_version=1,
        kind="observed_target_availability",
        policy="maximum_publication_per_fact_time_v1",
        targets=[
            dict(
                table=TASK_TABLE,
                target=TARGET,
                cutoffs_seconds=times.tolist(),
                available_seconds=pub.tolist(),
            )
        ],
    )
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(document) + "\n")
    pending.replace(path)
    return document


