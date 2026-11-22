"""Generate denormalised task and event columns from strictly earlier events."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openrfm.synth.worlds import panel as native
from openrfm.synth.worlds.unified import PriorConfig, generate_world
from openrfm.synth.worlds.unified_write import write_world

NAMES = (
    "count",
    "sum",
    "mean",
    "last",
    "age",
    "distinct",
    "window_count",
    "window_sum",
    "peer_count",
    "peer_sum",
    "peer_last",
    "peer_window_count",
)


def values(own, peer, cutoff, horizon):
    def before(stream):
        visible = stream["times"] < cutoff
        return (
            stream["times"][visible],
            stream["values"][visible],
            stream["categories"][visible],
        )

    times, marks, categories = before(own)
    peer_times, peer_marks, _ = before(peer)
    in_window = times > cutoff - horizon
    peer_in_window = peer_times > cutoff - horizon
    return np.array(
        [
            len(times),
            marks.sum(),
            marks.mean() if len(marks) else 0.0,
            marks[-1] if len(marks) else 0.0,
            cutoff - times[-1] if len(times) else cutoff,
            len(np.unique(categories)),
            in_window.sum(),
            marks[in_window].sum(),
            len(peer_times),
            peer_marks.sum(),
            peer_marks[-1] if len(peer_marks) else 0.0,
            peer_in_window.sum(),
        ],
        float,
    )


def world(root, slot):
    cfg = PriorConfig(
        entities=96,
        cutoffs_per_entity=4,
        width_range=(1, 3),
        relation_count_range=(1, 1),
        mean_events_range=(4.0, 384.0),
        text_probability=0.0,
    )
    d = generate_world(
        20261009, slot, config=cfg, task_kind="future_window", functional="count"
    )
    # Each target world fixes one independently sampled window and denormalised
    # field. All snapshot columns have the same lawful row-time availability.
    peer = np.array([p[0] if p else e for e, p in enumerate(d["peers"])])
    chosen = slot % len(NAMES)
    streams = d["streams"]
    crng = np.random.default_rng([slot, 71002])
    classes = int(crng.integers(3, 33))
    for s in streams:
        if len(s["times"]) > 1024:
            keep = np.sort(crng.choice(len(s["times"]), 1024, replace=False))
            for key in ("times", "values", "categories"):
                s[key] = s[key][keep]
        seconds, keep = np.unique(
            np.floor(s["times"] * d["seconds_per_unit"]).astype(np.int64), return_index=True
        )
        for key in ("values", "categories"):
            s[key] = s[key][keep]
        s["times"] = seconds / d["seconds_per_unit"]
        s["categories"] = crng.integers(classes, size=len(s["times"]))
    d["cutoffs"] = np.floor(d["cutoffs"] * d["seconds_per_unit"]) / d["seconds_per_unit"]
    derived = np.array(
        [
            values(streams[e], streams[peer[e]], cut, d["horizon"])
            for e, cut in zip(d["entities"], d["cutoffs"], strict=False)
        ]
    )
    d["labels"] = derived[:, chosen]
    d["prediction_type"] = "regression"
    d["task_kind"] = "autocomplete"
    d["target_covariates"] = np.zeros(len(derived))
    empty = np.array([], float)
    d["history_times"] = [empty for _ in d["history_times"]]
    d["history_values"] = [empty for _ in d["history_values"]]
    record = write_world(root, 20261009, slot, data=d)
    # The packer requires nonempty tables. These cases have no label history.
    (root / record["db_name"] / "db/label_history.parquet").unlink(missing_ok=True)
    db = root / record["db_name"]
    task = db / "db/a5_task_0.parquet"
    tab = pq.read_table(task)
    cols = {k: tab[k] for k in tab.column_names if k not in ("observed_attribute",)}

    def clock(t):
        seconds = np.rint(np.asarray(t) * d["seconds_per_unit"]).astype(np.int64)
        return pa.array((native.EPOCH_S + seconds) * np.int64(10**9), pa.timestamp("ns"))

    cols["date"] = clock(d["cutoffs"])
    cols["peer_id"] = pa.array(peer[d["entities"]], pa.int64())
    rng = np.random.default_rng([slot, 71001])
    names = [f"x_{x:08x}" for x in rng.integers(0, 2**32, len(NAMES))]
    for i, name in enumerate(names):
        cols[name] = derived[:, i]
    # Peer FK is schema-derived.
    entity_fkeys = {"entity_id": "entities", "peer_id": "entities"}
    if "parent_id" in cols:
        entity_fkeys["parent_id"] = "entities" if record["layout"]["compact"] else "parents"
    native._write_table(task, cols, "row_idx", entity_fkeys, "date")
    # Dense per-cell exposure also on historical event rows. Each derived value
    # excludes that event and every later event; raw marks remain ordinary cells.
    path = db / "db/events_0.parquet"
    events = pq.read_table(path)
    ecols = {k: events[k] for k in events.column_names}
    ecols["date"] = clock(np.concatenate([s["times"] for s in streams]))
    rows = []
    for e, s in enumerate(streams):
        rows.extend(values(s, streams[peer[e]], t, d["horizon"]) for t in s["times"])
    rows = np.array(rows).reshape(-1, len(NAMES))
    for i, name in enumerate(names):
        ecols[name] = rows[:, i]
    ecols["peer_id"] = pa.array(peer[events["entity_id"].to_numpy()], pa.int64())
    native._write_table(
        path, ecols, "row_idx", {"entity_id": "entities", "peer_id": "entities"}, "date"
    )
    meta = json.loads((db / "task_metadata.json").read_text())
    for entry in [*meta["tasks"], *meta[record["db_name"]].values()]:
        for key in list(entry):
            if key.startswith("forecast_") or key == "label_history_rows":
                entry.pop(key)
        entry.update(
            task_kind="autocomplete",
            task_view="autocomplete",
            label_horizon_seconds=0,
            public_task_horizon_seconds=int(round(d["horizon"] * d["seconds_per_unit"])),
            derived_cells=True,
            drop_columns=names,
        )
    # Hide sibling aggregates on the task row while keeping historical event attributes visible.
    (db / "task_metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    names_index = [
        f"{c} of {p.stem}"
        for p in sorted((db / "db").glob("*.parquet"))
        for c in pq.read_schema(p).names
    ]
    (db / "column_index.json").write_text(
        json.dumps({n: i for i, n in enumerate(names_index)}) + "\n"
    )
    audit = dict(
        recipe="derived cells",
        chosen=NAMES[chosen],
        opaque_columns=names,
        strict_prior=True,
        task_labels_used=False,
        own_and_peer=True,
        event_rows=len(rows),
        target_std=float(d["labels"].std()),
        horizon_seconds=int(round(d["horizon"] * d["seconds_per_unit"])),
    )
    (db / "derived_cells.json").write_text(json.dumps(audit, indent=1) + "\n")
    (db / "unified_private.npz").unlink()
    native._write_integrity_anchor(db)
    if not np.isfinite(derived).all() or not np.isfinite(rows).all():
        raise ValueError("Nonfinite derived columns")
    # Independent strict-time audit on actual parquet clock precision.
    dates = ecols["date"].cast(pa.int64()).to_numpy()
    for e in range(len(streams)):
        ix = events["entity_id"].to_numpy() == e
        if len(dates[ix]) > 1 and not np.all(np.diff(dates[ix] // 10**9) > 0):
            raise ValueError("Native second timestamp collision")
    return audit
