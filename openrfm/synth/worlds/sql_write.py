"""Write event-stream forecast worlds as relational databases."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from openrfm.synth.worlds.sql import NAMES, OPERATORS


def write_world(
    raw_root, seed, slot, entities=256, *, data, db_name=None
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from openrfm.synth.worlds import panel as w

    db = db_name or f"rel-synthetic-{8_300_000 + slot}"
    root = Path(raw_root) / db
    (root / "db").mkdir(parents=True, exist_ok=False)
    n = len(data["cases"])
    ts = lambda days: pa.array(
        ((w.EPOCH_S + np.asarray(days) * 86400) * 1e9).astype(np.int64), pa.timestamp("ns")
    )
    w._write_table(
        root / "db" / "parent.parquet",
        {"parent_id": np.arange(16), "feature_0": data["parent_values"]},
        "parent_id",
        {},
    )
    nuisance = np.random.default_rng([seed, slot, 83008]).normal(size=entities)[
        data["entities"]
    ]
    # Native identity admission requires at least one payload cell on the
    # declared entity. The horizon is public task metadata, not a statistic:
    # its column name survives world-wise normalization of constant values.
    w._write_table(
        root / "db" / "cases.parquet",
        {
            "case_id": np.arange(n),
            "parent_id": data["parents"][data["entities"]],
            "feature_0": nuisance,
            f"declared_horizon_{int(data['horizon'])}_days": np.ones(n),
        },
        "case_id",
        {"parent_id": "parent"},
    )
    for relation, peer, labels in [
        ("own_events", False, False),
        ("own_labels", False, True),
        ("hub_events", True, False),
        ("peer_links", True, True),
    ]:
        case_ids, times, values, categories = [], [], [], []
        for cid, (entity, cutoff, own, other) in enumerate(data["cases"]):
            history = other if peer else own
            if not labels and not peer:
                stream = data["streams"][entity]
                future_ix = (stream["times"] >= cutoff) & (
                    stream["times"] <= cutoff + 4 * data["horizon"]
                )
                history = {
                    key: np.concatenate([history[key], stream[key][future_ix]])
                    for key in ("times", "values", "categories")
                }
            ds = history["label_times" if labels else "times"]
            case_ids.extend([cid] * len(ds))
            times.extend(ds)
            values.extend(history["labels" if labels else "values"])
            if not labels:
                categories.extend(history["categories"])
        cols = {
            "row_idx": np.arange(len(times)),
            "case_id": case_ids,
            "date": ts(times),
            "label_value" if labels else "feature_0": values,
        }
        if not labels:
            cols["hub_category"] = categories
        w._write_table(
            root / "db" / f"case_{relation}.parquet",
            cols,
            "row_idx",
            {"case_id": "cases"},
            "date",
        )
    w._write_table(
        root / "db" / "a5_task_0.parquet",
        {
            "row_idx": np.arange(n),
            "entity_id": np.arange(n),
            "date": ts([c[1] for c in data["cases"]]),
            "a5_y_0": data["labels"],
        },
        "row_idx",
        {"entity_id": "cases"},
        "date",
    )
    metadata = w.history_task_metadata(db, "drift", data.get("task_type", "classification"))
    for task in metadata["tasks"]:
        task["label_horizon_seconds"] = int(data["horizon"] * 86400)
    for task in metadata[db].values():
        task["label_horizon_seconds"] = int(data["horizon"] * 86400)
    (root / "task_metadata.json").write_text(json.dumps(metadata, indent=1) + "\n")
    names = [
        f"{c} of {q.stem}"
        for q in sorted((root / "db").glob("*.parquet"))
        for c in pq.read_schema(q).names
    ]
    (root / "column_index.json").write_text(
        json.dumps({name: i for i, name in enumerate(names)}) + "\n"
    )
    schema = dict(
        version=1,
        horizon_seconds=int(data["horizon"] * 86400),
        own_event={
            "tables": ["case_own_events"],
            "value_column": "feature_0",
            "center": 0,
            "scale": 1,
        },
        own_label={
            "tables": ["case_own_labels"],
            "value_column": "label_value",
            "center": 0,
            "scale": 1,
            "binary_positive": True,
        },
        peer_event={
            "tables": ["case_hub_events"],
            "value_column": "feature_0",
            "center": 0,
            "scale": 1,
        },
        peer_label={
            "tables": ["case_peer_links"],
            "value_column": "label_value",
            "center": 0,
            "scale": 1,
            "binary_positive": True,
        },
    )
    for task in metadata["tasks"]:
        task["anchor_schema"] = schema
    for task in metadata[db].values():
        task["anchor_schema"] = schema
    (root / "task_metadata.json").write_text(json.dumps(metadata, indent=1) + "\n")
    (root / "anchor_schema.json").write_text(json.dumps(schema, indent=1) + "\n")
    record = dict(
        db_name=db,
        slot=slot,
        seed=seed,
        recipe="repair_streams_v1",
        family=data["group"],
        cases=n,
        operator=data["operator"],
        residual=data["residual"],
        rendering=data["rendering"],
        horizon_seconds=int(data["horizon"] * 86400),
        anchor_schema=schema,
        rank_q=1,
        rank={"q": 1, "recipe": "repair_streams_v1"},
        admission=data["admission"],
        forecast_auc=data["forecast_auc"],
        encoder_hidden_columns=[],
    )
    (root / "a5_world.json").write_text(json.dumps(record, indent=1) + "\n")
    w._write_integrity_anchor(root)
    np.savez_compressed(
        root / "repair_private.npz",
        facts=data["facts"],
        labels=data["labels"],
        score=data["score"],
        entities=data["entities"],
        future=data["future"],
        future_valid=data["future_valid"],
        fact_names=np.asarray(OPERATORS + ("parent_value",)),
    )
    return record


def export(data, root, seed, slot, entities, db):
    r = write_world(root, seed, slot, entities, data=data, db_name=db)
    p = root / db
    meta = json.loads((p / "task_metadata.json").read_text())
    if data.get("template") == "next_bucket":
        for t in meta["tasks"]:
            t.update(num_classes=3, task_type="multiclass_classification")
        for t in meta[db].values():
            t.update(num_classes=3, task_type="multiclass_classification")
        (p / "task_metadata.json").write_text(json.dumps(meta, indent=1))
    r.update(template=data.get("template"), peer=data.get("peer", False))
    (p / "a5_world.json").write_text(json.dumps(r, indent=1))
    # Dedicated, consistently named probe facts; never appear in Parquet.
    np.savez_compressed(
        p / "probe_private.npz",
        facts=data["facts"],
        labels=data["labels"],
        entities=data["entities"],
        names=NAMES,
    )
    return r

