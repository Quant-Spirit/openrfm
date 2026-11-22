"""Write relational event-stream worlds with one exported row per physical event."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from openrfm.synth.worlds.unified import FACT_NAMES, _layout, generate_world


def write_world(raw_root, seed, slot, entities=None, config=None, data=None):
    """Export each physical stream event once, with its relational identity."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from openrfm.synth.worlds import panel as native

    d = data if data is not None else generate_world(seed, slot, entities, config)
    db = f"rel-synthetic-{8_400_000 + slot}"
    root = Path(raw_root) / db
    (root / "db").mkdir(parents=True, exist_ok=False)
    ts = lambda t: pa.array(
        ((native.EPOCH_S + np.asarray(t) * d["seconds_per_unit"]) * 1e9).astype(np.int64),
        pa.timestamp("ns"),
    )
    write = lambda name, columns, pkey, fkeys, time=None: native._write_table(
        root / "db" / f"{name}.parquet", columns, pkey, fkeys, time
    )
    rng = np.random.default_rng([seed, slot, 84010])
    layout = _layout(seed, slot, len(d["streams_by_relation"]))
    nentities, nhubs = len(d["attributes"]), len(d["parent_values"])
    if layout["compact"]:
        # The same hierarchy in one self-referencing table; hub/ancestor
        # rows have stable disjoint IDs, and focal entity IDs are unchanged.
        features = np.r_[d["attributes"], d["parent_values"], d["ancestor_values"]]
        parent_ids = [int(nentities + p) for p in d["parents"]]
        parent_ids += [int(nentities + nhubs + a) for a in d["ancestor_ids"]]
        # Root IDs self-reference: native integer integrity hashing rejects nulls.
        parent_ids += list(
            range(nentities + nhubs, nentities + nhubs + len(d["ancestor_values"]))
        )
        cols = {
            "entity_id": np.arange(len(features)),
            "parent_id": pa.array(parent_ids, type=pa.int64()),
            "feature_0": features,
        }
        entity_fkeys = {"parent_id": "entities"}
    else:
        write(
            "ancestors",
            {
                "ancestor_id": np.arange(len(d["ancestor_values"])),
                "feature_0": d["ancestor_values"],
            },
            "ancestor_id",
            {},
        )
        write(
            "parents",
            {
                "parent_id": np.arange(nhubs),
                "ancestor_id": d["ancestor_ids"],
                "feature_0": d["parent_values"],
            },
            "parent_id",
            {"ancestor_id": "ancestors"},
        )
        features = d["attributes"]
        cols = {
            "entity_id": np.arange(nentities),
            "parent_id": d["parents"],
            "feature_0": features,
        }
        entity_fkeys = {"parent_id": "parents"}
    for j in range(1, d["width"]):
        cols[f"feature_{j}"] = rng.normal(size=len(features))
    if rng.random() < d["config"]["text_probability"]:
        cols["description"] = [
            f"type {int(x > 0)} group {i % 4}" for i, x in enumerate(features)
        ]
    for j in range(layout["extra_tables"]):
        count = int(rng.integers(4, 33))
        name = f"context_entities_{j}"
        key = f"context_id_{j}"
        write(name, {key: np.arange(count), "feature_0": rng.normal(size=count)}, key, {})
        cols[key] = rng.integers(count, size=len(features))
        entity_fkeys[key] = name
    write("entities", cols, "entity_id", entity_fkeys)
    for relation, streams in enumerate(d["streams_by_relation"]):
        source = np.concatenate(
            [np.full(len(st["times"]), e, dtype=np.int64) for e, st in enumerate(streams)]
        )
        times = np.concatenate([st["times"] for st in streams])
        values = np.concatenate([st["values"] for st in streams])
        categories = np.concatenate([st["categories"] for st in streams])
        columns = {
            "row_idx": np.arange(len(times)),
            "entity_id": source,
            "date": ts(times),
            "feature_0": values,
            "category": categories,
        }
        for j in range(1, d["width"]):
            columns[f"feature_{j}"] = rng.normal(size=len(times))
        write(f"events_{relation}", columns, "row_idx", {"entity_id": "entities"}, "date")
    label_times = np.concatenate(d["history_times"])
    label_sources = np.concatenate(
        [np.full(len(t), e, dtype=np.int64) for e, t in enumerate(d["history_times"])]
    )
    write(
        "label_history",
        {
            "row_idx": np.arange(len(label_times)),
            "entity_id": label_sources,
            "date": ts(label_times),
            "label_value": np.concatenate(d["history_values"]),
        },
        "row_idx",
        {"entity_id": "entities"},
        "date",
    )
    n = len(d["cases"])
    ids = d["entities"]
    horizon_seconds = (
        int(round(d["horizon"] * d["seconds_per_unit"]))
        if d["task_kind"] == "future_window"
        else 0
    )
    target = {
        "row_idx": np.arange(n),
        "entity_id": ids,
        "date": ts(d["cutoffs"]),
        "a5_y_0": d["labels"],
        f"declared_horizon_{horizon_seconds}_seconds": np.ones(n),
    }
    if d["task_kind"] == "autocomplete":
        target["observed_attribute"] = d["target_covariates"]
    target_fkeys = {"entity_id": "entities"}
    if d["parent_shortcut"]:
        target["parent_id"] = d["parents"][ids] + (nentities if layout["compact"] else 0)
        target_fkeys["parent_id"] = "entities" if layout["compact"] else "parents"
    write("a5_task_0", target, "row_idx", target_fkeys, "date")
    event_table = f"events_{d['target_relation']}"
    schema = dict(
        version=1,
        horizon_seconds=int(round(d["horizon"] * d["seconds_per_unit"])),
        own_event=dict(tables=[event_table], value_column="feature_0", center=0, scale=1),
        own_label=dict(
            tables=["label_history"],
            value_column="label_value",
            center=0,
            scale=1,
            binary_positive=d["prediction_type"] == "classification",
        ),
        peer_event=dict(tables=[event_table], value_column="feature_0", center=0, scale=1),
        peer_label=dict(
            tables=["label_history"],
            value_column="label_value",
            center=0,
            scale=1,
            binary_positive=d["prediction_type"] == "classification",
        ),
    )
    meta = native.history_task_metadata(db, "unified", d["prediction_type"])
    for task in [*meta["tasks"], *meta[db].values()]:
        task.update(
            label_horizon_seconds=horizon_seconds,
            anchor_schema=schema,
            task_kind="forecast" if d["task_kind"] == "future_window" else "autocomplete",
            task_view="forecast" if d["task_kind"] == "future_window" else "autocomplete",
            unified_task_kind=d["task_kind"],
            functional=d["functional"],
            forecast_entity_table="entities",
            forecast_entity_fk="entity_id",
            forecast_event_table=event_table,
        )
        if d["task_kind"] == "autocomplete":
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
        json.dumps({name: i for i, name in enumerate(names)}) + "\n"
    )
    record = dict(
        db_name=db,
        slot=slot,
        seed=seed,
        recipe=d["config"]["version"],
        family=d["task_kind"],
        task_kind=d["task_kind"],
        functional=d["functional"],
        prediction_type=d["prediction_type"],
        cases=n,
        horizon_seconds=horizon_seconds,
        anchor_schema=schema,
        rank_q=1,
        rank={"q": 1, "recipe": d["config"]["version"]},
        rendering="timestamps",
        config=d["config"],
        process=d["process"],
        encoder_hidden_columns=[],
        event_storage="unique physical stream rows; strict-prior native filter required",
        layout=layout,
    )
    (root / "a5_world.json").write_text(json.dumps(record, indent=1) + "\n")
    native._write_integrity_anchor(root)
    np.savez_compressed(
        root / "unified_private.npz",
        facts=d["facts"],
        labels=d["labels"],
        entities=ids,
        tiers=d["tiers"],
        cutoffs=d["cutoffs"],
        fact_names=np.asarray(FACT_NAMES),
    )
    return record
