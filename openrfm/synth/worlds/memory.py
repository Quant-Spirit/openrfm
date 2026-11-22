"""Generate temporal worlds with labels drawn from event, entity and peer histories."""

import json
from pathlib import Path

import numpy as np
import yaml

from openrfm.synth.worlds.memory_signals import generate


def _write_integrity_anchor(out, files, tasks):
    import pyarrow.parquet as pq

    from openrfm.synth.io import column_value_digest

    base, base_hashes, target_hashes = {}, {}, {}
    targets = {t["target_col"] for t in tasks}
    for path in files:
        table = pq.read_table(path)
        for name in table.column_names:
            digest = column_value_digest(table[name])
            if name in targets:
                target_hashes.setdefault(path.stem, {})[name] = digest
            else:
                base.setdefault(path.stem, []).append(name)
                base_hashes.setdefault(path.stem, {})[name] = digest
    (out / "generator_integrity.json").write_text(
        json.dumps(
            dict(
                schema_version=2,
                kind="real_world_anchor",
                written_by="openrfm.synth.worlds.memory",
                recipe="memory_worlds_v1",
                opaque_targets=sorted(targets),
                immutable_base_columns=base,
                immutable_base_value_sha256=base_hashes,
                published_target_value_sha256=target_hashes,
                checks={},
            )
        )
        + "\n"
    )


def write_world(root, db, seed, component, *, prior=None, signal=None):
    import pyarrow as pa

    from openrfm.synth.worlds.panel import _write_table

    config = yaml.safe_load(Path(component).read_text())
    if config["schema"] != "memory_worlds_v1":
        raise ValueError("unsupported memory world component")
    w = generate(seed, config, prior=prior, signal=signal)
    out = Path(root) / db
    out.mkdir(exist_ok=False)
    (out / "db").mkdir()
    rng = np.random.default_rng([seed, 31])
    # Schema aliases are independently randomized; no carrier names are inputs.
    entity_name, event_name, hub_name = [f"t{v}" for v in rng.choice(10000, 3, replace=False)]
    epoch = 1609459200

    def timestamps(days):
        return pa.array(
            ((epoch + np.asarray(days) * 86400) * 1e9).astype(np.int64), pa.timestamp("ns")
        )

    E = len(w["entity_hub"])
    _write_table(
        out / "db" / f"{hub_name}.parquet",
        {
            "id": np.arange(
                len(w["hub_cells"]) if "hub_cells" in w else w["entity_hub"].max() + 1
            ),
            "x": w["hub_cells"]
            if "hub_cells" in w
            else rng.normal(size=w["entity_hub"].max() + 1),
        },
        "id",
        {},
    )
    _write_table(
        out / "db" / f"{entity_name}.parquet",
        {
            "id": np.arange(E),
            "hub": w["entity_hub"],
            "x": w["entity_cells"] if "entity_cells" in w else rng.normal(size=E),
        },
        "id",
        {"hub": hub_name},
    )
    _write_table(
        out / "db" / f"{event_name}.parquet",
        {
            "id": np.arange(len(w["entity"])),
            "entity": w["entity"],
            "hub": w["event_hub"],
            "date": timestamps(w["time"]),
            **{f"x{j}": w["values"][:, j] for j in range(w["values"].shape[1])},
        },
        "id",
        {"entity": entity_name, "hub": hub_name},
        "date",
    )
    tasks = []
    for index, kind in enumerate(config["generator"]["task_types"]):
        table, target = f"v6_task_{index}", f"v6_y_{index}"
        label = (
            (w["score"] > w.get("threshold", np.median(w["score"])))
            if kind == "classification"
            else w["score"]
        )
        _write_table(
            out / "db" / f"{table}.parquet",
            {
                "id": np.arange(len(label)),
                "entity_id": w["task_entity"],
                "hub": w["entity_hub"][w["task_entity"]],
                "date": timestamps(w["cutoffs"]),
                target: label,
                **(
                    {f"u{j}": w["task_cells"][:, j] for j in range(3)}
                    if "task_cells" in w
                    else {}
                ),
            },
            "id",
            {"entity_id": entity_name, "hub": hub_name},
            "date",
        )
        task = dict(
            db=db,
            table=table,
            entity_table=table,
            target_column=target,
            target_col=target,
            task_type="binary_classification" if kind == "classification" else "regression",
            sem_type="boolean" if kind == "classification" else "number",
            task_view="forecast",
            task_kind="forecast",
            forecast_cutoff="row_timestamp",
            forecast_entity_table=entity_name,
            forecast_entity_fk="entity_id",
            forecast_event_table=event_name,
            cutoff_column="date",
            label_horizon_seconds=w["horizon"] * 86400,
            public_task_horizon_seconds=w["horizon"] * 86400,
            time_legality="strict_prior",
            label_history_rows=config["generator"]["label_history_rows"],
            v6_contract="v6_forecast_v1",
        )
        if kind == "classification":
            task["num_classes"] = 2
        tasks.append(task)
    (out / "task_metadata.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                exact_aggregates=[],
                tasks=tasks,
                **{db: {t["target_col"]: t for t in tasks}},
            )
        )
        + "\n"
    )
    import pyarrow.parquet as pq

    files = sorted((out / "db").glob("*.parquet"))
    columns = [f"{c} of {p.stem}" for p in files for c in pq.read_schema(p).names]
    (out / "column_index.json").write_text(
        json.dumps({c: i for i, c in enumerate(columns)}) + "\n"
    )
    _write_integrity_anchor(out, files, tasks)
    record = dict(
        schema="memory_world_v1",
        db=db,
        seed=seed,
        shared_labels=False,
        signal=signal,
        signal_form=w.get("signal_form"),
        long_world=w.get("long_world"),
        component=config,
        prior=prior,
        rows_per_key=w["counts"].tolist(),
        incidence="independent_balanced_event_hub",
        rows={p.stem: pq.ParquetFile(p).metadata.num_rows for p in files},
        delivery_accepted=False,
        delivery_required=config["delivery"],
    )
    (out / "generation.json").write_text(json.dumps(record) + "\n")
    return record
