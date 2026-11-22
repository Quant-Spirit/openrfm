"""Constants and the shared table, task-metadata and integrity writers of panel worlds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from openrfm.synth.io import _write_parquet, column_value_digest

RECIPE = "panel_worlds_v1"
STAGE = "s1"
WORLD_FILE = "a5_world.json"
EPOCH_S = 946_684_800
H_DAYS = 30
HORIZON_SECONDS = H_DAYS * 86400
PEER_ROWS, HUB_ROWS = 16, 16
PARENT_RANGE = (4, 16)
MAX_ATTEMPTS = 16
RATE_BOUNDS = (0.15, 0.85)
HISTORY_FAMILIES = ("drift", "regime", "meanrev", "window", "seasonal", "nohist")
HISTORY_RECIPE = "history_worlds_v2"
HISTORY_STEPS, HISTORY_STEP_DAYS, HISTORY_HORIZON_DAYS, HISTORY_BASE_DAY = 80, 7, 5, 360
HISTORY_OWN_EVENT_ROWS, HISTORY_OWN_LABEL_ROWS, HISTORY_MAX_EVENTS, HISTORY_EVENT_WINDOW_DAYS = 160, 96, 10, 28
HISTORY_ENTITY_RANGE, HISTORY_ROWS_RANGE = (200, 400), (5, 12)
HISTORY_PRED_NAMES = (
    "lab_all",
    "lab_last1",
    "lab_last4",
    "lab_last16",
    "ev_count28",
    "ev_recency",
    "raw_all",
    "raw_last4",
    "raw_last16",
    "raw_last64",
)
HISTORY_WINDOW_PREDS = ("lab_last4", "lab_last16", "raw_last4", "raw_last16")
HISTORY_ALL_PREDS = ("lab_all", "raw_all")
HISTORY_GAIN_MIN, HISTORY_NOHIST_GAIN_MAX, HISTORY_WINDOW_GAP_MIN = 0.05, 0.01, 0.03
HISTORY_SUPPLY_MEDIAN_MIN, HISTORY_COLD_SHARE_MIN = 16, 0.10


def _write_table(
    path: Path,
    values: Mapping[str, Any],
    pkey: str,
    fkeys: Mapping[str, str],
    time_col: str | None = None,
) -> None:
    import pyarrow as pa

    _write_parquet(path, pa.table(values), pkey=pkey, fkeys=fkeys, time_col=time_col)


def task_metadata(db: str, family: str | None = None) -> dict[str, Any]:
    task = dict(
        table="a5_task_0",
        target_column="a5_y_0",
        task_type="binary_classification",
        sem_type="discrete",
        num_classes=2,
        task_kind="forecast",
        task_view="forecast",
        forecast_entity_table="cases",
        forecast_entity_fk="entity_id",
        forecast_event_table="case_own_events",
        forecast_cutoff="row_timestamp",
        cutoff_column="date",
        time_legality="strict_prior",
        label_history_rows=[1, 1],
        label_horizon_seconds=HORIZON_SECONDS,
        drop_columns=[],
        world_source={"recipe": RECIPE, **({"family": family} if family else {})},
    )
    target = task["target_column"]
    view = dict(task, entity_table="a5_task_0", target_col=target)
    return {"schema_version": 1, "tasks": [task], "exact_aggregates": [], db: {target: view}}


def _write_integrity_anchor(root: Path) -> None:
    """Record per-column digests for the base tables and published targets."""
    import pyarrow.parquet as pq

    target = "a5_y_0"
    base, base_hashes, target_hashes = {}, {}, {}
    for path in sorted((root / "db").glob("*.parquet")):
        table = pq.read_table(path)
        for name in table.column_names:
            if name == target:
                target_hashes.setdefault(path.stem, {})[name] = column_value_digest(table[name])
            else:
                base.setdefault(path.stem, []).append(name)
                base_hashes.setdefault(path.stem, {})[name] = column_value_digest(table[name])
    (root / "generator_integrity.json").write_text(
        json.dumps(
            dict(
                schema_version=2,
                kind="real_world_anchor",
                written_by="openrfm.synth.worlds.panel",
                recipe=RECIPE,
                opaque_target=target,
                opaque_targets=[target],
                immutable_base_columns=base,
                immutable_base_value_sha256=base_hashes,
                published_target_value_sha256=target_hashes,
                checks={},
            ),
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )


def history_task_metadata(db: str, family: str, kind: str) -> dict[str, Any]:
    meta = task_metadata(db, family)
    task = meta["tasks"][0]
    task["world_source"]["history_recipe"] = HISTORY_RECIPE
    task["label_horizon_seconds"] = HISTORY_HORIZON_DAYS * 86400
    if kind == "regression":
        task.update(task_type="regression", sem_type="number")
        task.pop("num_classes")
    meta[db] = {"a5_y_0": dict(task, entity_table="a5_task_0", target_col="a5_y_0")}
    return meta
