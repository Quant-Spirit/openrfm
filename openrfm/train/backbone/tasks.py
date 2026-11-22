"""The training tasks of the corpus: one per (world, task table, target column), in mixture groups."""
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Task:
    db: str
    table: str
    target: str
    regression: bool
    target_column: int
    drop_columns: tuple[int, ...]
    node_offset: int
    num_nodes: int
    # Declared forecast view: the entity table key ('<table>:Db') and the label window in seconds.
    forecast_entity: str | None
    label_horizon: int | None
    # Window of the published target (``public_task_horizon_seconds``), else the label window.
    public_horizon: int | None
    world: str


@lru_cache(maxsize=None)
def _json(path: Path):
    return json.loads(path.read_text())


def _task_entry(root: Path, table: str, target: str) -> dict:
    entries = [e for e in _json(root / "task_metadata.json").get("tasks", [])
               if (e.get("table", e.get("entity_table")), e.get("target_column", e.get("target_col"))) == (table, target)]
    if len(entries) != 1:
        raise ValueError(f"{root.name}: task {table}/{target} is not declared exactly once")
    return entries[0]


def _context_drops(columns_by_table: dict, table: str, target: str) -> list:
    """Columns hidden from every episode of this target: relation-null, stream and aggregate readout families."""
    def readout(name, stems):
        return any(name == stem or name.startswith(stem + "_") for stem in stems)
    sj = [f + suffix for f in ("satfeat", "satnull", "junmix", "juncount", "hubhist", "hubnull", "xhub", "xhub_add",
                               "streamrec", "streamnull", "hubrate", "hubrate_null", "hubhistj", "hubhistj_null")
          for suffix in ("_r", "_k2")]
    null = target.startswith(("relnull_r", "relnull_k2", "relnull_imb"))
    aggregate = readout(target, ("aggcol_r", "aggcol_k2", "aggcol_imb"))
    drops = []
    for other, columns in columns_by_table.items():
        for column in columns:
            if other == table and column == target:
                continue
            allowed = column.startswith("relnull_evx_") or (other == table and column.startswith("relnull_x_"))
            privileged = other == "sj_stream_evidence_2" and target.startswith(("streamrec_", "streamnull_"))
            if (privileged or ((column.startswith("sj_") or readout(column, sj)) and not readout(target, sj))
                    or (null and not allowed) or (not null and column.startswith("relnull_"))
                    or (not aggregate and column.startswith("aggcol_"))):
                drops.append(column if other == table else [other, column])
    return drops


def _drop_columns(root: Path, table: str, target: str) -> list:
    """The declared drop columns of the target, then the context drops not already declared."""
    columns_by_table: dict[str, list[str]] = {}
    for name in sorted(_json(root / "column_index.json")):
        if " of " in name:
            column, owner = name.split(" of ", 1)
            columns_by_table.setdefault(owner, []).append(column)
    declared = list(_task_entry(root, table, target).get("drop_columns", []))
    return declared + [c for c in _context_drops(columns_by_table, table, target) if c not in declared]


def _task(corpus: Path, archives: Path, db: str, table: str, target: str, kind: str) -> Task:
    root = archives / db
    index = _json(root / "column_index.json")
    drops = tuple(index[f"{c[1]} of {c[0]}" if isinstance(c, list) else f"{c} of {table}"]
                  for c in _drop_columns(root, table, target))
    tables = _json(root / "table_info.json")
    info = tables.get(f"{table}:Db") or tables[f"{table}:Train"]
    entry = _task_entry(root, table, target)
    forecast = entry.get("task_view", "autocomplete") == "forecast"
    horizon = entry.get("label_horizon_seconds") if forecast else None
    public = entry.get("public_task_horizon_seconds")
    return Task(db, table, target, kind == "regression", index[f"{target} of {table}"], drops,
                int(info["node_idx_offset"]), int(info["num_nodes"]),
                f"{entry['forecast_entity_table']}:Db" if forecast else None, horizon,
                horizon if public is None else public, os.path.relpath(os.path.normpath(archives / os.readlink(root)), corpus))


def corpus_groups(corpus: Path, archives: Path, families: list) -> tuple[list[list[Task]], list[float]]:
    """One mixture group per catalog family in ``families`` order (classification first), and its weights."""
    catalog = _json(archives / "compact_task_catalog.json")["records"]
    unknown = {r["component"] for r in catalog} - {family for family, _ in families}
    if unknown:
        raise ValueError(f"catalog families missing from the configuration: {sorted(unknown)}")
    groups, weights = [], []
    for family, weight in families:
        selected = sorted((((r["db_name"], r["table"], r["target"]), r["task_type"]) for r in catalog
                           if r["component"] == family), key=lambda item: (item[1] != "classification", item[0]))
        if selected:
            groups.append([_task(corpus, archives, *key, kind) for key, kind in selected])
            weights.append(float(weight))
    return groups, weights
