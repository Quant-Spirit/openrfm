"""Task-independent admission of a decoded archive: the directed CSR, the composition metadata, query roots and the
archive metadata a task payload carries."""
import copy
import json
from pathlib import Path

import torch

from openrfm.archive.format import compact_tensor
from openrfm.sampler.composition import CompositionData, forecast_identity_mask
from openrfm.sampler.retrieval import RTGraph
from openrfm.sampler.v2 import DirectedCSR


def cold(value):
    """CPU index storage widened to int32 where uint8/uint16 cannot index."""
    return value.int() if value.dtype in (torch.uint8, torch.uint16) else value


def _compact_object(value):
    result = copy.copy(value)
    for name, item in vars(value).items():
        if torch.is_tensor(item):
            object.__setattr__(result, name, compact_tensor(item))
    return result


def table_names(state, decoded):
    """Table name -> row-table id of every nonempty archived table."""
    if "table_names" not in state:
        tables = cold(decoded["common"][3])
        state["table_names"] = {name: int(tables[int(info["node_idx_offset"])])
                                for name, info in decoded["meta"].items() if int(info["num_nodes"]) > 0}
    return state["table_names"]


def admit_shared(decoded, state, spec):
    """(directed CSR, shared composition metadata of `spec`), cached in `state`."""
    values, fmt = decoded["values"], decoded["archive_format"]
    if "directed_cpu" not in state:
        directed = DirectedCSR.admit(decoded["fptr"], decoded["fdst"], decoded["pptr"], decoded["pdst"],
                                     cold(decoded["row_time"]), device="cpu")
        state["directed_cpu"] = _compact_object(directed) if fmt == "v2" else directed
    directed = state["directed_cpu"]
    by_spec = state.setdefault("shared_composition_cpu", {})
    if spec not in by_spec:
        rows = cold(decoded["row_time"])
        zeros = torch.zeros_like(cold(decoded["cols"]), dtype=torch.int64)
        graph = RTGraph(*(cold(x) for x in decoded["common"]), rows[cold(decoded["nodes"])].long(), zeros, zeros)
        shared = CompositionData.admit_shared(
            graph, type(directed)(*(cold(x) for x in directed.tensors())), {k: cold(v) for k, v in values.items()},
            spec, dict(table_names(state, decoded)), schema_cache=state.setdefault("schema_cpu", {}))
        if fmt == "v2":
            shared = _compact_object(shared)
            shared.types = values["cell_types"]
            state["schema_cpu"] = {k: compact_tensor(v) if torch.is_tensor(v) else v
                                   for k, v in state["schema_cpu"].items()}
            state["schema_cpu"]["text_cardinality"] = shared.cardinality
        by_spec[spec] = shared
    return directed, by_spec[spec]


def admit_roots(decoded, state, spec, target_column, drops, device_nodes, device_cols, device):
    """Archive-ordered rows that carry a `target_column` cell, cached per (target, drops, spec)."""
    key = (target_column, tuple(drops), spec)
    admissions = state["admitted_cpu"]
    if key not in admissions:
        if device.type == "cuda":
            roots = device_nodes.long()[device_cols == target_column].cpu().numpy()
        else:
            roots = cold(decoded["nodes"])[cold(decoded["cols"]) == target_column].numpy()
        if len(roots) == 0:
            raise ValueError("target column has no archived cells")
        if spec.max_resident_task_views and len(admissions) >= spec.max_resident_task_views:
            admissions.popitem(last=False)
        admissions[key] = (roots,)
    admissions.move_to_end(key)
    return key, admissions[key][0]


def forecast_roots(root, state, key, graph, composition, spec, target_column):
    """The admitted roots whose entity identity is visible at the query cutoff."""
    roots, *checked = state["admitted_cpu"][key]
    if checked:
        return roots
    keep = forecast_identity_mask(graph, composition, roots, spec, target_column).cpu().numpy()
    excluded = int((~keep).sum())
    if excluded:
        print(f"[archive] db={Path(root).name} target_column={target_column} excluded {excluded}/{len(roots)} "
              "forecast query rows without a visible entity identity", flush=True)
    roots = roots[keep]
    if len(roots) == 0:
        raise ValueError("no forecast query row has a visible declared entity identity")
    state["admitted_cpu"][key] = (roots, True)
    return roots


def _summary_columns(root):
    derived, world = Path(root) / "derived_cells.json", Path(root) / "a5_world.json"
    names = set(json.loads(derived.read_text())["opaque_columns"]) if derived.is_file() else set()
    if world.is_file():
        names.update(json.loads(world.read_text()).get("summary_columns", []))
    index = json.loads((Path(root) / "column_index.json").read_text())
    return [v for k, v in index.items() if k.split(" of ", 1)[0] in names]


def _history_tables(root, names):
    """(history table ids, same-time table ids, explicit publication history) of a synthetic world."""
    schema_path = Path(root) / "anchor_schema.json"
    schema = json.loads(schema_path.read_text()) if schema_path.exists() else {}
    history = set()
    for role in ("own_label", "peer_label"):
        definition = schema.get(role, {})
        if isinstance(definition, dict):
            history.update(definition.get("tables", []))
    record = json.loads((Path(root) / "a5_world.json").read_text())
    same_time = [names[record["event_table"] + ":Db"]] if record.get("group") in ("F1", "F2", "F3") else []
    return ([v for k, v in names.items() if k.split(":", 1)[0] in history], same_time,
            record.get("group") in ("F1", "F2", "F3", "F4"))


def bind_metadata(root, state, payload, decoded, device):
    """Adds table identities and, for synthetic worlds, summary columns and history tables to `payload`."""
    world = (Path(root) / "a5_world.json").is_file()
    if world or (Path(root) / "derived_cells.json").is_file():
        if "summary_column_ids" not in state:
            state["summary_column_ids"] = _summary_columns(root)
        payload["summary_columns"] = torch.tensor(state["summary_column_ids"], device=device, dtype=torch.long)
    payload["table_names"] = table_names(state, decoded)
    if world:
        if "history_table_ids" not in state:
            (state["history_table_ids"], state["same_time_table_ids"],
             state["explicit_publication_history"]) = _history_tables(root, state["table_names"])
        for key in ("history_table_ids", "same_time_table_ids"):
            payload[key] = torch.tensor(state[key], device=device, dtype=torch.long)
        payload["explicit_publication_history"] = state["explicit_publication_history"]
