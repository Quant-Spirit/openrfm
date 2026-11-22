"""Load converted database archives into sampler graphs."""
import json
from collections import OrderedDict
from pathlib import Path

import torch

from openrfm.archive.admission import admit_roots, admit_shared, bind_metadata, forecast_roots
from openrfm.archive.decode import decode_archive
from openrfm.archive.format import compact_tensor, take, widen_index, widen_object
from openrfm.archive.policy import observed_availability
from openrfm.sampler.retrieval import STATIC, UNKNOWN, RTGraph, _assert

PAYLOAD_KEYS = ("cell_types", "cell_numbers", "cell_datetimes", "cell_booleans", "cell_text_ids", "cell_codes",
                "cell_num_classes")


def _build_parent_slots(fptr, fdst, n, parent_slots, device):
    """[n, parent_slots] foreign-key parents of each row, -1 padded."""
    fptr, fdst = fptr.to(device).long(), fdst.to(device).long()
    rows = torch.repeat_interleave(torch.arange(n, device=device), fptr[1:] - fptr[:-1])
    parents = torch.full((n, parent_slots), -1, dtype=torch.long, device=device)
    parents[rows, torch.arange(len(fdst), device=device) - fptr[rows]] = fdst
    return parents


def _resident_bytes(decoded, directed, shared, parent_slots):
    """(bytes of a full device residency, upper bound of the per-task bytes added to it)."""
    needed = [*decoded["common"], decoded["nodes"], decoded["embeddings"],
              *(decoded["values"][k] for k in PAYLOAD_KEYS), *directed.tensors(), *shared.tensors()]
    n = len(decoded["row_time"])
    dynamic = 25 * len(decoded["cols"]) + (4 * n + 1) * 8 + len(shared.table_ids) * 8 + 8
    return sum(x.numel() * x.element_size() for x in needed) + n * parent_slots * 8 + dynamic, dynamic


def _check_device_memory(device, state, resident, dynamic, reserve):
    admission = resident if state.get("device_key") != str(device) else dynamic
    free, _ = torch.cuda.mem_get_info(device)
    if admission + reserve > free:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(device)
    if admission + reserve > free:
        raise RuntimeError(f"RT graph needs {admission} new resident bytes plus {reserve} reserve; free={free}")


def _make_resident(state, decoded, directed, device, parent_slots):
    """Moves the shared graph, payload and directed CSR of the archive to `device`."""
    cuda = device.type == "cuda"
    values = decoded["values"]
    state["device_key"] = str(device)
    state["common_device"] = tuple(widen_index(x.to(device)) if cuda else x.to(device) for x in decoded["common"])
    payload = {k: values[k].to(device) for k in PAYLOAD_KEYS}
    payload.update(nodes=decoded["nodes"].to(device), embeddings=decoded["embeddings"].to(device))
    payload["parents"] = _build_parent_slots(decoded["fptr"], decoded["fdst"], len(decoded["row_time"]),
                                             parent_slots, device)
    state["payload_device"] = payload
    state["directions_device"] = type(directed)(*(x.to(device) for x in directed.tensors()))
    state["shared_composition_device"] = {}


def _label_availability(root, state, cell_rows, label, target_column, spec):
    """Availability time of every cell: its row time, or for a target cell its label maturity (unknown without a
    horizon, row time plus horizon with one, or the publication time the archive records)."""
    available = torch.where(label, UNKNOWN, cell_rows)
    if spec.label_horizon_seconds is not None:
        known = label & (cell_rows != STATIC)
        _assert(~(known & (cell_rows > UNKNOWN - spec.label_horizon_seconds)).any(), "label maturity overflow")
        available = torch.where(known, cell_rows + spec.label_horizon_seconds, available)
    observed = Path(root) / "observed_target_availability.json"
    if observed.is_file():
        if "observed_target_availability" not in state:
            state["observed_target_availability"] = json.loads(observed.read_text())
        index = json.loads((Path(root) / "column_index.json").read_text())
        publication = observed_availability(cell_rows, label, target_column, index,
                                            state["observed_target_availability"])
        available = torch.where(label, publication, available)
    return available


def load_rt_archive(root, embedding_model, d_text, *, device, spec, target_column, drops, parent_slots=5,
                    decoded_state=None):
    """Load a task graph, payload, eligible query roots, and estimated resident bytes."""
    state = {} if decoded_state is None else decoded_state
    if "cpu" not in state:
        state["cpu"] = decode_archive(root, embedding_model, d_text, parent_slots)
    decoded = state["cpu"]
    state.setdefault("admitted_cpu", OrderedDict())
    fmt = decoded["archive_format"]
    directed, shared = admit_shared(decoded, state, spec)
    resident, dynamic = _resident_bytes(decoded, directed, shared, parent_slots)
    device = torch.device(device)
    if device.type == "cuda":
        _check_device_memory(device, state, resident, dynamic, spec.reserve_bytes)
    if state.get("device_key") != str(device):
        _make_resident(state, decoded, directed, device, parent_slots)
    common = state["common_device"]
    nodes, cols = widen_index(state["payload_device"]["nodes"]), widen_index(common[7])
    key, roots = admit_roots(decoded, state, spec, target_column, drops, nodes, cols, device)

    cell_rows = take(common[4], nodes.long()).long()
    label = cols == target_column
    available = _label_availability(root, state, cell_rows, label, target_column, spec)
    declared = torch.isin(cols, torch.tensor(drops, dtype=torch.int64, device=device))
    label, drop = label.long(), declared.long()
    if fmt == "v2" and device.type == "cpu":
        available, label, drop = (compact_tensor(x) for x in (available, label, drop))
    graph = object.__new__(RTGraph)
    for name, value in zip(RTGraph.__dataclass_fields__, (*common, available, label, drop), strict=False):
        object.__setattr__(graph, name, value)

    payload = dict(state["payload_device"], declared_drop=declared, archive_format=fmt)
    bind_metadata(root, state, payload, decoded, device)
    payload["directions"] = state["directions_device"]
    resident_shared = state["shared_composition_device"]
    if spec not in resident_shared:
        moved = shared.to(device)
        resident_shared[spec] = widen_object(moved) if device.type == "cuda" else moved
    payload["composition"] = resident_shared[spec].bind_task(graph, payload["directions"], payload, spec)
    if spec.task_kind == "forecast":
        roots = forecast_roots(root, state, key, graph, payload["composition"], spec, target_column)
    return graph, payload, roots, resident
