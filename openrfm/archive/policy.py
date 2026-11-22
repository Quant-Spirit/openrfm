"""Archive sidecar validation and device admission."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

TAG = "own_time_col_v1"
SIDECAR = "node_time_own_time_col_v1.npz"
POLICY = "node_time_policy.json"
ARCHIVE_STATIC = np.iinfo(np.int32).min


class NodeTimePolicyError(ValueError):
    """The sidecar does not match this bundle."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_node_time_policy(root) -> dict | None:
    """The validated policy of one bundle directory, or None when it has no sidecar."""
    root = Path(root)
    sidecar, policy_path = root / SIDECAR, root / POLICY
    if not sidecar.exists() and not policy_path.exists():
        return None
    if not (sidecar.is_file() and policy_path.is_file()):
        raise NodeTimePolicyError(f"{root}: node-time sidecar and policy must both be present")
    policy = json.loads(policy_path.read_text())
    if policy.get("tag") != TAG:
        raise NodeTimePolicyError(f"{root}: unknown node-time policy {policy.get('tag')!r}")
    if policy.get("table_info_sha256") != _sha256(root / "table_info.json"):
        raise NodeTimePolicyError(f"{root}: node-time sidecar was built for another table_info.json")
    if policy.get("sidecar_sha256") != _sha256(sidecar):
        raise NodeTimePolicyError(f"{root}: node-time sidecar bytes changed after the policy was written")
    return policy


def apply_node_time_policy(root, row_time, *, static, clock_view="availability"):
    """Apply the selected clock to archived row times in place."""
    if clock_view not in ("availability", "benchmark"):
        raise ValueError("clock_view must be availability or benchmark")
    policy = load_node_time_policy(root)
    if policy is None:
        return None
    with np.load(Path(root) / SIDECAR, allow_pickle=False) as arrays:
        for name, record in policy["tables"].items():
            values = np.asarray(arrays[name], dtype=np.int64)
            offset, count = int(record["offset"]), int(record["num_nodes"])
            if len(values) != count or offset + count > len(row_time):
                raise NodeTimePolicyError(f"{root}: sidecar table {name} does not fit the archive")
            if clock_view == "benchmark":
                delay = record.get("availability_offset_seconds", 0)
                if type(delay) is not int or delay < 0:
                    raise NodeTimePolicyError(f"{root}: invalid availability offset for {name}")
                values = np.where(values == int(ARCHIVE_STATIC), values, values - delay)
            values = np.where(values == int(ARCHIVE_STATIC), int(static), values)
            try:
                import torch

                if torch.is_tensor(row_time):
                    row_time[offset : offset + count] = torch.from_numpy(values).to(row_time.dtype)
                    continue
            except ImportError:
                pass
            row_time[offset : offset + count] = values
    return policy["tag"] if clock_view == "availability" else policy["tag"] + ":benchmark"


def observed_availability(cell_rows, label, target_column, column_index, document):
    """Look up publication times from the observed target sidecar."""
    if (
        set(document) != {"schema_version", "kind", "policy", "targets"}
        or document["schema_version"] != 1
        or document["kind"] != "observed_target_availability"
        or document["policy"] != "maximum_publication_per_fact_time_v1"
    ):
        raise ValueError("strict observed availability contract required")
    matches = []
    for entry in document["targets"]:
        if set(entry) != {"table", "target", "cutoffs_seconds", "available_seconds"}:
            raise ValueError("strict observed target clock required")
        key = f"{entry['target']} of {entry['table']}"
        if key not in column_index:
            raise ValueError("observed target column missing")
        cuts, available = entry["cutoffs_seconds"], entry["available_seconds"]
        if (
            not cuts
            or len(cuts) != len(available)
            or any(type(v) is not int for v in cuts + available)
            or cuts != sorted(set(cuts))
            or any(a <= c for c, a in zip(cuts, available, strict=False))
        ):
            raise ValueError("observed publications must be ordered and strictly after facts")
        if column_index[key] == target_column:
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError("observed availability needs exactly one matching target")
    entry = matches[0]
    cuts = torch.tensor(entry["cutoffs_seconds"], device=cell_rows.device, dtype=torch.long)
    publication = torch.tensor(entry["available_seconds"], device=cell_rows.device, dtype=torch.long)
    ix = torch.searchsorted(cuts, cell_rows.long()).clamp_max(len(cuts) - 1)
    from openrfm.sampler.retrieval import _assert

    _assert((~label | (cuts[ix] == cell_rows)).all(), "target row missing observed publication time")
    return publication[ix]


def structural_table_ids(row_table, cell_ptr, metadata, text_map):
    """Restore table identities for featureless rows."""
    ptr = cell_ptr.detach().cpu().numpy().astype(np.int64, copy=False)
    current = row_table.detach().cpu().numpy().astype(np.int64, copy=False)
    if ptr.shape != (len(current) + 1,) or np.any(np.diff(ptr) < 0):
        raise ValueError("structural table IDs: invalid cell partition")
    empty = ptr[1:] == ptr[:-1]
    if not empty.any():
        return row_table
    repaired = None
    for name, info in metadata.items():
        start = info["node_idx_offset"]
        end = start + info["num_nodes"]
        if not 0 <= start <= end <= len(current):
            raise ValueError("structural table IDs: invalid table partition")
        rows = np.flatnonzero(empty[start:end]) + start
        if not len(rows):
            continue
        base = name.rsplit(":", 1)[0]
        expected = text_map.get(base)
        if type(expected) is not int or expected < 0:
            raise ValueError("structural table IDs: missing canonical table text identity")
        if np.any((current[rows] != 0) & (current[rows] != expected)):
            raise ValueError("structural table IDs: nonzero archived identity conflict")
        wrong = rows[current[rows] != expected]
        if len(wrong):
            if repaired is None:
                repaired = current.copy()
            repaired[wrong] = expected
    if repaired is None:
        return row_table
    from openrfm.archive.format import compact_tensor

    return compact_tensor(torch.from_numpy(repaired))
