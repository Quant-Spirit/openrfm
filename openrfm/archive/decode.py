"""Decode an archive into the task-independent row, cell and edge tensors that task views share."""
import json

import numpy as np
import torch

from openrfm.archive.format import archive_format, archive_root, index_dtype, load_arrays
from openrfm.archive.policy import apply_node_time_policy, structural_table_ids
from openrfm.sampler.retrieval import STATIC

CHUNK = 262144
SPLIT_TASK = {"db": 0, "train": 1, "val": 2, "test": 2}
NUMPY_INDEX = {"uint8": "u1", "uint16": "<u2", "int32": "<i4", "int64": "<i8"}


def _raw_arrays(root, embedding_model, d_text, fmt):
    if fmt == "v2":
        return load_arrays(root, embedding_model, d_text)
    from rustler_archive import ArchiveV1

    return dict(ArchiveV1.load_arrays(str(root), embedding_model, d_text))


def _row_tasks(meta, n):
    """Per row: 0 for a database table, 1 for TRAIN task rows, 2 for VAL/TEST task rows."""
    row_task = torch.zeros(n, dtype=torch.uint8)
    end = 0
    for name, info in sorted(meta.items(), key=lambda item: int(item[1]["node_idx_offset"])):
        start, stop = int(info["node_idx_offset"]), int(info["node_idx_offset"]) + int(info["num_nodes"])
        split = name.rsplit(":", 1)[-1].lower()
        if start != end or stop > n or split not in SPLIT_TASK:
            raise ValueError(f"table metadata must partition archived rows exactly: {name}")
        row_task[start:stop] = SPLIT_TASK[split]
        end = stop
    if end != n:
        raise ValueError("table metadata must partition archived rows exactly")
    return row_task


def _row_times(archived, root):
    row_time = archived.to(dtype=torch.int64, copy=True)
    for start in range(0, len(row_time), CHUNK):
        part = row_time[start : start + CHUNK]
        part[part == -(2**31)] = STATIC
    apply_node_time_policy(archive_root(root), row_time, static=STATIC)
    return row_time


def _cell_ptr(raw_nodes, n):
    """CSR row pointer of the row-sorted cells, in the narrowest index width."""
    if len(raw_nodes) >= 2**31:
        raise ValueError("archive cell address overflow")
    if (len(raw_nodes) and (int(raw_nodes[0]) < 0 or int(raw_nodes[-1]) >= n)) or any(
        np.any(part[1:] < part[:-1]) for start in range(0, len(raw_nodes), CHUNK)
        for part in (raw_nodes[start : start + CHUNK + 1],)
    ):
        raise ValueError("archive cells must have monotone, valid row addresses")
    ptr = np.empty(n + 1, dtype=NUMPY_INDEX[index_dtype(0, len(raw_nodes))])
    key_dtype = raw_nodes.dtype if n <= np.iinfo(raw_nodes.dtype).max else np.int64
    for start in range(0, n + 1, CHUNK):
        stop = min(n + 1, start + CHUNK)
        ptr[start:stop] = np.searchsorted(raw_nodes, np.arange(start, stop, dtype=key_dtype), side="left")
    return ptr


def _check_ids(values, embeddings, n, parent_slots):
    fptr = values["f2p_rowptr"].numpy()
    if any(np.any(np.diff(fptr[start : start + CHUNK + 1].astype(np.int64, copy=False)) > parent_slots)
           for start in range(0, n, CHUNK)):
        raise ValueError(f"FK degree exceeds declared parent_slots={parent_slots}; refusing truncation")
    cols = values["cell_columns"].numpy()
    if any(np.any((part < 0) | (part >= len(embeddings)))
           for start in range(0, len(cols), CHUNK) for part in (cols[start : start + CHUNK],)):
        raise ValueError("invalid column embedding ids")
    text_ids, types = values["cell_text_ids"].numpy(), values["cell_types"].numpy()
    if any(np.any((selected < 0) | (selected >= len(embeddings)))
           for start in range(0, len(text_ids), CHUNK)
           for selected in (text_ids[start : start + CHUNK][types[start : start + CHUNK] == 1],)):
        raise ValueError("invalid text embedding ids")


def _merged_edges(values, row_time, n):
    """(ptr, dst, time) of both edge directions, ordered per source by (destination time, destination)."""
    fptr, fdst, pptr, pdst = (values[k].long() for k in ("f2p_rowptr", "f2p_col_idx", "p2f_rowptr", "p2f_col_idx"))
    src = torch.cat((torch.repeat_interleave(torch.arange(n), fptr.diff()),
                     torch.repeat_interleave(torch.arange(n), pptr.diff())))
    dst = torch.cat((fdst, pdst))
    edge_time = row_time[dst]
    order = torch.argsort(dst, stable=True)
    order = order[torch.argsort(edge_time[order], stable=True)]
    order = order[torch.argsort(src[order], stable=True)]
    ptr = torch.cat((torch.zeros(1, dtype=torch.long), torch.bincount(src, minlength=n).cumsum(0)))
    return ptr, dst[order], edge_time[order]


def decode_archive(root, embedding_model, d_text, parent_slots, *, directed_only=False, repair_table_ids=True):
    """Decode and index the task-independent archive tensors.

    `common` holds the RTGraph row/cell fields in field order; v2 archives and `directed_only` decodes leave the merged
    edge CSR empty because the directed composer samples their own CSR."""
    fmt = archive_format(root)
    raw = _raw_arrays(root, embedding_model, d_text, fmt)
    if "cell_num_classes" not in raw:
        raise RuntimeError("archive lacks discrete cardinalities; rebuild it with the current converter")
    values = {k: v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v)) for k, v in raw.items() if k != "num_nodes"}
    if any(not value.is_contiguous() for value in values.values()):
        raise ValueError("archive decoder produced a noncontiguous array")
    meta = json.loads((archive_root(root) / "table_info.json").read_text())
    n = len(values["node_timestamps"])
    if n >= 2**31 or any(int(v["node_idx_offset"]) + int(v["num_nodes"]) > n for v in meta.values()):
        raise ValueError("archive row address overflow or invalid table metadata")
    row_task = _row_tasks(meta, n)
    row_time = _row_times(values["node_timestamps"], root)
    nodes, cols = values["cell_nodes"], values["cell_columns"]
    raw_ptr = _cell_ptr(nodes.numpy(), n)
    cell_ptr = torch.from_numpy(raw_ptr)
    if repair_table_ids and fmt == "v2" and np.any(raw_ptr[1:] == raw_ptr[:-1]):
        values["node_table_name_idxs"] = structural_table_ids(
            values["node_table_name_idxs"], cell_ptr, meta,
            json.loads((archive_root(root) / "text_map.json").read_text()))
    embeddings = values["embedding_bits"].view(torch.bfloat16).reshape(-1, d_text)
    _check_ids(values, embeddings, n, parent_slots)
    edges = dict(fptr=values["f2p_rowptr"], fdst=values["f2p_col_idx"],
                 pptr=values["p2f_rowptr"], pdst=values["p2f_col_idx"])
    if fmt == "v2" or directed_only:
        common = (torch.zeros(n + 1, dtype=torch.int32), torch.empty(0, dtype=torch.int32),
                  torch.empty(0, dtype=torch.int64), values["node_table_name_idxs"], row_time, row_task, cell_ptr, cols)
        for name in ("node_timestamps", "p2f_table_name_idxs", "p2f_table_types"):
            values.pop(name, None)
    else:
        common = (*_merged_edges(values, row_time, n), values["node_table_name_idxs"].long(), row_time,
                  row_task.long(), cell_ptr.long(), cols.long())
    return dict(archive_format=fmt, values=values, meta=meta, row_time=row_time, nodes=nodes, cols=cols,
                common=common, embeddings=embeddings, **edges)
