"""Packing sampled context cells into model batches, and the structural provenance of packed cells."""
import torch

from openrfm.archive.format import take
from openrfm.sampler.episodes import rt_cell_order_permutation
from openrfm.sampler.native import get_extension
from openrfm.sampler.path_streams import row_provenance
from openrfm.sampler.retrieval import STATIC

NO_TIME = -(2**31)
PROVENANCE = dict(path_sig_id="cell_path_sig", hop_depth="cell_hop", time_delta_bucket="cell_tdelta",
                  degree_in="cell_deg_in", degree_out="cell_deg_out")
SIDECAR_DTYPES = dict(cell_path_sig=torch.int32, cell_hop=torch.int16, cell_tdelta=torch.int8,
                      cell_deg_in=torch.int8, cell_deg_out=torch.int8)


def composition_provenance(receipt: torch.Tensor, selected: torch.Tensor) -> dict:
    """Per-cell structural features of the composer's rows (hop, path signature, time bucket, degrees)."""
    fields, rows = row_provenance(receipt)
    rows = rows[:, :selected.shape[1]]
    padding = selected < 0
    return {PROVENANCE[key]: torch.gather(value, 1, rows.clamp_min(0)).masked_fill(
        padding, -128 if key == "time_delta_bucket" else 0) for key, value in fields.items()}


def pack_cells(graph, payload, requests, selected, roles, dataset_id, *, provenance=None) -> dict:
    """The batch of ``[B, L]`` selected cell ids (``-1`` padding) in table/row/column order, with provenance."""
    permutation = rt_cell_order_permutation(selected, cell_order="table_row_col", cell_nodes=payload["nodes"],
                                            row_table=graph.row_table, cell_columns=graph.cell_column,
                                            seed_rows=requests[:, 0], row_time=graph.row_time)
    if permutation is not None:
        selected, roles = torch.gather(selected, 1, permutation), torch.gather(roles, 1, permutation)
        if provenance is not None:
            provenance = {k: torch.gather(v, 1, permutation) for k, v in provenance.items()}
    padding = selected < 0
    ix = selected.clamp_min(0)
    rows = take(payload["nodes"], ix)
    columns = take(graph.cell_column, ix)
    target = (rows == requests[:, :1]) & (columns == requests[:, 2:3]) & ~padding

    def ids(value):
        return value.masked_fill(padding, -1)

    out = dict(sampler_requests=requests.clone(), node_idxs=ids(rows), col_name_idxs=ids(columns),
               table_name_idxs=ids(take(graph.row_table, rows)),
               sem_types=ids(take(payload["cell_types"], ix).long()),
               class_value_idxs=ids(take(payload["cell_codes"], ix).long()),
               discrete_num_classes=ids(take(payload["cell_num_classes"], ix).long()),
               is_padding=padding, is_targets=target, masks=target, hidden_label_cells=torch.zeros_like(padding),
               is_task_nodes=((take(graph.row_task, rows) != 0) | (columns == requests[:, 2:3])) & ~padding,
               dataset_idxs=torch.full_like(rows, dataset_id),
               timestamps=ids(torch.where(rows == requests[:, :1], requests[:, 1:2], take(graph.row_time, rows))),
               f2p_nbr_idxs=take(payload["parents"], rows).masked_fill(padding[..., None], -1),
               context_role=roles.masked_fill(padding, 0),
               label_available_at=ids(take(graph.cell_available, ix)),
               retrieved_label_cells=(take(graph.cell_label, ix).bool() | (columns == requests[:, 2:3])) & ~padding)
    out["declared_drop_cells"] = payload["declared_drop"][ix] & ~out["is_targets"] & ~padding
    for field, source in (("number_values", "cell_numbers"), ("boolean_values", "cell_booleans")):
        out[field] = payload[source][ix].masked_fill(padding, 0).to(torch.bfloat16)[..., None]
    dates = payload["cell_datetimes"][ix].masked_fill(padding, 0)
    out["datetime_values"] = dates.to(torch.bfloat16)[..., None]
    if dates.dtype == torch.float32:
        out["datetime_values_fp32"] = dates[..., None]
    embeddings = payload["embeddings"]
    out["col_name_values"] = embeddings[columns].masked_fill(padding[..., None], 0)
    text_ids = take(payload["cell_text_ids"], ix).long().clamp(0, len(embeddings) - 1)
    out["text_values"] = embeddings[text_ids].masked_fill((padding | (out["sem_types"] != 1))[..., None], 0)
    out["timestamps"] = torch.where(out["timestamps"] == STATIC, NO_TIME, out["timestamps"])
    if provenance is not None:
        out.update(provenance)
    return observation_view(payload, out, requests, graph)


def _publication_history(payload) -> bool:
    tables = payload.get("same_time_table_ids")
    return bool(payload.get("explicit_publication_history", False)) and tables is not None and tables.numel() > 0


def request_flags(payload, nodes: torch.Tensor) -> torch.Tensor:
    """Request column 4: 1 for the strict as-of cut, 0 where same-time publication rows are admitted."""
    return torch.full_like(nodes, 0 if _publication_history(payload) else 1)


def observation_view(payload, batch: dict, requests, graph) -> dict:
    """Remove published history rows at or after the cutoff when the composer admitted same-time rows."""
    if not _publication_history(payload):
        return batch
    tables = payload.get("history_table_ids")
    hidden = (torch.isin(batch["table_name_idxs"].long(), tables.long()) & (batch["timestamps"] >= requests[:, 1:2])
              & ~batch["is_padding"]) if tables is not None else torch.zeros_like(batch["is_padding"])
    out = dict(batch, is_padding=batch["is_padding"] | hidden, masks=batch["masks"].masked_fill(hidden, 0),
               hidden_label_cells=batch["hidden_label_cells"] | hidden)
    for name in ("number_values", "boolean_values", "datetime_values", "datetime_values_fp32", "text_values",
                 "col_name_values"):
        if name in out:
            out[name] = out[name].masked_fill(hidden[..., None], 0)
    return archive_provenance(out, payload["directions"], graph.row_table)


def archive_provenance(batch: dict, directions, row_table, *, native: bool = False) -> dict:
    """Recompute the structural features of the packed cells from the packed foreign keys."""
    if not native or not batch["node_idxs"].is_cuda:
        return _archive_provenance(batch, directions, row_table)
    hop, sig, delta, din, dout = get_extension(batch["node_idxs"].device).rt_provenance(
        batch["node_idxs"].long().contiguous(), batch["f2p_nbr_idxs"].long().contiguous(),
        batch["is_padding"].bool().contiguous(), batch["is_targets"].bool().contiguous(),
        batch["table_name_idxs"].long().contiguous(), batch["timestamps"].long().contiguous(),
        batch["sampler_requests"].long().contiguous(), row_table.contiguous())
    return dict(batch, cell_hop=hop, cell_path_sig=sig, cell_tdelta=delta, cell_deg_in=din, cell_deg_out=dout)


def _mix(x):
    x = x + 0x9e3779b97f4a7c15
    x = (x ^ ((x >> 30) & ((1 << 34) - 1))) * 0xbf58476d1ce4e5b9
    x = (x ^ ((x >> 27) & ((1 << 37) - 1))) * 0x94d049bb133111eb
    return x ^ ((x >> 31) & ((1 << 33) - 1))


def _archive_provenance(batch: dict, directions, row_table) -> dict:
    b = dict(batch)
    node, live = b["node_idxs"].long(), ~b["is_padding"]
    B, S = node.shape
    N = len(directions.fptr) - 1
    idx = node.clamp(0, N - 1)
    target = b["is_targets"] & live
    root = idx.gather(1, target.long().argmax(1)[:, None])
    parents = b["f2p_nbr_idxs"].long()
    valid_edge = (parents >= 0) & (parents < N) & live[..., None]
    physical_parent = parents.clamp(0, N - 1)
    # Compact per-sequence row index: rows of the sequence in sorted id order.
    physical = idx.sort(dim=1).values.contiguous()
    local = torch.searchsorted(physical, idx.contiguous())
    parent = torch.searchsorted(physical, physical_parent.flatten(1).contiguous()).clamp_max(S - 1)
    member = physical.gather(1, parent).reshape_as(physical_parent) == physical_parent
    parent = parent.reshape_as(physical_parent)
    idx, root, width = local, torch.searchsorted(physical, root.contiguous()), S
    distance = torch.full((B, width), 65, dtype=torch.int64, device=node.device)
    distance.scatter_(1, root, 0)
    signatures = torch.zeros_like(distance)
    present = torch.zeros((B, width), dtype=torch.int32, device=node.device)
    present.scatter_reduce_(1, idx, live.int(), reduce="amax")
    valid_edge &= member & present.gather(1, parent.flatten(1)).reshape_as(parent).bool()
    batch_ids = torch.arange(B, device=node.device)[:, None, None].expand_as(parent)
    child = idx[..., None].expand_as(parent)
    witness = torch.unique(torch.stack((batch_ids, child, parent), -1)[valid_edge & (parent != child)], dim=0)
    outgoing = torch.bincount(witness[:, 0] * width + witness[:, 1], minlength=B * width).reshape(B, width)
    incoming = torch.bincount(witness[:, 0] * width + witness[:, 2], minlength=B * width).reshape(B, width)
    tables = b["table_name_idxs"].long()
    root_table = tables.gather(1, target.long().argmax(1)[:, None])
    parent_table = take(row_table, physical_parent).long()

    def path_hash(prefix, direction, table):
        d = torch.full((), direction + 1, dtype=torch.long, device=node.device)
        return 1 + (_mix(prefix ^ _mix(d) ^ _mix(table + 17)) & ((1 << 63) - 1)).remainder(0x7fffffff)

    never = torch.iinfo(torch.int64).max
    for _ in range(6):
        source = distance.gather(1, idx)
        pd = distance.gather(1, parent.flatten(1)).reshape_as(parent)
        ss, ps = signatures.gather(1, idx), signatures.gather(1, parent.flatten(1)).reshape_as(parent)
        distance.scatter_reduce_(1, idx, (pd + 1).masked_fill(~valid_edge, 65).amin(2), reduce="amin")
        rev = (source[..., None] + 1).expand_as(parent).masked_fill(~valid_edge, 65)
        distance.scatter_reduce_(1, parent.flatten(1), rev.flatten(1), reduce="amin")
        candidate = torch.full_like(signatures, never)
        child_hash = path_hash(torch.where(ps != 0, ps, root_table[..., None]), 1, tables[..., None])
        child_ok = valid_edge & (pd < 65) & (source[..., None] == 65) & ((pd + 1) == distance.gather(1, idx)[..., None])
        candidate.scatter_reduce_(1, idx, child_hash.masked_fill(~child_ok, never).amin(2), reduce="amin")
        parent_hash = path_hash(torch.where(ss != 0, ss, root_table)[..., None], 0, parent_table)
        parent_ok = (valid_edge & (source[..., None] < 65) & (pd == 65)
                     & ((source[..., None] + 1) == distance.gather(1, parent.flatten(1)).reshape_as(parent)))
        candidate.scatter_reduce_(1, parent.flatten(1), parent_hash.masked_fill(~parent_ok, never).flatten(1),
                                  reduce="amin")
        signatures = torch.where(candidate != never, candidate, signatures)
    hops = distance.gather(1, idx).clamp_max(65)
    sig = torch.where(hops == 65, 1 + tables.remainder(0x7fffffff), signatures.gather(1, idx))
    timestamp = b["timestamps"].long()
    delta = timestamp - b["sampler_requests"][:, 1:2]
    log = torch.floor(torch.log2(delta.abs().clamp_min(1).double())).long().clamp_max(63) + 1
    td = torch.where(delta == 0, 0, delta.sign() * log).masked_fill((timestamp == STATIC) | (timestamp == NO_TIME), -128)
    for name, value in dict(cell_hop=hops, cell_path_sig=sig, cell_tdelta=td,
                            cell_deg_in=torch.floor(torch.log2(1 + incoming.gather(1, idx))).long().clamp_max(31),
                            cell_deg_out=torch.floor(torch.log2(1 + outgoing.gather(1, idx))).long().clamp_max(31)).items():
        b[name] = value.to(SIDECAR_DTYPES[name]).masked_fill(~live, 0)
    return b
