"""Encoded features of the support and query rows: the R1 export (backbone taps, focal FP32 scalars, memory columns)
and the R2 record bank."""
import numpy as np
import torch
import torch.nn.functional as F

from openrfm.infer.snapshot import snapshot_child_cells
from openrfm.models.memory.context import (block_states, column_key, focal_positions, focal_scalars, row_context_cells,
                                           scalar_matrix, slot_rows, typed_numeric_scalars)

TAPS = (3, 6)
# Depth of the record-bank row states: after the last block (depth 0 is the block-0 input).
ROW_DEPTH = 6


@torch.no_grad()
def forward(rt, batch):
    """One frozen bf16 backbone forward: the states at every depth (block-0 input first), the live mask, the focal
    FP32 scalars and the target cell's residual stream."""
    with torch.autocast('cuda', dtype=torch.bfloat16):
        with block_states(rt.blocks) as states:
            rt.encode_hidden(batch, support_axis_enabled=False)
    full = torch.stack([states[-1]]+[states[b] for b in range(len(rt.blocks))], 2)
    _, pos = focal_positions(batch)
    target = full[torch.arange(len(pos), device=pos.device), pos].float()
    return full, ~batch['is_padding'].bool(), focal_scalars(batch), target


def slot_records(batch, states, source, ranges, bank, out):
    """Per row: the (table, column) keys, name vectors, row states and FP32 scalars of its slot-row cells, first
    occurrence of each key, appended to `out`; with `source == 'snapshot'` the latest lawful snapshot child's cells
    count as slot-row cells, with their states but without scalars."""
    has, pos = focal_positions(batch)
    ix = torch.arange(len(pos), device=pos.device)
    if source == 'snapshot':
        child = snapshot_child_cells(batch, ranges)
        slot = slot_rows(batch, batch['node_idxs'][ix, pos, None], pos, has)
        batch = dict(batch, node_idxs=torch.where(child, slot.to(batch['node_idxs'].dtype), batch['node_idxs']),
                     value_visible_cells=batch['value_visible_cells'].bool() & ~child)
    focal = batch['node_idxs'][ix, pos, None]
    slot = slot_rows(batch, focal, pos, has)
    live = (batch['node_idxs'] == slot) & row_context_cells(batch) & has[:, None]
    ceiling = torch.iinfo(torch.int64).max
    keys = column_key(batch['table_name_idxs'], batch['col_name_idxs']).masked_fill(~live, ceiling)
    ordered, order = keys.sort(dim=1, stable=True)
    first = ordered != ceiling
    first[:, 1:] &= ordered[:, 1:] != ordered[:, :-1]
    take = ordered.masked_fill(~first, ceiling).argsort(dim=1, stable=True)[:, :bank]
    positions = order.gather(1, take)
    keys = ordered.gather(1, take).masked_fill(~first.gather(1, take), ceiling)
    scalars = typed_numeric_scalars(batch)[ix[:, None], positions]
    for i in range(len(pos)):
        live = keys[i] != ceiling
        out.append((int(focal[i, 0]), keys[i][live].cpu(), batch['col_name_values'][i, positions[i][live]].cpu(),
                        states[i, positions[i][live], ROW_DEPTH].float().cpu(), scalars[i][live].float().cpu()))


@torch.no_grad()
def encode(sampler, rt, memory, world, splits, records=None, ranges=None, bank=96):
    """R1 features [taps ; focal FP32 scalars ; memory columns] of every row of `splits` (lists of roots, each split
    encoded in its own batches of 8 from a fresh workspace) and, when `records` is 'slot' or 'snapshot', the slot
    records of every row."""
    taps, triples, reads, out, n = [], [], [], [], 0
    for rows in splits:
        sampler.restart()
        for start in range(0, len(rows), sampler.batch_size):
            roots = torch.as_tensor(rows[start:start+sampler.batch_size])
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in sampler.contexts(roots).items()}
            states, live, (r, k, v), target = forward(rt, batch)
            reads.append(memory.columns(world, roots.cuda(), sampler.graph.row_time[roots].long().cuda(),
                                        context=(states.to(torch.bfloat16), live)).cpu())
            triples.append((r.cpu()+n, k.cpu(), v.cpu()))
            taps.append(target[:, list(TAPS)].flatten(1).cpu())
            if records is not None:
                slot_records(batch, states, records, ranges, bank, out)
            n += len(roots)
    scalars, _ = scalar_matrix(tuple(torch.cat(t) for t in zip(*triples)), n)
    x = torch.cat((torch.cat(taps), scalars.float(), torch.cat(reads).float().nan_to_num()), 1).numpy()
    return x, out


@torch.no_grad()
def record_bank(rt, handoff, records):
    """[N, 7S] record-bank columns over the sorted union of the records' keys: per key the learned 4-dim cell
    projection, the FP32 value and its signed log (absent: 0), and a presence bit."""
    names = {}
    for _, keys, name, _, _ in sorted(records, key=lambda r: r[0]):
        for key, vector in zip(keys.tolist(), name):
            names.setdefault(key, vector)
    keys = sorted(names)
    position = {key: i for i, key in enumerate(keys)}
    device = handoff['ecol_weight'].device
    embeddings = rt.enc_dict['col_name'](torch.stack([names[k] for k in keys]).to(device, rt.enc_dict['col_name'].weight.dtype)).float()
    out = []
    for start in range(0, len(records), 512):
        chunk = records[start:start+512]
        states = torch.zeros((len(chunk), len(keys), embeddings.shape[1]))
        scalars = torch.zeros((len(chunk), len(keys), 2))
        present = torch.zeros((len(chunk), len(keys)), dtype=torch.bool)
        for i, (_, row_keys, _, row_states, row_scalars) in enumerate(chunk):
            j = [position[k] for k in row_keys.tolist()]
            states[i, j], scalars[i, j], present[i, j] = row_states, row_scalars, True
        states, scalars, present = states.to(device), scalars.to(device), present.to(device).float()
        s = F.layer_norm(states, (states.shape[-1],))
        e = F.layer_norm(embeddings, (embeddings.shape[-1],))
        proj = F.linear(torch.cat((s, s*e[None]), -1), handoff['ecol_weight'].float(), handoff['ecol_bias'].float())
        cells = torch.cat((proj*present[..., None], scalars*present[..., None], present[..., None]), -1)
        out.append(cells.flatten(1).cpu())
    return torch.cat(out).numpy()


def hybrid(r1, bank, width):
    """R2 features: the R1 features and, per record-bank key, the first `width` projection columns, the FP32 value,
    its signed log and the presence bit; columns constant over all support and query rows are dropped."""
    cells = bank.shape[1]//7
    keep = (7*np.arange(cells)[:, None]+np.r_[np.arange(width), 4, 5, 6][None, :]).ravel()
    x = np.concatenate((r1, bank[:, keep]), 1)
    live = x.max(0) != x.min(0)
    return x[:, live].astype(np.float32)
