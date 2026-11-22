"""The context side of the memory read: one frozen backbone forward over a packed batch."""
from contextlib import contextmanager

import torch

SEM_NUMBER, SEM_DISCRETE = 0, 4
# A (table, column) pair packed into one int64 key; both ids are < 2**31.
KEY_SHIFT = 2**31
LABEL_MASKS = ('retrieved_label_cells', 'is_task_nodes', 'hidden_label_cells', 'declared_drop_cells')


def signed_log(x):
    return x.sign()*x.abs().log1p()


def column_key(table_ids, col_ids):
    """Injective int64 key of (table, column)."""
    return table_ids.long()*KEY_SHIFT+col_ids.long()


def focal_positions(batch):
    """(has_target [B], position [B]) of each sequence's target cell; position 0 where there is none."""
    flags = batch['is_targets'].bool()
    return flags.any(1), flags.long().argmax(1)


def row_context_cells(batch):
    """[B, L] cells that carry evidence: not padding, not the target, not a label, task-table or dropped cell."""
    cells = ~batch['is_padding'].bool() & ~batch['is_targets'].bool()
    for key in LABEL_MASKS:
        if torch.is_tensor(batch.get(key)):
            cells &= ~batch[key].bool()
    return cells


def typed_numeric_visibility(batch):
    """[B, L] visible numeric evidence cells (a visible zero counts)."""
    return (batch['sem_types'] == SEM_NUMBER) & batch['value_visible_cells'].bool() & row_context_cells(batch)


def typed_numeric_scalars(batch):
    """[B, L, 2] (v, signed_log v) of the original FP32 numbers; 0 where not visible."""
    values = batch['number_values_fp32'].squeeze(-1).masked_fill(~typed_numeric_visibility(batch), 0)
    return torch.stack((values, signed_log(values)), -1)


def slot_rows(batch, focal, pos, has):
    """[B, 1] entity-row node of each sequence: the focal row, or its smallest foreign-key parent."""
    nodes = batch['node_idxs']
    sem = batch['sem_types']
    payload = row_context_cells(batch) & ((sem == SEM_NUMBER) | (sem == SEM_DISCRETE)) & has[:, None]
    own = ((nodes == focal) & payload).any(1, keepdim=True)
    parents = batch['f2p_nbr_idxs'][torch.arange(len(nodes), device=nodes.device), pos].long()
    ceiling = torch.iinfo(torch.int64).max
    parent = parents.masked_fill(parents < 0, ceiling).min(1, keepdim=True).values
    return torch.where(own | (parent == ceiling), focal, parent.to(focal.dtype))


@contextmanager
def block_states(blocks):
    """Keeps the input of block 0 under -1 and the output of block b under b; hooks are removed on exit."""
    states = {}

    def keep(index):
        def hook(_module, _inputs, output):
            states[index] = output[0] if isinstance(output, tuple) else output
        return hook

    def keep_input(_module, args):
        states[-1] = args[0]

    handles = [block.register_forward_hook(keep(b)) for b, block in enumerate(blocks)]
    handles.append(blocks[0].register_forward_pre_hook(keep_input))
    try:
        yield states
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def encode_context(rt, batch):
    """One frozen bf16 backbone forward: states, live mask, focal FP32 scalars and the target's residual stream."""
    with torch.autocast('cuda', dtype=torch.bfloat16):
        with block_states(rt.blocks) as states:
            rt.encode_hidden(batch, support_axis_enabled=False)
    full = torch.stack([states[-1]]+[states[b] for b in range(len(rt.blocks))], 2)
    _, pos = focal_positions(batch)
    target = full[torch.arange(len(pos), device=pos.device), pos].float()
    return full.to(torch.bfloat16), ~batch['is_padding'].bool(), focal_scalars(batch), target


def focal_scalars(batch):
    """(row, key, value[2]) of visible numeric cells on the focal and entity rows, first per (table, column)."""
    has, pos = focal_positions(batch)
    nodes = batch['node_idxs']
    rows = torch.arange(len(pos), device=pos.device)
    focal = nodes[rows, pos][:, None]
    slot = slot_rows(batch, focal, pos, has)
    mask = ((nodes == focal) | (nodes == slot)) & typed_numeric_visibility(batch) & has[:, None]
    values = typed_numeric_scalars(batch)
    keys = column_key(batch['table_name_idxs'], batch['col_name_idxs'])
    i, p = mask.nonzero(as_tuple=True)
    order = torch.argsort(i*(keys.shape[1]+1)+p)
    i, p = i[order], p[order]
    k = keys[i, p]
    _, inverse = torch.stack((i, k), 1).unique(dim=0, return_inverse=True)
    first = torch.full((int(inverse.max())+1 if len(inverse) else 0,), len(inverse), device=i.device)
    first = first.scatter_reduce(0, inverse, torch.arange(len(inverse), device=i.device), 'amin')
    return i[first], k[first], values[i[first], p[first]].float()


def scalar_matrix(triples, n):
    """[n, 2S] FP32 scalars over the sorted union of the keys in `triples` (absent is 0), and the keys [S]."""
    row, key, value = triples
    bank, column = key.unique(sorted=True, return_inverse=True)
    out = value.new_zeros((n, len(bank), 2))
    out[row, column] = value
    return out.flatten(1), bank
