"""The latest lawful snapshot child of the slot row, for the R2 record bank."""
import json

import torch

from openrfm.archive.format import archive_root
from openrfm.models.memory.context import focal_positions, row_context_cells, slot_rows
from openrfm.sampler.pack import NO_TIME as STATIC_TIME


def table_ranges(base):
    """{table: (first node, end node)} of the archive's Db tables, in table_info order."""
    info = json.loads((archive_root(base)/'table_info.json').read_text())
    return {name.rsplit(':', 1)[0]: (v['node_idx_offset'], v['node_idx_offset']+v['num_nodes'])
            for name, v in info.items() if name.rsplit(':', 1)[-1] == 'Db'}


def table_of(nodes, ranges):
    out = torch.full_like(nodes, -1)
    for i, (lo, hi) in enumerate(ranges.values()):
        out[(nodes >= lo) & (nodes < hi)] = i
    return out


def snapshot_child_cells(batch, ranges):
    """[B, L] cells of the latest lawful child (as-of strictly before the cutoff, in the context) in each
    `<P>__snapshot` table, where P is the slot row's table or a foreign-key parent's table of the focal row."""
    names = list(ranges)
    snapshot_of = torch.tensor([names.index(n+'__snapshot') if n+'__snapshot' in ranges else -1 for n in names]+[-1],
                               device=batch['node_idxs'].device)
    nodes = batch['node_idxs'].long()
    has, pos = focal_positions(batch)
    ix = torch.arange(len(nodes), device=nodes.device)
    f2p = batch['f2p_nbr_idxs'].long()
    slot = slot_rows(batch, nodes[ix, pos, None], pos, has).long()
    owners = torch.cat((slot, f2p[ix, pos]), 1)
    wanted = snapshot_of[table_of(owners, ranges)].masked_fill(owners < 0, -1)
    times = batch['timestamps'].long()
    cutoff = times[ix, pos, None]
    table = table_of(nodes, ranges)
    owned = ((table[:, :, None] == wanted[:, None, :]) & (wanted[:, None, :] >= 0)
             & (f2p[:, :, :, None] == owners[:, None, None, :]).any(2)).any(-1)
    candidate = (row_context_cells(batch) & has[:, None] & owned & (times != STATIC_TIME) & (cutoff != STATIC_TIME)
                 & (times < cutoff))
    out = torch.zeros_like(candidate)
    for t in wanted[wanted >= 0].unique().tolist():
        mine = candidate & (table == t)
        latest = times.masked_fill(~mine, STATIC_TIME).max(1, keepdim=True).values
        rows = mine & (times == latest)
        # Two rows at one as-of are content-identical by construction of the schema programs; the larger node is kept.
        chosen = nodes.masked_fill(~rows, -1).max(1, keepdim=True).values
        out |= mine & (nodes == chosen)
    return out
