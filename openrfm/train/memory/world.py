"""The history world of one training task, with time-sorted key indexes for the label programs.

A `Relation` indexes the rows that carry one key (a primary key, a foreign key, or a low-cardinality categorical
column used as a virtual key) by (key, time); `owner` is each row's key value, -1 outside its table."""
from dataclasses import dataclass

import torch

from openrfm.archive.history import history_world
from openrfm.archive.keys import KeyIndex

# Distinct values up to which a categorical column becomes a virtual key.
VIRTUAL_CARDINALITY = 32


@dataclass
class Relation:
    index: KeyIndex
    owner: torch.Tensor


def build_world(decoded, rt, graph):
    """The history world under the task graph `graph`, all on the backbone's device, with its key relations."""
    if any(not table.lower().endswith(':db') for table in decoded['meta']):
        raise ValueError('memory training reads databases whose task tables are stored as tables')
    device = next(rt.parameters()).device
    world = history_world(decoded, graph, rt, device)
    readable = ~graph.cell_drop.cpu().bool() & ~graph.cell_label.cpu().bool()
    world.relations = key_relations(decoded, world.parents.cpu(), world.times.cpu(), readable,
                                    world.types.cpu(), world.columns.cpu(), device)
    return world


def key_relations(decoded, parents, times, readable, sem, columns, device):
    """Relations of every table: its primary key, its foreign keys (resolved by the destination table's row range)
    and its low-cardinality categorical columns, in table order."""
    meta, column_index, codes = decoded['meta'], decoded['column_index'], decoded['values']['cell_codes']
    n = len(times)
    ranges = {name: (int(m['node_idx_offset']), int(m['num_nodes'])) for name, m in meta.items()}
    ptr_cells, nodes = decoded['common'][6].long(), decoded['nodes'].long()
    relations = []

    def add(keys, rows):
        owner = torch.full((n,), -1, dtype=torch.long)
        owner[rows] = keys
        keep = keys >= 0
        if keep.any():
            index = KeyIndex(keys[keep].to(device), times[rows[keep]].to(device), rows[keep].to(device))
            relations.append(Relation(index, owner.to(device)))

    for table, info in meta.items():
        start, count = ranges[table]
        if not count:
            continue
        bare = table.rsplit(':', 1)[0]
        row = torch.arange(start, start+count)
        for column, target in [(info.get('pkey_col'), None), *info['fkey_col_to_pkey_table'].items()]:
            if column is None:
                continue
            if f'{column} of {bare}' not in column_index:
                raise ValueError(f'archive has no key column {bare}.{column}')
            if target is None:
                add(row, row)
                continue
            (a, length), = [value for name, value in ranges.items() if name.rsplit(':', 1)[0] == target]
            candidates = parents[row]
            matching = (candidates >= a) & (candidates < a+length)
            if (matching.sum(1) > 1).any():
                raise ValueError(f'{bare}.{column}: parallel foreign keys to {target} need named key resolution')
            keys = candidates.gather(1, matching.long().argmax(1)[:, None]).squeeze(1).masked_fill(~matching.any(1), -1)
            add(keys, row)
        a, b = int(ptr_cells[start]), int(ptr_cells[start+count])
        cells = a+torch.where(readable[a:b] & (sem[a:b] == 4))[0]
        cells = cells[columns[cells].argsort(stable=True)]
        groups, sizes = columns[cells].unique_consecutive(return_counts=True)
        for selected in cells.split(sizes.tolist()):
            unique, inverse = codes[selected].long().unique(sorted=True, return_inverse=True)
            if 1 < len(unique) <= VIRTUAL_CARDINALITY:
                # Virtual keys lie outside the physical row address space.
                base = n+len(relations)*(VIRTUAL_CARDINALITY+1)
                add(base+inverse, nodes[selected])
    return relations
