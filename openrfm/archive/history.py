"""The history world the memory module reads (`openrfm.models.memory.history`), built from a decoded archive and the
task graph a sampler admitted."""
from dataclasses import dataclass, field
import json

import numpy as np
import torch

from openrfm.archive.format import archive_root
from openrfm.archive.decode import decode_archive

PARENT_SLOTS = 5


@dataclass
class World:
    ptr: torch.Tensor
    nodes: torch.Tensor
    columns: torch.Tensor
    types: torch.Tensor
    values: torch.Tensor
    text_ids: torch.Tensor
    visible: torch.Tensor
    names: torch.Tensor
    text: torch.Tensor
    times: torch.Tensor
    parents: torch.Tensor
    context_rows: torch.Tensor
    relations: list = field(default_factory=list)

    @property
    def nrows(self):
        return len(self.times)


def decode(archive):
    """The decoded archive with its column-name index."""
    decoded = decode_archive(archive, 'all-MiniLM-L12-v2', 384, PARENT_SLOTS, directed_only=True,
                                    repair_table_ids=False)
    decoded['column_index'] = json.loads((archive_root(archive)/'column_index.json').read_text())
    return decoded


def cell_values(v):
    """One float per cell: the number, boolean, date or category code of its type (0 for text)."""
    sem = v['cell_types'].long()
    numbers = v['cell_numbers'].float()
    values = torch.where(sem == 0, numbers, torch.where(sem == 3, v['cell_booleans'].float(), torch.where(
        sem == 2, v['cell_datetimes'].float(), torch.where(sem == 4, v['cell_codes'].float(), torch.zeros_like(numbers)))))
    return torch.nan_to_num(values)


@torch.no_grad()
def history_world(decoded, graph, rt, cells='cuda'):
    """The world under the task graph `graph`, with its rows on the backbone's device and its cells on `cells`.

    Dropped cells are not visible, a label-carrying row is timed at its latest label availability (so its covariates
    appear together with its label), and rows of other tasks' tables are not history."""
    device = next(rt.parameters()).device
    v, meta, index = decoded['values'], decoded['meta'], decoded['column_index']
    nodes, n = decoded['nodes'].long(), len(decoded['row_time'])
    name_ids = torch.tensor(sorted(set(index.values())), dtype=torch.long)
    columns = torch.from_numpy(np.searchsorted(name_ids.numpy(), decoded['cols'].long().numpy()))
    times = graph.row_time.cpu().long().clone()
    visible = ~graph.cell_drop.cpu().bool()
    label = graph.cell_label.cpu().bool()
    times.scatter_reduce_(0, nodes[label & visible], graph.cell_available.cpu().long()[label & visible], reduce='amax')
    parents = torch.full((n, PARENT_SLOTS), -1, dtype=torch.long)
    ptr, dst = decoded['fptr'].long(), decoded['fdst'].long()
    source = torch.repeat_interleave(torch.arange(n), ptr.diff())
    parents[source, torch.arange(len(dst))-ptr[source]] = dst
    labelled = torch.zeros(n, dtype=torch.bool)
    labelled[nodes[label]] = True
    own = {name.rsplit(':', 1)[0] for name, m in meta.items()
           if labelled[m['node_idx_offset']:m['node_idx_offset']+m['num_nodes']].any()}
    context_rows = torch.ones(n, dtype=torch.bool)
    for name, m in meta.items():
        if name.rsplit(':', 1)[-1].lower() != 'db' and name.rsplit(':', 1)[0] not in own:
            context_rows[m['node_idx_offset']:m['node_idx_offset']+m['num_nodes']] = False
    emb = decoded['embeddings']
    names = rt.enc_dict['col_name'](emb[name_ids].to(device).float()).detach()
    return World(decoded['common'][6].long(), nodes.to(cells), columns.to(cells), v['cell_types'].long().to(cells),
                 cell_values(v).to(cells), v['cell_text_ids'].long().to(cells), visible.to(cells), names, emb.to(cells),
                 times.to(device), parents.to(device), context_rows.to(device))
