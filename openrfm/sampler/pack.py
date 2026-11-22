"""Packed contexts: the native composer's selected cells of a task archive in the backbone's batch layout."""
from dataclasses import replace
from pathlib import Path

import torch

from openrfm.archive.format import take
from openrfm.archive.loader import load_rt_archive
from openrfm.sampler.episodes import rt_cell_order_permutation
from openrfm.sampler.path_streams import row_provenance
from openrfm.sampler.retrieval import STATIC, context_settings
from openrfm.sampler.v2 import Workspace

EMBEDDING_MODEL, D_TEXT = 'all-MiniLM-L12-v2', 384
NO_TIME = -(2**31)
# Composer row provenance -> batch keys of the backbone's structural features.
PROVENANCE = dict(path_sig_id='cell_path_sig', hop_depth='cell_hop', time_delta_bucket='cell_tdelta',
                  degree_in='cell_deg_in', degree_out='cell_deg_out')


def forecast_view(spec, entity_table, horizon):
    """The context policy of a forecast task: its matured label history is readable. Labels of a window of
    `horizon` seconds are stored as available at t+horizon-1, so a strict `available < cutoff` admits exactly
    t+horizon <= cutoff."""
    return replace(spec, label_visibility='history_available', label_horizon_seconds=max(0, horizon-1),
                   context_text_policy='keep', task_kind='forecast', entity_table=f'{entity_table}:Db',
                   filler_percent=min(spec.filler_percent, 20))


class ContextSampler:
    """Contexts of rows of one task archive, composed on the CPU in batches of at most `batch_size` roots.

    `spec` admits the archive and `compose` (default `spec`) composes the contexts; an `inclusive` cutoff also admits
    rows at the cutoff time."""

    def __init__(self, archive, spec, target, drops, *, compose=None, batch_size=8, inclusive=False):
        self.archive, self.target, self.batch_size = Path(archive), target, batch_size
        self.inclusive = inclusive
        self.spec = compose or spec
        self.graph, self.payload, roots, _ = load_rt_archive(self.archive, EMBEDDING_MODEL, D_TEXT, device='cpu',
                                                             spec=spec, target_column=target, drops=drops)
        self.roots = torch.as_tensor(roots).long()
        self.directions = self.payload['directions']
        self.restart()

    def restart(self):
        """A new workspace."""
        self.workspace = Workspace(self.graph, self.directions, self.batch_size, self.spec, self.payload['composition'])

    def sample(self, roots, seed=0):
        """The packed context of each root; request salts are `seed` plus the root's position in the batch."""
        roots = torch.as_tensor(roots, dtype=torch.int64)
        requests = torch.stack([roots, self.graph.row_time[roots].long(), torch.full_like(roots, self.target),
                                torch.arange(len(roots))+int(seed), torch.full_like(roots, int(not self.inclusive))], 1)
        sampled = self.workspace.sample(requests, context_settings(requests, self.spec))
        order = 'canonical_recent' if self.spec.canonical_recent else 'table_row_col'
        batch = pack_cells(self.graph, self.payload, requests, sampled[3], sampled[4], sampled[5], order)
        live = ~batch['is_padding'] & (batch['timestamps'] != NO_TIME)
        context = live & (batch['node_idxs'] != requests[:, :1])
        if not bool(((batch['timestamps'] <= requests[:, 1:2]) | ~context).all()):
            raise AssertionError('a context cell is later than its cutoff')
        return batch


def pack_cells(graph, payload, requests, selected, roles, receipt, cell_order):
    """[B, L] backbone batch of the composer's selected cells, ordered by `cell_order` and then stably by column."""
    fields, slots = row_provenance(receipt)
    slots = slots[:, :selected.shape[1]]
    padding = selected < 0
    sidecars = {PROVENANCE[key]: torch.gather(value, 1, slots.clamp_min(0)).masked_fill(
        padding, -128 if key == 'time_delta_bucket' else 0) for key, value in fields.items()}
    permutation = rt_cell_order_permutation(selected, cell_order=cell_order, cell_nodes=payload['nodes'],
                                            row_table=graph.row_table, cell_columns=graph.cell_column,
                                            seed_rows=requests[:, 0], row_time=graph.row_time)
    keys = take(graph.cell_column, torch.gather(selected, 1, permutation).clamp_min(0)).masked_fill(
        torch.gather(padding, 1, permutation), torch.iinfo(torch.int64).max)
    order = torch.gather(permutation, 1, torch.argsort(keys, dim=1, stable=True))
    selected, roles = torch.gather(selected, 1, order), torch.gather(roles, 1, order)
    sidecars = {k: torch.gather(v, 1, order) for k, v in sidecars.items()}
    padding = selected < 0
    ix = selected.clamp_min(0)
    rows, columns = take(payload['nodes'], ix), take(graph.cell_column, ix)
    target = (rows == requests[:, :1]) & (columns == requests[:, 2:3]) & ~padding

    def ids(value):
        return value.masked_fill(padding, -1)

    def values(source):
        return payload[source][ix].masked_fill(padding, 0)
    emb = payload['embeddings']
    text_ids = take(payload['cell_text_ids'], ix).long().clamp(0, len(emb)-1)
    timestamps = ids(torch.where(rows == requests[:, :1], requests[:, 1:2], take(graph.row_time, rows)))
    dates = values('cell_datetimes')
    batch = dict(
        sampler_requests=requests.clone(), node_idxs=ids(rows), col_name_idxs=ids(columns),
        table_name_idxs=ids(take(graph.row_table, rows)), sem_types=ids(take(payload['cell_types'], ix).long()),
        class_value_idxs=ids(take(payload['cell_codes'], ix).long()),
        discrete_num_classes=ids(take(payload['cell_num_classes'], ix).long()),
        is_padding=padding, is_targets=target, masks=target, hidden_label_cells=torch.zeros_like(padding),
        is_task_nodes=((take(graph.row_task, rows) != 0) | (columns == requests[:, 2:3])) & ~padding,
        timestamps=torch.where(timestamps == STATIC, NO_TIME, timestamps),
        f2p_nbr_idxs=take(payload['parents'], rows).masked_fill(padding[..., None], -1),
        context_role=roles.masked_fill(padding, 0), label_available_at=ids(take(graph.cell_available, ix)),
        retrieved_label_cells=(take(graph.cell_label, ix).bool() | (columns == requests[:, 2:3])) & ~padding,
        number_values_fp32=values('cell_numbers')[..., None],
        number_values=values('cell_numbers').to(torch.bfloat16)[..., None],
        boolean_values=values('cell_booleans').to(torch.bfloat16)[..., None],
        datetime_values=dates.to(torch.bfloat16)[..., None], datetime_values_fp32=dates[..., None],
        col_name_values=emb[columns].masked_fill(padding[..., None], 0),
        text_values=emb[text_ids].masked_fill(padding[..., None], 0), **sidecars)
    if 'declared_drop' in payload:
        batch['declared_drop_cells'] = payload['declared_drop'][ix] & ~target & ~padding
    return batch
