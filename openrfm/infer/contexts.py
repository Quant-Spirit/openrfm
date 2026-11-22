"""Inference contexts: the sampler of one task archive and its 8-row context batches under the inference label
visibility."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import torch

from openrfm.archive.format import archive_root
from openrfm.sampler.pack import NO_TIME as STATIC_TIME, ContextSampler, forecast_view
from openrfm.sampler.spec import RTRetrievalSpec


def task_view(spec, task):
    """The label view of a task: the forecast view of its entity table, the matured label history alone (`history`) or
    no labels (`entity_row`)."""
    if task.get('view') == 'entity_row':
        return replace(spec, label_visibility='strict_drop', label_horizon_seconds=None, context_text_policy='keep')
    if task.get('view') == 'history':
        return replace(spec, label_visibility='history_available', label_horizon_seconds=max(0, task['horizon_s']-1),
                       context_text_policy='keep')
    return forecast_view(spec, task['entity_table'], task['horizon_s'])


def sampler_spec(config, length, task):
    """(admission spec, composition spec) of one task: the protocol under the task's label view, then the own-history
    policy."""
    spec = replace(RTRetrievalSpec.parse(config['retrieval']), _model_requires_provenance=True, reserve_bytes=0)
    if length != spec.seq_len_set[0]:
        spec = replace(spec, seq_len_set=(length,))
    view = task_view(spec, task)
    own = dict(config['own_history'], seq_len_set=(length,), local_cell_budget_set=(length,),
               protected_core_len=min(length, 1024))
    return view, replace(view, **own)


class TaskSampler(ContextSampler):
    """Contexts of one task archive; an archive of the original schema also samples along the task's `indirect`
    paths."""

    def __init__(self, base, task, config, length, *, original_schema=True):
        index = json.loads((archive_root(base)/'column_index.json').read_text())
        # Peer-order counts of relation materializers are structure, not features.
        drops = [index[key] for key in index if key.startswith('shared_members of ')]
        view, compose = sampler_spec(config, length, task)
        super().__init__(base, view, index[f"{task['target']} of {task['table']}"], drops, compose=compose,
                         inclusive=task.get('view') == 'entity_row')
        self.base, self.horizon = Path(base), task['horizon_s']
        info = json.loads((archive_root(base)/'table_info.json').read_text())
        if original_schema and task.get('indirect'):
            self.directions = indirect_streams(self.graph, self.directions, task['indirect'], info)
            self.restart()
        self.test_ranges = [(v['node_idx_offset'], v['node_idx_offset']+v['num_nodes']) for name, v in info.items()
                            if name.rsplit(':', 1)[-1].lower() == 'test']

    def contexts(self, roots):
        """The context batch of up to 8 roots, as the encoder reads it."""
        batch = dict(self.sample(roots), task_horizon=torch.tensor(float(self.horizon), dtype=torch.float64),
                     task_horizon_valid=True, timestamp_units_per_day=86400.)
        batch = contract_view(hide_test_labels(batch, self.test_ranges))
        batch = hide_test_labels(batch, self.test_ranges, restore=True)
        hidden = batch['masks'].bool() | batch['encoder_hidden_cells'] | batch['is_targets'] | batch['hidden_label_cells']
        if 'declared_drop_cells' in batch:
            hidden |= batch['declared_drop_cells']
        # The FP32 channel carries visible numbers only.
        numbers = batch['number_values_fp32'].masked_fill((hidden | (batch['sem_types'] != 0))[..., None], 0)
        return dict(batch, number_values_fp32=numbers)


def indirect_streams(graph, directions, paths, info):
    """Sampling CSR with an extra reverse edge from the entity to each event row reached through an intermediate table.

    A path [event table, intermediate, ..., entity table] follows each row's foreign keys; the row's sampling clock
    becomes the latest clock of the intermediate rows it joins, and the reverse edges are re-sorted by
    (source, clock, destination)."""
    ranges = {}
    for name, v in info.items():
        ranges.setdefault(name.split(':')[0], []).append((v['node_idx_offset'], v['node_idx_offset']+v['num_nodes']))
    inside = lambda ids, table: torch.stack([(ids >= lo) & (ids < hi) for lo, hi in ranges[table]]).any(0)
    n = len(graph.row_time)
    ids = torch.arange(n)
    ptr, dst = directions.fptr.long().cpu(), directions.fdst.long().cpu()
    src = torch.repeat_interleave(ids, ptr.diff())
    row_time = graph.row_time.cpu().long()
    available = row_time.clone()
    added = []
    for path in paths:
        nodes = ids[inside(ids, path[0])]
        current, valid, time = nodes.clone(), torch.ones(len(nodes), dtype=torch.bool), row_time[nodes].clone()
        for parent in path[1:]:
            edge = inside(dst, parent)
            edge = torch.nonzero(edge).flatten()
            # A row with several foreign keys into `parent` follows its last one in CSR order.
            last = torch.full((n,), -1, dtype=torch.long).scatter_reduce(0, src[edge], edge, 'amax')
            mapping = torch.where(last >= 0, dst[last.clamp_min(0)], -1)
            step = mapping[current]
            valid &= step >= 0
            current = step.clamp_min(0)
            if parent != path[-1]:
                time = torch.maximum(time, row_time[current])
        available[nodes] = time
        direct = inside(dst, path[-1])
        new = valid & ~torch.isin(nodes*n+current, src[direct]*n+dst[direct])
        added.append(torch.stack((current[new], nodes[new]), 1))
    reverse = torch.repeat_interleave(ids, directions.pptr.long().cpu().diff())
    pairs = torch.cat([torch.stack((reverse, directions.pdst.long().cpu()), 1), *added])
    times = available[pairs[:, 1]]
    order = torch.argsort(pairs[:, 1], stable=True)
    order = order[torch.argsort(times[order], stable=True)]
    order = order[torch.argsort(pairs[order, 0], stable=True)]
    pairs = pairs[order]
    out = copy.copy(directions)
    out.pptr = torch.cat((ids.new_zeros(1), torch.bincount(pairs[:, 0], minlength=n).cumsum(0))).to(directions.pptr.dtype)
    out.pdst = pairs[:, 1].to(directions.pdst.dtype).contiguous()
    out.ptime = available[pairs[:, 1]].to(directions.ptime.dtype).contiguous()
    return out


def hide_test_labels(batch, test_ranges, *, restore=False):
    """Hides every label of a TEST-range row that is not mature before its query's cutoff.

    `restore` un-hides the mature labels an earlier pass hid, except the focal, decoder, dropped and hidden-label cells."""
    test = torch.zeros_like(batch['is_padding'])
    for lo, hi in test_ranges:
        test |= (batch['node_idxs'] >= lo) & (batch['node_idxs'] < hi)
    labels = test & batch['retrieved_label_cells'].bool() & ~batch['is_padding'].bool()
    focal = batch['is_targets'].bool() & ~batch['is_padding'].bool()
    times = batch['timestamps']
    cutoff = times.gather(1, focal.long().argmax(1, keepdim=True))
    legal = (times != STATIC_TIME) & (cutoff != STATIC_TIME) & (times < cutoff) & (batch['label_available_at'] < cutoff)
    hidden = batch.get('encoder_hidden_cells', torch.zeros_like(labels)).bool()
    if restore:
        protected = batch['masks'].bool() | focal | batch['hidden_label_cells'].bool()
        if 'declared_drop_cells' in batch:
            protected |= batch['declared_drop_cells'].bool()
        hidden = hidden & ~(labels & legal & ~protected)
    return dict(batch, encoder_hidden_cells=hidden | (labels & ~legal))


def contract_view(batch):
    """The targeted-inference encoder input: the focal answer and its same-row or not-yet-earlier copies hidden."""
    view = dict(batch)
    for key in ('number_values', 'boolean_values'):
        view[key] = torch.nan_to_num(view[key], nan=0.0, posinf=0.0, neginf=0.0)
    padding = view['is_padding'].bool()
    targets = view['is_targets'].bool() & ~padding
    view['masks'] = targets.to(batch['masks'].dtype)
    view['is_support_seed'] = torch.zeros(len(targets), dtype=torch.bool)
    pos = targets.long().argmax(1, keepdim=True)
    same_column = ((view['col_name_idxs'] == view['col_name_idxs'].gather(1, pos))
                   & (view['table_name_idxs'] == view['table_name_idxs'].gather(1, pos))
                   & targets.any(1, keepdim=True) & ~padding)
    same_row = view['node_idxs'] == view['node_idxs'].gather(1, pos)
    hidden = targets | (same_column & same_row)
    times = view['timestamps']
    focal_time = times.gather(1, pos)
    unknown = (times == STATIC_TIME) | (times == -(2**63))
    hidden |= same_column & ~same_row & (unknown | unknown.gather(1, pos) | (times >= focal_time))
    view['encoder_hidden_cells'] = hidden | view['encoder_hidden_cells'].bool()
    return view
