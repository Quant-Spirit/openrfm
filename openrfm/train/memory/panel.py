"""Backbone contexts of one training task's rows (`openrfm.sampler.pack`)."""
from dataclasses import replace
import json
from pathlib import Path

import torch

from openrfm.archive.format import archive_root
from openrfm.sampler.pack import ContextSampler, forecast_view
from openrfm.sampler.spec import RTRetrievalSpec


def task_spec(base, entry):
    """The context policy of one task: a forecast task reads its matured label history."""
    if entry.get('task_view') != 'forecast':
        return base
    return forecast_view(base, entry['forecast_entity_table'], entry.get('label_horizon_seconds') or 0)


def public_horizon(entry):
    """The label window (seconds) the backbone's time features are scaled by; None when the task declares none."""
    horizon = entry.get('public_task_horizon_seconds')
    if horizon is None and entry.get('task_view') == 'forecast':
        horizon = entry.get('label_horizon_seconds')
    return horizon if horizon else None


class PanelSampler(ContextSampler):
    """Contexts of `length` cells for rows of task (db, table, target, drops) of archive root `root`."""

    def __init__(self, root, task, sampler, *, length, batch_size=8):
        db, table, column, drops = task
        archive = Path(root)/db
        column_index = json.loads((archive_root(archive)/'column_index.json').read_text())
        entry, = [t for t in json.loads((archive/'task_metadata.json').read_text())['tasks']
                  if (t['table'], t['target_column']) == (table, column)]
        base = replace(RTRetrievalSpec.parse(sampler), seq_len_set=(length,),
                       protected_core_len=min(length, sampler['protected_core_len']), _model_requires_provenance=True)
        self.horizon = public_horizon(entry)
        drop_ids = [column_index[f'{c} of {table}' if isinstance(c, str) else f'{c[1]} of {c[0]}'] for c in drops]
        super().__init__(archive, task_spec(base, entry), column_index[f'{column} of {table}'], drop_ids,
                         batch_size=batch_size)

    def sample(self, seed, roots):
        """The packed context of each root (at most `batch_size`); request salts are `seed`+position."""
        batch = super().sample(roots, seed)
        known = self.horizon is not None
        return dict(batch, task_horizon=torch.as_tensor(self.horizon if known else 1., dtype=torch.float64),
                    task_horizon_valid=known)
