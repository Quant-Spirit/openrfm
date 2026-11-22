"""`openrfm predict`: one benchmark task end to end under a released recipe."""
import json
from pathlib import Path

import numpy as np
import torch

from openrfm import hub
from openrfm.archive.history import decode, history_world
from openrfm.infer import catalog, download, features, predict, prepare, recipe, residual
from openrfm.infer.contexts import TaskSampler
from openrfm.infer.snapshot import table_ranges
from openrfm.models import load_backbone
from openrfm.models.memory import load_memory


def models(checkpoints):
    """The frozen backbone, the memory module bound to it and the record-bank projection."""
    path = hub.checkpoint('backbone', checkpoints)
    rt = load_backbone(path, 'cuda').requires_grad_(False)
    memory = load_memory(hub.checkpoint('memory', checkpoints)).cuda().eval().bind(rt)
    handoff = torch.load(path, map_location='cuda', weights_only=True)['rt_handoff']['state_dict']
    return rt, memory, handoff


def exports(task, cell, proto, data, work, checkpoints):
    """(support features, support labels, query features, query labels, query entities, cohort rows) of `task`."""
    from openrfm.sampler import episodes
    episodes.configure_backend('native', build=True)
    base, raw = prepare.archive(task, cell, proto, data, work)
    sampler = TaskSampler(base, proto, recipe.load('sampler.yaml'), cell['L'], original_schema=cell['schema'] != 'program')
    caps = recipe.load('tasks.yaml')
    extend = caps['expanded_support_cap'] if cell['support'] == 'expanded' else 0
    rows = catalog.cohort(sampler, raw, proto, caps['support_cap'], extend)
    rt, memory, handoff = models(checkpoints)
    records = cell.get('records')
    splits = [rows['train'], rows['test'], rows['val_support']]
    world = history_world(decode(base), sampler.graph, rt, 'cpu')
    x, out = features.encode(sampler, rt, memory, world, splits, records, table_ranges(base))
    a, b = len(rows['train']), len(rows['train'])+len(rows['test'])
    order = np.r_[np.arange(a), np.arange(b, len(x)), np.arange(a, b)]
    x = x[order]
    if records is not None:
        x = features.hybrid(x, features.record_bank(rt, handoff, [out[i] for i in order]), cell['bank_width'])
    n = len(rows['support'])
    return x[:n], rows['support_labels'], x[n:], rows['test_labels'], rows['test_entities'], rows


def run(task, recipe_name, data, work, *, checkpoints=None, spec=None):
    """Predicts every TEST row of benchmark task `task` with recipe `recipe_name` and prints the metric. With `spec`, a
    YAML file declaring a task of one's own database `data/<db>` (the keys of openrfm/configs/infer/tasks.yaml), the
    task is predicted under the recipe's default cell and scored when test.parquet holds the target."""
    cell, proto = recipe.task_config(task, recipe_name, spec)
    if spec is None:
        download.download(proto['db'], data)
    else:
        task = f"{proto['db']}/{proto['table']}"
    # R3 reads the R2 exports of the task.
    sx, sy, qx, qy, entity, rows = exports(task, cell, proto, data, Path(work)/('r2' if recipe_name == 'r3' else recipe_name),
                                           checkpoints)
    work = Path(work)/recipe_name
    kind = proto['task_type']
    draws = predict.draw_predictions(dict(sx=sx, sy=sy, qx=qx), cell, kind)
    if 'head' in cell:
        head = residual.train(sx, sy, rows['support_times'], rows['support_val'], proto['horizon_s'], kind, cell)
        p = residual.apply(head, draws, sx, sy, qx, cell, kind)
    else:
        p = predict.ensemble(draws, kind)
    result = dict(task=task, recipe=recipe_name, split='test', metric='AUROC' if kind == 'classification' else 'MAE',
                  rows=len(p))
    if qy is not None:
        result.update(value=predict.metric(qy, p, kind), SE=predict.metric_se(qy, p, entity, kind))
    out = work/task.replace('/', '__')/'result.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1)+'\n')
    np.savez_compressed(out.with_name('predictions.npz'), p=p, entity=entity, row=rows['test_rows'],
                        **({} if qy is None else dict(y=qy)))
    print(json.dumps(result), flush=True)
    return result
