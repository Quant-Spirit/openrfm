"""Training tasks of one episode: the world's own task, a past and a future label program, and a column of the
LLM-written SQL target bank. Bank and program values are labels only, never inputs."""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch

from openrfm.models.memory.context import signed_log
from openrfm.train.memory.programs import draw_live_programs, execute_world, forward_programs

# Bank columns that measure a time since an event; a missing value means "never" and becomes 100 years.
FILL_LARGE = re.compile('since|recency|age|gap|last')
REQUEST_KEYS = ('entity_id', 'timestamp')


def task_rows(root, db):
    """(db, table, target column, drop columns) of each task in the archive's task metadata."""
    data = json.loads((Path(root)/db/'task_metadata.json').read_text())
    return [(db, t['table'], t['target_column'], t.get('drop_columns', [])) for t in data['tasks']]


def program_tasks(world, rows, cutoffs, k, seed, bank_targets, cfg):
    """[(kind, y)] for a past program, a future program (labels over (cutoff, cutoff+H]) and an LLM bank column,
    each a random live column, as a regression or a median-split classification task."""
    rng = np.random.default_rng([seed, 7])
    options = dict(scoped_share=cfg['scoped_share'], count_share=cfg['count_share'])
    past = draw_live_programs(world, rows[:k], cutoffs[:k], cfg['programs'], seed, groups=cfg['program_groups'],
                              **options)
    ahead = forward_programs(world, rows[:k], cutoffs[:k], cfg['programs'], seed+3, cfg['forward_days'], **options)
    targets = [execute_world(world, rows, cutoffs, past),
               execute_world(world, rows, cutoffs+int(cfg['forward_days']*86400), ahead)]
    if bank_targets is not None and bank_targets.shape[1]:
        targets.append(bank_targets.float())
    tasks = []
    for target in targets:
        target = signed_log(target.float().nan_to_num(0., 0., 0.))
        live = torch.where((target[:k].std(0) > 1e-6) & (target[k:].std(0) > 1e-6))[0]
        if not len(live):
            continue
        y = target[:, int(live[int(rng.integers(len(live)))])]
        if rng.random() < .5:
            tasks.append(('regression', y))
            continue
        label = (y > y[:k].median()).long()
        if 0 < int(label[:k].sum()) < k and 0 < int(label[k:].sum()) < len(label)-k:
            tasks.append(('classification', label))
    return tasks


def scorable(tasks, k):
    """Tasks the Kumo head can score: a classification task needs every query class in the support and at most 10
    classes; its labels become codes 0..C-1."""
    out = []
    for kind, y in tasks:
        if kind == 'classification':
            if not bool(torch.isin(y[k:], y[:k]).all()) or len(y.unique()) > 10:
                continue
            y = y.unique(return_inverse=True)[1]
        out.append((kind, y))
    return out


def load_bank(path):
    doc = json.loads(Path(path).read_text())
    if doc['schema'] != 'temporal_llm_bank_v1':
        raise ValueError('unsupported LLM bank')
    return doc['blocks']


def task_views(con, raw_db, task):
    """Canonical views (hub, entity, event, labels) over a memory world's Parquet tables, and its task requests."""
    tables = {p.stem: str(p) for p in (Path(raw_db)/'db').glob('*.parquet')}
    entity, event, table = task['forecast_entity_table'], task['forecast_event_table'], task['table']
    hub = next(t for t in tables if t not in (entity, event) and not t.startswith('v6_task_'))
    available = f"(date::TIMESTAMP + INTERVAL ({int(task['label_horizon_seconds'])}) SECOND)"
    con.execute(f"CREATE OR REPLACE VIEW hub AS SELECT * FROM read_parquet('{tables[hub]}')")
    con.execute(f"CREATE OR REPLACE VIEW entity AS SELECT * FROM read_parquet('{tables[entity]}')")
    con.execute(f"CREATE OR REPLACE VIEW event AS SELECT * REPLACE (date::TIMESTAMP AS date) "
                f"FROM read_parquet('{tables[event]}')")
    con.execute(f"CREATE OR REPLACE VIEW labels AS SELECT entity_id, hub, date::TIMESTAMP AS date, {available} AS "
                f"available_at, CAST({task['target_column']} AS DOUBLE) AS y FROM read_parquet('{tables[table]}')")
    con.execute(f"CREATE OR REPLACE TEMP TABLE eval_table AS SELECT DISTINCT entity_id, date::TIMESTAMP AS timestamp "
                f"FROM read_parquet('{tables[table]}') ORDER BY 1, 2")


def run_block(con, sql):
    """{column: values} in request order, or None when the block fails or does not return one row per request."""
    try:
        frame = con.execute(sql).df()
    except Exception:
        return None
    n = con.execute('SELECT count(*) FROM eval_table').fetchone()[0]
    if not set(REQUEST_KEYS) <= set(frame.columns) or len(frame) != n or frame.duplicated(list(REQUEST_KEYS)).any():
        return None
    frame = frame.sort_values(list(REQUEST_KEYS)).reset_index(drop=True)
    columns = {}
    for name in frame.columns:
        if name in REQUEST_KEYS:
            continue
        values = pd.to_numeric(frame[name], errors='coerce') if frame[name].dtype == object else frame[name]
        if pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values):
            columns[name] = values.to_numpy(dtype=np.float64, na_value=np.nan)
    return columns


def fill(name, values):
    values = np.where(np.isfinite(values), values, np.nan)
    return np.nan_to_num(values, nan=36500. if FILL_LARGE.search(name) else 0.)


def execute_bank(raw_db, tasks, blocks, seed, *, columns):
    """Per task table: up to `columns` target columns from a seeded order of the bank's blocks, with the request
    keys. Blocks that do not read labels are run once per world."""
    import duckdb
    order = np.random.default_rng([seed, 84099]).permutation(len(blocks))
    con = duckdb.connect(':memory:', config=dict(threads=1, memory_limit='4GB'))
    results, shared = {}, {}
    for task in tasks:
        task_views(con, raw_db, task)
        keys = con.execute('SELECT entity_id, epoch(timestamp) FROM eval_table ORDER BY 1, 2').fetchnumpy()
        output, used = [], 0
        for index in order:
            if used >= columns:
                break
            block = blocks[index]
            if block['id'] in shared:
                out = shared[block['id']]
            else:
                out = run_block(con, block['sql'])
                if 'labels' not in block['sql']:
                    shared[block['id']] = out
            kept = {name: fill(name, values) for name, values in (out or {}).items() if name in block['columns']}
            if kept:
                output.append(np.stack(list(kept.values()), 1))
                used += len(kept)
        results[task['table']] = dict(entity=keys['entity_id'].astype(np.int64),
                                      time=keys[list(keys)[1]].astype(np.float64),
                                      targets=np.concatenate(output, 1).astype(np.float32) if output else None)
    con.close()
    return results


def align_targets(result, entity, seconds):
    """Row of the bank matrix for each (entity, cutoff) root, -1 when absent. Clocks are whole seconds, truncated."""
    lookup = {(e, int(t)): i for i, (e, t) in enumerate(zip(result['entity'].tolist(), result['time'].tolist()))}
    return np.array([lookup.get((e, int(t)), -1) for e, t in zip(np.asarray(entity).tolist(),
                                                                  np.asarray(seconds).tolist())])
