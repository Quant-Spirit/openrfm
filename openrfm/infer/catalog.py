"""Task rows of an archive (labels, node ids, cutoffs, entities per split) and the inference cohort."""
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from openrfm.archive.format import archive_root
from openrfm.infer.predict import nested_order

SPLITS = ('train', 'val', 'test')


def first_time(data, task, split):
    """The earliest row time of one task split, as `YYYY-MM-DD HH:MM:SS`."""
    path = Path(data)/task['db']/'tasks'/task['table']/f'{split}.parquet'
    clock = json.loads(pq.read_schema(path).metadata[b'time_col'])
    return pc.min(pq.read_table(path, columns=[clock])[clock]).as_py().strftime('%Y-%m-%d %H:%M:%S')


def split_rows(base, data, task, split, row_time):
    """(labels, node ids, cutoffs, entity keys) of one split; archived task rows follow the parquet row order. Labels are
    None for a split without the target column."""
    info = json.loads((archive_root(base)/'table_info.json').read_text())
    offset = info[f"{task['table']}:{split.title()}"]['node_idx_offset']
    path = Path(data)/'tasks'/task['table']/f'{split}.parquet'
    keys = json.loads(pq.read_schema(path).metadata[b'fkey_col_to_pkey_table'])
    entity = next(column for column, parent in keys.items() if parent == task['entity_table'])
    table = pq.read_table(path, columns=[c for c in (task['target'], entity) if c in pq.read_schema(path).names])
    nodes = offset+np.arange(table.num_rows, dtype=np.int64)
    labels = None
    if task['target'] in table.column_names:
        labels = table[task['target']].to_numpy()
        # Regression targets are FP32 values, as the archive stores them.
        labels = (labels.astype(np.int64) if task['task_type'] == 'classification'
                  else labels.astype(np.float32).astype(np.float64))
    return labels, nodes, row_time[nodes].numpy(), table[entity].to_numpy()


def support_pool(ids, labels, cap, task_type, extend=0):
    """At most `cap` rows: a class-alternating seeded prefix, in node order. With `extend` > `cap`, further rows chosen
    the same way among the rest follow in node order, up to `extend` rows in all."""
    pool = np.sort(ids[nested_order(labels, 0, task_type)[:cap]])
    if extend <= cap:
        return pool
    rest = ~np.isin(ids, pool)
    more = np.sort(ids[rest][nested_order(labels[rest], 0, task_type)[:extend-len(pool)]])
    return np.concatenate((pool, more))


def cohort(sampler, data, task, cap, extend=0):
    """Support pools and TEST queries.

    TRAIN rows whose label window ends by the first VAL cutoff and VAL rows whose window ends by the first TEST cutoff
    form the support (each split capped at `cap`, or extended to `extend` rows); only roots the sampler admits enter.
    TEST queries are all TEST rows the sampler admits."""
    rows = {s: split_rows(sampler.base, data, task, s, sampler.graph.row_time) for s in SPLITS}
    admitted = sampler.roots.numpy()
    horizon = task['horizon_s']
    (y, ids, t, _), (vy, vi, vt, _), (ty, ti, tt, te) = (rows[s] for s in SPLITS)
    keep = (t+horizon <= vt.min()) & np.isin(ids, admitted)
    train = support_pool(ids[keep], y[keep], cap, task['task_type'], extend)
    keep = (vt+horizon <= tt.min()) & np.isin(vi, admitted)
    extra = (support_pool(vi[keep], vy[keep], cap, task['task_type'], extend) if keep.any()
             else np.zeros(0, dtype=np.int64))
    labels = {**dict(zip(ids.tolist(), y.tolist())), **dict(zip(vi.tolist(), vy.tolist()))}
    test_rows = np.flatnonzero(np.isin(ti, admitted))
    test = ti[test_rows]
    support = np.concatenate((train, extra))
    times = {**dict(zip(ids.tolist(), t.tolist())), **dict(zip(vi.tolist(), vt.tolist()))}
    return dict(train=train, val_support=extra, test=test, support=support,
                support_labels=np.asarray([labels[n] for n in support.tolist()]),
                support_times=np.asarray([times[n] for n in support.tolist()], dtype=np.int64),
                support_val=np.r_[np.zeros(len(train), bool), np.ones(len(extra), bool)],
                test_labels=None if ty is None else ty[test_rows], test_entities=te[test_rows], test_rows=test_rows)
