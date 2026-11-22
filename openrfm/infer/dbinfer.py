"""4DBInfer databases in RelBench layout.

`convert` rewrites one extracted 4DBInfer download (`<source>/metadata.yaml` and its Parquet files) as
`<db>/db/*.parquet` tables and `<db>/tasks/<task>/{train,val,test}.parquet` splits. Keys become row positions; a key
that is referenced but absent from its table (or from a table the download does not ship) appends a row holding only
that key; columns that store a row's final state are dropped; labels and clocks are kept as given. An event outcome
column that is also a task label is written to `<db>/history/<table>.parquet`, keeping only the values whose row
time + 1 s is before the first VAL cutoff; the `repair` schema reads those tables."""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from openrfm.data.layout import write

# Per database: the download directory; per official task its name, entity table, entity key and label window
# (seconds); columns holding a row's final state; event outcomes per table (column, stored type).
DATABASES = {
    'amazon-4db': dict(source='amazon', tasks={'churn': ('user-churn', 'Customer', 'customer_id', 92*86400)}),
    'outbrain-4db': dict(source='outbrain-small', tasks={'ctr': ('ad-ctr', 'Event', 'display_id', 1)},
                         final_state={'Click': {'clicked'}}, outcomes={'Click': ('clicked', pa.int64())}),
    'retailrocket-4db': dict(source='retailrocket', tasks={'cvr': ('item-cvr', 'Item', 'itemid', 1)},
                             final_state={'View': {'added_to_cart'}}, outcomes={'View': ('added_to_cart', pa.bool_())}),
    'stackexchange-4db': dict(source='stackexchange', tasks={'upvote': ('post-upvote', 'Posts', 'Id', 366*86400),
                                                             'churn': ('user-churn', 'Users', 'Id', 731*86400)},
                              final_state={'Posts': {'AcceptedAnswerId', 'LastEditorUserId'}}),
}
SPLITS = (('train', 'train'), ('val', 'validation'), ('test', 'test'))


def task_keys(task):
    """(column, parent table, parent key) of each key column of a task."""
    for c in task['columns']:
        if c['dtype'] == 'foreign_key':
            yield (c['name'], *c['link_to'].split('.'))
        elif c['dtype'] == 'primary_key':
            yield c['name'], task['target_table'], c['name']


def key_universes(tables, pkeys, fkeys, tasks, splits):
    """{parent table: all its keys}: the table's own keys, then each referenced key it lacks, in reference order.
    Absent rows are appended to `tables`; a table the download does not ship is created from its keys."""
    refs = {}
    for name, cols in fkeys.items():
        for col, (parent, pk) in cols.items():
            refs.setdefault((parent, pk), []).append(tables[name][col])
    for task in tasks:
        for col, parent, pk in task_keys(task):
            refs.setdefault((parent, pk), []).extend(splits[task['name'], s][col] for s, _ in SPLITS)
    keys = {}
    for (parent, pk), arrays in refs.items():
        original = tables[parent][pk] if parent in tables else pa.chunked_array([], type=arrays[0].type)
        kind = original.type if len(original) else arrays[0].type
        chunks = [c for a in arrays for c in pc.cast(a, kind).chunks]
        union = pc.drop_null(pc.unique(pa.chunked_array(chunks, type=kind)))
        extra = union.filter(pc.is_null(pc.index_in(union, value_set=original)))
        keys[parent] = pa.chunked_array(original.chunks+[extra], type=kind)
        if parent in tables:
            old = tables[parent]
            tables[parent] = pa.concat_tables([old, pa.table({c: extra if c == pk else pa.nulls(len(extra), old[c].type)
                                                              for c in old.column_names})])
        else:
            tables[parent], fkeys[parent] = pa.table({pk: keys[parent]}), {}
        pkeys[parent] = pk
    return keys


def positions(table, col, keys):
    return table.set_column(table.schema.get_field_index(col), col,
                            pc.cast(pc.index_in(pc.cast(table[col], keys.type), value_set=keys), pa.int64()))


def history(raw, table, outcome, kind, clock, cutoff):
    """The table with its outcome column restored where row time + 1 s < `cutoff` (seconds), null elsewhere."""
    values = pq.read_table(raw, columns=[outcome])[outcome].cast(kind)
    seconds = table[clock].cast(pa.timestamp('s'), safe=False).cast(pa.int64())
    mature = pc.less(pc.add(seconds, 1), cutoff)
    return table.append_column(outcome, pc.if_else(mature, values, pa.scalar(None, kind)))


def convert(db, raw, out):
    """Writes the 4DBInfer database `db`, extracted under `raw`, in RelBench layout to `out/<db>`."""
    spec = DATABASES[db]
    source, dest = Path(raw)/spec['source'], Path(out)/db
    meta = yaml.safe_load((source/'metadata.yaml').read_text())
    final_state = spec.get('final_state', {})
    tables, pkeys, fkeys, clocks, files = {}, {}, {}, {}, {}
    for t in meta['tables']:
        name = t['name']
        cols = [c['name'] for c in t['columns'] if c['name'] not in final_state.get(name, set())]
        files[name], clocks[name] = source/t['source'], t.get('time_column')
        tables[name] = pq.read_table(files[name], columns=cols)
        pkeys[name] = next((c['name'] for c in t['columns'] if c['dtype'] == 'primary_key'), None)
        fkeys[name] = {c['name']: c['link_to'].split('.') for c in t['columns']
                       if c['dtype'] == 'foreign_key' and c['name'] in cols}
    tasks = [t for t in meta['tasks'] if t['name'] in spec['tasks']]
    splits = {(t['name'], s): pq.read_table(source/t['source'].format(split=official),
                                            columns=[c['name'] for c in t['columns']])
              for t in tasks for s, official in SPLITS}
    keys = key_universes(tables, pkeys, fkeys, tasks, splits)
    for name, table in tables.items():
        for col, (parent, _) in fkeys[name].items():
            table = positions(table, col, keys[parent])
        if pkeys[name]:
            table = table.set_column(table.schema.get_field_index(pkeys[name]), pkeys[name],
                                     pa.array(np.arange(len(table), dtype=np.int64)))
        tables[name] = table
        write(table, dest/'db'/f'{name}.parquet', pkeys[name], {c: p[0] for c, p in fkeys[name].items()},
              clocks.get(name))
    metadata = {}
    for t in tasks:
        alias, entity, entity_col, horizon = spec['tasks'][t['name']]
        metadata[alias] = dict(task_type='binary_classification', entity_table=alias, focal_entity_table=entity,
                               entity_col=entity_col, target_col=t['target_column'], time_col=t['time_column'],
                               drop_columns=[], label_horizon_seconds=horizon, official_task=t['name'],
                               official_metric=t['evaluation_metric'])
        for s, _ in SPLITS:
            split, parents = splits[t['name'], s], {}
            for col, parent, _ in task_keys(t):
                split, parents[col] = positions(split, col, keys[parent]), parent
            write(split, dest/'tasks'/alias/f'{s}.parquet', None, parents, t['time_column'])
    (dest/'task_metadata.json').write_text(json.dumps({db: metadata}, indent=2))
    first_val = min(pc.min(splits[t['name'], 'val'][t['time_column']]).as_py() for t in tasks)
    cutoff = int(np.datetime64(first_val, 's').astype(np.int64))
    for name, (outcome, kind) in spec.get('outcomes', {}).items():
        (dest/'history').mkdir(exist_ok=True)
        restored = history(files[name], pq.read_table(dest/'db'/f'{name}.parquet'), outcome, kind, clocks[name], cutoff)
        pq.write_table(restored, dest/'history'/f'{name}.parquet', compression='zstd')
    return dest
