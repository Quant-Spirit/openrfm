"""Validation of schema programs: LLM-written relational rewrites, materialized as focused RT databases.

Per-case objects are evaluated at each case cutoff c on source views filtered to `time < c` and stamped
`as_of = c - 1 s`; matured labels are the only task data a program reads.
"""
from __future__ import annotations

import hashlib
import importlib.resources
import json
import re
from pathlib import Path

CONFIG = Path(str(importlib.resources.files('openrfm')/'configs/rewrite'))
KIND = 'schema_program_v1'
NAME = re.compile(r'^[A-Za-z][A-Za-z0-9_]{0,47}$')
TYPES = ('keep_table', 'snapshot_table', 'evidence_table', 'derived_table')
PER_CUTOFF_MAX = 400          # distinct cutoffs evaluated on clock-filtered views; above it SQL filters itself
MAX_ROWS_PER_CASE = 64
# Final-state study fields (published after the study ends); masked in every derived read (strict_final_state=true).
FINAL_STATE = {'rel-trial': {'studies': ('enrollment', 'enrollment_type', 'fdaaa801_violation', 'plan_to_share_ipd')}}
# Columns rustler pre rewrites in place; a kept table must carry them.
RUSTLER_REQUIRED = {('rel-trial', 'studies'): ('has_dmc',), ('rel-trial', 'eligibilities'): ('adult', 'child', 'older_adult'),
                    ('rel-stack', 'postLinks'): ('LinkTypeId',), ('rel-amazon', 'product'): ('category',)}
NUMERIC = ('TINYINT', 'SMALLINT', 'INTEGER', 'BIGINT', 'HUGEINT', 'UTINYINT', 'USMALLINT', 'UINTEGER', 'UBIGINT',
           'FLOAT', 'DOUBLE', 'DECIMAL', 'BOOLEAN')


def ident(x):
    return '"' + str(x).replace('"', '""') + '"'


def lit(x):
    return "'" + str(x).replace("'", "''") + "'"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def parquet_meta(path):
    import pyarrow.parquet as pq
    schema = pq.read_schema(path)
    m = schema.metadata or {}
    get = lambda k: json.loads(m[k]) if k in m else None
    return dict(pkey=get(b'pkey_col'), fkeys=get(b'fkey_col_to_pkey_table') or {}, time=get(b'time_col'),
                columns={f.name: str(f.type) for f in schema})


def task_info(key, source):
    """Task schema of `key` ('<db>/<task>') read from the dataset directory `source` (RelBench layout)."""
    db, task = key.split('/')
    spec = json.loads((CONFIG / 'tasks.json').read_text())[key]
    src = Path(source)
    m = parquet_meta(src / 'tasks' / task / 'train.parquet')
    tables = {p.stem: parquet_meta(p) for p in sorted((src / 'db').glob('*.parquet'))}
    for col, parent in list(m['fkeys'].items()):
        if parent == 'entity_table':  # rel-avito task files name their parent generically; it is the table keyed by that column
            m['fkeys'][col] = next(t for t, x in tables.items() if x['pkey'] == col)
    return dict(key=key, db=db, task=task, target=spec['target'], task_type=spec['task_type'],
                horizon_s=float(spec['horizon_s']), source=str(src), task_fkeys=m['fkeys'],
                task_time=m['time'], task_columns=m['columns'], tables=tables)


def validate(doc, info):
    """Structural checks; SQL is checked by execution (unknown tables fail) and by the mutation test."""
    err = lambda msg: ValueError('Invalid schema program: ' + msg)
    if doc.get('kind') != KIND or doc.get('dataset') != info['db'] or doc.get('task') != info['task']:
        raise err('kind/dataset/task must match the supplied task exactly')
    objects = doc.get('objects')
    if not isinstance(objects, list):
        raise err('objects must be a list ([] means no rewrite)')
    tables, names = info['tables'], set()
    kept = {o['table'] for o in objects if isinstance(o, dict) and o.get('type') == 'keep_table'}
    if objects and not set(info['task_fkeys'].values()) <= kept:
        raise err('every table the task row points to must be kept: ' + str(sorted(set(info['task_fkeys'].values()))))
    for o in objects:
        if not isinstance(o, dict) or o.get('type') not in TYPES:
            raise err(f'object type must be one of {TYPES}')
        if o['type'] == 'keep_table':
            t = o.get('table')
            if t not in tables:
                raise err(f'unknown table {t}')
            cols = o.get('columns', 'all')
            if cols != 'all' and (not isinstance(cols, list) or not set(cols) <= set(tables[t]['columns'])):
                raise err(f'unknown columns in keep_table {t}')
            if o.get('rows', 'linked') not in ('all', 'linked'):
                raise err('keep_table rows must be all or linked')
            m = tables[t]
            payload = [c for c in m['columns'] if c != m['pkey'] and c not in m['fkeys'] and c != m['time']]
            if t in info['task_fkeys'].values() and payload and cols != 'all' and not set(cols) & set(payload):
                # RT admits a forecast row only when its entity row shows at least one feature cell
                raise err(f'keep_table {t} is a task entity table: keep at least one of its own columns {payload}')
            continue
        name = o.get('name')
        if not isinstance(name, str) or not NAME.match(name) or name in tables or name in names or name == info['task']:
            raise err(f'new table name {name!r} must be a fresh identifier')
        names.add(name)
        if o.get('entity') not in info['task_fkeys']:
            raise err(f'{name}: entity must be one of the task FK columns {sorted(info["task_fkeys"])}')
        if o['type'] == 'snapshot_table':
            if name != info['task_fkeys'][o['entity']] + '__snapshot':
                raise err(f'snapshot table of entity {o["entity"]} must be named {info["task_fkeys"][o["entity"]]}__snapshot (one per entity)')
            blocks = o.get('blocks')
            if not isinstance(blocks, list) or not blocks:
                raise err(f'{name}: blocks required')
            for b in blocks:
                if not isinstance(b, dict) or not NAME.match(str(b.get('name', ''))) or not isinstance(b.get('sql'), str):
                    raise err(f'{name}: each block needs a name and sql')
        elif o['type'] == 'evidence_table':
            src = o.get('source')
            if src not in tables or not tables[src]['time'] or o.get('fk') not in tables[src]['fkeys'] \
                    or tables[src]['fkeys'][o['fk']] != info['task_fkeys'][o['entity']]:
                raise err(f'{name}: source must be a timed table with an FK column to the entity table')
            if not isinstance(o.get('columns'), list) or not set(o['columns']) <= set(tables[src]['columns']):
                raise err(f'{name}: unknown evidence columns')
            if type(o.get('k')) is not int or not 1 <= o['k'] <= MAX_ROWS_PER_CASE:
                raise err(f'{name}: k in [1, {MAX_ROWS_PER_CASE}]')
        else:
            if not isinstance(o.get('sql'), str) or type(o.get('max_rows_per_case')) is not int \
                    or not 1 <= o['max_rows_per_case'] <= MAX_ROWS_PER_CASE:
                raise err(f'{name}: sql and max_rows_per_case in [1, {MAX_ROWS_PER_CASE}] required')
            for col, parent in (o.get('fkeys') or {}).items():
                if parent not in kept:
                    raise err(f'{name}: FK {col} must point to a kept table')
    for o in objects:
        sql = ' '.join(b.get('sql', '') for b in o.get('blocks', [])) + ' ' + o.get('sql', '')
        if re.search(r'\b(train_table|val_table|test_table|task_train|task_val|task_test|all_labels|cases|pool_\w+|read_parquet|read_csv|copy|attach|install|load)\b', sql, re.I):
            raise err('SQL may read only eval_table, matured_labels and database tables')
    return doc


def focal_keys(info):
    """{parent table: [task FK columns]} for the task row's 1-hop parents that have a time column."""
    out = {}
    for fk, parent in info['task_fkeys'].items():
        if info['tables'][parent]['time'] and info['tables'][parent]['pkey']:
            out.setdefault(parent, []).append(fk)
    return out


