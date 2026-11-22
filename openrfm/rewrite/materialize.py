"""Write an evaluated schema program as an RT database."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from openrfm.rewrite.engine import Engine, evaluate
from openrfm.rewrite.schema import FINAL_STATE, PER_CUTOFF_MAX, RUSTLER_REQUIRED, digest, ident, lit, parquet_meta


def _write(con, query, path, pkey, fkeys, time_col):
    import pyarrow.parquet as pq
    # rustler reads timestamps, doubles/ints, booleans and strings: DATE -> TIMESTAMP, DECIMAL/HUGEINT -> DOUBLE,
    # INTERVAL/TIME -> seconds
    casts = []
    for name, typ, *_ in con.execute(f'DESCRIBE ({query})').fetchall():
        c = ident(name)
        if typ == 'DATE':
            casts.append(f'CAST({c} AS TIMESTAMP) AS {c}')
        elif typ.startswith('DECIMAL') or typ in ('HUGEINT', 'UHUGEINT'):
            casts.append(f'CAST({c} AS DOUBLE) AS {c}')
        elif typ in ('INTERVAL', 'TIME'):
            casts.append(f'CAST(epoch({c}) AS DOUBLE) AS {c}')
    if casts:
        query = f'SELECT * REPLACE ({", ".join(casts)}) FROM ({query})'
    table = con.execute(query).fetch_arrow_table()
    meta = dict(table.schema.metadata or {})
    meta.update({b'pkey_col': json.dumps(pkey).encode(), b'fkey_col_to_pkey_table': json.dumps(fkeys).encode(),
                 b'time_col': json.dumps(time_col).encode()})
    pq.write_table(table.replace_schema_metadata(meta), path, compression='zstd')
    return table.num_rows


def materialize(program, info, out, *, case_rows=None, memory='24GB'):
    """Write <out>/data/<db>/{db,tasks/<task>} plus schema_manifest.json; the task files are copied unchanged."""
    import numpy as np
    out = Path(out)
    data = out / 'data' / info['db']
    if data.exists():
        shutil.rmtree(data)
    (data / 'db').mkdir(parents=True)
    (data / 'tasks' / info['task']).mkdir(parents=True)
    start = time.monotonic()
    engine = Engine(info, memory, spill=str(out / 'duckdb_spill'))
    engine.load_tasks(case_rows)
    results = evaluate(engine, program)
    tcol = info['task_time']
    for split in ('train', 'val', 'test'):
        shutil.copy2(Path(info['source']) / 'tasks' / info['task'] / f'{split}.parquet', data / 'tasks' / info['task'] / f'{split}.parquet')
    manifest = dict(kind='schema_archive_v1', task=info['key'], program=program, program_sha256=digest(program),
                    case_rows=str(case_rows) if case_rows else 'all_task_rows', strict_final_state=True,
                    final_state_masked=FINAL_STATE.get(info['db'], {}), focal_own_row_readable=True,
                    tables={}, generated_columns={})
    entity_tables = {}
    for o in program['objects']:
        if o['type'] != 'keep_table':
            continue
        t, m = o['table'], info['tables'][o['table']]
        cols = list(m['columns']) if o.get('columns', 'all') == 'all' else list(o['columns'])
        cols = list(dict.fromkeys(([m['pkey']] if m['pkey'] else []) + list(m['fkeys']) + ([m['time']] if m['time'] else []) + cols
                                  + [c for c in RUSTLER_REQUIRED.get((info['db'], t), ()) if c in m['columns']]))
        engine.source_views(None)
        # All rows are kept: RelBench primary keys are dense row indices that the archive addresses directly,
        # so a row subset would need re-keying, which would change the fixed task table's FK values.
        where = ''
        kept_fk = {c: p for c, p in m['fkeys'].items() if p in {k['table'] for k in program['objects'] if k['type'] == 'keep_table'}}
        # Rows keep the source order: rustler addresses a row by its position, and FK values are those positions.
        engine.x('SET preserve_insertion_order=true')
        order = f' ORDER BY {ident(m["pkey"])}' if m['pkey'] else ''
        n = _write(engine.con, f'SELECT {", ".join(ident(c) for c in cols if c in kept_fk or c not in m["fkeys"])} FROM {ident(t)}{where}{order}',
                   data / 'db' / f'{t}.parquet', m['pkey'], kept_fk, m['time'])
        engine.x('SET preserve_insertion_order=false')
        if m['pkey']:
            import pyarrow.parquet as pq
            pk_values = pq.read_table(data / 'db' / f'{t}.parquet', columns=[m['pkey']]).column(0).to_numpy()
            assert (pk_values == np.arange(len(pk_values))).all(), f'{t}: primary key is not the row position'
        manifest['tables'][t] = dict(origin='original', rows=n, rows_policy='all')
        entity_tables[t] = m['pkey']
    for o in program['objects']:
        if o['type'] == 'keep_table':
            continue
        name, ent = o['name'], o['entity']
        parent = info['task_fkeys'][ent]
        src = results[name]
        if engine.x(f'SELECT count(*) FROM {src}').fetchone()[0] == 0:  # rustler cannot build an empty table
            manifest.setdefault('dropped_empty', []).append(name)
            continue
        value_cols = [r[0] for r in engine.x(f'DESCRIBE {src}').fetchall() if r[0] not in (ent, tcol)]
        fks = {ent: parent} | {c: p for c, p in (o.get('fkeys') or {}).items()}
        pk = f'{name}_id'
        values = []
        for c in value_cols:
            if c in fks:  # an FK value without a kept parent row becomes NULL
                parent_pk = info['tables'][fks[c]]['pkey']
                path = data / 'db' / f'{fks[c]}.parquet'
                values.append(f'CASE WHEN {ident(c)} IN (SELECT {ident(parent_pk)} FROM read_parquet({lit(path)})) THEN {ident(c)} END AS {ident(c)}')
            else:
                values.append(ident(c))
        # rows of one (entity, time) get keys in content order, so the archive does not depend on scan/join order
        order = ', '.join([ident(ent), ident(tcol)] + [ident(c) for c in value_cols])
        query = (f'SELECT row_number() OVER (ORDER BY {order}) - 1 AS {ident(pk)}, {ident(ent)}, '
                 f'{ident(tcol)} - INTERVAL 1 SECOND AS as_of{"".join(", " + v for v in values)} FROM {src} ORDER BY 1')
        n = _write(engine.con, query, data / 'db' / f'{name}.parquet', pk, fks, 'as_of')
        manifest['tables'][name] = dict(origin='generated', type=o['type'], rows=n, entity_fk=ent, entity_table=parent,
                                        time_col='as_of', grain='entity_asof' if o['type'] == 'snapshot_table' else 'entity_asof_multi',
                                        hops_from_task_row=2)
        manifest['generated_columns'][name] = [c for c in value_cols if c not in fks]
        bad = engine.x(f'SELECT count(*) FROM {src} WHERE {ident(tcol)} - INTERVAL 1 SECOND >= {ident(tcol)}').fetchone()[0]
        assert bad == 0
        if o['type'] == 'snapshot_table':  # one row per (entity, as_of) at RT's whole-second time resolution
            dup = engine.x(f"SELECT count(*) FROM (SELECT 1 FROM {src} GROUP BY {ident(ent)}, date_trunc('second', {ident(tcol)}) HAVING count(*) > 1)").fetchone()[0]
            if dup:
                raise ValueError(f'{name}: {dup} (entity, as_of second) pairs hold several rows; a snapshot needs one row per entity and second')
    meta_path = Path(info['source']) / 'task_metadata.json'
    tasks = [info['task']]
    if meta_path.exists():
        # 4DBInfer registry, kept byte-identical with every listed task's files: the loader derives the dataset's
        # val/test timestamps from all listed tasks (a dropped sibling shrinks them below a task's horizon).
        shutil.copy2(meta_path, data / 'task_metadata.json')
        tasks = list(json.loads(meta_path.read_text())[info['db']])
        for t in tasks:
            if t != info['task']:
                shutil.copytree(Path(info['source']) / 'tasks' / t, data / 'tasks' / t)
                # a sibling task's parent the program did not keep: key and time only, no FK edges, so it is reachable
                # only from that sibling's rows
                for parent in parquet_meta(Path(info['source']) / 'tasks' / t / 'train.parquet')['fkeys'].values():
                    if parent not in manifest['tables']:
                        m = info['tables'][parent]
                        cols = [c for c in (m['pkey'], m['time']) if c]
                        engine.source_views(None)
                        n = _write(engine.con, f'SELECT {", ".join(ident(c) for c in cols)} FROM {ident(parent)} ORDER BY {ident(m["pkey"])}',
                                   data / 'db' / f'{parent}.parquet', m['pkey'], {}, m['time'])
                        manifest['tables'][parent] = dict(origin='sibling_task_parent', rows=n, columns=cols)
        manifest['sibling_tasks'] = [t for t in tasks if t != info['task']]
    import duckdb
    splits = [Path(info['source']) / 'tasks' / t / f'{s}.parquet' for t in tasks for s in ('val', 'test')]
    fit = duckdb.sql('SELECT min(t) - INTERVAL 1 SECOND FROM (' + ' UNION ALL '.join(  # earliest val/test time of any task
        f'SELECT min({ident(parquet_meta(f)["time"])}) t FROM read_parquet({lit(f)})' for f in splits) + ')').fetchone()[0]
    (out / 'fit_time_cutoffs.json').write_text(json.dumps({info['db']: fit.strftime('%Y-%m-%d %H:%M:%S')}, indent=2) + '\n')  # floor; rustler parses whole seconds
    manifest.update(seconds=time.monotonic() - start, cases=engine.x('SELECT count(*) FROM cases').fetchone()[0],
                    cutoffs=len(engine.cutoffs()), per_cutoff=len(engine.cutoffs()) <= PER_CUTOFF_MAX)
    (out / 'schema_manifest.json').write_text(json.dumps(manifest, indent=1, default=str) + '\n')
    shutil.rmtree(out / 'duckdb_spill', ignore_errors=True)
    return manifest
