"""Mutation test: perturbing or deleting post-cutoff rows must leave every materialized value unchanged."""
from __future__ import annotations

import json
import re
from pathlib import Path

from openrfm.rewrite.engine import Engine, evaluate
from openrfm.rewrite.schema import focal_keys, ident, lit


FLOAT_ORDER_TOL = 1e-12


def _float_drift(before, after, changed):
    """Largest relative change of each changed float column, rows matched on the non-float columns."""
    import numpy as np
    import pyarrow as pa
    keys = [c for c in before.column_names if not pa.types.is_floating(before.schema.field(c).type)]
    floats = [c for c in changed if c in after.column_names and pa.types.is_floating(before.schema.field(c).type)]
    if not floats or before.num_rows != after.num_rows or not keys or set(keys) & set(changed):
        return None
    order = [(c, 'ascending') for c in keys]
    b, a = before.sort_by(order), after.sort_by(order)
    out = {}
    for c in floats:
        x = np.nan_to_num(b.column(c).to_numpy(zero_copy_only=False))
        y = np.nan_to_num(a.column(c).to_numpy(zero_copy_only=False))
        out[c] = float((np.abs(x - y) / np.maximum(np.abs(x), 1e-12)).max())
    return out


def mutation_test(program, info, workdir, *, case_rows=None, n_cutoffs=3, memory='24GB', seed=0,
                  variants=('perturb', 'delete')):
    """Two future-row mutations (perturb, delete) at each tested cutoff must leave the derived rows bit-identical."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    base = Engine(info, memory)
    base.load_tasks(case_rows)
    allc = base.cutoffs()
    rng = np.random.default_rng(seed)
    picks = sorted({allc[0], allc[-1], allc[len(allc) // 2]} | set(rng.choice(allc, min(n_cutoffs, len(allc)), replace=False).tolist()))[:max(n_cutoffs, 3)]
    report = dict(cutoffs=[str(c) for c in picks], identical=True, objects={})
    focal = focal_keys(info)
    for cutoff in picks:
        before = evaluate(base, program, cutoffs=[cutoff])
        snap = {k: base.x(f'SELECT * FROM {v} ORDER BY ALL').fetch_arrow_table() for k, v in before.items()}
        mutated, deleted = {}, {}
        text = json.dumps([b['sql'] for o in program['objects'] for b in o.get('blocks', [])]
                          + [o['sql'] if o['type'] == 'derived_table' else o.get('source', '') for o in program['objects']])
        for t, m in info['tables'].items():
            if not m['time'] or not re.search(r'\b' + re.escape(t) + r'\b', text):
                continue
            table = pq.read_table(Path(info['source']) / 'db' / f'{t}.parquet')
            at = pa.scalar(cutoff, table[m['time']].type)
            future = pc.fill_null(pc.greater_equal(table[m['time']], at), False)
            for fk in focal.get(t, ()):  # the focal entity's own row at time == cutoff is readable
                own = pa.array([r[0] for r in base.x(f'SELECT DISTINCT {ident(fk)} FROM cases WHERE {ident(info["task_time"])} = {lit(cutoff)}::TIMESTAMP').fetchall()])
                keep = pc.and_(pc.fill_null(pc.equal(table[m['time']], at), False), pc.is_in(table[m['pkey']], value_set=own.cast(table[m['pkey']].type)))
                future = pc.and_(future, pc.invert(keep))
            if not pc.any(future).as_py():
                continue
            deleted[t] = workdir / f'{t}.deleted.parquet'
            pq.write_table(table.filter(pc.invert(future)), deleted[t])
            cols = []
            for name in table.column_names:
                col = table[name]
                # The time and the primary key stay: a renamed future row could take a focal entity's key and pass
                # the readable-own-row rule at time == cutoff, a test artefact; every other cell is still perturbed.
                if name in (m['time'], m['pkey']):
                    cols.append(col)
                    continue
                if pa.types.is_integer(col.type) or pa.types.is_floating(col.type):
                    noise = pa.array(rng.integers(1, 1000, len(table)).astype(col.type.to_pandas_dtype()))
                    alt = pc.add(pc.fill_null(col, 0), noise)
                elif pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
                    alt = pc.binary_join_element_wise(pc.fill_null(col, pa.scalar('', col.type)), pa.scalar('#m', col.type), pa.scalar('', col.type))
                elif pa.types.is_timestamp(col.type):
                    alt = pa.array([cutoff] * len(table), col.type)
                else:
                    cols.append(col)
                    continue
                cols.append(pc.if_else(future, alt.cast(col.type), col))
            path = workdir / f'{t}.parquet'
            pq.write_table(pa.table(cols, names=table.column_names), path)
            mutated[t] = path
        h = int(info['horizon_s'])
        tcol, target = ident(info['task_time']), ident(info['target'])
        unmatured = f'{tcol} + INTERVAL {h} SECOND >= {lit(cutoff)}::TIMESTAMP'
        # Both callbacks run synchronously for the current cutoff.
        labels = dict(perturb=lambda q: f'SELECT * REPLACE (CASE WHEN {unmatured} THEN NULL ELSE {target} END AS {target}) FROM ({q})',  # noqa: B023
                      delete=lambda q: f'SELECT * FROM ({q}) WHERE NOT ({unmatured})')  # noqa: B023
        for variant in variants:
            alt = Engine(info, memory)
            alt.load_tasks(case_rows, mutate_labels=labels[variant])
            after = evaluate(alt, program, cutoffs=[cutoff], mutate=mutated if variant == 'perturb' else deleted)
            for k in snap:
                got = alt.x(f'SELECT * FROM {after[k]} ORDER BY ALL').fetch_arrow_table() if k in after else snap[k].slice(0, 0)
                same = got.equals(snap[k])
                entry = dict(cutoff=str(cutoff), variant=variant, identical=same, rows=snap[k].num_rows)
                if not same:  # which columns changed (per-column multiset), so a repair can target them
                    entry['rows_after'] = got.num_rows
                    entry['changed_columns'] = [c for c in snap[k].column_names if c not in got.column_names
                                                or not got.select([c]).sort_by(c).equals(snap[k].select([c]).sort_by(c))]
                    entry['float_drift'] = drift = _float_drift(snap[k], got, entry['changed_columns'])
                    # Perturbed rows that the SQL filters out can still reorder a float sum (hash order), which moves the
                    # last bits. A perturbation adds 1..1000 per value, so a leak moves a value far above 1e-12 relative.
                    if drift and set(drift) == set(entry['changed_columns']) and max(drift.values()) <= FLOAT_ORDER_TOL:
                        entry['identical'] = same = True
                        entry['float_summation_order_only'] = True
                report['objects'].setdefault(k, []).append(entry)
                report['identical'] &= same
        for p in [*mutated.values(), *deleted.values()]:
            p.unlink()
    return report
