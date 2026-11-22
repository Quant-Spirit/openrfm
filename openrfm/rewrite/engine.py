"""DuckDB evaluation of schema programs on clock-filtered source views, and the value-rule audit."""
from __future__ import annotations

import json
import re
from pathlib import Path

from openrfm.rewrite.schema import FINAL_STATE, NUMERIC, PER_CUTOFF_MAX, focal_keys, ident, lit


class Engine:
    def __init__(self, info, memory='24GB', spill=None):
        import duckdb
        self.info, self.con = info, duckdb.connect()
        # One thread in insertion order: window rankings and LIMITs with tied sort keys (e.g. two results of one driver
        # in one race) and float aggregates then come out the same in every run.
        self.con.execute(f"SET threads=1; SET memory_limit='{memory}'; SET preserve_insertion_order=true")
        if spill:
            self.con.execute(f"SET temp_directory={lit(spill)}")
        self.src = Path(info['source'])

    def x(self, sql):
        return self.con.execute(sql)

    def source_views(self, cutoff=None, mutate=None):
        """Every table as a view; with a cutoff, timed tables keep rows strictly before it plus the focal entity's own row."""
        masked = FINAL_STATE.get(self.info['db'], {})
        focal = focal_keys(self.info)
        for t, m in self.info['tables'].items():
            path = (mutate or {}).get(t, self.src / 'db' / f'{t}.parquet')
            repl = [f'NULL AS {ident(c)}' for c in masked.get(t, ()) if c in m['columns']]
            proj = '*' + (' REPLACE (' + ','.join(repl) + ')' if repl else '')
            where = ''
            if cutoff is not None and m['time']:
                c = f'{lit(cutoff)}::TIMESTAMP'
                where = f' WHERE {ident(m["time"])} < {c}' + ''.join(
                    f' OR ({ident(m["time"])} = {c} AND {ident(m["pkey"])} IN (SELECT {ident(fk)} FROM cases WHERE {ident(self.info["task_time"])} = {c}))'
                    for fk in focal.get(t, ()))
            self.x(f'CREATE OR REPLACE TEMP VIEW {ident(t)} AS SELECT * FROM (SELECT {proj} FROM read_parquet({lit(path)})){where}')

    def load_tasks(self, case_rows=None, mutate_labels=None):
        i = self.info
        keys = list(i['task_fkeys']) + [i['task_time']]
        for split in ('train', 'val', 'test'):
            path = self.src / 'tasks' / i['task'] / f'{split}.parquet'
            self.x(f'CREATE OR REPLACE TEMP TABLE task_{split} AS SELECT *, file_row_number AS _row '
                   f'FROM read_parquet({lit(path)}, file_row_number=true)')
        if case_rows is not None:
            import pyarrow as pa
            rows = json.loads(Path(case_rows).read_text())
            for split in ('train', 'val', 'test'):
                self.con.register(f'pool_{split}', pa.table({'r': pa.array(sorted(set(rows.get(split, []))), pa.int64())}))
            pool = ' UNION ALL '.join(f'SELECT {",".join(ident(k) for k in keys)} FROM task_{s} WHERE _row IN (SELECT r FROM pool_{s})'
                                      for s in ('train', 'val', 'test'))
        else:
            pool = ' UNION ALL '.join(f'SELECT {",".join(ident(k) for k in keys)} FROM task_{s}' for s in ('train', 'val', 'test'))
        self.x(f'CREATE OR REPLACE TEMP TABLE cases AS SELECT DISTINCT * FROM ({pool})')
        target = ident(i['target'])
        labels = f'SELECT {",".join(ident(k) for k in keys)}, {target} FROM task_train UNION ALL ' \
                 f'SELECT {",".join(ident(k) for k in keys)}, {target} FROM task_val'
        if mutate_labels is not None:
            labels = mutate_labels(labels)
        self.x(f'CREATE OR REPLACE TEMP TABLE all_labels AS {labels}')

    def labels_view(self, cutoff):
        """matured_labels: TRAIN/VAL task rows whose label window (t, t+h] ended strictly before the cutoff."""
        t, h = ident(self.info['task_time']), int(self.info['horizon_s'])
        self.x(f"CREATE OR REPLACE TEMP VIEW matured_labels AS SELECT *, {t} + INTERVAL {h} SECOND AS label_window_end "
               f"FROM all_labels WHERE {t} + INTERVAL {h} SECOND < {lit(cutoff)}::TIMESTAMP")

    def cutoffs(self):
        return [r[0] for r in self.x(f'SELECT DISTINCT {ident(self.info["task_time"])} FROM cases ORDER BY 1').fetchall()]


# Value rule: generated columns are raw quantities; no hand-designed value transforms,
# hand-set smoothing priors or label-correlation feature choice. Checked on the SQL text.
_TRANSFORM = re.compile(r'\b(ln|log|log2|log10|log1p|sqrt|cbrt|exp|pow|power|sign|tanh|atan|atan2|ntile|percent_rank|'
                        r'cume_dist|width_bucket|corr|covar_pop|covar_samp|regr_\w+)\s*\(', re.I)
# time units and window lengths (days, hours, seconds) are not weights
_UNIT = {0.0, 1.0, 7.0, 14.0, 24.0, 28.0, 30.0, 31.0, 60.0, 90.0, 91.0, 100.0, 180.0, 182.0, 365.0, 366.0, 1000.0, 3600.0,
         86400.0}


def hard_rule_audit(program):
    """Violations of the value rule per object/block, with the offending text: [] = pass."""
    out = []
    for o in program.get('objects', []):
        for b in [o] + list(o.get('blocks', [])):
            sql = b.get('sql', '')
            if not sql:
                continue
            text = ' '.join(re.sub(r"'(?:[^']|'')*'", "''", re.sub(r'--[^\n]*', '', sql)).split())   # no literals/comments
            where = f"{o.get('name', o.get('table'))}" + (f".{b['name']}" if b is not o else '')
            # All snippet lookups finish before the next block.
            snip = lambda m: text[max(0, m.start() - 40):m.end() + 25]  # noqa: B023
            for m in _TRANSFORM.finditer(text):
                out.append(f'{where}: value transform {m.group(1).lower()}() in "{snip(m)}"')
            # a weight = a non-integer factor, or a factor on one term of a sum (a hand-weighted score); a plain integer
            # rescale of one quantity (per-30-day rate, a size band) changes no information
            for m in re.finditer(r'(?<![\w.])(\d+(?:\.\d+)?)\s*\*(?!\s*interval)|\*\s*(\d+(?:\.\d+)?)(?![\w.])', text, re.I):
                v = float(m.group(1) or m.group(2))
                summed = re.search(r'[+-]\s*$', text[:m.start()]) or re.match(r'\s*[\w.]+(?:\([^()]*\))?\s*[+-]', text[m.end():])
                if v not in _UNIT and (v != int(v) or summed):
                    out.append(f'{where}: hand-set weight in "{snip(m)}"')
            # smoothing toward a prior: a denominator (or NULLIF argument) that adds a constant other than 0/1
            for m in re.finditer(r'/\s*(?:NULLIF\s*)?\(', text, re.I):
                depth, top = 0, []
                for ch in text[m.end():]:
                    if depth == 0 and ch in '),':
                        break
                    depth += (ch == '(') - (ch == ')')
                    top.append(ch if depth == 0 else ' ')
                c = re.search(r'\+\s*(\d+(?:\.\d+)?)\s*$', ''.join(top))
                if c and float(c.group(1)) not in (0.0, 1.0):
                    out.append(f'{where}: hand-set smoothing prior in "{text[max(0, m.start() - 40):m.end() + len(top) + 1]}"')
            if re.search(r'(?<![\w.])\^|\*\*', text):
                out.append(f'{where}: power operator')
    return sorted(set(out))


def _numeric_projection(engine, table, alias, keep):
    """Numeric and boolean outputs become DOUBLE, dates stay timestamps; text outputs are dropped."""
    out = []
    for col, dtype, *_ in engine.x(f'DESCRIBE {table}').fetchall():
        if col in keep:
            continue
        if dtype in NUMERIC or dtype.startswith('DECIMAL'):
            out.append((col, f'CAST({alias}.{ident(col)} AS DOUBLE)'))
        elif dtype.startswith(('TIMESTAMP', 'DATE')):
            out.append((col, f'CAST({alias}.{ident(col)} AS TIMESTAMP)'))
    return out


def evaluate(engine, program, *, cutoffs=None, mutate=None):
    """Run every per-case object. Returns {name: DuckDB table name} with key columns (entity FK, cutoff)."""
    i = engine.info
    tcol = i['task_time']
    allc = engine.cutoffs()
    todo = allc if cutoffs is None else [c for c in allc if c in set(cutoffs)]
    per_cutoff = len(allc) <= PER_CUTOFF_MAX
    results = {}
    parts = {}
    groups = [[c] for c in todo] if per_cutoff else [todo]
    for group in groups:
        cutoff = group[0] if per_cutoff else None
        engine.source_views(cutoff, mutate)
        cond = f'{ident(tcol)} = {lit(cutoff)}::TIMESTAMP' if per_cutoff else \
            f'{ident(tcol)} IN ({",".join(lit(c) + "::TIMESTAMP" for c in group)})'
        engine.x(f'CREATE OR REPLACE TEMP TABLE eval_exact AS SELECT * FROM cases WHERE {cond}')
        # RT archives keep times in whole seconds, so snapshot rows are evaluated at the whole-second floor of each case
        # time: cases of one entity within one second share one row, stamped floor - 1 s (before each of them). Per-cutoff
        # views are bounded at c itself, so that mode needs whole-second cutoffs.
        if per_cutoff and any(o['type'] == 'snapshot_table' for o in program['objects']) and engine.x(
                f"SELECT {lit(cutoff)}::TIMESTAMP <> date_trunc('second', {lit(cutoff)}::TIMESTAMP)").fetchone()[0]:
            raise ValueError(f'cutoff {cutoff} is not a whole second; snapshot tables need whole-second cutoffs')
        engine.x(f"CREATE OR REPLACE TEMP TABLE eval_second AS SELECT * REPLACE (date_trunc('second', {ident(tcol)}) AS {ident(tcol)}) FROM eval_exact")
        if per_cutoff:
            engine.labels_view(cutoff)
        else:
            # Self-filtering mode: matured_labels carries no global bound, so each SQL must compare
            # label time + horizon with its own eval_table time; the mutation test checks this.
            h = int(i['horizon_s'])
            engine.x(f'CREATE OR REPLACE TEMP VIEW matured_labels AS SELECT *, {ident(tcol)} + INTERVAL {h} SECOND AS label_window_end FROM all_labels')
        for o in program['objects']:
            if o['type'] == 'keep_table':
                continue
            ent = o['entity']
            key = [ent, tcol]
            engine.x(f'CREATE OR REPLACE TEMP VIEW eval_table AS SELECT * FROM {"eval_second" if o["type"] == "snapshot_table" else "eval_exact"}')
            if o['type'] == 'snapshot_table':
                sel, joins = [], []
                for b, block in enumerate(o['blocks']):
                    tmp = f'blk_{o["name"]}_{b}'
                    engine.x(f'CREATE OR REPLACE TEMP TABLE {tmp} AS {block["sql"]}')
                    cols = [r[0] for r in engine.x(f'DESCRIBE {tmp}').fetchall()]
                    if ent not in cols or tcol not in cols:
                        raise ValueError(f'block {block["name"]} must return {ent} and {tcol}')
                    dup = engine.x(f'SELECT count(*) FROM (SELECT {ident(ent)},{ident(tcol)} FROM {tmp} GROUP BY ALL HAVING count(*)>1)').fetchone()[0]
                    if dup:
                        raise ValueError(f'block {block["name"]} returns several rows per case')
                    for col, expr in _numeric_projection(engine, tmp, f'b{b}', set(key) | set(o.get('drop_columns', []))):
                        sel.append(f'{expr} AS {ident(block["name"] + "__" + col)}')
                    joins.append(f'LEFT JOIN {tmp} b{b} ON b{b}.{ident(ent)} = e.{ident(ent)} AND b{b}.{ident(tcol)} = e.{ident(tcol)}')
                query = f'SELECT DISTINCT e.{ident(ent)}, e.{ident(tcol)}, {", ".join(sel)} FROM (SELECT DISTINCT {ident(ent)}, {ident(tcol)} FROM eval_table) e {" ".join(joins)}'
            elif o['type'] == 'evidence_table':
                src, fk = o['source'], o['fk']
                stime = i['tables'][src]['time']
                cols = ', '.join(f's.{ident(c)}' for c in dict.fromkeys(o['columns'] + [stime]) if c != fk)
                query = (f'SELECT e.{ident(ent)}, e.{ident(tcol)}, {cols} FROM (SELECT DISTINCT {ident(ent)}, {ident(tcol)} FROM eval_table) e '
                         f'JOIN {ident(src)} s ON s.{ident(fk)} = e.{ident(ent)} AND s.{ident(stime)} < e.{ident(tcol)} '
                         # ties at one source time are broken on every selected value, so the kept rows do not depend on
                         # scan or join order (which deleting future rows changes)
                         f'QUALIFY row_number() OVER (PARTITION BY e.{ident(ent)}, e.{ident(tcol)} ORDER BY s.{ident(stime)} DESC, '
                         f'{", ".join(f"s.{ident(c)}" for c in dict.fromkeys(o["columns"]) if c not in (fk, stime))}) <= {o["k"]}')
            else:
                query = o['sql']
            tmp = f'part_{o["name"]}_{len(parts.get(o["name"], []))}'
            engine.x(f'CREATE OR REPLACE TEMP TABLE {tmp} AS {query}')
            cols = [r[0] for r in engine.x(f'DESCRIBE {tmp}').fetchall()]
            if ent not in cols or tcol not in cols:
                raise ValueError(f'{o["name"]} must return {ent} and {tcol}')
            if o['type'] == 'derived_table':
                worst = engine.x(f'SELECT coalesce(max(n),0) FROM (SELECT count(*) n FROM {tmp} GROUP BY {ident(ent)},{ident(tcol)})').fetchone()[0]
                if worst > o['max_rows_per_case']:
                    raise ValueError(f'{o["name"]} returns {worst} rows for one case (max {o["max_rows_per_case"]})')
            parts.setdefault(o['name'], []).append(tmp)
    for name, tmps in parts.items():
        out = f'out_{name}'
        engine.x(f'CREATE OR REPLACE TEMP TABLE {out} AS ' + ' UNION ALL BY NAME '.join(f'SELECT * FROM {t}' for t in tmps))
        results[name] = out
    return results
