"""Schema programs: LLM-written relational rewrites, materialized as focused RT databases.

Per-case objects are evaluated at each case cutoff c on source views filtered to `time < c` and stamped
`as_of = c - 1 s`; matured labels are the only task data a program reads.
"""
from __future__ import annotations

import json
from pathlib import Path

from openrfm.rewrite.engine import hard_rule_audit
from openrfm.rewrite.materialize import materialize
from openrfm.rewrite.mutation import mutation_test
from openrfm.rewrite.schema import task_info, validate


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('materialize', 'mutation', 'validate', 'audit'))
    p.add_argument('--task', required=True)
    p.add_argument('--dataset', type=Path, required=True, help='dataset directory (db/*.parquet, tasks/<task>/*.parquet)')
    p.add_argument('--program', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--case-rows', type=Path)
    p.add_argument('--memory', default='24GB')
    p.add_argument('--variants', default='perturb,delete', help='mutation variants; delete alone writes mutation_delete.json')
    a = p.parse_args()
    info = task_info(a.task, a.dataset)
    program = validate(json.loads(a.program.read_text()), info)
    if not program['objects']:  # no rewrite: the original tables, but only this task's table (no other task's labels)
        program = dict(program, no_rewrite=True,
                       objects=[dict(type='keep_table', table=t, columns='all', rows='all') for t in info['tables']])
    if a.action == 'validate':
        print('valid')
    elif a.action == 'audit':
        v = hard_rule_audit(program)
        (a.out / 'hard_rule_audit.json').write_text(json.dumps(dict(verdict='pass' if not v else 'fail', violations=v), indent=1) + '\n')
        print('pass' if not v else 'fail', json.dumps(v))
    elif a.action == 'materialize':
        print(json.dumps({k: v for k, v in materialize(program, info, a.out, case_rows=a.case_rows, memory=a.memory).items()
                          if k not in ('program',)}, default=str)[:4000])
    else:
        variants = tuple(a.variants.split(','))
        r = mutation_test(program, info, a.out / 'mutation', case_rows=a.case_rows, memory=a.memory,
                          variants=variants)
        (a.out / ('mutation_delete.json' if variants == ('delete',) else 'mutation_test.json')).write_text(json.dumps(r, indent=1) + '\n')
        print(json.dumps(r))
        raise SystemExit(0 if r['identical'] else 9)


if __name__ == '__main__':
    main()
