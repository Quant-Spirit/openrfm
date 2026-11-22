# openrfm.rewrite

Executor of the schema programs in `openrfm/configs/rewrite/programs/`. A program rewrites a benchmark database so that the
evidence a task needs sits close to the task row; the executor materializes it with DuckDB into a new database that
`rustler pre` then converts like any other.

`program.py` is the command line; `schema.py` validates programs and reads task info, `engine.py` runs the DuckDB
queries, `materialize.py` writes the rewritten database and `mutation.py` runs the leakage test.

```bash
python -m openrfm.rewrite.program ACTION --task rel-f1/driver-dnf --dataset RELBENCH/rel-f1 \
    --program openrfm/configs/rewrite/programs/rel-f1__driver-dnf.json --out WORK
```

`--dataset` is a RelBench dataset directory with `db/*.parquet` and `tasks/<task>/*.parquet`.

| Action | Result |
|---|---|
| `validate` | Structural check of the program against the task |
| `audit` | Checks the value rule (raw quantities only: no hand-set value transforms, weights or smoothing priors); writes `WORK/hard_rule_audit.json` |
| `materialize` | Writes `WORK/data/<db>/{db,tasks/<task>}` and `schema_manifest.json`; the task files are copied unchanged |
| `mutation` | Leakage test, writes `WORK/mutation_test.json` and exits with 9 if it fails |

**What a program can read.** Each generated object is computed per case at its cutoff `c` from source rows with
`time < c` and is stamped `as_of = c - 1 s`. The only task labels a program can read are matured ones, whose label
window ended before `c`. A program with `"objects": []` keeps the original tables.

**Mutation test.** At several cutoffs the test perturbs every value of the source rows at or after the cutoff (and the
labels of rows not yet matured), then deletes those rows, and requires the generated rows of the cases at that cutoff to
stay bit-identical. Options: `--variants perturb,delete`, `--memory` (DuckDB memory limit).
