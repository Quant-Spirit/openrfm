# openrfm/configs/rewrite

The final schema programs of the 32 benchmark tasks, as written by the LLM harness. Run them with `openrfm.rewrite.program`
(see [`openrfm/rewrite`](../../rewrite)).

| Path | Contents |
|---|---|
| `programs/<dataset>__<task>.json` | One program per task: 23 rewrites and 9 programs with `"objects": []` that keep the original schema |
| `tasks.json` | Target column, task type and label horizon of each task |

A program has the form `{"kind": "schema_program_v1", "dataset", "task", "objects": [...]}`. The
object types are `keep_table`, `snapshot_table`, `evidence_table` and `derived_table`; `openrfm/rewrite/schema.py` validates each one.
