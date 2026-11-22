# openrfm.infer

`openrfm predict --task rel-f1/driver-dnf --data RELBENCH --recipe r2` runs one benchmark task end to end:

0. `download.py`: `RELBENCH/<db>` when it is absent (`openrfm/configs/infer/downloads.yaml`), each file checked by sha256:
   RelBench zips from the RelBench server, or a 4DBInfer archive written in RelBench layout by `dbinfer.py`.
1. `prepare.py`: converts `RELBENCH/<db>` with `rustler_archive.convert` (adding the node-time sidecar of
   `openrfm.archive.node_time` for the databases `openrfm/configs/infer/databases.yaml` lists), or first materializes the
   task's schema program (`openrfm.rewrite`) and converts its output. The `repair` schema reads a database's restored
   event-outcome tables (`<db>/history/`) in place of its own.
2. `catalog.py`: task rows per split; support pools (TRAIN rows mature by the first VAL cutoff and VAL rows mature by
   the first TEST cutoff, capped per split, or extended for an `expanded` pool) and every admitted TEST row as a
   query.
3. `contexts.py`: the task's context sampler (`openrfm.sampler.pack`) with the inference label visibility.
4. `features.py`, `snapshot.py`: one frozen backbone forward per batch over the history world of
   `openrfm.archive.history`; R1 features are the target-cell
   states at depths 3 and 6, the focal FP32 scalars and the memory columns; R2 adds the record bank of the slot row.
5. `predict.py`: E support draws of K rows (consecutive blocks of one permutation with `disjoint_draws`), frozen
   Kumo-Tabular small per draw, mean (classification) or median (regression) over draws, AUROC or MAE and its
   entity-bootstrap SE.
6. `residual.py` (recipe `r3`): a zero-initialised residual head on the Kumo-Tabular prediction of the R2 draws, trained
   on episodes of the support rows, with the backbone, memory module and Kumo-Tabular frozen.

Label visibility follows the Relational Transformer protocol: the encoder sees the label of an earlier task row (TEST
rows included) when its label window ended strictly before the query's cutoff, never the query's own label; the
support pool holds TRAIN and VAL rows only, and schema programs read matured TRAIN and VAL labels.

`run.py` writes `WORK/<recipe>/<task>/result.json` and `predictions.npz` (`p` the predictions, `row` their row
numbers in test.parquet, `entity` and, when known, `y`). Recipes and settings: `openrfm/configs/infer`.

`openrfm predict --spec TASK.yaml` runs a task of one's own database `RELBENCH/<db>`: the YAML file holds the keys of a
`openrfm/configs/infer/tasks.yaml` entry, plus optionally `cutoff` (the normalization cutoff, by default the first VAL time) and
`own_row_time`. The recipe's `default` cell is used, every TEST row is predicted, and the metric is reported when
test.parquet holds the target.

`dbinfer.py` writes a 4DBInfer database (`amazon-4db`, `outbrain-4db`, `retailrocket-4db`, `stackexchange-4db`) in
RelBench layout from its extracted official archive (`20240304-<name>.tar`). The archives were served at
`https://data.dgl.ai/mtbench/`, which no longer resolves; `openrfm download` takes the unchanged archives (same sha256)
from the Hugging Face dataset [`zkchen/openrfm-4dbinfer`](https://huggingface.co/datasets/zkchen/openrfm-4dbinfer), each under the terms of its source. RelBench's own 4DBInfer conversion is not used: it renumbers
primary keys as row positions without remapping the foreign keys that reference them. With an archive already
extracted to `RAW/<name>/`, `openrfm convert --dbinfer RAW` runs the conversion directly:

```bash
openrfm convert --db stackexchange-4db --dbinfer RAW --data RELBENCH --out ARCHIVES --time-cutoffs cutoffs.json
openrfm predict --task stackexchange-4db/user-churn --data RELBENCH --recipe r1
```
