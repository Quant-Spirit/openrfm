# openrfm/configs/infer

Inference settings read by `openrfm predict` ([`openrfm/infer`](../../infer)).

| File | Contents |
|---|---|
| `r3.yaml` | Per task the residual-head setting of R3 (learning rate, regression loss, maximum updates, holdout checkpoint steps, episode sizes and width) on top of the task's R2 cell |
| `r1.yaml`, `r2.yaml` | Per task the recipe cell: schema (`program` reads `openrfm/configs/rewrite/programs/<task>.json`; `plain` keeps the original schema, `repair` also restores the matured event outcomes of a 4DBInfer database), context length `L`, support budget `K`, support draws `E`, support selection, regression target transform, support pool (`standard` or `expanded`, with `disjoint_draws`), and R2 record source and bank width |
| `tasks.yaml` | Per task: database, task table, target column, task type, entity table, label window, the support caps and, for two tasks, the foreign-key paths sampled indirectly |
| `sampler.yaml` | The context sampler's retrieval spec and the own-history policy |
| `databases.yaml` | Normalization cutoff of each benchmark database and the databases whose archives carry the node-time sidecar |

The support pool holds the TRAIN rows whose label window ends by the first VAL cutoff and the VAL rows whose window ends
by the first TEST cutoff. `standard` keeps at most `support_cap` rows per split; `expanded` keeps up to
`expanded_support_cap` rows per split, the `standard` rows first. A draw takes `K` rows of the pool (all of it when
`K` is larger); with `disjoint_draws` draw `d` is the `d`-th block of `K` rows of one seeded permutation, so the draws
do not overlap until they wrap around the pool.
