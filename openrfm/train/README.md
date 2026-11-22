# openrfm.train

Training of the two learned parts of the released model. Both read tracked configurations under `openrfm/configs/train/`. Backbone training needs the `train` extra
(`pip install ".[train]"`, which adds `tabicl==2.0.1`).

| Package | Command | Output |
|---|---|---|
| `backbone` | `openrfm train backbone corpus --config openrfm/configs/train/backbone.yaml --generated GEN --out CORPUS` | The training corpus under `CORPUS` |
| `backbone` | `openrfm train backbone train --config openrfm/configs/train/backbone.yaml --corpus CORPUS --out DIR [--resume CKPT] [--steps N]` | Training checkpoints under `DIR` |
| `backbone` | `openrfm train backbone export --checkpoint CKPT --out openrfm_backbone.pt` | The backbone file that `openrfm.models.load_backbone` reads |
| `memory` | `openrfm train memory --config openrfm/configs/train/memory.yaml --corpus CORPUS --out DIR [--resume] [--steps N]` | `DIR/latest.pt` (resumable) and `DIR/memory.pt` |

## Backbone

Training runs in four steps.

```bash
openrfm generate --config openrfm/configs/data/backbone_corpus.yaml --out GEN \
    --worldgen native/worldgen/target/release/worldgen \
    --worldgen-materialize native/worldgen/target/release/worldgen-materialize
openrfm train backbone corpus --config openrfm/configs/train/backbone.yaml --generated GEN --out CORPUS
openrfm train backbone train --config openrfm/configs/train/backbone.yaml --corpus CORPUS --out RUN
openrfm train backbone export --checkpoint RUN/steps=40000.pt --out openrfm_backbone.pt
```

`corpus` converts every generated world with `rustler_archive.convert(..., fp32_datetimes=True)` (needs a GPU for the
text encoder) and writes:

| Path under `CORPUS` | Contents |
|---|---|
| `sources/<world>/` | The v2 archive of each world, with fp32 datetimes and the world's JSON sidecars |
| `corpus/rel-synthetic-<n>` | Links to the archives, numbered family by family in the order of `families:` |
| `corpus/compact_task_catalog.json` | One record per training task: alias, family, table, target and task type |
| `consumers/{classifier,regressor}.ckpt` | The frozen TabICL v2 models, downloaded from `jingang/TabICL` and checked by sha256 |

A rerun skips worlds already converted. Each family with worlds becomes one mixture group with its weight from
`families:`; families absent from `GEN` (for example after `openrfm generate --limit N --families ...`) are left out.

Default training configuration: 50,000 updates of 32 contexts with 8 accumulation steps, Muon for the
matrices and AdamW for the rest, masked cell prediction with a column-slot auxiliary loss, and a coupling loss through
the frozen TabICL consumer every fourth update. The exported backbone uses the step-40,000 checkpoint.

## Memory module

The memory module trains over the frozen backbone on memory worlds generated on the fly (`openrfm/configs/data/memory_worlds.yaml`)
plus replayed backbone-corpus archives. Each world yields program tasks whose labels the frozen Kumo-Tabular heads
fit in context; the loss is their query error. The replay archives and their selection are read from the corpus of
`openrfm train backbone fetch` (`--corpus`), the backbone is downloaded (or read from `--checkpoints`), and `inputs:`
in the configuration names the `rustler` binary (`cargo install --path native/rustler`).
