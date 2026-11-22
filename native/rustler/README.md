# rustler

`rustler pre` converts a database (a synthetic world or a RelBench database) into the archive that
`openrfm.archive.loader` reads. The same crate builds `rustler_archive`, the Python extension that runs the same
conversion (`rustler_archive.convert`, used by `openrfm convert`) and decodes format-v1 archives.

## `rustler pre`

```bash
cargo build --release --manifest-path native/rustler/Cargo.toml
```

Text columns are embedded with a sentence-transformers model called through Python, so the Python that has
`sentence-transformers` must be on the library path at run time (for example
`LD_LIBRARY_PATH=$PYTHONHOME/lib`).

| Variable | Contents |
|---|---|
| `RELBENCH_DATA_ROOT` | Required input root; the database is read from `<root>/<db>/` |
| `RELBENCH_PRE_ROOT` | Required output root; the archive is written to `<root>/<db>/` |
| `RUSTLER_TEXT_EMB_CACHE` | Optional cache of text embeddings, shared across databases |

```bash
RELBENCH_DATA_ROOT=OUT/raw RELBENCH_PRE_ROOT=PRE RUSTLER_TEXT_EMB_CACHE=TEXT \
    rustler pre rel-synthetic-40000000 --archive-format v2 --text-encoder all-MiniLM-L12-v2
```

| Option | Meaning |
|---|---|
| `--archive-format v2` / `v1` | v2 (default) for the backbone corpus and RelBench, v1 for the memory-module worlds |
| `--text-encoder` | Embedding model; the released checkpoints use `all-MiniLM-L12-v2` |
| `--text-encoder-device`, `--allow-cpu` | Text encoding runs on CUDA by default (`auto`); `--allow-cpu` permits CPU encoding |
| `--text-encoder-batch`, `--text-encoder-max-length` | Encoder batch size (8192) and token limit (256) |
| `--text-embedding-cache-only` | Fail on a cache miss instead of loading the model |
| `--fp32-datetimes` | Keep cell datetimes in fp32 in a v2 archive (the backbone training corpus); bf16 by default |
| `--time-cutoffs-file` | Optional JSON map from database name to its validation start (`"YYYY-MM-DD[ HH:MM:SS]"` or an integer timestamp); an empty string marks a database without a temporal split |

### Input database

`<root>/<db>/db/<table>.parquet` holds the tables and `<root>/<db>/tasks/<task>/{train,val,test}.parquet` the task
splits, if any. Every Parquet file carries the key-value metadata `pkey_col`, `fkey_col_to_pkey_table` and `time_col`
(JSON values), as RelBench writes them; foreign-key values are row positions in the parent table. Each table is a node
type, each row a node with one cell per non-key column; foreign keys become edges. An optional
`<root>/<db>/task_metadata.json` declares the tasks; a declared classification target is encoded with one class
vocabulary fitted on the training split (2 to 10 classes as a discrete target, more classes as class ids stored as numbers).
Without it, target columns keep their stored values.
The other top-level JSON files of `<root>/<db>/` (generator sidecars such as `a5_world.json` or `derived_cells.json`)
are copied into a v2 archive unchanged.

Numbers, booleans and timestamps are normalised with statistics from training rows only: rows earlier than the
database's cutoff from `--time-cutoffs-file` or the earliest test-split timestamp, whichever is earlier. Without
either, all rows count.

- **Synthetic worlds** (`openrfm.synth`) ship `db/` and `task_metadata.json` with the targets on the entity table;
  they need no cutoffs file.
- **RelBench databases** are converted as saved by RelBench (`db/` plus `tasks/`). Link-prediction tasks and
  list-valued key columns are skipped, and a few tables of `rel-stack`, `rel-trial`, `rel-amazon`, `rel-event` and
  `rel-avito` are adjusted (binarised flags, first product category, dropped nulls and index columns, foreign-key
  targets).
- **R2 rewrites**: `python -m openrfm.rewrite.program materialize --task rel-f1/driver-dnf --dataset <rel-f1>
  --program openrfm/configs/rewrite/programs/rel-f1__driver-dnf.json --out WORK` writes `WORK/data/rel-f1/{db,tasks}` and
  `WORK/fit_time_cutoffs.json`; convert with `RELBENCH_DATA_ROOT=WORK/data` and
  `--time-cutoffs-file WORK/fit_time_cutoffs.json`.

## `rustler_archive`

```bash
pip install ./native/rustler        # needs a Rust toolchain (rustup) and builds with maturin
```

`convert` takes the options of `rustler pre` as keywords and writes the same archive. Text is embedded by
`sentence-transformers` in the calling interpreter.

```python
import rustler_archive

rustler_archive.convert("rel-f1", "RELBENCH", "PRE", archive_format="v2", time_cutoffs_file="cutoffs.json")
arrays = dict(rustler_archive.ArchiveV1.load_arrays("PRE/rel-synthetic-7200000", "all-MiniLM-L12-v2", 384))
```

`openrfm.archive.loader` uses `ArchiveV1` to read v1 archives; v2 archives are read without it.
