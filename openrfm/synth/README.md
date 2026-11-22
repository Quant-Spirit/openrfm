# openrfm.synth

Python generators of the synthetic databases ("worlds") and the entry point that builds the backbone corpus.

A world is a directory `rel-synthetic-<n>/` with Parquet tables under `db/` and a `task_metadata.json` that declares
its prediction tasks (entity table, target column, task type, cutoffs and label horizon). Generation is CPU only.

## Generate the backbone corpus

```bash
PYTHONHASHSEED=0 python -m openrfm.synth.generate --config openrfm/configs/data/backbone_corpus.yaml --out OUT \
    --worldgen native/worldgen/target/release/worldgen \
    --worldgen-materialize native/worldgen/target/release/worldgen-materialize \
    [--families NEW.future.count F1 ...] [--limit N] [--workers 8]
```

| Option | Meaning |
|---|---|
| `--config` | Family mix and generator settings (default `openrfm/configs/data/backbone_corpus.yaml`) |
| `--out` | Output root: `raw/` holds the worlds, `receipts/` one JSON per finished world |
| `--families` | Generate only these families; family names are the keys of `worlds:` in the config, `F1`–`F4`, and `OLD.tabular.tabicl/support/operator` |
| `--limit N` | At most N worlds per family, for a smoke run |
| `--workers` | Parallel processes |

`PYTHONHASHSEED=0` is needed for byte-identical output, because some families iterate over Python sets of strings.

**How a world is admitted.** World `i` of a family tries seed slots `first_slot + 1000·i + attempt` for up to
`attempts` attempts. A candidate is rejected when its generator refuses the seed, when it draws another family of the
same generator, or when it has no task of the requested family. The first admitted candidate is renamed to
`rel-synthetic-<40000000+i>` and its receipt records the slot, the number of attempts and the rejection reasons.
Because the receipt is written last, rerunning the same command resumes an interrupted run.

## Families

| Config key | Generator | What the labels depend on |
|---|---|---|
| `NEW.auto.*` | `worlds/unified.py`, `worlds/unified_write.py` | Autocomplete: a masked attribute or event mark of an entity, given its relational history |
| `NEW.future.*` | `worlds/unified.py`, `worlds/unified_write.py` | Future window: count, existence, sum, mean, max, next category or time to next event after the cutoff |
| `NEW.derived` | `worlds/derived.py` | A denormalised column computed from strictly earlier events |
| `OLD.PH.*` | `worlds/panel.py`, `worlds/history_*.py` | Panel forecasts with drift, mean reversion, regimes, seasonality, windows or no history |
| `OLD.rank.q*` | `worlds/rank.py` | Quantile classes of selected facts about an entity and its peers |
| `OLD.SQL.own.*`, `OLD.SQL.peer.*` | `worlds/sql.py`, `worlds/sql_oracle.py`, `worlds/sql_write.py` | SQL-style aggregates over the entity's own events or its peers' events in a future window |
| `F1`–`F4` (`fact_rows:`) | `worlds/fact/` | Fact rows (F1, F2), same-time slates (F3) and dated event windows (F4) |
| `OLD.nuisance.*`, `OLD.A4.*`, `OLD.long.*`, `OLD.frank.*`, `OLD.v6_own_history` | `native/worldgen` | Native Rust worlds, see that crate |
| `OLD.tabular.*` | `native/worldgen` (`worldgen-materialize`) | Static tabular worlds from the fixed manifests in `openrfm/configs/data/tabular/` |

`io.py` writes tables with key and timestamp metadata and integrity digests; `stats.py` holds shared scalar helpers.

## Memory-module worlds

`worlds/memory.py` (label mechanisms in `worlds/memory_signals.py`) generates the temporal worlds used to train the memory module. Training writes them online, one
world per 40 updates:

```python
import yaml
from openrfm.synth.worlds.memory import write_world

prior = yaml.safe_load(open("openrfm/configs/data/temporal_prior.yaml"))
write_world("RAW", "rel-synthetic-7200000", 7200000, "openrfm/configs/data/memory_worlds.yaml",
            prior=prior, signal="hawkes_signal")
```

`signal` picks the label mechanism, one of the groups in `mix:` of `openrfm/configs/data/memory_worlds.yaml`: `memory_signal`,
`near_signal`, `joint_signal`, `silent_signal`, `exact_signal`, `hawkes_signal`, `regime_signal`, `censored_signal` or
`tail_signal`; `signal=None` writes a plain event stream. `RAW/<db>` must not exist yet. Convert the world with
`rustler pre --archive-format v1`.
