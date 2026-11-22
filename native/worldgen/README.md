# worldgen

Rust generator of the native world families of the backbone corpus. A world is sampled as a relational schema, a
structural causal model over its tables and a label family, and written as Parquet plus `task_metadata.json`.

```bash
cargo build --release --manifest-path native/worldgen/Cargo.toml
```

The crate builds two binaries. `openrfm.synth.generate` calls both, so they are rarely run by hand.

## `worldgen`: temporal worlds

```bash
worldgen --axis temporal-kernel --seed-start SEED --num-dbs 1 --output-root OUT [options]
```

| Corpus family | Extra options |
|---|---|
| `OLD.nuisance.*` | `--min-tables 3 --max-tables 20 --min-cols 3 --max-cols 24 --min-entity-rows 128 --max-entity-rows 512 --min-activity-rows 512 --max-activity-rows 4096 --exact-aggregates --nuisance-text` |
| `OLD.v6_own_history` | `--v6-stage s1 --v6-archetype a1` |
| `OLD.A4.*` | `--v6-stage s1 --v6-archetype a4` |
| `OLD.long.*` | `--v6-stage s2 --v6-variant long-history` |
| `OLD.frank.*` | `--v6-stage s1 --v6-variant f-rank` |

Each run writes one world directory `OUT/rel-synthetic-*`. Exit code 1 means the seed was refused (no table of the sampled
world can carry the label); `openrfm.synth.generate` then moves on to the next seed. One world contains several label
columns; `openrfm.synth.generate` keeps the columns of the requested family.

## `worldgen-materialize`: static tabular worlds

```bash
worldgen-materialize --manifest openrfm/configs/data/tabular/replica0.json \
    --expected-manifest-sha256 SHA --output-root OUT --split train \
    --family tabicl_v2_like_discrete --workers 8 --publish-gated-subset [--limit N]
```

The manifest fixes every world: its name, seed, family, target and class count. `SHA` is the `manifest_sha256` field
stored in the manifest. The three families are `tabicl_v2_like_discrete`, `support_required_multiclass` and
`relational_operator_discrete`. Worlds that fail a construction-time shortcut gate are not published;
`OUT/materialization_audit.json` lists every record and its outcome.
