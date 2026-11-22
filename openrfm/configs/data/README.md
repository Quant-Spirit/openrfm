# openrfm/configs/data

Configurations of the synthetic data.

| File | Used by | Contents |
|---|---|---|
| `backbone_corpus.yaml` | `openrfm.synth.generate` | Family mix of the backbone corpus (world count per family under `worlds:` and `fact_rows:`), seeds, attempt budget, coverage and temporal settings, and the label-column rules of the nuisance families |
| `tabular/replica0.json`, `tabular/replica1.json` | `worldgen-materialize` | Fixed manifests of the static tabular worlds, one per replica; each record fixes a world's name, seed, family and target. The `manifest_sha256` field is checked at load time, so do not edit these files |
| `memory_worlds.yaml` | `openrfm.synth.worlds.memory.write_world` | Starting shares of the memory-world groups (`mix:`, reweighted during training by measured reducible loss) and the stream generator settings |
| `temporal_prior.yaml` | `openrfm.synth.worlds.memory.write_world` | Share of worlds with irregular label publication times and the jitter range |

To generate a smaller corpus with the same families, lower the counts under `worlds:` and `fact_rows: worlds:`, or pass
`--limit N` to `openrfm.synth.generate`.
