# openrfm.archive

Reads an archive written by `rustler pre` into the graph and cell payload that [`openrfm.sampler`](../sampler) samples
from.

```python
from pathlib import Path
from openrfm.archive.loader import load_rt_archive
from openrfm.sampler.spec import RTRetrievalSpec

spec = RTRetrievalSpec(version=2)
graph, payload, roots, resident_bytes = load_rt_archive(
    Path("PRE/rel-synthetic-40000000"), "all-MiniLM-L12-v2", 384, device="cuda",
    spec=spec, target_column=TARGET, drops=[])
```

`load_rt_archive` reads v2 archives through memory-mapped arrays and v1 archives through the `rustler_archive` extension
(`native/rustler`). It returns the cell graph, the cell payload, the eligible query roots for `target_column` and
the estimated resident memory. `drops` lists column indices that are removed from the context.

| Module | Role |
|---|---|
| `loader.py` | `load_rt_archive`: the task graph and payload of one target on one device |
| `decode.py` | Decodes an archive into the task-independent row, cell and edge tensors |
| `admission.py` | Directed CSR, composition metadata, query roots and archive metadata shared by task views |
| `format.py` | On-disk layout of format-v2 arrays (little-endian, individually hashed, memory-mappable) |
| `policy.py` | Node-time and observed-availability sidecars, structural table identities |
| `node_time.py` | Writes the node-time sidecar: rows that no dated foreign key dates take their own time column |
| `history.py` | The history world the memory module reads, from a decoded archive and an admitted task graph |
