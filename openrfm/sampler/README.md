# openrfm.sampler

Relational context sampler. It draws bounded relational contexts from the graph that
[`openrfm.archive`](../archive) reads. The C++/CUDA kernels are in [`kernels/`](kernels) and
are built on first use as the extension `_openrfm_episodes`, cached under `~/.cache/openrfm/episodes`.

## Sample contexts

```python
from openrfm.sampler import episodes

episodes.configure_backend("native", build=True)   # compile kernels/ once (CUDA if available)
contexts = episodes.sample_rt_contexts(graph, requests, spec)
```

`configure_backend("torch")` selects the pure-PyTorch reference implementation, which tests use to check the native
kernels. `RTRetrievalSpec` fixes the context construction: sequence lengths, walk and breadth-first budgets, the order
in which the budget is spent (`seed`, `own_history`, `peers`, `filler`) and label visibility. `mixture`
samples these settings per batch during training.

| Module | Role |
|---|---|
| `spec.py` | `RTRetrievalSpec`, the context-construction policy |
| `episodes.py`, `native.py`, `native_build.py` | Public sampler entry points, backend selection and the native build |
| `retrieval.py`, `v2.py`, `path_streams.py`, `composition.py` | Retrieval along the two axes and composition of seed, history, peer and filler cells, each with a CPU reference |
| `admission.py` | Vectorized CPU reductions used when admitting roots |
| `mixture.py` | Distribution over sampler settings for training |
| `pack.py` | `ContextSampler`: contexts of one task archive packed into the backbone's batch layout (inference and memory training) |

