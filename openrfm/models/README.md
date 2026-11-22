# openrfm.models

Model code of OpenRFM. The backbone is a relational transformer (RT) pretrained on synthetic relational data;
it encodes the contexts that [`openrfm.sampler`](../sampler) draws. `memory/` reads the frozen backbone's
states and `consumers/` holds the frozen tabular models that predict from the exported columns.

## Load the backbone

```python
from openrfm.models import load_backbone

rt = load_backbone("openrfm_backbone.pt", "cuda")       # fp32 weights, eval mode
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    hidden = rt.encode_hidden(batch, support_axis_enabled=False)   # [B, S, 128]
```

`openrfm_backbone.pt` is the `zkchen/openrfm` checkpoint (3.5 M parameters). `load_backbone` loads it strictly
into the architecture of `RTConfig()`; the training-only tensors of a training checkpoint are dropped when it is
exported (`UNUSED_PREFIXES` in `openrfm/train/backbone/export.py`).

## Architecture

A batch holds B sampled contexts of S cells. A cell is the value of one column of one row; its input is the
column-name embedding plus a typed value encoding (number, text, datetime, boolean or class label), or the
type's mask embedding when the value is hidden, plus structural (hop, path, degree) and time features. Six
pre-norm blocks then apply, in order:

| axis | a cell attends to |
|---|---|
| col | cells of the same table and column |
| feat | cells of its own row and of its row's parents |
| nbr | cells of rows whose parents include its row |
| support | one slot per other context of the batch (designated support targets that are strictly earlier, or pooled features) |
| entity | one pooled token per row of the focal entity's neighbourhood (softmax and sigmoid-sum heads) |

followed by a SwiGLU feed-forward. Training masks cells and predicts them with per-type decoders; the loss is
normalized per context (`backbone/readouts.py`).

| Module | Role |
|---|---|
| `backbone/config.py` | `RTConfig` and the axis and type constants |
| `backbone/transformer.py` | `RelationalTransformer`: module wiring, `encode_hidden`, training `forward` |
| `backbone/embedding.py` | value visibility, typed value encoding, the block-0 cell input |
| `backbone/time.py` | context age and role features, horizon-relative cell age |
| `backbone/structure.py` | structural cell features, entity rows and the entity axis |
| `backbone/masks.py` | focal positions, timestamp sentinels, dense attention masks |
| `backbone/attention.py` | masked attention, the support axis and its time validation |
| `backbone/blocks.py` | SwiGLU FFN, `RelationalBlock`, the block stack |
| `backbone/readouts.py`, `backbone/losses.py` | masked-cell decoders and losses |
| `backbone/checkpoint.py` | `load_backbone` |
| `kernels/sparse_attention.py` | Triton sparse attention for the col/feat/nbr axes (`attention="sparse"`) |
| `kernels/graph_stack.py` | CUDA-graph replay of the block stack for training (`rt.graph_stack()`) |

`attention="dense"` (default) uses SDPA with `[B, S, S]` boolean masks; `attention="sparse"` computes the same
function with the Triton kernels, which need memory linear in S. Both kernel modules are imported only when
selected.

## Memory module

`memory/` reads the frozen backbone with one learned cross-attention. A request is a focal row at a cutoff;
its tokens are every live context cell at every backbone depth (the block-0 input and each block's output) and
every cell of the `rows` nearest visible rows of its temporal foreign-key neighbourhood (`hops` hops, timestamp
<= cutoff), encoded by the backbone's frozen cell encoders. Each of the `queries` learned queries exports its
softmax-weighted value and its log-partition: `2*queries` columns per request.

```python
from openrfm.models.memory import encode_context, load_memory, scalar_matrix

memory = load_memory("openrfm_memory_v1.pt").cuda().eval().bind(rt)     # strict load; rt is the frozen backbone
states, live, triples, target = encode_context(rt, batch)            # one bf16 backbone forward
columns = memory.columns(world, rows, cutoffs, context=(states, live))   # [n, 128]
```

The consumer reads `[target[:, (3, 6)], FP32 focal scalars (scalar_matrix), columns]`, under its 500-column cut.

| Module | Role |
|---|---|
| `memory/config.py` | `MemoryConfig` |
| `memory/context.py` | `encode_context`, focal FP32 scalars, `scalar_matrix` |
| `memory/history.py` | the history world, its foreign-key reach and the frozen cell encoding |
| `memory/module.py` | `MemoryModule`: tokens, the read, `columns` |
| `memory/checkpoint.py` | `save_memory`, `load_memory` (also reads the released checkpoint) |

## Consumers

`consumers/kumo.py` wraps the frozen Kumo-Tabular model (`structured-data-models`, weights `nvidia/Kumo-Tabular`).
`KumoHead` is the differentiable training consumer (FP32; cross-entropy or pinball loss); `KumoPredictor` predicts
in context with the package's own recipe:

```python
from openrfm.models.consumers import KumoPredictor

torch.manual_seed(1008)                         # the recipe draws its permutations from the torch RNG
kumo = KumoPredictor("classification", "small").fit(support_x, support_y)
log_p = kumo.predict(query_x)                   # [Q, classes]; regression returns [Q, 999] quantiles
```
