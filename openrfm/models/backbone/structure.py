# Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
"""Structural inputs: per-cell graph features and the entity axis over the focal entity's rows."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .masks import focal_targets


class StructuralCellFeatures(nn.Module):
    """Embeddings of the sampler's per-cell hop, path signature, time delta and row degrees."""

    def __init__(self, d_model: int, path_buckets: int):
        super().__init__()
        self.hop = nn.Embedding(66, d_model)
        self.path = nn.Embedding(path_buckets, d_model)
        self.time = nn.Embedding(128, d_model)
        self.in_degree = nn.Embedding(32, d_model)
        self.out_degree = nn.Embedding(32, d_model)
        for module in self.children():
            nn.init.zeros_(module.weight)

    def forward(self, batch: dict) -> torch.Tensor:
        x = self.hop(batch["cell_hop"].long().clamp(0, 65))
        x = x + self.path(batch["cell_path_sig"].long().clamp_min(0).remainder(self.path.num_embeddings))
        # cell_tdelta lies in [-63, 63]; -128 means no time delta.
        x = x + self.time(torch.where(batch["cell_tdelta"] == -128, 127, batch["cell_tdelta"].long() + 63).clamp(0, 127))
        x = x + self.in_degree(batch["cell_deg_in"].long().clamp(0, 31))
        x = x + self.out_degree(batch["cell_deg_out"].long().clamp(0, 31))
        return x * (~batch["is_padding"])[..., None]


def entity_rows(batch: dict, bucket: int = 0) -> tuple[torch.Tensor, ...]:
    """Rows of the focal entity's neighbourhood: ``(inverse, live, counts, membership)``."""
    nodes, parents = batch["node_idxs"], batch["f2p_nbr_idxs"]
    _, pos = focal_targets(batch["is_targets"])
    ix = torch.arange(len(nodes), device=nodes.device)
    focal = nodes[ix, pos]
    selected = (nodes == focal[:, None]) | (parents == focal[:, None, None]).any(-1)
    for key in parents[ix, pos].unbind(-1):
        selected |= (key >= 0)[:, None] & ((nodes == key[:, None]) | (parents == key[:, None, None]).any(-1))
    live = selected & ~batch["is_padding"].bool() & ~batch["is_targets"].bool()
    for name in ("encoder_hidden_cells", "hidden_label_cells", "declared_drop_cells"):
        if name in batch:
            live &= ~batch[name].bool()
    ordered, order = nodes.masked_fill(~live, -1).sort(dim=1, stable=True)
    first = ordered >= 0
    first[:, 1:] &= ordered[:, 1:] != ordered[:, :-1]
    group = (first.long().cumsum(1) - 1).clamp_min(0)
    inverse = torch.zeros_like(group).scatter(1, order, group)
    width = max(1, int(first.sum(1).max()))
    if bucket:
        width = bucket << (-(-width // bucket) - 1).bit_length()
    counts = torch.zeros((len(nodes), width), device=nodes.device).scatter_add_(1, inverse, live.float())
    with torch.autocast(device_type=live.device.type, enabled=False):
        membership = torch.zeros((*inverse.shape, width), dtype=torch.float32, device=inverse.device)
        membership.scatter_(2, inverse[..., None], 1.)
        membership = membership.transpose(1, 2)
        membership *= live[:, None, :]
    return inverse, live, counts, membership


class EntityAxis(nn.Module):
    """Attention from each cell to the mean-pooled row tokens; half the heads softmax, half sigmoid-sum."""

    def __init__(self, d_model: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm = nn.RMSNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x: torch.Tensor, rows: tuple[torch.Tensor, ...]) -> torch.Tensor:
        _, _, counts, membership = rows
        h = self.norm(x)
        with torch.autocast(device_type=h.device.type, enabled=False):
            row = torch.bmm(membership, h.float()) / counts.clamp_min(1)[..., None]
        B, S, D = h.shape
        q = self.q(h).view(B, S, self.heads, -1).transpose(1, 2)
        k, v = self.kv(row.to(h.dtype)).view(B, row.shape[1], 2, self.heads, -1).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        mask = counts[:, None, None, :] > 0
        split = self.heads // 2
        mean = F.scaled_dot_product_attention(q[:, :split], k[:, :split], v[:, :split], attn_mask=mask)
        # The sum heads count: sigmoid weights are not normalized, and a signed log1p bounds the total.
        weights = torch.sigmoid(q[:, split:] @ k[:, split:].transpose(-1, -2) / math.sqrt(q.shape[-1])) * mask
        total = weights @ v[:, split:]
        total = total.sign() * total.abs().log1p()
        y = torch.cat((mean, total), 1)
        return self.out(y.transpose(1, 2).reshape(B, S, D))
