# Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
"""The attention axes: masked self-attention over cells and the cross-sequence support axis."""
import torch
import torch.nn.functional as F
from torch import nn

from .masks import cell_timestamps, focal_targets, is_no_time


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm computed in fp32 and cast back before the weight."""
    dtype = x.dtype
    x_f = x.float()
    return (x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)).to(dtype) * weight


class MaskedAttention(nn.Module):
    """Multi-head attention with RMS-normalized queries and keys; queries without any key return zeros."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.q_norm_weight = nn.Parameter(torch.ones(self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim))

    def heads(self, x: torch.Tensor) -> torch.Tensor:
        """``[..., S, H*D] -> [..., H, S, D]``."""
        return x.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)

    def merge(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, H, S, D] -> [B, S, H*D]`` through the output projection."""
        return self.wo(x.transpose(1, 2).flatten(2))

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        q = rms_norm(self.heads(self.wq(x)), self.q_norm_weight)
        k = rms_norm(self.heads(self.wk(x)), self.k_norm_weight)
        v = self.heads(self.wv(x))
        if torch.is_tensor(mask):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1))
        else:
            out = mask.attend(q, k, v)
        return self.merge(out)


def validate_support_timestamps(batch: dict) -> None:
    """Refuse a batch whose readable support rows lack a real time.

    i32::MIN orders before every real time, so a support row without a time would be readable by every query.
    """
    _, pos = focal_targets(batch["is_targets"])
    row_time = torch.gather(cell_timestamps(batch).long(), 1, pos[:, None]).squeeze(1)
    if batch.get("support_timeless"):
        if not bool(is_no_time(row_time).all()):
            raise ValueError("a batch marked support_timeless carries real timestamps")
        return
    support = batch.get("is_support_seed")
    if support is not None:
        row_time = row_time[support.bool().reshape(-1)]
    missing = is_no_time(row_time)
    if bool(missing.any()):
        raise ValueError(f"{int(missing.sum())} of {missing.numel()} support rows have no timestamp")


class SupportAxisAttention(MaskedAttention):
    """Cross-entity attention from every cell to one memory slot per sequence of the batch.

    A query reads only support slots whose seed time is strictly earlier, never its own slot.
    """

    def memory(self, x: torch.Tensor, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Slot states ``[B, D]`` and the readable ``[query, slot]`` pairs."""
        B = x.shape[0]
        is_targets = batch["is_targets"].to(device=x.device, dtype=torch.bool)
        support = batch.get("is_support_seed")
        timeless = bool(batch.get("support_timeless"))
        if not timeless:
            _, pos = focal_targets(is_targets)
            row_time = torch.gather(cell_timestamps(batch).to(device=x.device).long(), 1, pos[:, None]).squeeze(1)
        if support is None:
            # Task-table rows keep their own labels, so they stay out of the feature pool.
            pooled = ~is_targets & ~batch["is_task_nodes"].to(device=x.device, dtype=torch.bool)
            pooled = pooled & ~batch["is_padding"].to(device=x.device, dtype=torch.bool)
            weights = pooled.unsqueeze(-1).to(dtype=x.dtype)
            memory = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            readable = torch.ones(B, device=x.device, dtype=torch.bool)
            earlier = (torch.ones(B, B, dtype=torch.bool, device=x.device) if timeless
                       else row_time[None, :] <= row_time[:, None])
        else:
            has_target, pos = focal_targets(is_targets)
            memory = torch.gather(x, 1, pos[:, None, None].expand(-1, 1, x.shape[-1])).squeeze(1)
            memory = memory * has_target[:, None].to(dtype=memory.dtype)
            readable = support.to(device=x.device, dtype=torch.bool) & has_target
            if timeless:
                earlier = torch.ones(B, B, dtype=torch.bool, device=x.device)
            else:
                earlier = row_time[None, :] < row_time[:, None]
                known = batch.get("support_label_available_at")
                if known is not None:
                    earlier = earlier & (known.to(device=x.device).long()[None, :] < row_time[:, None])
        pairs = ~torch.eye(B, device=x.device, dtype=torch.bool) & earlier & readable[None, :]
        if "episode_donor_mask" in batch:
            pairs = pairs & batch["episode_donor_mask"].to(device=x.device, dtype=torch.bool)
        return memory, pairs

    def forward(self, x: torch.Tensor, batch: dict) -> torch.Tensor:
        memory, pairs = self.memory(x, batch)
        queries = ~batch["is_padding"].to(device=x.device, dtype=torch.bool)
        mask = pairs[:, None, :] & queries[:, :, None]  # [query sequence, query cell, slot]
        q = rms_norm(self.heads(self.wq(x)), self.q_norm_weight)
        k = rms_norm(self.heads(self.wk(memory)).unsqueeze(0).expand(x.shape[0], -1, -1, -1), self.k_norm_weight)
        v = self.heads(self.wv(memory)).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1))
        # Rows with no readable slot are pinned to zero whatever the kernel does with an all-masked softmax.
        out = out.masked_fill(~mask.any(dim=-1)[:, None, :, None], 0.0)
        return self.merge(out)
