# Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
"""Time features of context cells: age and role relative to the sampler cutoff, and age relative to the horizon."""
import math

import torch
from torch import nn

from .masks import cell_timestamps, is_no_time


class ContextTimeRole(nn.Module):
    """Age (relative to the sampler cutoff) and context-role features of live context cells."""

    def __init__(self, d_model: int):
        super().__init__()
        self.time_proj = nn.Linear(6, d_model, bias=False)
        self.role_embedding = nn.Embedding(3, d_model)
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.role_embedding.weight)

    def forward(self, batch: dict) -> torch.Tensor:
        role = batch["context_role"]
        timestamps = cell_timestamps(batch)
        static = is_no_time(timestamps)
        cutoff = batch["sampler_requests"][:, 1].long().reshape(-1, 1)
        # int64 before subtracting: i32::MIN minus a cutoff overflows int32. Timestamps are seconds, age is days.
        age = (cutoff - timestamps.long()).clamp_min(0).float() / 86400.0
        real = ~static
        age = torch.where(real, age, torch.zeros_like(age))
        features = torch.stack((torch.log1p(age) / torch.log1p(age.new_full((), 3650.0)),
                                (real & (age < 1)).to(age.dtype), (real & (age < 7)).to(age.dtype),
                                (real & (age < 30)).to(age.dtype), (real & (age < 365)).to(age.dtype),
                                static.to(age.dtype)), dim=-1)
        role_index = torch.where(role == 1, 0, torch.where(role == 2, 1, 2)).long()
        active = ~batch["is_padding"].bool() & (role != 0)
        out = self.time_proj(features.to(self.time_proj.weight.dtype)) + self.role_embedding(role_index)
        return out * active[..., None].to(out.dtype)


class CellTimeEncoder(nn.Module):
    """Gaussian radial basis over log(age / horizon), plus a missing-time flag."""

    def __init__(self, dim: int, n_basis: int = 8):
        super().__init__()
        self.centres = nn.Parameter(torch.linspace(-3, 3, n_basis))
        self.log_widths = nn.Parameter(torch.full((n_basis,), math.log(.7)))
        self.proj = nn.Linear(n_basis + 1, dim)

    def forward(self, age: torch.Tensor, horizon: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
        age, horizon = torch.broadcast_tensors(age, horizon)
        ratio = (age / horizon.clamp_min(1e-12)).float().clamp_min(0)
        x = ratio.clamp_min(1e-6).log()[..., None]
        basis = torch.exp(-.5 * ((x - self.centres) / self.log_widths.exp().clamp_min(.02)) ** 2)
        features = torch.cat((missing.float()[..., None], basis), -1)
        return self.proj(features.to(self.proj.weight.dtype))


class HorizonCellTimeResidual(nn.Module):
    """Gated residual of each context cell's age relative to the task horizon."""

    def __init__(self, d_model: int, dim: int):
        super().__init__()
        self.encoder = CellTimeEncoder(dim)
        self.projection = nn.Linear(dim, d_model, bias=False)
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(self, batch: dict) -> torch.Tensor:
        timestamps = cell_timestamps(batch)
        cutoff = batch["sampler_requests"][:, 1].double()[:, None]
        horizon = per_sequence(batch["task_horizon"], timestamps, torch.float64)
        missing = ~torch.isfinite(timestamps) | is_no_time(timestamps)
        age = (cutoff - timestamps.double()).clamp_min(0)
        age = torch.where(missing, torch.zeros_like(age), age)
        projected = self.projection(self.encoder(age, horizon, missing).to(self.projection.weight.dtype))
        active = (batch["context_role"] != 0) & ~batch["is_padding"].bool() & ~batch["is_targets"].bool()
        if "task_horizon_valid" in batch:
            active = active & per_sequence(batch["task_horizon_valid"], timestamps, torch.bool)
        return projected * active[..., None].to(projected.dtype) * self.residual_gate


def per_sequence(value, like: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """A scalar or per-sequence value as a ``[B, 1]`` tensor on ``like``'s device.

    Python values are filled on the device, because a CUDA graph capture cannot contain a host copy.
    """
    if torch.is_tensor(value):
        value = value.to(device=like.device, dtype=dtype)
    elif isinstance(value, (tuple, list)):
        value = torch.stack([like.new_full((), item, dtype=dtype) for item in value])
    else:
        value = like.new_full((), value, dtype=dtype)
    return value.expand(like.shape[0]).reshape(-1, 1) if value.numel() == 1 else value.reshape(-1, 1)
