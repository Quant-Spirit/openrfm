"""Opaque column names: synthetic column-name embeddings are replaced by per-episode random aliases."""
import hashlib
from functools import lru_cache

import torch

from openrfm.sampler.native import get_extension

BANK_VERSION = "opaque_rademacher_v1"
SYNTHETIC = 2  # the native alias law's code of a synthetic (opaque-named) task


@lru_cache(maxsize=8)
def alias_bank(embedding_dim: int, alias_count: int) -> torch.Tensor:
    """``[1 + alias_count, dim]`` fixed sign vectors scaled by dim^-1/2: the target-column vector, then the aliases."""
    scale = float(embedding_dim) ** -0.5

    def vector(kind: str, index: int) -> torch.Tensor:
        bits = hashlib.shake_256(f"{BANK_VERSION}:{kind}:{index}".encode("ascii")).digest(embedding_dim)
        return torch.tensor([scale if value & 1 else -scale for value in bits], dtype=torch.float32)

    return torch.stack([vector("target_column", 0)] + [vector("opaque_column", i) for i in range(alias_count)])


def neutralize_names(batch: dict, *, seed: int, batch_index: int, task_count: int, bank: torch.Tensor) -> dict:
    """Replace each non-focal physical column's name embedding by an alias drawn from the episode's seed."""
    # Episode e is seeded by a hash of seed + batch_index * 1_000_003 + e * 97_409, its task and its sorted columns.
    values = batch["col_name_values"]
    b = values.shape[0]
    modulus = (1 << 63) - 1
    base = (seed + batch_index * 1_000_003) % modulus
    seeds = torch.tensor([(base + episode * 97_409) % modulus for episode in range(b)], dtype=torch.long)
    device_bank = bank.to(device=values.device, dtype=values.dtype)
    valid = ~batch["is_padding"]
    counts = valid.long().sum(1, keepdim=True)
    destinations = torch.where(valid, valid.long().cumsum(1) - 1, counts + (~valid).long().cumsum(1) - 1)
    metadata = torch.stack((batch["dataset_idxs"], batch["table_name_idxs"], batch["col_name_idxs"],
                            batch["is_targets"].long(), valid.long()), -1)
    receipt = torch.empty_like(metadata).scatter(1, destinations[..., None].expand_as(metadata), metadata).cpu()
    kinds = torch.full((task_count,), SYNTHETIC, dtype=torch.long)
    slots, synthetic = get_extension().rt_alias_slots(receipt, kinds, seeds, bank.shape[0] - 1)
    codes = torch.where(synthetic[:, None], slots + 1, 0).to(values.device).gather(1, destinations)
    replacement = device_bank[(codes - 1).clamp_min(0)]
    return dict(batch, col_name_values=torch.where((valid & (codes > 0))[..., None], replacement, values))
