# Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
"""Focal positions, timestamp sentinels and the dense intra-entity attention masks."""
import torch

from openrfm.archive.policy import ARCHIVE_STATIC

# Timestamps that mean "this row has no time": -1, and i32::MIN for rows of tables without a time column.
NO_TIME = (-1, int(ARCHIVE_STATIC))


def is_no_time(timestamps: torch.Tensor) -> torch.Tensor:
    return (timestamps == NO_TIME[0]) | (timestamps == NO_TIME[1])


def cell_timestamps(batch: dict) -> torch.Tensor:
    """``[B, S]`` timestamps (the batch may carry them as ``[B, S, 1]``)."""
    timestamps = batch["timestamps"]
    return timestamps.squeeze(-1) if timestamps.dim() == 3 else timestamps


def focal_targets(is_targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(has_target, position)`` of each sequence's target cell; position is 0 where there is none."""
    flags = is_targets.bool()
    return flags.any(dim=1), flags.long().argmax(dim=1)


def dense_attention_masks(batch: dict) -> dict[str, torch.Tensor]:
    """``[B, S, S]`` boolean masks of the col, feat and nbr axes (query, key).

    feat: the key is a cell of the query's row or of one of the query row's parents.
    nbr:  the query's row is a parent of the key's row.
    col:  same table and column.
    Padding cells neither query nor key.
    """
    node, parents = batch["node_idxs"], batch["f2p_nbr_idxs"]
    col, table = batch["col_name_idxs"], batch["table_name_idxs"]
    live = ~batch["is_padding"]
    pair = live[:, :, None] & live[:, None, :]
    feat = node[:, :, None] == node[:, None, :]
    nbr = torch.zeros_like(pair)
    # One parent slot at a time: a [B, S, S, P] comparison would be P times the mask.
    for slot in range(parents.shape[-1]):
        feat |= node[:, None, :] == parents[:, :, None, slot]
        nbr |= node[:, :, None] == parents[:, None, :, slot]
    same_column = (col[:, :, None] == col[:, None, :]) & (table[:, :, None] == table[:, None, :])
    return {"col": same_column & pair, "feat": feat & pair, "nbr": nbr & pair}
