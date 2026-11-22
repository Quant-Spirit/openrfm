# Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
"""Per-cell training losses of the masked-cell objective."""
import torch
import torch.nn.functional as F


def mask_discrete_logits(logits: torch.Tensor, cardinalities: torch.Tensor) -> torch.Tensor:
    """Set the logits of class slots at or beyond each cell's class count to -inf."""
    slots = torch.arange(logits.shape[-1], device=logits.device)
    return logits.masked_fill(slots >= cardinalities.to(device=logits.device).unsqueeze(-1), float("-inf"))


def discrete_loss(logits: torch.Tensor, batch: dict, selected: torch.Tensor) -> torch.Tensor:
    """``[B, S]`` cross-entropy of the selected discrete cells over their own classes, zero elsewhere."""
    labels = batch["class_value_idxs"].to(device=selected.device, dtype=torch.long)
    cardinalities = batch["discrete_num_classes"].to(device=selected.device, dtype=torch.long)
    # Unselected cells carry the -1 sentinel; give them all slots and label 0 so their (discarded) loss is finite.
    cardinalities = torch.where(selected, cardinalities, torch.full_like(cardinalities, logits.shape[-1]))
    labels = torch.where(selected, labels, torch.zeros_like(labels))
    loss = F.cross_entropy(mask_discrete_logits(logits, cardinalities).transpose(1, 2), labels, reduction="none")
    return loss * selected


def point_loss(prediction: torch.Tensor, target: torch.Tensor, selected: torch.Tensor, binary: bool) -> torch.Tensor:
    """``[B, S]`` Huber (or logistic, for booleans) loss of the selected cells, zero elsewhere."""
    prediction = torch.where(selected[..., None], prediction, 0.0)
    target = torch.where(selected[..., None], target, 0.0)
    if binary:
        loss = F.binary_cross_entropy_with_logits(prediction, (target > 0).float(), reduction="none")
    else:
        loss = F.huber_loss(prediction, target, reduction="none")
    return loss.mean(-1) * selected


def sequence_mean(cell_loss_sums: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Mean over sequences with a masked cell of each sequence's loss divided by its masked-cell count."""
    per_sequence = cell_loss_sums / masks.sum(dim=1).float().clamp(min=1.0)
    return per_sequence[masks.any(dim=1)].mean()
