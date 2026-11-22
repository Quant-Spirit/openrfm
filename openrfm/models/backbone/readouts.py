# Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
"""Training readout: per-type decoders of the masked cells and their loss."""
import torch

from .config import DISCRETE, VALUE_TYPES
from .losses import discrete_loss, point_loss, sequence_mean


def masked_cell_loss(decoders, x: torch.Tensor, batch: dict) -> tuple[torch.Tensor, dict]:
    """Loss and per-cell predictions for the cells selected by ``batch["masks"]``.

    Each sequence's loss is divided by its masked-cell count; the batch averages sequences with a masked cell.
    """
    sem_types, masks = batch["sem_types"], batch["masks"].bool()
    regression_rows = batch.get("is_regression_task")
    if regression_rows is None:
        regression_rows = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    regression_rows = regression_rows.to(device=x.device, dtype=torch.bool).flatten()
    outputs = {"hidden": x}
    loss = x.new_zeros(x.shape[0])
    for index, name in enumerate(VALUE_TYPES):
        prediction = outputs[name] = decoders[name](x)
        selected = (sem_types == index) & masks
        regression = None
        if name == "number":
            regression = selected & batch["is_targets"].bool() & regression_rows[:, None]
            outputs["regression"] = decoders["regression"](x)
            selected = selected & ~regression
        # A type without supervised cells adds an exact zero that keeps its decoder in the graph; skipping its
        # (all-zero) loss also keeps the accumulation in the decoders' dtype.
        if not selected.any() and not (regression is not None and regression.any()):
            loss = loss + prediction.sum() * 0.0
            continue
        target = batch[f"{name}_values"]
        if regression is not None:
            loss = loss + point_loss(outputs["regression"], target, regression, binary=False).sum(dim=1)
        loss = loss + point_loss(prediction, target, selected, binary=name == "boolean").sum(dim=1)
    outputs["discrete"] = decoders["discrete"](x)
    selected = (sem_types == DISCRETE) & masks
    if selected.any():
        loss = loss + discrete_loss(outputs["discrete"], batch, selected).sum(dim=1)
    else:
        loss = loss + outputs["discrete"].sum(dim=(1, 2)) * 0.0
    return sequence_mean(loss, masks), outputs
