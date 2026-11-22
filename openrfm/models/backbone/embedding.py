# Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
"""Cell embedding: value visibility, typed value encoding and the block-0 input."""
import torch
from torch import nn

from .config import DISCRETE, VALUE_TYPES
from .masks import cell_timestamps, focal_targets


def contemporaneous_labels(batch: dict) -> torch.Tensor:
    """``[B, S]`` cells of other rows in the target column whose timestamp is at or after the seed's."""
    is_targets = batch["is_targets"].bool()
    node, col, table = batch["node_idxs"], batch["col_name_idxs"], batch["table_name_idxs"]
    rows = torch.arange(node.shape[0], device=node.device)
    _, pos = focal_targets(is_targets)
    timestamps = cell_timestamps(batch)
    candidates = ((col == col[rows, pos, None]) & (table == table[rows, pos, None]) & (node != node[rows, pos, None])
                  & ~batch["is_padding"].bool() & is_targets.any(dim=1, keepdim=True))
    return candidates & (timestamps >= timestamps[rows, pos, None])


def value_visibility(batch: dict) -> torch.Tensor:
    """``[B, S]`` cells whose value the encoder may read."""
    hidden = batch["masks"].bool() | contemporaneous_labels(batch)
    if "encoder_hidden_cells" in batch:
        hidden = hidden | batch["encoder_hidden_cells"].bool()
    return ~hidden & ~batch["is_padding"].bool()


def encode_cell_value(rt, name: str, value: torch.Tensor) -> torch.Tensor:
    """Encoded value ``[..., d_model]`` of cells of type ``name`` from their raw values ``[..., width]``."""
    if name == "datetime":
        # fp32 end to end: autocast would round epoch-scale datetime values to bf16 before the GEMM.
        with torch.autocast(device_type=value.device.type, enabled=False):
            layer = rt.enc_dict[name]
            encoded = nn.functional.linear(value.float(), layer.weight.float(), layer.bias.float())
            return rt.norm_dict[name](encoded).to(layer.weight.dtype)
    return rt.norm_dict[name](rt.enc_dict[name](value))


def embed_cells(rt, batch: dict) -> torch.Tensor:
    """Block-0 input ``[B, S, d_model]``; also stores the visibility in ``batch["value_visible_cells"]``."""
    padding, sem_types = batch["is_padding"], batch["sem_types"]
    x = rt.norm_dict["col_name"](rt.enc_dict["col_name"](batch["col_name_values"])) * (~padding)[..., None]
    visible_cells = batch["value_visible_cells"] = value_visibility(batch)
    hidden_cells = ~visible_cells & ~padding
    for index, name in enumerate(VALUE_TYPES):
        visible = (sem_types == index) & visible_cells
        raw = batch["datetime_values_fp32"] if name == "datetime" else batch[f"{name}_values"]
        x = x + encode_cell_value(rt, name, torch.where(visible[..., None], raw, 0.0)) * visible[..., None]
        x = x + rt.mask_embs[name] * ((sem_types == index) & hidden_cells)[..., None]
    discrete = (sem_types == DISCRETE) & ~padding
    # Skipped without discrete cells, so the class embedding then receives no gradient at all.
    if bool(discrete.any()):
        visible = discrete & visible_cells
        labels = torch.where(visible, batch["class_value_idxs"].long(), 0)
        x = x + rt.norm_dict["discrete"](rt.enc_dict["discrete"](labels)) * visible[..., None]
    x = x + rt.mask_embs["discrete"] * (discrete & hidden_cells)[..., None]
    x = x + rt.structural_features(batch)
    x = x + rt.context_time_role(batch)
    return x + rt.cell_time_residual(batch).to(x.dtype)
