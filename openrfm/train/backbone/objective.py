"""The column-slot auxiliary objective and the row/cell selections it shares with the handoff.

For up to ``slots`` payload columns of the focal entity's row, a zero-initialized linear head maps
``[z_target ; e_col ; family one-hot]`` to four statistics of that column over the admitted context: cells that are
not labels, targets, task rows, hidden or the entity's own rows, and are older than the query's cutoff.

=======  =======================  ==========================
channel  numeric column            categorical column
=======  =======================  ==========================
0        log1p(finite count)       log1p(classed count)
1        signed log1p(mean)        log1p(distinct classes)
2        signed log1p(last value)  mode share
3        log1p(cutoff - last time) log1p(cutoff - last time)
=======  =======================  ==========================

Undefined channels (no timed cell, no finite value, undated query) are left out of the loss.
"""
import torch

SEM_NUMBER, SEM_DISCRETE = 0, 4
NO_TIME = -(2**31)
SLOT_CHANNELS = 4
MAX_CLASSES = 16


def focal_targets(is_targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flags = is_targets.bool()
    return flags.any(dim=1), flags.long().argmax(dim=1)


def payload_cells(batch: dict) -> torch.Tensor:
    """Numeric and categorical cells that are not padding, targets, labels, task rows, hidden or dropped."""
    payload = ~batch["is_padding"].bool() & ~batch["is_targets"].bool()
    for key in ("retrieved_label_cells", "is_task_nodes", "hidden_label_cells", "declared_drop_cells"):
        if torch.is_tensor(batch.get(key)):
            payload &= ~batch[key].bool()
    sem = batch["sem_types"]
    return payload & ((sem == SEM_NUMBER) | (sem == SEM_DISCRETE))


def slot_row_nodes(batch: dict, focal_node, target_pos, has_target) -> torch.Tensor:
    """``[B, 1]`` the focal row, or its first parent when the focal row is a payload-free task row."""
    node_idxs = batch["node_idxs"]
    payload = payload_cells(batch) & has_target.unsqueeze(1)
    focal_has_payload = ((node_idxs == focal_node) & payload).any(dim=1, keepdim=True)
    parents = batch["f2p_nbr_idxs"][torch.arange(node_idxs.shape[0], device=node_idxs.device), target_pos].long()
    ceiling = torch.iinfo(torch.int64).max
    parent = torch.where(parents >= 0, parents, torch.full_like(parents, ceiling)).min(dim=1, keepdim=True).values
    return torch.where(focal_has_payload | (parent == ceiling), focal_node, parent.to(focal_node.dtype))


def _signed_log1p(value):
    return torch.sign(value) * torch.log1p(value.abs())


def _statistics(batch: dict, slots: int):
    is_targets = batch["is_targets"].bool()
    has_target, target_pos = focal_targets(is_targets)
    if not bool(has_target.any()):
        return None
    rows = torch.arange(is_targets.shape[0], device=is_targets.device)
    node_idxs, timestamps = batch["node_idxs"], batch["timestamps"]
    focal_node = node_idxs[rows, target_pos].unsqueeze(1)
    cutoff = timestamps[rows, target_pos].unsqueeze(1)
    cutoff_known = (cutoff != NO_TIME) & has_target.unsqueeze(1)
    slot_node = slot_row_nodes(batch, focal_node, target_pos, has_target)
    admitted = (~batch["is_padding"].bool() & ~is_targets & ~batch["masks"].bool()
                & ~batch["retrieved_label_cells"].bool() & ~batch["is_task_nodes"].bool())
    for key in ("declared_drop_cells", "hidden_label_cells", "encoder_hidden_cells"):
        if torch.is_tensor(batch.get(key)):
            admitted &= ~batch[key].bool()
    admitted &= (node_idxs != focal_node) & (node_idxs != slot_node)
    admitted &= ((timestamps == NO_TIME) | (timestamps < cutoff)) & has_target.unsqueeze(1)
    on_row = (node_idxs == slot_node) & ~batch["is_padding"].bool() & ~is_targets & has_target.unsqueeze(1)
    on_row &= payload_cells(batch)
    length = on_row.shape[1]
    position = torch.arange(length, device=on_row.device).expand_as(on_row)
    ordered = torch.sort(torch.where(on_row, position, torch.full_like(position, length)), dim=1)[0][:, :slots]
    slot_pos, slot_valid = ordered.clamp_max(length - 1), ordered < length
    if not bool(admitted.any()) or not bool(slot_valid.any()):
        return None
    sem = batch["sem_types"].gather(1, slot_pos)
    numeric_slot = (sem == SEM_NUMBER) & slot_valid
    categorical_slot = (sem == SEM_DISCRETE) & slot_valid
    slot_valid = numeric_slot | categorical_slot
    if not bool(slot_valid.any()):
        return None
    # A column is (table, column): two tables may reuse a column name.
    member = ((batch["col_name_idxs"].unsqueeze(1) == batch["col_name_idxs"].gather(1, slot_pos).unsqueeze(-1))
              & (batch["table_name_idxs"].unsqueeze(1) == batch["table_name_idxs"].gather(1, slot_pos).unsqueeze(-1))
              & admitted.unsqueeze(1) & slot_valid.unsqueeze(-1))
    values = batch["number_values"].float()[..., 0]
    finite = torch.isfinite(values)
    values = torch.where(finite, values, torch.zeros_like(values))
    timed = member & (timestamps != NO_TIME).unsqueeze(1) & cutoff_known.unsqueeze(-1)
    numeric_member = member & finite.unsqueeze(1) & numeric_slot.unsqueeze(-1)
    numeric_timed = timed & finite.unsqueeze(1) & numeric_slot.unsqueeze(-1)
    count = member.sum(-1)
    numeric_count, timed_count = numeric_member.sum(-1), timed.sum(-1)
    mean = (values.unsqueeze(1) * numeric_member).sum(-1) / torch.where(numeric_count > 0, numeric_count,
                                                                       torch.ones_like(numeric_count))
    floor = torch.full_like(timestamps, torch.iinfo(timestamps.dtype).min).unsqueeze(1)
    last_pos = torch.where(numeric_timed, timestamps.unsqueeze(1), floor).argmax(-1)
    last_value = values.gather(1, last_pos.reshape(values.shape[0], -1)).reshape(count.shape)
    recent_pos = torch.where(timed, timestamps.unsqueeze(1), floor).argmax(-1)
    last_time = timestamps.gather(1, recent_pos.reshape(timestamps.shape[0], -1)).reshape(count.shape)
    since = torch.log1p((cutoff - last_time).clamp_min(0).float())
    classes = batch["class_value_idxs"].long()
    bound = max(1, min(int(batch["discrete_num_classes"].max()), MAX_CLASSES))
    # Classes beyond the cap share the last bucket, so the histogram still sums to the classed count.
    bucket = classes.clamp(max=bound - 1)
    categorical_member = member & (classes >= 0).unsqueeze(1) & categorical_slot.unsqueeze(-1)
    histogram = torch.stack([(categorical_member & (bucket == k).unsqueeze(1)).sum(-1) for k in range(bound)], dim=-1)
    categorical_count = categorical_member.sum(-1)
    mode_share = histogram.max(-1).values.float() / torch.where(categorical_count > 0, categorical_count,
                                                                torch.ones_like(categorical_count))
    numeric_target = torch.stack([torch.log1p(numeric_count.float()), _signed_log1p(mean), _signed_log1p(last_value),
                                  since], dim=-1)
    categorical_target = torch.stack([torch.log1p(categorical_count.float()), torch.log1p((histogram > 0).sum(-1).float()),
                                      mode_share, since], dim=-1)
    numeric_channels = torch.stack([numeric_count > 0, numeric_count > 0, numeric_timed.any(-1), timed_count > 0],
                                   dim=-1) & numeric_slot.unsqueeze(-1)
    categorical_channels = torch.stack([categorical_count > 0] * 3 + [timed_count > 0], dim=-1) & categorical_slot.unsqueeze(-1)
    target = (torch.where(numeric_channels, numeric_target, torch.zeros_like(numeric_target))
              + torch.where(categorical_channels, categorical_target, torch.zeros_like(categorical_target)))
    return rows, target_pos, slot_pos, slot_valid, numeric_slot, categorical_slot, target, numeric_channels | categorical_channels


def column_slot_loss(batch: dict, hidden: torch.Tensor, col_embed: torch.nn.Module, head: torch.nn.Linear):
    """Mean Huber loss over the defined channels, or None when no sequence has a supervisable slot."""
    stats = _statistics(batch, head.slots)
    if stats is None:
        return None
    rows, target_pos, slot_pos, slot_valid, numeric_slot, categorical_slot, target, channel_valid = stats
    denominator = channel_valid.float().sum()
    if not bool(denominator > 0):
        return None
    z_target = hidden[rows, target_pos].float()
    with torch.no_grad():
        col_emb = col_embed(batch["col_name_values"]).float()
    col_emb = col_emb.gather(1, slot_pos.unsqueeze(-1).expand(-1, -1, col_emb.shape[-1]))
    family = torch.stack([numeric_slot, categorical_slot], dim=-1).to(col_emb.dtype)
    features = torch.cat([z_target.unsqueeze(1).expand(-1, col_emb.shape[1], -1), col_emb, family], dim=-1)
    # Excluded slots must not reach the head, or their non-finite values would enter its weight gradient.
    features = torch.where(slot_valid.unsqueeze(-1), features, torch.zeros_like(features))
    prediction = head(features.to(head.weight.dtype)).float()
    per_channel = torch.nn.functional.huber_loss(prediction, target.float(), reduction="none", delta=1.0)
    return (per_channel * channel_valid).sum() / denominator


class ColumnSlotHead(torch.nn.Linear):
    """``Linear(2 * d_model + 2 -> 4)``, zero-initialized so the backbone receives no gradient at the first step."""

    def __init__(self, d_model: int, slots: int):
        super().__init__(2 * d_model + 2, SLOT_CHANNELS)
        torch.nn.init.zeros_(self.weight)
        torch.nn.init.zeros_(self.bias)
        self.slots = slots
