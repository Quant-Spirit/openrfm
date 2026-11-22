"""The RT -> TabICL handoff: per-entity feature rows read from the backbone at the target cell and on the entity's row.

An entity's row is ``[z ; target-cell states of the tapped blocks ; per bank column (k projections, present)]``. The
bank is the sorted union of the members' slot-row columns (table, column); a member's projections of a column are
``Linear([LN(s) ; LN(s) * LN(e_col)])`` of the early-depth state ``s`` of its first cell in that column.
"""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .objective import focal_targets, slot_row_nodes

KEY_SHIFT = 2**31
INIT_SEED = 20260926


@dataclass
class Record:
    entity: int
    z: torch.Tensor
    extras: torch.Tensor
    row_cells: dict


class Handoff(torch.nn.Module):
    """The trainable per-column projector of the handoff (``rt_handoff`` in checkpoints)."""

    def __init__(self, d_model: int, *, slots: int, taps: tuple[int, ...], k: int, row_depth: int):
        super().__init__()
        self.d_model, self.slots, self.taps, self.k, self.row_depth = d_model, slots, tuple(taps), k, row_depth
        generator = torch.Generator(device="cpu")
        generator.manual_seed(INIT_SEED)
        self.ecol_weight = torch.nn.Parameter(torch.randn((k, 2 * d_model), generator=generator) / float(2 * d_model) ** 0.5)
        self.ecol_bias = torch.nn.Parameter(torch.zeros(k))

    @property
    def capture_blocks(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.taps) | {self.row_depth - 1}))

    def export(self) -> dict:
        return dict(schema="rt_handoff_v1", mode="h3e", d_model=self.d_model, slots=self.slots, g_dim=16,
                    slot_target_source=None,
                    state_dict={name: value.detach().cpu().clone() for name, value in self.state_dict().items()},
                    h3e=dict(taps=list(self.taps), k=self.k, row_depth=self.row_depth))

    def forward(self, z, extras, column_embeddings, row_states, row_present) -> torch.Tensor:
        n = int(z.shape[0])
        target = [z.float(), extras.float().to(z.device).reshape(n, len(self.taps) * self.d_model)]
        if column_embeddings is None:
            return torch.cat(target, dim=-1)
        states = F.layer_norm(row_states.float().to(z.device), (self.d_model,))
        names = F.layer_norm(column_embeddings.float().to(z.device), (self.d_model,))
        present = row_present.to(device=z.device, dtype=torch.float32)
        cols = F.linear(torch.cat([states, states * names.unsqueeze(0)], dim=-1), self.ecol_weight.float(),
                        self.ecol_bias.float()) * present.unsqueeze(-1)
        row = torch.cat([cols, present.unsqueeze(-1)], dim=-1).reshape(n, column_embeddings.shape[0] * (self.k + 1))
        return torch.cat(target + [row], dim=-1)


@contextmanager
def capture_block_outputs(blocks, indices):
    """Keep ``blocks[i]``'s output for every ``i`` in ``indices`` while the context is open."""
    outputs, handles = {}, []

    def hook_for(index):
        def hook(_module, _inputs, output):
            outputs[index] = output
        return hook

    try:
        for index in indices:
            handles.append(blocks[index].register_forward_hook(hook_for(index)))
        yield outputs
    finally:
        for handle in handles:
            handle.remove()


def _row_context_cells(batch: dict) -> torch.Tensor:
    cells = ~batch["is_padding"].bool() & ~batch["is_targets"].bool()
    for key in ("retrieved_label_cells", "is_task_nodes", "hidden_label_cells", "declared_drop_cells"):
        if torch.is_tensor(batch.get(key)):
            cells &= ~batch[key].bool()
    return cells


def extract_records(batch: dict, hidden: torch.Tensor, states: dict, handoff: Handoff) -> list[Record]:
    """One record per sequence: final and tapped target states and the slot row's early-depth cells."""
    rows = list(range(hidden.shape[0]))
    has_target, target_pos = focal_targets(batch["is_targets"])
    device = has_target.device
    focal_node = batch["node_idxs"][torch.arange(len(rows), device=device), target_pos].unsqueeze(1)
    slot_node = slot_row_nodes(batch, focal_node, target_pos, has_target)
    keys = batch["table_name_idxs"].long() * KEY_SHIFT + batch["col_name_idxs"].long()
    index = torch.as_tensor(rows, dtype=torch.long, device=device)
    meta_focal, meta_keys = focal_node.index_select(0, index).squeeze(1).cpu(), keys.index_select(0, index).cpu()
    local = torch.arange(len(rows), device=hidden.device)
    picked_pos = target_pos.index_select(0, index).to(hidden.device)
    z = hidden.index_select(0, index)[local, picked_pos].float()
    extras = torch.stack([states[tap].index_select(0, index)[local, picked_pos].float() for tap in handoff.taps], dim=1)
    names = batch["col_name_values"].index_select(0, index)
    row_state = states[handoff.row_depth - 1].index_select(0, index)
    row_mask = ((batch["node_idxs"] == slot_node) & _row_context_cells(batch) & has_target.unsqueeze(1)).index_select(0, index).cpu()
    positions = [row_mask[i].nonzero(as_tuple=False).flatten().tolist() for i in rows]
    flat_i = [i for i, p in enumerate(positions) for _ in p]
    flat_p = [pos for p in positions for pos in p]
    gathered = row_state[torch.as_tensor(flat_i, dtype=torch.long, device=row_state.device),
                         torch.as_tensor(flat_p, dtype=torch.long, device=row_state.device)].float() if flat_i else None
    gathered_names = names[torch.as_tensor(flat_i, dtype=torch.long, device=names.device),
                           torch.as_tensor(flat_p, dtype=torch.long, device=names.device)] if flat_i else None
    records, offset = [], 0
    for i in rows:
        cells = {}
        for j, pos in enumerate(positions[i]):
            key = int(meta_keys[i, pos])
            if key not in cells:
                cells[key] = (gathered_names[offset + j], gathered[offset + j])
        offset += len(positions[i])
        records.append(Record(int(meta_focal[i]), z[i], extras[i], cells))
    return records


def detached(record: Record) -> Record:
    return Record(record.entity, record.z.detach().clone(), record.extras.detach().clone(),
                  {key: (name.detach().clone(), state.detach().clone()) for key, (name, state) in record.row_cells.items()})


def handoff_features(handoff: Handoff, col_embed: torch.nn.Module, support: list[Record], query: list[Record]):
    """``(support rows, query rows)`` of the consumer, over the bank of all members' slot-row columns."""
    members = (*support, *query)
    chosen = {}
    for record in members:
        for key, (name, _) in record.row_cells.items():
            if key not in chosen or record.entity < chosen[key][0]:
                chosen[key] = (record.entity, name)
    bank = tuple(sorted(chosen))
    if len(bank) > handoff.slots:
        raise RuntimeError(f"the handoff bank needs {len(bank)} columns, more than its {handoff.slots} slots")
    device = handoff.ecol_weight.device
    z = torch.stack([record.z.to(device) for record in members])
    extras = torch.stack([record.extras.to(device) for record in members])
    embeddings = row_states = row_present = None
    if bank:
        col = col_embed.weight
        with torch.no_grad():
            embeddings = col_embed(torch.stack([chosen[key][1].detach() for key in bank]).to(device=col.device, dtype=col.dtype)).float().to(device)
        row_states = torch.zeros((len(members), len(bank), handoff.d_model), device=device)
        row_present = torch.zeros((len(members), len(bank)), dtype=torch.bool, device=device)
        slot = {key: position for position, key in enumerate(bank)}
        filled = [(m, slot[key], state) for m, record in enumerate(members) for key, (_, state) in record.row_cells.items()]
        if filled:
            index_m = torch.as_tensor([m for m, _, _ in filled], dtype=torch.long, device=device)
            index_s = torch.as_tensor([s for _, s, _ in filled], dtype=torch.long, device=device)
            row_states = row_states.index_put((index_m, index_s), torch.stack([state.to(device).float() for _, _, state in filled]))
            row_present[index_m, index_s] = True
    features = handoff(z, extras, embeddings, row_states, row_present)
    if not bool(torch.isfinite(features).all()):
        raise FloatingPointError("non-finite handoff features")
    return features[:len(support)], features[len(support):]
