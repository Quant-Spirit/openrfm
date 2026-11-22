"""The consumer objective: every ``every`` updates, one episode of one world's task is encoded by the backbone under the
evaluation contract and scored by the frozen TabICL consumer; its gradients join the update's gradients.

Supports are encoded without gradients (detached records) at the current weights; the query records' gradient is
replayed through the backbone chunk by chunk. Each member's own label cell is hidden in every context of the episode.
"""
import json
import math
from contextlib import contextmanager

import torch

from openrfm.archive.format import take
from openrfm.sampler.retrieval import STATIC, context_settings

from .consumer import group_loss
from .context import draw_programs, pack_context
from .episodes import Entity, EpisodeSources, cursor_seed, draw_geometry, episode_identity, select_episode
from .handoff import capture_block_outputs, detached, extract_records, handoff_features
from .pack import request_flags

TRAINING_ONLY = ("is_support_seed", "support_label_available_at")


@contextmanager
def preserved_rng():
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state_all(cuda)


def eval_view(batch: dict, members) -> dict:
    """The encoder input of an episode member: no support axis, other members' answers and label cells hidden."""
    view = {k: v for k, v in batch.items() if k not in TRAINING_ONLY}
    for key in ("number_values", "boolean_values"):
        view[key] = torch.nan_to_num(view[key], nan=0.0, posinf=0.0, neginf=0.0)
    padding = view["is_padding"].bool()
    targets = view["is_targets"].bool() & ~padding
    view["masks"] = targets.to(dtype=batch["masks"].dtype)
    pos = targets.long().argmax(dim=1, keepdim=True)
    same_column = ((view["col_name_idxs"] == view["col_name_idxs"].gather(1, pos))
                   & (view["table_name_idxs"] == view["table_name_idxs"].gather(1, pos))
                   & targets.any(dim=1, keepdim=True) & ~padding)
    same_entity = view["node_idxs"] == view["node_idxs"].gather(1, pos)
    hidden = targets | (same_column & same_entity)
    others = same_column & ~same_entity
    if bool(others.any()):
        timestamps = view["timestamps"]
        unknown = ~torch.isfinite(timestamps) | (timestamps == -(2**31)) | (timestamps == -(2**63))
        hidden = hidden | (others & (unknown | unknown.gather(1, pos) | (timestamps >= timestamps.gather(1, pos))))
    if "encoder_hidden_cells" in batch:
        hidden = hidden | batch["encoder_hidden_cells"].bool()
    dataset, table, column, nodes = members
    nodes = torch.as_tensor(sorted(set(nodes)), dtype=view["node_idxs"].dtype, device=view["node_idxs"].device)
    same = (view["table_name_idxs"] == table) & (view["col_name_idxs"] == column) & (view["dataset_idxs"] == dataset)
    view["encoder_hidden_cells"] = (hidden | (same & torch.isin(view["node_idxs"], nodes))) & ~padding
    view["is_support_seed"] = torch.zeros(int(targets.shape[0]), dtype=torch.bool, device=targets.device)
    return view


def _lineage(panel):
    """Copies of earlier outcomes in a fact world (case row, published history row, value columns -> fact row)."""
    root = panel["archives"] / panel["task"].db
    source = json.loads((root / "a5_world.json").read_text()) if (root / "a5_world.json").exists() else {}
    payload, graph = panel["payload"], panel["graph"]
    facts = payload["table_names"].get(source.get("event_table", "") + ":Db")
    history_tables = payload.get("history_table_ids", torch.empty(0)).detach().cpu().tolist()
    columns = json.loads((root / "column_index.json").read_text())
    schema = json.loads((root / "anchor_schema.json").read_text()) if (root / "anchor_schema.json").exists() else {}
    value_columns = set()
    for role in ("own_label", "peer_label"):
        channel = schema.get(role, {})
        if isinstance(channel, dict) and isinstance(channel.get("value_column"), str):
            for table in channel.get("tables", []):
                if channel["value_column"] + " of " + table in columns:
                    value_columns.add(columns[channel["value_column"] + " of " + table])
    case_to_fact, history_to_fact = {}, {}
    if facts is not None and history_tables and source.get("group") in ("F1", "F2", "F3"):
        tables = graph.row_table.detach().cpu().long().tolist()
        times = graph.row_time.detach().cpu().long().tolist()
        parents = payload["parents"].detach().cpu().long().tolist()

        def key(node):
            return times[node], tuple(sorted(p for p in parents[node] if p >= 0))
        lookup = {}
        for node, table in enumerate(tables):
            if table == facts:
                if key(node) in lookup:
                    raise ValueError("ambiguous fact/case lineage")
                lookup[key(node)] = node
        for node in panel["roots"]:
            case_to_fact[int(node)] = lookup[key(int(node))]
        for node, table in enumerate(tables):
            if table in history_tables:
                linked = [p for p in parents[node] if p >= 0 and tables[p] == facts]
                if len(linked) != 1:
                    raise ValueError("published history needs one fact witness")
                history_to_fact[node] = linked[0]
    return case_to_fact, history_to_fact, sorted(value_columns)


def hide_query_copies(lineage, batch, query_nodes):
    cases, history, columns = lineage
    if not cases:
        return batch
    hidden_facts = {cases[int(node)] for node in query_nodes}
    rows = [node for node, fact in history.items() if fact in hidden_facts]
    device = batch["node_idxs"].device
    mask = (torch.isin(batch["node_idxs"], torch.tensor(rows, device=device, dtype=batch["node_idxs"].dtype))
            & torch.isin(batch["col_name_idxs"], torch.tensor(columns, device=device, dtype=batch["col_name_idxs"].dtype))
            & ~batch["is_padding"])
    return dict(batch, encoder_hidden_cells=batch["encoder_hidden_cells"] | mask,
                hidden_label_cells=batch["hidden_label_cells"] | mask)


def replay_query_gradients(chunks, encode, loss_fn, parameters):
    """Loss and parameter gradients of ``loss_fn(records)``, pulled back through re-encoded query chunks."""
    # Each chunk is re-encoded at the same weights and RNG state as its gradient-free first pass.
    device = torch.cuda.current_device()
    pieces, leaves = [], []
    for chunk in chunks:
        cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        with torch.no_grad():
            records = encode(chunk)
        local = [type(r)(r.entity, r.z.detach().clone().requires_grad_(True), r.extras.detach().clone().requires_grad_(True),
                         {k: (r.row_cells[k][0].detach().clone(), r.row_cells[k][1].detach().clone().requires_grad_(True))
                          for k in sorted(r.row_cells)}) for r in records]
        pieces.append((chunk, cpu, cuda, local))
        leaves.extend(local)

    def tensors(record):
        return [record.z, record.extras] + [record.row_cells[k][1] for k in sorted(record.row_cells)]

    flat = [t for record in leaves for t in tensors(record)]
    loss = loss_fn(leaves)
    grads = torch.autograd.grad(loss, [*flat, *parameters], allow_unused=True)
    record_grads, total, offset = grads[:len(flat)], list(grads[len(flat):]), 0
    for chunk, cpu, cuda, local in pieces:
        with torch.random.fork_rng(devices=[device]):
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state(cuda, device)
            outputs = [t for record in encode(chunk) for t in tensors(record)]
        vjp = record_grads[offset:offset + len(outputs)]
        offset += len(outputs)
        active = [(out, g) for out, g in zip(outputs, vjp) if g is not None and out.requires_grad]
        if active:
            part = torch.autograd.grad([a for a, _ in active], parameters, grad_outputs=[g for _, g in active],
                                       allow_unused=True)
            for i, g in enumerate(part):
                if g is not None:
                    total[i] = g if total[i] is None else total[i] + g
    return loss.detach(), total


class Coupling:
    """The scheduled consumer episodes of the run (cursor-only state)."""

    def __init__(self, config, groups, samplers, weights, archives_root, consumers, handoff):
        self.config, self.episodes = config["coupling"], config["coupling"]["episodes"]
        self.samplers, self.archives_root, self.consumers, self.handoff = samplers, archives_root, consumers, handoff
        self.document = json.dumps(config["context_prior"], sort_keys=True, separators=(",", ":"))
        self.cursor, self.start_step, self.facts, self.lineage = 0, None, {}, {}
        worlds, kinds, populations = [], {}, {}
        for gi, tasks in enumerate(groups):
            sources = {}
            for index, task in enumerate(tasks):
                sources.setdefault(task.world, {}).setdefault((task.table, task.target), index)
                kinds[gi, index] = "forecast_or_other" if task.forecast_entity else "autocomplete"
                populations[gi, index] = task.num_nodes
            worlds.append(sources)
        self.sources = EpisodeSources(weights, worlds, kinds, populations, self.episodes)

    def weight(self, step: int) -> float:
        return float(self.config["weight"]) * min(1.0, max(0, step - self.start_step) / float(self.config["ramp_steps"]))

    def _entities(self, sampler, task_index, graph, payload, roots):
        key = (id(sampler), task_index)
        if key not in self.facts:
            task = sampler.tasks[task_index]
            nodes = torch.as_tensor(roots, device=graph.row_time.device).long()
            cells = payload["composition"].target.long()[nodes]
            types = take(payload["cell_types"], cells).long()
            labels = torch.where(types == 0, take(payload["cell_numbers"], cells).to(torch.bfloat16).float(),
                                 torch.where(types == 3, (take(payload["cell_booleans"], cells).to(torch.bfloat16) > 0).float(),
                                             take(payload["cell_codes"], cells).float()))
            times = take(graph.row_time, nodes).long()
            times = torch.where(times == STATIC, -(2**31), times)
            values = torch.stack((nodes, take(graph.row_table, nodes).long(), types, times,
                                  take(graph.cell_available, cells).long()), 1).cpu().tolist()
            index = sampler.task_offset + task_index
            self.facts[key] = [Entity(i, n, (index, t, task.target_column, s), y, c, a)
                               for i, ((n, t, s, c, a), y) in enumerate(zip(values, labels.cpu().tolist()))
                               if s in (0, 3, 4) and math.isfinite(y)]
        return self.facts[key]

    def _chunks(self, panel, episode, members, program, salt):
        """Packed evaluation views of ``members``, ``encoding_chunk_cells`` cells at a time."""
        sampler, graph, payload, workspace = panel["sampler"], panel["graph"], panel["payload"], panel["workspace"]
        size, device = self.episodes["encoding_chunk_cells"] // episode.length, graph.row_time.device
        hidden = (episode.task[0], episode.task[1], panel["task"].target_column, [e.entity for e in (*episode.support, *episode.query)])
        known = panel["task"].public_horizon is not None
        for start in range(0, len(members), size):
            nodes = torch.tensor([e.entity for e in members[start:start + size]], device=device, dtype=torch.long)
            q = torch.stack((nodes, graph.row_time[nodes].long(), torch.full_like(nodes, panel["task"].target_column),
                             nodes * 1000003 ^ salt, request_flags(payload, nodes)), 1)
            settings = context_settings(q, panel["spec"])
            selected, roles, receipt = workspace.sample(q, settings, graph=graph, composition=payload["composition"])[3:6]
            batch = pack_context(self.document, graph, payload, q, selected, roles, receipt, panel["task_index"],
                                 labels_visible=panel["spec"].label_visibility == "history_available",
                                 program=torch.tensor(program, device=device, dtype=torch.long)[None].expand(len(q), -1),
                                 schema_key=salt)
            batch = sampler.finalize(batch, panel["task_index"], self.cursor)
            batch = dict(batch, task_horizon=torch.as_tensor(panel["task"].public_horizon if known else 1.,
                                                             device=device, dtype=torch.float64), task_horizon_valid=known)
            yield hide_query_copies(panel["lineage"], eval_view(batch, hidden), [e.entity for e in episode.query])

    def _encode(self, model, chunk):
        with capture_block_outputs(model.blocks, self.handoff.capture_blocks) as states:
            hidden = model.encode_hidden(chunk, support_axis_enabled=False)
        return extract_records(chunk, hidden, states, self.handoff)

    def accumulate(self, step: int, units: torch.Tensor, model, parameters) -> dict:
        """At the last micro-batch of an update: add ``units * w(step) * grad(L_icl)`` to the parameter gradients."""
        if self.start_step is None:
            self.start_step = step
        weight = self.weight(step)
        if step % self.config["every"] or weight <= 0:
            return {}
        k, length = draw_geometry(self.episodes, self.cursor)
        source = self.sources.coordinate(self.cursor, k)
        if source is None:  # a small corpus has no target with k supports plus the query cohort
            self.cursor += 1
            return dict(icl_weight=weight, icl_skip="no_world_for_stratum")
        group, task_index = source
        program = draw_programs(self.document, torch.tensor([self.cursor])).flatten().tolist()
        sampler = self.samplers[group][length]
        task, spec = sampler.tasks[task_index], sampler.specs[task_index]
        graph, payload, roots, workspace = sampler.archives.runtime(task, spec, sampler.batch_size)
        entities = self._entities(sampler, task_index, graph, payload, roots)
        seed = cursor_seed(self.episodes["seed"], self.cursor)
        self.cursor += 1
        episode, reason = select_episode(entities, length=length, k=k, config=self.episodes, seed=seed, coupling=self.config)
        if episode is None:
            return dict(icl_weight=weight, icl_skip=reason)
        identity = episode_identity(task.world, episode.task, task.public_horizon, tuple(program), length, k, episode)
        salt = int(identity[:15], 16)
        key = (id(sampler), task_index)
        if key not in self.lineage:
            self.lineage[key] = _lineage(dict(archives=self.archives_root, task=task, payload=payload, graph=graph, roots=roots))
        panel = dict(sampler=sampler, task=task, task_index=task_index, spec=spec, graph=graph, payload=payload,
                     workspace=workspace, lineage=self.lineage[key])

        supports = []

        def loss_fn(queries):
            sx, qx = handoff_features(self.handoff, model.enc_dict["col_name"], supports, queries)
            sx = sx.detach()
            mean = sx.mean(dim=0)
            raw_std = sx.std(dim=0, unbiased=False)
            std = torch.where(raw_std > 1e-8, raw_std, torch.ones_like(raw_std))
            return group_loss(self.consumers, episode.classes, [e.label for e in episode.support],
                              [e.label for e in episode.query], (sx - mean) / std, (qx - mean) / std).mean()

        with preserved_rng():
            for chunk in self._chunks(panel, episode, episode.support, program, salt):
                with torch.no_grad():
                    supports.extend(detached(r) for r in self._encode(model, chunk))
            loss, grads = replay_query_gradients(self._chunks(panel, episode, episode.query, program, salt),
                                                 lambda chunk: self._encode(model, chunk), loss_fn, parameters)
        with torch.no_grad():
            for p, g in zip(parameters, grads):
                if g is not None:
                    scaled = (g.float() * units.to(device=p.device, dtype=torch.float32) * weight).to(p.dtype)
                    if p.grad is None:
                        p.grad = scaled
                    else:
                        p.grad.add_(scaled)
        return dict(icl_weight=weight, loss_icl=loss)

    def state_dict(self) -> dict:
        return dict(cursor=self.cursor, start_step=self.start_step)

    def load_state_dict(self, state: dict) -> None:
        self.cursor, self.start_step = int(state["cursor"]), state["start_step"]
