"""Corpus archives on the device and the per-(group, length) samplers that turn batch indices into batches."""
import hashlib
import math
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from openrfm.archive.decode import decode_archive
from openrfm.archive.loader import load_rt_archive
from openrfm.sampler.episodes import sample_rt_contexts
from openrfm.sampler.retrieval import context_settings
from openrfm.sampler.v2 import Workspace

from .context import batch_salts, pack_context
from .neutralize import alias_bank, neutralize_names
from .pack import request_flags

EMBEDDING_MODEL, D_TEXT, PARENT_SLOTS = "all-MiniLM-L12-v2", 384, 5


def task_spec(spec, task):
    """The sampler policy of one task: strict-drop autocomplete, or its declared forecast view."""
    if task.forecast_entity is None:
        return replace(spec, task_kind="autocomplete", entity_table=None, label_visibility="strict_drop",
                       label_horizon_seconds=None)
    offset = 0 if task.label_horizon is None else max(0, task.label_horizon - 1)
    return replace(spec, label_visibility="history_available", label_horizon_seconds=offset, context_text_policy="keep",
                   task_kind="forecast", entity_table=task.forecast_entity, filler_percent=min(spec.filler_percent, 20))


class Archives:
    """Decoded archives (host) and task runtimes and workspaces (device), loaded on first use."""

    def __init__(self, archives: Path, device, resident_views: int):
        self.root, self.device, self.capacity = archives, device, resident_views
        self.decoded, self.views = {}, OrderedDict()

    def _state(self, db: str) -> dict:
        if db not in self.decoded:
            decoded = decode_archive(self.root / db, EMBEDDING_MODEL, D_TEXT, PARENT_SLOTS)
            if decoded["values"]["cell_datetimes"].dtype != torch.float32:
                raise ValueError(f"{db}: training archives keep fp32 datetimes (convert with fp32_datetimes=True)")
            self.decoded[db] = {"cpu": decoded}
        return self.decoded[db]

    def view(self, task, spec, batch_size: int) -> dict:
        """The device view of a world under one policy: its task runtimes and its sampling workspace."""
        key = (task.db, spec, batch_size)
        if key not in self.views:
            self.views[key] = {"tasks": {}}
            while len(self.views) > self.capacity:
                self.views.popitem(last=False)
        self.views.move_to_end(key)
        return self.views[key]

    def runtime(self, task, spec, batch_size: int):
        """``(graph, payload, admitted roots, workspace)`` of a task under ``spec``."""
        view = self.view(task, spec, batch_size)
        runtime_key = (task.target_column, task.drop_columns)
        if runtime_key not in view["tasks"]:
            graph, payload, valid, _ = load_rt_archive(
                self.root / task.db, EMBEDDING_MODEL, D_TEXT, device=self.device, spec=spec,
                target_column=task.target_column, drops=list(task.drop_columns), parent_slots=PARENT_SLOTS,
                decoded_state=self._state(task.db))
            view["tasks"][runtime_key] = (graph, payload, valid)
        graph, payload, valid = view["tasks"][runtime_key]
        if "workspace" not in view:
            view["workspace"] = Workspace(graph, payload["directions"], batch_size, spec, payload["composition"])
        return graph, payload, valid, view["workspace"]


class GroupSampler:
    """Batches of one mixture group at one length: tasks in order, seed rows in archive order, wrapping."""

    def __init__(self, archives, tasks, *, spec, batch_size: int, seed: int, document: str, task_offset: int):
        self.archives, self.tasks, self.batch_size, self.seed = archives, tasks, batch_size, seed
        self.document, self.task_offset, self.length = document, task_offset, spec.seq_len_set[0]
        self.specs = [task_spec(spec, task) for task in tasks]
        self.cumulative = np.cumsum([math.ceil(task.num_nodes / batch_size) for task in tasks])
        self.bank = alias_bank(D_TEXT, self.length)

    def __len__(self) -> int:
        return int(self.cumulative[-1])

    def coordinate(self, index: int) -> tuple[int, int]:
        task = int(np.searchsorted(self.cumulative, index, side="right"))
        return task, index - (int(self.cumulative[task - 1]) if task else 0)

    def requests(self, task_index: int, epoch: int):
        """``(runtime, requests [n, 5], settings)`` of a task: one request per scheduled seed slot."""
        task, spec = self.tasks[task_index], self.specs[task_index]
        graph, payload, valid, workspace = self.archives.runtime(task, spec, self.batch_size)
        if payload.get("training_epoch") != epoch:
            eligible = valid[(valid >= task.node_offset) & (valid < task.node_offset + task.num_nodes)]
            roots = np.resize(eligible, task.num_nodes)
            q = torch.empty((len(roots), 5), dtype=torch.int64, device=graph.row_time.device)
            q[:, 0] = torch.as_tensor(roots, device=q.device)
            q[:, 1] = graph.row_time[q[:, 0]]
            q[:, 2] = task.target_column
            q[:, 3] = (q[:, 0] * 1000003) ^ ((self.seed ^ (epoch * 1000033)) & (2**63 - 1))
            q[:, 4] = request_flags(payload, q[:, 0])
            payload.update(training_epoch=epoch, training_requests=q, training_settings=context_settings(q, spec))
        return (graph, payload, workspace), payload["training_requests"], payload["training_settings"]

    def finalize(self, batch: dict, task_index: int, batch_index: int) -> dict:
        """Task horizon fields, opaque column names (seeded by ``batch_index``) and global task indices."""
        horizon = self.tasks[task_index].public_horizon
        known = horizon is not None and float(horizon) > 0
        count, device = batch["is_padding"].shape[0], batch["is_padding"].device
        batch["task_horizon"] = torch.full((count,), float(horizon) if known else 1., dtype=torch.float64, device=device)
        batch["task_horizon_valid"] = torch.full((count,), known, dtype=torch.bool, device=device)
        batch = neutralize_names(batch, seed=self.seed, batch_index=batch_index, task_count=len(self.tasks), bank=self.bank)
        batch["dataset_idxs"] = batch["dataset_idxs"] + self.task_offset
        return batch

    def run(self, indices: list[int], epoch: int) -> list[dict]:
        """Batches of consecutive indices; consecutive batches of one task share one native composition."""
        out = []
        for index in indices:
            task_index, local = self.coordinate(index)
            if out and out[-1][0] == task_index:
                out[-1][1].append((index, local))
            else:
                out.append((task_index, [(index, local)]))
        return [batch for task_index, members in out for batch in self._compose(task_index, members, epoch)]

    def _compose(self, task_index: int, members, epoch: int) -> list[dict]:
        task, spec = self.tasks[task_index], self.specs[task_index]
        (graph, payload, workspace), requests, settings = self.requests(task_index, epoch)
        if len(members) > 1:
            view = self.archives.view(task, spec, self.batch_size)
            size = len(members) * self.batch_size
            if size not in view.setdefault("run_workspaces", {}):
                view["run_workspaces"][size] = Workspace(graph, payload["directions"], size, spec, payload["composition"])
            workspace = view["run_workspaces"][size]
        rows = torch.cat([torch.arange(local * self.batch_size, (local + 1) * self.batch_size, device=requests.device)
                          for _, local in members]) % task.num_nodes
        q, chosen = requests.index_select(0, rows), settings.index_select(0, rows)
        world_salt = int(hashlib.sha256(task.db.encode()).hexdigest()[:12], 16)
        salt = torch.cat([batch_salts(world_salt, epoch, local, self.batch_size, q.device) for _, local in members])
        selected, roles, receipt = sample_rt_contexts(graph, q, spec, settings=chosen, directions=payload["directions"],
                                                      workspace=workspace, composition=payload["composition"])[3:6]
        packed = pack_context(self.document, graph, payload, q, selected, roles, receipt, task_index,
                              labels_visible=spec.label_visibility == "history_available", salt=salt)
        pieces = []
        for position, (index, _) in enumerate(members):
            part = slice(position * self.batch_size, (position + 1) * self.batch_size)
            piece = {k: v[part] if torch.is_tensor(v) and v.ndim and v.shape[0] == len(q) else v for k, v in packed.items()}
            pieces.append(self.finalize(piece, task_index, index))
        return pieces
