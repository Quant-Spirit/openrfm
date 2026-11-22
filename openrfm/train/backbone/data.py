"""The micro-batch stream: mixture group, then context length, then the group's task batch, per batch index."""
import json
from dataclasses import replace
from pathlib import Path

import torch

from openrfm.sampler.spec import RTRetrievalSpec

from .sampler import Archives, GroupSampler
from .schedule import LengthSchedule, SystematicGroupSchedule
from .tasks import corpus_groups


def retrieval_spec(config) -> RTRetrievalSpec:
    """The sampler policy of the recipe, with the composer settings the backbone requires (cell provenance)."""
    return replace(RTRetrievalSpec.parse(config["sampler"]), _model_requires_provenance=True)


def length_batch_sizes(config) -> dict[int, int]:
    """Micro-batch sequences per length: the cells of ``batch_size`` contexts of ``context_length`` cells."""
    cells = config["context_length"] * config["batch_size"]
    return {int(length): max(1, round(cells / int(length))) for length in sorted(config["length_mix"])}


class BatchSource:
    """``batch(index)``: the training micro-batch of a global micro-batch index (deterministic, resumable)."""

    def __init__(self, config, corpus: Path, device):
        archives_root = corpus / config["archives"]
        self.groups, weights = corpus_groups(corpus, archives_root, config["families"])
        self.archives = Archives(archives_root, device, config["resident_views"])
        self.weights = [w / sum(weights) for w in weights]
        self.schedule = SystematicGroupSchedule(weights, window=config["group_window"], seed=config["seed"])
        spec, sizes = retrieval_spec(config), length_batch_sizes(config)
        lengths = sorted(sizes)
        document = json.dumps(config["context_prior"], sort_keys=True, separators=(",", ":"))
        self.samplers, self.length_schedules, offset = [], [], 0
        for g, tasks in enumerate(self.groups):
            seed = config["seed"] + g * 100003
            self.samplers.append({length: GroupSampler(
                self.archives, tasks, spec=replace(spec, seq_len_set=(length,), protected_core_len=min(spec.protected_core_len, length)),
                batch_size=sizes[length], seed=seed, document=document, task_offset=offset) for length in lengths})
            self.length_schedules.append((lengths, LengthSchedule([float(config["length_mix"][l]) for l in lengths], seed=seed)))
            offset += len(tasks)
        self.group_lengths = [max(LengthSchedule.SIZE, sum(len(s) for s in samplers.values())) for samplers in self.samplers]
        self.size = max(config["group_window"], sum(self.group_lengths))
        self.regression_tasks = torch.tensor([t.regression for tasks in self.groups for t in tasks], device=device)

    def window(self, start: int, count: int, epoch: int) -> list[dict]:
        """The micro-batches ``start .. start + count - 1`` (runs of one group and length are composed together)."""
        runs = []
        for index in range(start, start + count):
            group, occurrence = self.schedule.coordinate(index)
            lengths, schedule = self.length_schedules[group]
            selected, local = schedule.coordinate(occurrence % self.group_lengths[group])
            sampler = self.samplers[group][lengths[selected]]
            if runs and runs[-1][0] is sampler:
                runs[-1][1].append(local % len(sampler))
            else:
                runs.append((sampler, [local % len(sampler)]))
        return [batch for sampler, locals_ in runs for batch in sampler.run(locals_, epoch)]
