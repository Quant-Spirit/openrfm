"""Consumer episodes: which world and task an episode reads, its support size and length, and its members."""
import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .schedule import SystematicGroupSchedule, uniform_draw

SOURCE_SEED = 20261012
HASH_SCALE = 1_000_000
NO_TIME = -(2**31)
RT_STATIC_AVAILABLE, RT_UNKNOWN_AVAILABLE = -(2**62), 2**63 - 1
MAX_CLASSES = 10


@dataclass(frozen=True)
class Entity:
    row: int
    entity: int
    task: tuple  # (dataset index, table id, target column, semantic type)
    label: float
    cutoff: int
    available: int


@dataclass(frozen=True)
class Episode:
    task: tuple
    length: int
    support: tuple
    query: tuple
    classes: tuple | None
    identity: str


class EpisodeSources:
    """Balanced (group, task) draws per episode cursor: task type, then group weight, then world cycles."""

    def __init__(self, weights, groups, kinds, populations, config):
        self.weights, self.groups, self.kinds, self.populations, self.config = weights, groups, kinds, populations, config
        self.category = SystematicGroupSchedule(list(config["task_type_mix"].values()), window=2, seed=SOURCE_SEED + 1)
        self.strata, self.permutations = {}, {}

    def coordinate(self, cursor: int, k: int) -> tuple[int, int] | None:
        """``(group, task)`` of episode ``cursor`` with ``k`` supports, or None when no world supplies its stratum."""
        ci, local = self.category.coordinate(cursor)
        category = list(self.config["task_type_mix"])[ci]
        stratum = (category, k)
        if stratum not in self.strata:
            eligible, weights, indices = [], [], []
            for gi, worlds in enumerate(self.groups):
                subset = {world: {target: i for target, i in targets.items()
                                  if self.kinds[gi, i] == category
                                  and self.populations[gi, i] >= k + self.config["query_cohort"]}
                          for world, targets in worlds.items()}
                subset = {w: t for w, t in subset.items() if t}
                if subset:
                    eligible.append(subset)
                    weights.append(self.weights[gi])
                    indices.append(gi)
            schedule = SystematicGroupSchedule(weights, window=2, seed=SOURCE_SEED + ci + int(k)) if eligible else None
            self.strata[stratum] = (eligible, indices, schedule) if eligible else None
        if not self.strata[stratum]:
            return None
        eligible, indices, schedule = self.strata[stratum]
        position, occurrence = schedule.coordinate(local)
        group, worlds = indices[position], eligible[position]
        names = sorted(worlds)
        cycle, place = divmod(occurrence, len(names))
        key = stratum, group, cycle
        if key not in self.permutations:
            self.permutations[key] = np.random.default_rng([SOURCE_SEED, group, cycle]).permutation(len(names))
        world = names[int(self.permutations[key][place])]
        targets = sorted(worlds[world])
        phase = int.from_bytes(hashlib.blake2b(world.encode(), digest_size=8).digest(), "little")
        return group, worlds[world][targets[(cycle + phase) % len(targets)]]


def draw_geometry(config, cursor: int) -> tuple[int, int]:
    """``(support size K, context length L)`` of episode ``cursor``; K is drawn conditionally on L."""
    def draw(mix, salt):
        point, cumulative = uniform_draw(config["seed"], cursor, salt), 0.0
        for value, weight in sorted(mix.items()):
            cumulative += weight
            if point < cumulative:
                return int(value)
        return int(max(mix))
    length = draw(config["length_mix"], 11)
    return draw(config["support_mix_by_length"][length], 7), length


def role_hash(entity: int, task, seed: int) -> int:
    payload = b"".join(int(part).to_bytes(16, "little", signed=True) for part in (entity, *task, seed))
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def is_query(entity: int, task, seed: int, fraction: float) -> bool:
    return role_hash(entity, task, seed) % HASH_SCALE < int(round(fraction * HASH_SCALE))


def matured(support_cutoff: int, support_available: int, query_cutoff: int) -> bool:
    """Whether a support label is known strictly before the query (from its availability, else its cutoff)."""
    if support_cutoff == NO_TIME:
        return True
    if query_cutoff == NO_TIME:
        return False
    horizon = support_available not in (NO_TIME, RT_STATIC_AVAILABLE, RT_UNKNOWN_AVAILABLE) and support_available > support_cutoff
    return (support_available if horizon else support_cutoff) < query_cutoff


def select_episode(entities, *, length, k, config, seed, coupling) -> tuple[Episode | None, str | None]:
    """The latest ``query_cohort`` query-role entities and the first ``k`` matured support-role entities by hash."""
    if not entities:
        return None, "empty_world"
    task = entities[0].task
    seen, ranked = set(), []
    for e in entities:
        if (e.entity, e.cutoff) not in seen:
            seen.add((e.entity, e.cutoff))
            ranked.append(e)
    fraction = coupling["query_fraction"]
    queries = sorted((e for e in ranked if is_query(e.entity, task, seed, fraction)),
                     key=lambda e: (-e.cutoff, role_hash(e.entity, task, seed), e.entity))[:config["query_cohort"]]
    if len(queries) < coupling["q_min"]:
        return None, "too_few_query"
    cutoff = min(e.cutoff for e in queries)
    supports = sorted((e for e in ranked if not is_query(e.entity, task, seed, fraction) and matured(e.cutoff, e.available, cutoff)),
                      key=lambda e: (role_hash(e.entity, task, seed), e.entity, e.cutoff))
    if len(supports) < k:
        return None, "insufficient_mature_support"
    supports, classes = supports[:k], None
    if task[3] in (3, 4):
        classes = tuple(sorted({int(e.label) for e in supports}))
        if len(classes) < 2:
            return None, "single_class_support"
        if len(classes) > MAX_CLASSES:
            return None, "too_many_classes"
        queries = [e for e in queries if int(e.label) in classes]
    elif len({e.label for e in supports}) < 2:
        return None, "constant_support_target"
    if len(queries) < coupling["q_min"]:
        return None, "too_few_scored_query"
    return Episode(task, length, tuple(supports), tuple(queries), classes, ""), None


def episode_identity(world, task, horizon, program, length, k, episode) -> str:
    return hashlib.sha256(json.dumps(dict(world=world, task=task, horizon=horizon, program=program, length=length, k=k,
                                          support=[(e.entity, e.cutoff) for e in episode.support],
                                          query=[(e.entity, e.cutoff) for e in episode.query]),
                                     sort_keys=True).encode()).hexdigest()


def cursor_seed(seed: int, cursor: int) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}/{cursor}".encode(), digest_size=8).digest(), "little")
