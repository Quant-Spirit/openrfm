"""Deterministic schedules: which mixture group, which context length and which task batch a micro-batch draws."""
import bisect
import hashlib
import itertools
import random


class SystematicGroupSchedule:
    """Weighted systematic sampling in shuffled windows: index -> (group, occurrence so far)."""
    # Each window draws one phase and visits its even, then odd, slots in seeded orders; slot s falls in the group
    # whose CDF interval holds (phase + s / window) mod 1.

    def __init__(self, weights, *, window: int, seed: int):
        total = sum(weights)
        self.cdf = list(itertools.accumulate(w / total for w in weights))
        self.cdf[-1] = 1.0
        self.window, self.seed = window, seed
        self.coordinates, self.served = [], [0] * len(weights)

    def coordinate(self, index: int) -> tuple[int, int]:
        while len(self.coordinates) <= index:
            rng = random.Random(self.seed + len(self.coordinates) // self.window * 1_000_003)
            phase = rng.random()
            even, odd = list(range(0, self.window, 2)), list(range(1, self.window, 2))
            rng.shuffle(even)
            rng.shuffle(odd)
            for slot in even + odd:
                group = bisect.bisect_right(self.cdf, (phase + slot / self.window) % 1.0)
                self.coordinates.append((group, self.served[group]))
                self.served[group] += 1
        return self.coordinates[index]


class LengthSchedule:
    """Seeded i.i.d. draws of a context length per batch of one group, and the occurrence index of each draw."""

    SIZE = 262144

    def __init__(self, weights, *, seed: int):
        self.schedule = random.Random(seed).choices(range(len(weights)), weights=weights, k=self.SIZE)
        self.positions = [[i for i, value in enumerate(self.schedule) if value == j] for j in range(len(weights))]

    def coordinate(self, index: int) -> tuple[int, int]:
        cycles, remainder = divmod(index, self.SIZE)
        selected = self.schedule[remainder]
        return selected, cycles * len(self.positions[selected]) + bisect.bisect_left(self.positions[selected], remainder)


def uniform_draw(seed: int, step: int, draw_index: int) -> float:
    """A uniform number in [0, 1) that depends only on its three integer arguments."""
    payload = b"\0".join(str(int(value)).encode("ascii") for value in (seed, step, draw_index))
    digest = hashlib.blake2b(payload, digest_size=8, person=b"support-mix-v1").digest()
    return int.from_bytes(digest, "little") / float(1 << 64)
