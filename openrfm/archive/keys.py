"""A sorted (key, time, row) index with per-key time searches on the device."""
import torch
import torch.nn.functional as F


class KeyIndex:
    """Entries sorted by key, then time (stable). Every search stays inside its key's span."""

    def __init__(self, keys, times, rows):
        order = times.argsort(stable=True)
        order = order[keys[order].argsort(stable=True)]
        self.rows, self.times = rows[order], times[order]
        self.keys, counts = keys[order].unique_consecutive(return_counts=True)
        self._set_counts(counts)

    def _set_counts(self, counts):
        self.offsets = F.pad(counts.cumsum(0), (1, 0))
        self.max_rows = int(counts.max()) if len(counts) else 0

    def subset(self, keep):
        """The entries whose row is set in the boolean `keep`, in the same order."""
        live = keep[self.rows]
        index = KeyIndex.__new__(KeyIndex)
        index.rows, index.times = self.rows[live], self.times[live]
        index.keys, counts = self.keys.repeat_interleave(self.offsets.diff())[live].unique_consecutive(return_counts=True)
        index._set_counts(counts)
        return index

    def bounds(self, keys):
        """[start, stop) entry span of each key; empty for a key not in the index."""
        if not len(self.keys):
            return torch.zeros_like(keys), torch.zeros_like(keys)
        p = torch.searchsorted(self.keys, keys).clamp_max(len(self.keys)-1)
        start = self.offsets[p]
        return start, torch.where(self.keys[p] == keys, self.offsets[p+1], start)

    def search(self, keys, times, *, inclusive=True, bounds=None):
        """First entry of each key's span with time > `times` (>= when not `inclusive`)."""
        lo, hi = self.bounds(keys) if bounds is None else bounds
        # A fixed number of bisection steps, so no step waits on the host.
        for _ in range((self.max_rows+1).bit_length()):
            mid = (lo+hi)//2
            t = self.times[mid.clamp_max(max(0, len(self.times)-1))]
            before = t <= times if inclusive else t < times
            open_ = lo < hi
            hi = torch.where(open_ & ~before, mid, hi)
            lo = torch.where(open_ & before, mid+1, lo)
        return lo

    def locate(self, keys, times, rows, *, bounds):
        """Entry of each (key, time, row) already in the index; equal times keep row order."""
        lo = self.search(keys, times, inclusive=False, bounds=bounds)
        hi = self.search(keys, times, inclusive=True, bounds=bounds)
        for _ in range((self.max_rows+1).bit_length()):
            mid = (lo+hi)//2
            before = self.rows[mid.clamp_max(max(0, len(self.rows)-1))] < rows
            open_ = lo < hi
            hi = torch.where(open_ & ~before, mid, hi)
            lo = torch.where(open_ & before, mid+1, lo)
        return lo
