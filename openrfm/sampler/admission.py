"""Vectorized CPU admission reductions; no Python objects per cell or edge."""
import numpy as np


def distinct_pair_counts(first, second):
    """Sorted first keys and number of distinct second keys for each."""
    first, second = np.asarray(first), np.asarray(second)
    if not len(first):
        return first.copy(), np.empty(0, dtype=np.int64)
    if min(first.min(), second.min()) >= 0 and max(first.max(), second.max()) < 2**31:
        keys = (first.astype(np.int64) << 32) | second.astype(np.int64)
        keys.sort()
        keys = keys[np.r_[True, keys[1:] != keys[:-1]]]
        a = (keys >> 32).astype(first.dtype)
    else:
        order = np.lexsort((second, first))
        a, b = first[order], second[order]
        distinct = np.empty(len(a), dtype=np.bool_)
        distinct[0] = True
        distinct[1:] = (a[1:] != a[:-1]) | (b[1:] != b[:-1])
        a = a[distinct]
    starts = np.r_[0, np.flatnonzero(a[1:] != a[:-1]) + 1]
    return a[starts], np.diff(np.r_[starts, len(a)])
