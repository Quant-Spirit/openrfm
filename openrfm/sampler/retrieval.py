"""Native RT retrieval plus an independent CPU oracle."""
from dataclasses import dataclass
from collections import Counter, deque
import torch
from torch import Tensor
from .spec import RTRetrievalSpec

STATIC = -(2**62)
UNKNOWN = 2**63 - 1


@dataclass(frozen=True)
class RTGraph:
    edge_ptr: Tensor
    edge_dst: Tensor
    edge_time: Tensor
    row_table: Tensor
    row_time: Tensor
    row_task: Tensor
    cell_ptr: Tensor
    cell_column: Tensor
    cell_available: Tensor
    cell_label: Tensor
    cell_drop: Tensor

    def __post_init__(self):
        self.validate()

    def tensors(self):
        return list(vars(self).values())

    def to(self, device):
        return RTGraph(*(x.to(device) for x in self.tensors()))

    def validate(self):
        allowed = (torch.uint8, torch.uint16, torch.int32, torch.int64)
        if any(x.dtype not in allowed for x in self.tensors()):
            raise ValueError('unsupported RT index dtype')
        if any(x.dtype in (torch.uint8, torch.uint16) for x in self.tensors()):
            import copy
            scratch = copy.copy(self)
            for name, x in vars(self).items():
                object.__setattr__(scratch, name, x.int() if x.dtype in (torch.uint8, torch.uint16) else x)
            scratch._validate_signed()
        else:
            self._validate_signed()
        return self

    def _validate_signed(self):
        arrays = self.tensors()
        if any(x.dtype not in (torch.int32, torch.int64) or x.ndim != 1 or not x.is_contiguous()
               or x.device != arrays[0].device for x in arrays):
            raise ValueError('RT graph fields must be contiguous int64 vectors on one device')
        n, m = len(self.row_table), len(self.cell_column)
        if n < 1 or m < 1 or any(len(x) != n for x in (self.row_time, self.row_task)):
            raise ValueError('empty or unaligned RT graph')
        for ptr, size in ((self.edge_ptr, len(self.edge_dst)), (self.cell_ptr, m)):
            if len(ptr) != n+1 or ptr[0] != 0 or ptr[-1] != size or (ptr.diff() < 0).any():
                raise ValueError('invalid RT CSR pointers')
        if len(self.edge_dst) != len(self.edge_time) or ((self.edge_dst < 0) | (self.edge_dst >= n)).any():
            raise ValueError('invalid RT edge addresses')
        if any(len(x) != m for x in (self.cell_available, self.cell_label, self.cell_drop)):
            raise ValueError('unaligned RT cell arrays')
        if ((self.row_task < 0) | (self.row_task > 2)).any() or (self.cell_column < 0).any():
            raise ValueError('invalid RT table role or column id')
        if any(((x != 0) & (x != 1)).any() for x in (self.cell_label, self.cell_drop)):
            raise ValueError('RT label/drop flags must be Boolean integers')
        if len(self.edge_dst) > 1:
            boundary = torch.zeros(len(self.edge_dst), dtype=torch.bool, device=self.edge_dst.device)
            boundary[self.edge_ptr[:-1].clamp_max(len(boundary)-1)] = True
            if ((self.edge_time[1:] < self.edge_time[:-1]) & ~boundary[1:]).any():
                raise ValueError('RT edges must be sorted by timestamp within each row')
        rows = torch.repeat_interleave(torch.arange(n, device=self.cell_ptr.device), self.cell_ptr.diff())
        if rows.device.type == 'cpu':
            from openrfm.sampler.admission import distinct_pair_counts
            _, counts = distinct_pair_counts(rows.numpy(), self.cell_column.numpy())
            distinct_cells = int(counts.sum())
        else:
            distinct_cells = len(torch.unique(torch.stack((rows, self.cell_column), 1), dim=0))
        if distinct_cells != m:
            raise ValueError('duplicate (row,column) cells')
        return self


SETTINGS_WIDTH = 5
RETRIEVAL_SETTINGS_WIDTH = 4


def context_settings(requests: Tensor, spec: RTRetrievalSpec) -> Tensor:
    """Per-context deterministic choices, independent of batch order and padding."""
    key = requests[:, 3] % 2147483647
    result = []
    for i, choices in enumerate((spec.seq_len_set, spec.local_cell_budget_set,
                                 spec.bfs_width_set, spec.recency_set)):
        key = (key * 48271 + 104729 * (i+1)) % 2147483647
        values = torch.tensor(choices, dtype=torch.int64, device=requests.device)
        result.append(values[key % len(choices)])
    result.append(result[0])
    return torch.stack(result, 1)


def _assert(value: Tensor, message: str):
    if value.is_cuda:
        torch._assert_async(value, message)
    elif not bool(value):
        raise ValueError(message)


def sample_rt_contexts(graph: RTGraph, requests: Tensor, spec: RTRetrievalSpec,
                       *, settings: Tensor | None = None, directions=None, workspace=None, composition=None):
    """Return visits, ranked rows/counts, selected cell ids, roles, and receipts."""
    from .native import get_extension, _MODE
    if requests.dtype != torch.int64 or requests.ndim != 2 or requests.shape[1] != 5 or not requests.is_contiguous():
        raise ValueError('requests must be contiguous int64 [B,5]')
    if requests.device != graph.edge_ptr.device:
        raise ValueError('graph/request devices differ')
    settings = context_settings(requests, spec) if settings is None else settings
    if settings.dtype != torch.int64 or settings.shape != (len(requests), SETTINGS_WIDTH) or settings.device != requests.device or not settings.is_contiguous():
        raise ValueError(f'settings must be contiguous int64 [B,{SETTINGS_WIDTH}]')
    _assert(((requests[:, 0] >= 0) & (requests[:, 0] < len(graph.row_table))).all(), 'invalid seed row')
    _assert(((settings[:, :3] > 0).all() & (settings[:, 0] <= max(spec.seq_len_set)).all()
             & (settings[:, 4] > 0).all() & (settings[:, 4] <= max(spec.seq_len_set)).all()
             & ((settings[:, 3] == 0) | (settings[:, 3] == 1)).all()), 'invalid context settings')
    _assert(((requests[:, 4] == 0) | (requests[:, 4] == 1)).all(), 'invalid temporal policy')
    if spec.version in (2, 3):
        if directions is None:
            raise ValueError('v2 requires admitted f2p and p2f CSR')
        if _MODE == 'torch':
            from .v2 import oracle
            return oracle(graph,directions,requests,spec,settings,composition=composition)
        if _MODE != 'native' or workspace is None:
            raise RuntimeError('v2 requires explicit native backend and reusable workspace')
        result = workspace.sample(requests,settings,graph=graph,composition=composition)
        _assert((result[5][:,5] == 1).all(),'each v2 context requires exactly one focal target')
        return result
    if _MODE == 'torch':
        if requests.is_cuda:
            raise RuntimeError('RT oracle is CPU-only; native CUDA is required in production')
        result = retrieve_oracle(graph, requests, settings, spec)
        _assert((result[5][:, 5] == 1).all(), 'each RT context requires exactly one focal target')
        return result
    if _MODE != 'native':
        raise RuntimeError("RT requires explicit configure_backend('native'); no auto fallback")
    ext = get_extension(requests.device)
    if not hasattr(ext, 'rt_retrieve'):
        raise RuntimeError('native RT kernels unavailable; rebuild episodes on the remote host')
    result = ext.rt_retrieve(graph.tensors(), requests, settings[:, :RETRIEVAL_SETTINGS_WIDTH].contiguous(), spec.walk_count,
                          spec.walk_steps, spec.candidate_count, max(spec.seq_len_set),
                          spec.max_hops, int(spec.label_visibility == 'history_available'),
                          0)
    _assert((result[5][:, 5] == 1).all(), 'each RT context requires exactly one focal target')
    return result


def mix(x):
    mask = 2**64-1
    x = (x + 0x9e3779b97f4a7c15) & mask
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & mask
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & mask
    return x ^ (x >> 31)


def retrieve_oracle(graph, requests, settings, spec):
    """List/Counter/deque oracle: no native kernel helpers or extension dependency."""
    ptr, dst, time, table, rowtime, task, cp, col, avail, label, drop = [x.tolist() for x in graph.tensors()]
    B, capacity = len(requests), max(spec.seq_len_set)
    visits = torch.full((B, spec.walk_count, spec.walk_steps), -1, dtype=torch.long)
    candidates = torch.full((B, spec.candidate_count), -1, dtype=torch.long)
    scores = torch.zeros_like(candidates)
    cells = torch.full((B, capacity), -1, dtype=torch.long)
    roles = torch.zeros_like(cells)
    receipts = torch.zeros((B, 7), dtype=torch.long)
    for b, (root, cut, target, key, strict) in enumerate(requests.tolist()):
        L, ell, width, rho = settings[b, :RETRIEVAL_SETTINGS_WIDTH].tolist()
        def edges(row):
            return [j for j in range(ptr[row], ptr[row+1])
                    if (time[j] < cut if strict else time[j] <= cut)]
        def row_ok(row):
            return row == root or not task[row] or (task[row] == 1 and table[row] == table[root])
        for w in range(spec.walk_count):
            row = root
            for k in range(spec.walk_steps):
                chosen, n = -1, 0
                for j in edges(row):
                    if row_ok(dst[j]):
                        n += 1
                        if mix(key ^ mix(w) ^ mix(k+spec.walk_steps) ^ mix(j)) % n == 0:
                            chosen = dst[j]
                row = root if chosen < 0 else chosen
                if chosen >= 0 and row != root and table[row] == table[root]:
                    visits[b, w, k] = row
        counts = Counter(x for x in visits[b].flatten().tolist() if x >= 0)
        ranked = sorted(counts, key=lambda x: (-counts[x], -rowtime[x] if rho else 0, x))[:spec.candidate_count]
        for i, row in enumerate(ranked):
            candidates[b, i], scores[b, i] = row, counts[row]
        cursors, row_roles, donor_rows = {}, {}, set()
        output, output_rows = [], []
        for c in range(cp[root], cp[root+1]):
            if col[c] == target:
                output.append(c); output_rows.append(root)
                cursors[root], row_roles[root] = cp[root], 0
        reason = labels = 0
        budget, starts = ell, [root, *ranked]
        for pass_index in range(L+1):
            before = len(output)
            if before == L:
                break
            for launch, start in enumerate(starts):
                if len(output) == L:
                    break
                frontier, seen = deque([(start, 0, 0 if launch == 0 else 2)]), {start}
                spent, admitted = (len(output) if pass_index == 0 and launch == 0 else 0), 1
                while frontier and spent < budget and len(output) < L:
                    row, depth, role = frontier.popleft()
                    cursor = cursors.get(row, cp[row])
                    if row in cursors and role == 1 and row != root:
                        row_roles[row] = 1
                    while cursor < cp[row+1] and spent < budget and len(output) < L:
                        c = cursor; cursor += 1
                        if row == root and col[c] == target:
                            continue
                        is_label = bool(label[c]) or col[c] == target
                        legal = not drop[c] and (avail[c] <= cut if not is_label else
                            spec.label_visibility == 'history_available' and row != root and avail[c] < cut)
                        if not legal:
                            continue
                        if row not in cursors:
                            cursors[row], row_roles[row] = cp[row], role
                            if role == 2:
                                donor_rows.add(row)
                        output.append(c); output_rows.append(row); spent += 1; labels += is_label
                    if row in cursors:
                        cursors[row] = cursor
                    if spent == budget or len(output) == L:
                        break
                    if spec.max_hops and depth >= spec.max_hops:
                        continue
                    legal_edges = edges(row)
                    n, accepted = len(legal_edges), 0
                    offset = mix(key ^ mix(row)) % n if n else 0
                    for j in range(n):
                        child = dst[legal_edges[(offset+j) % n]]
                        if not row_ok(child) or child in seen:
                            continue
                        if admitted >= 2*L:
                            reason = 2; break
                        seen.add(child); frontier.append((child, depth+1, 0 if child == root else 1))
                        admitted += 1; accepted += 1
                        if accepted == width:
                            break
            if len(output) == before:
                break
        role_output = [row_roles[row] for row in output_rows]
        emitted = set(cursors)
        cells[b, :len(output)] = torch.tensor(output, dtype=torch.long)
        roles[b, :len(output)] = torch.tensor(role_output, dtype=torch.long)
        receipts[b] = torch.tensor([L, len(output), len(emitted), len(donor_rows), labels,
                                   sum(col[c] == target and c in range(cp[root], cp[root+1]) for c in output),
                                   0 if len(output) == L else reason or 1])
    return visits, candidates, scores, cells, roles, receipts
