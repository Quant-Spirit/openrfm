"""Label programs: random aggregates over a request's lawful foreign-key reach, executed on the history world.

Program groups: 0 aggregate, 1 filter, 2 window, 3 ratio, 4 recency, 5 last-k, 6 shared key, 7 composition,
8 k-th newest value, 9 mean gap, 10 overdue, 11 last-k trend, 12 distinct other keys. Reducers: count, sum, mean,
variance, min, max."""
from dataclasses import dataclass

import numpy as np
import torch

# Groups whose columns need dated rows.
DATED_GROUPS = (2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
NO_TIME = -(2**31)


@dataclass
class LabelPrograms:
    column: torch.Tensor
    predicate_column: torch.Tensor
    threshold: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    last_k: torch.Tensor
    reducer: torch.Tensor
    group: torch.Tensor
    key_slot: torch.Tensor
    scoped: torch.Tensor


def reachable_rows(world, rows, cutoffs, exclude_focal=False):
    """[n, w] distinct rows each request reaches through one key (its own row or a parent) at its cutoff, with all
    history, in physical row order, and their presence."""
    keys = torch.cat((rows[:, None], world.parents[rows]), 1)
    pieces = []
    for relation in world.relations:
        index = relation.index
        local = torch.cat((keys, relation.owner[rows, None]), 1)
        start, _ = index.bounds(local)
        stop = index.search(local, cutoffs[:, None].expand_as(local).contiguous())
        width = int((stop-start).max())
        if not width:
            continue
        pos = start[..., None]+torch.arange(width, device=rows.device)
        live = (pos < stop[..., None]) & (local[..., None] >= 0)
        picked = index.rows[pos.clamp_max(len(index.rows)-1)].masked_fill(~live, world.nrows)
        pieces.append(picked.flatten(1))
    candidates = torch.cat(pieces, 1).sort(1).values
    valid = candidates < world.nrows
    valid[:, 1:] &= candidates[:, 1:] != candidates[:, :-1]
    if exclude_focal:
        valid &= candidates != rows[:, None]
    ordered = candidates.masked_fill(~valid, world.nrows).sort(1).values
    selected = ordered[:, :max(1, int(valid.sum(1).max()))]
    return selected.clamp_max(world.nrows-1), selected < world.nrows


def draw_live_programs(world, rows, cutoffs, n, seed, *, scoped_share, count_share, sample=64, groups=8):
    """`n` programs that can vary on these requests: columns with more than one reachable row per request,
    predicates in the column's table, thresholds at value quantiles and windows within the reachable ages."""
    device = rows.device
    rng = np.random.default_rng(seed)
    chosen = torch.as_tensor(rng.choice(len(rows), min(sample, len(rows)), replace=False), device=device)
    selected, live = reachable_rows(world, rows[chosen], cutoffs[chosen])
    reached = selected[live]
    weight = torch.zeros(world.nrows, device=device).index_add_(
        0, reached, torch.ones_like(reached, dtype=torch.float))/len(chosen)
    dated = world.times > NO_TIME
    nodes = world.nodes
    cells = torch.where(world.visible & (weight[nodes] > 0))[0]
    columns, slot = world.columns[cells].unique(return_inverse=True)

    def per_column(values, reduce='sum'):
        return values.new_zeros(len(columns)).scatter_reduce_(0, slot, values, reduce)
    reach = per_column(weight[nodes[cells]])
    dated_reach = per_column((weight*dated)[nodes[cells]])
    numeric = per_column(((world.types[cells] == 0) | (world.types[cells] == 3)).float(), 'amax') > 0
    first = world.columns[world.ptr[:-1].to(device)]
    tables = per_column(first[nodes[cells]].float(), 'amax').long()
    eligible = reach > 1 if bool((reach > 1).any()) else reach > 0
    reach, dated_reach, numeric, tables, eligible = [x.cpu().numpy() for x in (reach, dated_reach, numeric, tables, eligible)]
    ages = ((cutoffs[chosen][:, None]-world.times[selected]).double()/86400.).float()[live & dated[selected]]
    ages = ages[ages > 0]
    if len(ages):
        ages = ages[torch.randperm(len(ages), device=device)[:1 << 16]]
        low, high = torch.quantile(ages, torch.tensor([.02, .98], device=device)).tolist()
    else:
        low, high = 1., 365.
    low, high = max(low, 1/24), max(high, low*2, 1/12)
    pool = np.flatnonzero(eligible)
    dated_pool = pool[dated_reach[pool] > 1] if (dated_reach[pool] > 1).any() else pool
    group = np.arange(n) % groups
    column, predicate = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    for i, g in enumerate(group):
        candidates = dated_pool if g in DATED_GROUPS else pool
        table = rng.choice(np.unique(tables[candidates]))
        column[i] = rng.choice(candidates[tables[candidates] == table])
        numbers = pool[(tables[pool] == table) & numeric[pool]]
        predicate[i] = rng.choice(numbers) if len(numbers) else column[i]
        if g == 11:
            column[i] = predicate[i]
    reducer = rng.integers(0, 6, n)
    reducer[(~numeric[column]) | (rng.random(n) < count_share)] = 0
    reducer[group == 8] = 1
    threshold = np.zeros(n, dtype=np.float32)
    order = slot.argsort()
    values_by_column = world.values[cells[order]].split(slot.bincount(minlength=len(columns)).tolist())
    for c in np.unique(predicate):
        values = values_by_column[c].float()[:1 << 16]
        draw = predicate == c
        threshold[draw] = torch.quantile(values, torch.as_tensor(rng.uniform(.1, .9, int(draw.sum())),
                                                                 device=values.device, dtype=values.dtype)).cpu().numpy()
    upper = np.exp(rng.uniform(np.log(low), np.log(high), n))
    keys = torch.cat((world.parents[rows], rows[:, None]), 1).long()
    slots = torch.where((keys >= 0).float().mean(0) > .5)[0].cpu().numpy()
    if not len(slots):
        slots = np.array([keys.shape[1]-1])
    names = columns.cpu().numpy()

    def tensor(x, dtype):
        return torch.as_tensor(x, device=device, dtype=dtype)
    return LabelPrograms(tensor(names[column], torch.long), tensor(names[predicate], torch.long),
                         tensor(threshold, torch.float), tensor(upper*rng.uniform(0, .5, n), torch.float),
                         tensor(upper, torch.float), tensor(rng.integers(1, 17, n), torch.long),
                         tensor(reducer, torch.long), tensor(group, torch.long),
                         tensor(rng.choice(slots, n), torch.long), tensor(rng.random(n) < scoped_share, torch.bool))


def forward_programs(world, rows, cutoffs, n, seed, days, *, scoped_share, count_share):
    """Window programs over (cutoff, cutoff+H], H <= `days`, to execute at cutoff+`days`. Labels only: inputs are
    read at the cutoff."""
    programs = draw_live_programs(world, rows, cutoffs, n, seed, scoped_share=scoped_share, count_share=count_share)
    rng = np.random.default_rng([seed, 5])
    device = programs.lower.device
    horizon = torch.as_tensor(np.exp(rng.uniform(0, np.log(days), n)), device=device, dtype=programs.lower.dtype)
    programs.lower, programs.upper = days-horizon, torch.full_like(horizon, days)
    programs.group = torch.as_tensor(rng.choice((2, 3), n, p=(.75, .25)), device=device)
    return programs


@torch.no_grad()
def execute_world(world, rows, cutoffs, programs, *, chunk=32, exclude_focal=False):
    """[n, programs] program values per request; chunks only bound the padded history."""
    columns, inverse = torch.cat((programs.column, programs.predicate_column)).unique(return_inverse=True)
    cells = torch.where(world.visible & torch.isin(world.columns, columns))[0]
    slot = torch.searchsorted(columns, world.columns[cells])
    dense = world.values.new_zeros((len(columns), world.nrows)).index_put_(
        (slot, world.nodes[cells]), world.values[cells], accumulate=True)
    present = world.values.new_zeros((len(columns), world.nrows)).index_put_(
        (slot, world.nodes[cells]), torch.ones_like(world.values[cells]), accumulate=True)
    dense, present = (dense/present.clamp_min(1))[inverse], present[inverse]
    return torch.cat([execute_requests(world, rows[a:a+chunk], cutoffs[a:a+chunk], programs, dense, present,
                                       exclude_focal) for a in range(0, len(rows), chunk)])


def execute_requests(world, rows, cutoffs, programs, dense, present, exclude_focal=False):
    selected, live = reachable_rows(world, rows, cutoffs, exclude_focal)
    values = dense[:, selected].permute(1, 0, 2)
    present = (present[:, selected] > 0).permute(1, 0, 2) & live[:, None]
    n = len(programs.group)
    values, predicate = values[:, :n], values[:, n:]
    selected_parents = world.parents[selected][:, None]

    def incident(keys):
        return ((selected_parents == keys[..., None, None]) & (keys[..., None, None] >= 0)).any(-1)
    focal = torch.cat((world.parents[rows], rows[:, None]), 1)[:, programs.key_slot]
    base = present[:, :n] & (incident(focal) | ~programs.scoped[None, :, None])
    age = ((cutoffs[:, None]-world.times[selected]).double()/86400.).float()[:, None]
    dated = world.times[selected] > NO_TIME
    temporal = base & dated[:, None]
    filtered = base & present[:, n:] & (predicate > programs.threshold[None, :, None])
    window = temporal & (age >= programs.lower[None, :, None]) & (age < programs.upper[None, :, None])
    g = programs.group[None, :, None]
    mask = torch.where(g == 1, filtered, torch.where((g == 2) | (g == 3) | (g == 7), window, base))
    mask &= torch.where((g == 4) | (g == 5) | (g >= 8), temporal, True)
    rank = age.expand_as(mask).masked_fill(~mask, float('inf')).argsort(dim=-1, stable=True).argsort(-1)
    mask = torch.where(g == 5, mask & (rank < programs.last_k[None, :, None]), mask)
    mask = torch.where(g == 8, mask & (rank == programs.last_k[None, :, None]-1), mask)
    count = mask.sum(-1).float()
    total = values.masked_fill(~mask, 0).sum(-1)
    mean = total/count.clamp_min(1)
    variance = (values-mean[..., None]).square().masked_fill(~mask, 0).sum(-1)/count.clamp_min(1)
    minimum = values.masked_fill(~mask, float('inf')).amin(-1).masked_fill(count == 0, 0)
    maximum = values.masked_fill(~mask, -float('inf')).amax(-1).masked_fill(count == 0, 0)
    reductions = torch.stack((count, total, mean, variance, minimum, maximum), -1)
    target = reductions.gather(-1, programs.reducer[None, :, None].expand(len(rows), -1, 1)).squeeze(-1)
    ratio = count/base.sum(-1).clamp_min(1)
    recency = age.expand_as(mask).masked_fill(~mask, float('inf')).amin(-1).masked_fill(count == 0, 0)
    composition = values.masked_fill(~(filtered & window), 0).sum(-1)/count.clamp_min(1)+ratio*mean
    out = torch.where(g[..., 0] == 3, ratio, torch.where(g[..., 0] == 4, recency,
                      torch.where(g[..., 0] == 7, composition, target)))
    if not (programs.group > 8).any():
        return out
    oldest = age.expand_as(mask).masked_fill(~mask, -float('inf')).amax(-1).masked_fill(count == 0, 0)
    gap = (oldest-recency)/(count-1).clamp_min(1)
    recent = mask & (rank < programs.last_k[None, :, None])
    trend = values.masked_fill(~recent, 0).sum(-1)/recent.sum(-1).clamp_min(1)-mean
    out = torch.where(g[..., 0] == 9, gap, torch.where(g[..., 0] == 10, recency/(gap+1e-3)*(count > 1),
                      torch.where(g[..., 0] == 11, trend, out)))
    distinct = torch.where(programs.group == 12)[0]
    if len(distinct):
        slot = programs.last_k[distinct] % world.parents.shape[1]
        keys = world.parents[selected][:, :, slot].permute(0, 2, 1).masked_fill(~mask[:, distinct], -1).sort(-1).values
        fresh = keys >= 0
        fresh[..., 1:] &= keys[..., 1:] != keys[..., :-1]
        out[:, distinct] = fresh.sum(-1).float()
    return out
