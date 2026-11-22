"""The temporal history of a request: the visible rows of its foreign-key neighbourhood and their frozen cell
encodings. The history `world` is a row graph on the device: `ptr` [rows+1] cell offsets, `parents` [rows, F]
foreign-key parents (-1 none), `times` [rows], `context_rows` [rows] or None, and per cell `columns`, `types`,
`values`, `text_ids`, `visible`, with the column-name embeddings `names` and text embeddings `text`."""
import torch
import torch.nn.functional as F

from openrfm.archive.keys import KeyIndex

# Cell types in the archive's type codes, in the backbone's encoder names; type 4 is the discrete code.
VALUE_TYPES = ('number', 'text', 'datetime', 'boolean')
DISCRETE = 4


def lawful(world, keys, cutoffs):
    """`keys` [n, k] with every row whose timestamp is after its request's cutoff set to -1."""
    ok = world.times[keys.clamp_min(0)] <= cutoffs[:, None]
    return keys.masked_fill(~((keys >= 0) & ok), -1)


def dense_cells(world, ids, width):
    """[len(ids), width] cell addresses of rows `ids`, -1 past each row's cells; `width` is the widest row."""
    host = ids.to(world.ptr.device)
    begin, end = world.ptr[host].long(), world.ptr[host+1].long()
    slot = begin[:, None]+torch.arange(width, device=host.device)
    return torch.where(slot < end[:, None], slot, -1)


def context_children(world):
    """Children of each row under any foreign key in time order, restricted to `world.context_rows`."""
    parents = world.parents.long()
    live = parents >= 0
    if world.context_rows is not None:
        live &= world.context_rows[:, None]
    child = torch.arange(len(parents), device=parents.device)[:, None].expand_as(parents)[live]
    return KeyIndex(parents[live], world.times[child], child)


def newest_children(children, keys, cutoffs, width):
    """[n, k*width] the newest `width` visible children of each key in `keys` [n, k]; -1 where none."""
    k = keys.flatten()
    cuts = cutoffs[:, None].expand_as(keys).flatten()
    start, stop = children.bounds(k.clamp_min(0))
    start, stop = torch.where(k >= 0, start, 0), torch.where(k >= 0, stop, 0)
    end = children.search(k.clamp_min(0), cuts, bounds=(start, stop))
    slot = end[:, None]-1-torch.arange(width, device=k.device)
    found = children.rows[slot.clamp_min(0).clamp_max(max(0, len(children.rows)-1))]
    return found.masked_fill(slot < start[:, None], -1).view(len(keys), -1)


def reach(world, children, rows, cutoffs, width, hops):
    """[n, width] visible rows of each focal row's temporal neighbourhood, their reach class and presence.

    A row's reach class is 2*hop for a parent step and 2*hop+1 for a child step, at the first hop reaching it.
    """
    parents = world.parents.long()
    frontier = rows[:, None]
    found = torch.full_like(rows[:, None], -1)
    kind = torch.zeros_like(found)
    for h in range(hops):
        up = lawful(world, parents[frontier.clamp_min(0)].masked_fill(frontier[..., None] < 0, -1).flatten(1), cutoffs)
        down = newest_children(children, frontier, cutoffs, width)
        found = torch.cat((found, up, down), 1)
        kind = torch.cat((kind, torch.full_like(up, 2*h), torch.full_like(down, 2*h+1)), 1)
        # The next frontier: the rows this hop reached, parents before children, then newest first.
        frontier, _, _ = nearest(world, rows, found, kind, width, 2*hops, start=2*h)
    return nearest(world, rows, found, kind, width, 2*hops)


def nearest(world, rows, found, kind, width, classes, start=0):
    """[n, width] distinct rows of `found` besides the focal row at their smallest class >= `start`, newest first."""
    found = found.masked_fill(found == rows[:, None], -1)
    # Exact integer sort keys: row then class, so the first entry of each row has its smallest class.
    order = (found.clamp_min(0)*classes+kind).argsort(dim=1, stable=True)
    found, kind = found.gather(1, order), kind.gather(1, order)
    keep = found >= 0
    keep[:, 1:] &= found[:, 1:] != found[:, :-1]
    keep &= kind >= start
    big = torch.iinfo(torch.long).max
    newest = torch.where(keep, -world.times[found.clamp_min(0)], big).argsort(dim=1, stable=True)
    found, kind, keep = found.gather(1, newest), kind.gather(1, newest), keep.gather(1, newest)
    best = torch.where(keep, kind, big).argsort(dim=1, stable=True)[:, :width]
    found, kind, keep = found.gather(1, best), kind.gather(1, best), keep.gather(1, best)
    if found.shape[1] < width:
        pad = width-found.shape[1]
        found, kind, keep = F.pad(found, (0, pad), value=-1), F.pad(kind, (0, pad)), F.pad(keep, (0, pad))
    return found.masked_fill(~keep, -1), kind, keep


@torch.no_grad()
def encode_cells(rt, world, cells):
    """[M, d] frozen backbone encoding of cell addresses [M] (-1 absent); every encoder runs on every cell."""
    pick = lambda x: x[cells.clamp_min(0).to(x.device)].to(cells.device)
    columns, types, values = pick(world.columns).long(), pick(world.types).long(), pick(world.values).float()
    text_ids, visible = pick(world.text_ids).long(), pick(world.visible).bool()
    x = rt.norm_dict['col_name'](world.names[columns].float())
    for t, name in enumerate(VALUE_TYPES):
        raw = world.text[text_ids.clamp_min(0).to(world.text.device)].to(x.device).float() if t == 1 else values[:, None]
        encoded = rt.encode_cell_value(name, raw).float()
        x = x+torch.where(((types == t) & visible)[:, None], encoded, 0.)
        x = x+torch.where(((types == t) & ~visible)[:, None], rt.mask_embs[name].float(), 0.)
    table = rt.enc_dict['discrete']
    code = values.long()
    shown = (types == DISCRETE) & visible & (code >= 0) & (code < table.num_embeddings)
    encoded = rt.norm_dict['discrete'](table(code.clamp(0, table.num_embeddings-1))).float()
    x = x+torch.where(shown[:, None], encoded, 0.)
    x = x+torch.where(((types == DISCRETE) & ~shown)[:, None], rt.mask_embs['discrete'].float(), 0.)
    return x*(cells >= 0)[:, None]
