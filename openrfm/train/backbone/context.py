"""Context programs: per-request random views of the composed context (column/table subsets, row order,
own-history and peer quotas, matured label donors), packed into training batches."""
import json
from functools import lru_cache

import torch

from openrfm.archive.format import take
from openrfm.sampler.retrieval import UNKNOWN, mix

from .pack import archive_provenance, composition_provenance, pack_cells

FIELDS = ("link", "bridge", "column_percent", "table_percent", "order", "peers", "own_labels", "own_events",
          "peer_labels", "core", "summary", "pack")
INT64_MAX = torch.iinfo(torch.int64).max


@lru_cache(maxsize=4)
def _choices(document: str) -> dict:
    return json.loads(document)["choices"]


@lru_cache(maxsize=16)
def _constants(document: str, device):
    choices = _choices(document)
    return dict(salts=torch.tensor([(j + 1) * 104729 for j in range(len(FIELDS))], device=device),
                sizes=torch.tensor([len(choices[name]) for name in FIELDS], device=device),
                numeric={name: torch.tensor(values, device=device) for name, values in choices.items()
                         if isinstance(values[0], int)})


def draw_programs(document: str, keys: torch.Tensor) -> torch.Tensor:
    """``[B, fields]`` choice indices, a hash of each request key; a 'default' pack zeroes every other field."""
    c = _constants(document, keys.device)
    program = mix(keys[:, None] ^ c["salts"]).remainder(c["sizes"])
    program[:, :-1] = program[:, :-1].masked_fill(_is(document, program, "pack", "default")[:, None], 0)
    return program


def _is(document, program, name, value):
    values = _choices(document)[name]
    if value not in values:
        return torch.zeros(len(program), dtype=torch.bool, device=program.device)
    return program[:, FIELDS.index(name)] == values.index(value)


def _numeric(document, program, name):
    return _constants(document, program.device)["numeric"][name][program[:, FIELDS.index(name)]]


def batch_salts(world_salt: int, epoch: int, local_batch: int, size: int, device) -> torch.Tensor:
    return torch.full((size,), world_salt ^ (local_batch + epoch * 1000033), dtype=torch.int64, device=device)


def _row_quota(rows, valid, score, quota):
    """Cells whose row ranks within ``quota`` rows by the row's best ``score`` among valid cells."""
    order = rows.clamp_min(0).argsort(dim=1, stable=True)
    physical = rows.clamp_min(0).gather(1, order)
    first = torch.cat((torch.ones_like(physical[:, :1], dtype=torch.bool), physical[:, 1:] != physical[:, :-1]), 1)
    group = torch.empty_like(order).scatter_(1, order, first.long().cumsum(1) - 1)
    costs = torch.full_like(rows, INT64_MAX)
    costs.scatter_reduce_(1, group, score.masked_fill(~valid, INT64_MAX), reduce="amin")
    order = costs.argsort(dim=1, stable=True)
    rank = torch.empty_like(order).scatter_(1, order, torch.arange(rows.shape[1], device=rows.device)[None].expand_as(order))
    return rank.gather(1, group) < quota[:, None]


def _parent_rows(payload, roots, nodes):
    keys = take(payload["parents"], roots).long()
    row_keys = take(payload["parents"], nodes.clamp_min(0)).long()
    shared = ((row_keys[..., None] == keys[:, None, None, :]) & (keys[:, None, None, :] >= 0)).any((-1, -2))
    witnesses = ((nodes[..., None] == keys[:, None, :]) & (keys[:, None, :] >= 0)).any(-1)
    return shared | witnesses, witnesses


def _siblings(graph, payload, requests, mode, length):
    """Same-time covariate cells of rows sharing a foreign key with the request row (parent-first programs)."""
    if "nh_siblings" not in payload:
        tables, pool = payload.get("same_time_table_ids"), None
        if tables is not None and tables.numel():
            nodes = payload["nodes"].long()
            allowed = torch.isin(take(graph.row_table, nodes).long(), tables)
            allowed &= ~graph.cell_label.bool() & ~graph.cell_drop.bool() & ~payload["declared_drop"].bool()
            cells = allowed.nonzero().flatten()
            pool = (cells, nodes[cells]) if len(cells) else None
        payload["nh_siblings"] = pool
    if payload["nh_siblings"] is None:
        return None
    cells, rows = payload["nh_siblings"]
    shared, _ = _parent_rows(payload, requests[:, 0], rows[None].expand(len(requests), -1))
    legal = shared & (take(graph.row_time, rows).long()[None] == requests[:, 1:2]) & mode[:, None]
    chosen = cells[None].expand(len(requests), -1).masked_fill(~legal, INT64_MAX).sort(dim=1, stable=True).values[:, :length]
    return chosen.masked_fill(chosen == INT64_MAX, -1)


def _covariates(payload, graph):
    """Per-row mean of its non-label numeric cells, and whether it has any (the near/far peer distance)."""
    if "numeric_covariates" not in payload:
        n, nodes = len(graph.row_table), payload["nodes"].long()
        numeric = (payload["cell_types"].long() == 0) & ~graph.cell_label.bool() & ~graph.cell_drop.bool()
        numeric &= ~payload["declared_drop"].bool() & (take(graph.row_task, nodes) == 0)
        histories = payload.get("history_table_ids", torch.empty(0, device=nodes.device, dtype=torch.long))
        numeric &= ~torch.isin(take(graph.row_table, nodes).long(), histories)
        values = payload["cell_numbers"].float()
        numeric &= torch.isfinite(values)
        total = torch.zeros(n, device=nodes.device).scatter_add_(0, nodes, values.masked_fill(~numeric, 0))
        count = torch.zeros(n, device=nodes.device).scatter_add_(0, nodes, numeric.float())
        payload["numeric_covariates"] = (total / count.clamp_min(1), count > 0)
    return payload["numeric_covariates"]


def _donors(graph, payload, m):
    """Task rows that can lend their label (and date) cells, with their entity's parent-row cells."""
    if "nh_donors" in payload:
        return payload["nh_donors"]
    device = m.target.device
    pool = (m.target.long() >= 0).nonzero().flatten()
    label_cells = m.target.long()[pool]
    available = take(graph.cell_available, label_cells).long()
    names = [v for k, v in payload["table_names"].items() if k.endswith(":Train") or k.endswith(":Db")]
    tables_ok = (available != UNKNOWN) & torch.isin(take(graph.row_table, pool).long(), torch.tensor(names, device=device))
    owner = m.entity.long()[pool]
    ptr = graph.cell_ptr.long()
    width = int(ptr.diff().max())
    ix = ptr[pool, None] + torch.arange(width, device=device)[None]
    date = (ix < ptr[pool + 1, None]) & (take(payload["cell_types"], ix.clamp_max(len(graph.cell_column) - 1)).long() == 2)
    date_id = ix.masked_fill(~date, -1).amax(1)
    parents = owner[owner >= 0].unique()
    pix = ptr[parents, None] + torch.arange(width, device=device)[None]
    pcells = pix.clamp_max(len(graph.cell_column) - 1).flatten()
    pallowed = ((pix < ptr[parents + 1, None]).flatten() & ~take(graph.cell_label, pcells).bool()
                & ~take(graph.cell_drop, pcells).bool() & ~take(payload["declared_drop"], pcells).bool())
    payload["nh_donors"] = dict(
        pool=pool, available=available, tables_ok=tables_ok, owner=owner,
        owner_slot=torch.where(owner >= 0, torch.searchsorted(parents, owner.clamp_min(0)), len(parents)),
        label_ix=torch.stack((label_cells, date_id), 1).flatten(), parents=parents, width=width, pcells=pcells,
        pcols=take(graph.cell_column, pcells).long(), pallowed=pallowed,
        ptables=take(graph.row_table, parents).long().repeat_interleave(width))
    return payload["nh_donors"]


def _donor_cells(graph, payload, requests, program, document, entity, parent_first, projection, cp, tp, B):
    """Matured task-row label/date cells (own entity first, then peers) and the kept peers' parent-row cells."""
    m, roots, cuts, device = payload["composition"], requests[:, 0], requests[:, 1], requests.device
    donors = _donors(graph, payload, m)
    pool = donors["pool"]
    legal = ((graph.row_time[pool].long()[None] < cuts[:, None]) & (pool[None] != roots[:, None])
             & (donors["available"][None] < cuts[:, None]) & donors["tables_ok"][None])
    if payload.get("explicit_publication_history", False):
        # Published history is the label channel of these worlds; the case table does not lend its labels too.
        legal &= ~parent_first[:, None]
    own_label = (donors["owner"][None] == entity[:, None]) & (entity[:, None] >= 0)
    peer = ~own_label & (donors["owner"][None] >= 0)
    cov, has = _covariates(payload, graph)
    safe_entity, safe_peer = entity.clamp_min(0), donors["owner"].clamp_min(0)
    distance = (cov[safe_entity, None] - cov[safe_peer][None]).abs()
    has_cov = has[safe_entity, None] & has[safe_peer][None]
    donor_score = mix(pool[None] ^ requests[:, 3:4]).remainder(2**30).double()
    donor_score = torch.where(_is(document, program, "peers", "near")[:, None] & has_cov, distance.double(), donor_score)
    donor_score = torch.where(_is(document, program, "peers", "far")[:, None] & has_cov, -distance.double(), donor_score)
    peer_order = donor_score.masked_fill(~legal | ~peer, float("inf")).argsort(dim=1, stable=True)
    arange = torch.arange(len(pool), device=device)[None]
    peer_rank = torch.empty_like(peer_order).scatter_(1, peer_order, arange.expand_as(peer_order))
    label_score = (cuts[:, None] - graph.row_time[pool].long()[None]).clamp_min(0)
    label_order = label_score.masked_fill(~legal | ~own_label, 2**62).argsort(dim=1, stable=True)
    label_rank = torch.empty_like(label_order).scatter_(1, label_order, arange.expand_as(label_order))
    keep_labels = legal & ((own_label & (label_rank < _numeric(document, program, "own_labels")[:, None]))
                           | (peer & (peer_rank < _numeric(document, program, "peer_labels")[:, None])
                              & ~_is(document, program, "peers", "absent")[:, None]))
    label_ix = donors["label_ix"]
    label_valid = keep_labels.repeat_interleave(2, 1) & (label_ix[None] >= 0)
    # Parent-row cells of the kept peer donors.
    membership = torch.zeros((B, len(donors["parents"]) + 1), dtype=torch.int32, device=device)
    membership.scatter_reduce_(1, donors["owner_slot"][None].expand(B, -1), (keep_labels & peer).int(), reduce="amax")
    pkeep = membership[:, :-1].bool().repeat_interleave(donors["width"], 1) & donors["pallowed"][None]
    pkeep &= mix(donors["pcols"][None] ^ projection).remainder(100) < cp[:, None]
    pkeep &= mix(donors["ptables"][None] ^ (projection + 4099)).remainder(100) < tp[:, None]
    cells = torch.cat((label_ix[None].expand(B, -1), donors["pcells"][None].expand(B, -1)), 1)
    roles = torch.cat((torch.where(own_label, 1, 2).repeat_interleave(2, 1), torch.full_like(pkeep, 2, dtype=torch.long)), 1)
    return cells, torch.cat((label_valid, pkeep), 1), roles, torch.cat((own_label.repeat_interleave(2, 1), torch.zeros_like(pkeep)), 1)


def _overlay(document, graph, payload, requests, selected, roles, program, schema_key, labels_visible):
    """Re-select the composed cells under each request's program; returns cell ids and roles."""
    B, L = selected.shape
    device, m = selected.device, payload["composition"]
    parent_first = _is(document, program, "order", "parent_first")
    siblings = _siblings(graph, payload, requests, parent_first, L)
    if siblings is not None:
        selected = torch.cat((selected, siblings), 1)
        roles = torch.cat((roles, torch.ones_like(siblings)), 1)
    roots, cuts = requests[:, 0], requests[:, 1]
    nodes = take(payload["nodes"], selected.clamp_min(0)).long()
    cols = take(graph.cell_column, selected.clamp_min(0)).long()
    tables, times = take(graph.row_table, nodes).long(), take(graph.row_time, nodes).long()
    entity = take(m.entity, roots).long()
    root_cell = (nodes == roots[:, None]) & take(graph.cell_label, selected.clamp_min(0)).bool() & (selected >= 0)
    root_rows = nodes == roots[:, None]
    # The focal entity row is the case itself, not an intermediate bridge.
    bridges = (nodes == entity[:, None]) & ~root_rows
    own = (((take(m.entity, nodes).long() == entity[:, None]) & (entity[:, None] >= 0)) | bridges) & ~root_rows
    shared, witnesses = _parent_rows(payload, roots, nodes)
    own |= parent_first[:, None] & shared & ~root_rows
    bridges |= parent_first[:, None] & witnesses & ~root_rows
    # Context task rows leave here; legal labels come back once, through the matured donor channel below.
    task_rows = take(m.target, nodes).long() >= 0
    valid = (selected >= 0) & (~task_rows | root_rows)
    cp, tp = _numeric(document, program, "column_percent"), _numeric(document, program, "table_percent")
    projection = torch.full_like(requests[:, 3:4], schema_key) if schema_key is not None else requests[:, 3:4]
    valid &= (mix(cols ^ projection).remainder(100) < cp[:, None]) | root_cell
    valid &= (mix(tables ^ (projection + 4099)).remainder(100) < tp[:, None]) | root_rows | bridges
    valid &= ~bridges | ~_is(document, program, "bridge", "empty")[:, None] | root_cell
    history_tables = payload.get("history_table_ids", torch.empty(0, device=device, dtype=torch.long))
    history = torch.isin(tables, history_tables) & ~task_rows
    valid &= ~history | (times < cuts[:, None])
    event = own & ~bridges & ~task_rows & ~history
    recent = (cuts[:, None] - times).clamp(0, 2**31 - 1) * (len(graph.row_table) + 1) + nodes
    valid &= ~event | _row_quota(nodes, valid & event, recent, _numeric(document, program, "own_events"))
    own_history = own & history & parent_first[:, None]
    valid &= ~own_history | _row_quota(nodes, valid & own_history, recent, _numeric(document, program, "own_labels"))
    peer_history = history & ~own & parent_first[:, None]
    valid &= ~peer_history | (~_is(document, program, "peers", "absent")[:, None] & _row_quota(
        nodes, valid & peer_history, mix(nodes ^ requests[:, 3:4]).remainder(2**30), _numeric(document, program, "peer_labels")))
    candidate_roles = torch.where(own, 1, roles)
    if labels_visible:
        cells, cell_valid, cell_roles, cell_own = _donor_cells(graph, payload, requests, program, document, entity,
                                                               parent_first, projection, cp, tp, B)
        selected = torch.cat((selected, cells), 1)
        valid = torch.cat((valid, cell_valid), 1)
        candidate_roles = torch.cat((candidate_roles, cell_roles), 1)
        root_cell = torch.cat((root_cell, torch.zeros_like(cell_valid)), 1)
        own = torch.cat((own, cell_own), 1)
        root_rows = torch.cat((root_rows, torch.zeros_like(cell_valid)), 1)
    # Each cell once.
    ids = selected.masked_fill(~valid, INT64_MAX)
    ixorder = ids.argsort(dim=1, stable=True)
    sorted_ids = ids.gather(1, ixorder)
    unique = torch.cat((torch.ones((B, 1), dtype=torch.bool, device=device), sorted_ids[:, 1:] != sorted_ids[:, :-1]), 1)
    valid &= torch.zeros_like(valid).scatter_(1, ixorder, unique)
    ix = selected.clamp_min(0)
    nodes = take(payload["nodes"], ix).long()
    columns = payload.get("summary_columns", torch.empty(0, device=device, dtype=torch.long))
    summary = torch.isin(take(graph.cell_column, ix).long(), columns)
    valid &= ~(_is(document, program, "summary", "raw")[:, None] & summary & ~root_cell)
    # Keep order: focal cell, focal row, protected own core, then own rows (own-first) or a random order.
    ages = (cuts[:, None] - take(graph.row_time, nodes).long()).clamp(0, 2**31 - 1)
    own_first = (_is(document, program, "order", "own_first") | parent_first)[:, None]
    score = torch.where(own_first, ages, mix(nodes ^ requests[:, 3:4]).remainder(2**30)).double()
    score -= (own_first & own).double() * 2**39
    _, witnesses = _parent_rows(payload, roots, nodes)
    score -= (parent_first[:, None] & witnesses).double() * 2**38
    history = torch.isin(take(graph.row_table, nodes).long(), history_tables)
    score -= (parent_first[:, None] & own & history).double() * 2**38
    core = _numeric(document, program, "core").clamp_max(L - 1)
    own_order = score.masked_fill(~own | ~valid, float("inf")).argsort(dim=1, stable=True)
    own_rank = torch.empty_like(own_order).scatter_(
        1, own_order, torch.arange(selected.shape[1], device=device)[None].expand_as(own_order))
    score -= (own & (own_rank < core[:, None]) & valid).double() * 2**40
    score -= root_rows.double() * 2**41
    score -= root_cell.double() * 2**42
    order = score.masked_fill(~valid, float("inf")).argsort(dim=1, stable=True)[:, :L]
    selected = selected.gather(1, order).masked_fill(~valid.gather(1, order), -1)
    return selected, candidate_roles.gather(1, order).masked_fill(selected < 0, 0)


def pack_context(document, graph, payload, requests, selected, roles, receipt, dataset_id, *, labels_visible,
                 salt=None, program=None, schema_key=None) -> dict:
    """The training batch of composed contexts under their programs (drawn from ``requests[:, 3] ^ salt``)."""
    # A 'default' program packs the composer's own cells and provenance; an 'overlay' program packs its re-selected
    # cells, adds the direct entity link as an extra foreign key and recomputes provenance from the archive.
    if program is None:
        program = draw_programs(document, requests[:, 3] ^ salt)
    native = _is(document, program, "pack", "default")
    native_batch = None
    if native.any():
        native_batch = pack_cells(graph, payload, requests, selected, roles, dataset_id,
                                  provenance=composition_provenance(receipt, selected))
    if native.all():
        batch = native_batch
    else:
        cells, cell_roles = _overlay(document, graph, payload, requests, selected, roles, program, schema_key,
                                     labels_visible)
        batch = pack_cells(graph, payload, requests, cells, cell_roles, dataset_id)
        m, roots = payload["composition"], requests[:, 0]
        entity = m.entity.long()[roots]
        node = batch["node_idxs"].long()
        own = ((m.entity.long()[node.clamp_min(0)] == entity[:, None]) & (entity[:, None] >= 0)
               & (node != roots[:, None]) & ~batch["is_padding"])
        shared, _ = _parent_rows(payload, roots, node)
        own |= _is(document, program, "order", "parent_first")[:, None] & shared & (node != roots[:, None]) & ~batch["is_padding"]
        direct = torch.where(_is(document, program, "link", "direct")[:, None] & own, roots[:, None], -1)
        batch["f2p_nbr_idxs"] = torch.cat((batch["f2p_nbr_idxs"], direct[..., None]), -1)
        batch = archive_provenance(batch, payload["directions"], graph.row_table, native=True)
        if native_batch is not None:
            for key, value in native_batch.items():
                if not torch.is_tensor(value) or not value.ndim or value.shape[0] != len(requests):
                    continue
                if key == "f2p_nbr_idxs":
                    value = torch.nn.functional.pad(value, (0, batch[key].shape[-1] - value.shape[-1]), value=-1)
                mask = native.reshape((len(native),) + (1,) * (value.ndim - 1))
                batch[key] = torch.where(mask, value, batch[key])
    batch["context_prior_program"] = program
    if schema_key is not None:
        batch["context_prior_schema_key"] = torch.full((len(program),), schema_key, dtype=torch.long, device=program.device)
    return batch
