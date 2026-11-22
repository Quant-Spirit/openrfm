"""Independent path-stream oracle of openrfm/sampler/kernels/rt_composition.h."""
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from heapq import nsmallest
from math import isqrt

import torch
import numpy as np

from openrfm.sampler.retrieval import STATIC, context_settings, mix
from openrfm.sampler.spec import MAX_SEQ_LEN

EXPLORATION_CAP = 32768
OWN_HISTORY_ROWS = 128
SEED_POOL_FACTOR, SEED_LOCAL_SCAN, SEED_CLASS_CAP, SEED_LOCAL_TABLES = 4, 64, 4095, 7
ROUTE_CHILDREN, ROUTE_TASKS, ROUTE_SCAN = 256, 4, 64
PROGRAM_PATHS = 16
STREAM_FIELDS = ('signature', 'parent', 'direction', 'table', 'eligible',
                 'discovered', 'selected', 'time', 'cap', 'budget', 'charged', 'weight',
                 'same_entity')
SUMMARY_FIELDS = ('path_stream_count', 'path_eligible', 'path_discovered',
                  'path_selected', 'path_time_drops', 'path_cap_drops',
                  'packing_slack', 'path_exhausted', 'schema_stream_count',
                  'path_streams_pruned', 'path_capacity', 'schema_count_saturated',
                  'seed_ego_rows', 'seed_ego_cap_drops')
STREAM_OFFSET = 22 + len(SUMMARY_FIELDS)
PATH_FRONTIER = 22 + SUMMARY_FIELDS.index('path_capacity')
PATH_CAP_DROPS, PATH_STREAMS_PRUNED = 27, 31
SIGN_MASK = (1 << 63) - 1
SCHEMA_HEADER = 6
SCHEMA_WIDTH = 6
WEIGHT_ONE = 1 << 20


def schema_record(stream, spec):
    """One flat SCHEMA_WIDTH record; field 5 is the demand bound, or the weight scale under 'selectivity'."""
    selective = spec.stream_weight_policy == 'selectivity'
    return (stream['parent'], stream['direction'], stream['table'], stream['id'],
            stream['signature'], stream['weight_scale'] if selective else stream['demand'])


def validate_weight_scales(flat, spec):
    """Fail closed unless every record's field 5 is a scale in [1, WEIGHT_ONE]."""
    if spec.stream_weight_policy != 'selectivity':
        return
    flat = tuple(int(x) for x in flat)
    roots = flat[1] // SCHEMA_HEADER if len(flat) >= SCHEMA_HEADER else 0
    for root in range(roots):
        header = flat[SCHEMA_HEADER * root:SCHEMA_HEADER * (root + 1)]
        offset, size = header[1], header[2]
        if offset < 0 or size < 0 or offset + SCHEMA_WIDTH * size > len(flat):
            raise ValueError('schema header addresses records outside the flat schema')
        for record in range(size):
            scale = flat[offset + SCHEMA_WIDTH * record + 5]
            if not 1 <= scale <= WEIGHT_ONE:
                raise ValueError(f'schema record weight scale {scale} outside [1, {WEIGHT_ONE}] '
                                 f'(root table {header[0]}, record {record})')


def receipt_size(n_streams, provenance=False):
    return STREAM_OFFSET + n_streams * len(STREAM_FIELDS) + (3 if provenance else 1) * MAX_SEQ_LEN


def boundary_offset(receipt):
    tail = 3 if has_row_provenance(receipt) else 1
    return receipt.shape[-1] - tail * MAX_SEQ_LEN


def is_path_receipt(receipt):
    return receipt.shape[-1] >= STREAM_OFFSET + len(STREAM_FIELDS) + MAX_SEQ_LEN


def has_row_provenance(receipt):
    size = receipt.shape[-1] - STREAM_OFFSET - 3 * MAX_SEQ_LEN
    return size >= len(STREAM_FIELDS) and size % len(STREAM_FIELDS) == 0


def row_provenance(receipt):
    """Compact per-selected-row fields and trace-cell -> row-slot mapping."""
    if not has_row_provenance(receipt):
        raise ValueError('composition receipt lacks row provenance; rebuild the sampler remotely')
    start = boundary_offset(receipt) + MAX_SEQ_LEN
    words = receipt[..., start:start + MAX_SEQ_LEN]
    rows = receipt[..., start + MAX_SEQ_LEN:start + 2 * MAX_SEQ_LEN]
    fields = dict(path_sig_id=(words & 0x7fffffff).to(torch.int32),
                  hop_depth=((words >> 31) & 127).to(torch.int16),
                  time_delta_bucket=((words >> 38) & 255).to(torch.int8),
                  degree_in=((words >> 46) & 63).to(torch.int8),
                  degree_out=((words >> 52) & 63).to(torch.int8))
    fields = {k: v.masked_fill(words < 0, -128 if k == 'time_delta_bucket' else 0)
              for k, v in fields.items()}
    return fields, rows


def integer_log_bucket(value):
    """floor(log2(1+n)), evaluated with integers on both backends."""
    return (int(value) + 1).bit_length() - 1


def time_delta_bucket(time, cut, *, query=False):
    if query:
        return 0
    if time == STATIC or cut == STATIC:
        return -128
    delta = int(time) - int(cut)
    return (1 if delta >= 0 else -1) * min(63, integer_log_bucket(abs(delta)))


def provenance_word(signature, hop, time, degree_in, degree_out):
    return (signature | (hop << 31) | ((time & 255) << 38)
            | (degree_in << 46) | (degree_out << 52))


def compact_signature(signature):
    return 1 + signature % 0x7fffffff


def signature_id(parent, direction, table):
    return mix(parent ^ mix(direction + 1) ^ mix(table + 17)) & SIGN_MASK


def extension_time_bucket(row_time, cut, unit_seconds, strata):
    """Return the bounded temporal-mix bucket used by oracle/native paths."""
    if row_time == STATIC or cut == STATIC:
        return strata
    n = abs(int(row_time) - int(cut)) // int(unit_seconds)
    bucket = 0
    while n and bucket < strata - 1:
        bucket += 1
        n >>= 2
    return bucket


def extension_window(neighbors, row, budget):
    """Deterministic midpoint CSR windows shared by CPU/native extension logic."""
    forward, reverse = (len(neighbors[0][row]), len(neighbors[1][row]))
    first = min(forward, (budget + 1) // 2)
    second = min(reverse, budget - first)
    output = []
    for direction, count in ((0, first), (1, second)):
        degree = (forward, reverse)[direction]
        for index in range(count):
            offset = ((2 * index + 1) * degree) // (2 * count)
            output.append((direction, neighbors[direction][row][offset]))
    return output, (forward + reverse), (first + second)


@dataclass
class SchemaPlan:
    __resident_storage_free__ = True
    roots: dict
    n_streams: int
    capacity: int
    flat: tuple


@dataclass
class PathSchema:
    """Small CPU schema statistics owned by CompositionData, never a hot copy."""
    __resident_storage_free__ = True
    populations: dict
    edges: dict
    query_tables: frozenset | None = None
    plans: dict = field(default_factory=dict)
    edge_counts: dict | None = None

    @classmethod
    def admit(cls, graph, directions, targets=None, *, schema_cache=None):
        if graph.row_table.is_cuda or any(t.is_cuda for t in directions.tensors()):
            raise ValueError('path schema requires CPU cold admission before residency')
        from openrfm.sampler.admission import distinct_pair_counts
        cache = {} if schema_cache is None else schema_cache
        table = graph.row_table.numpy()
        if 'path_statistics' not in cache or 'path_edge_counts' not in cache:
            ids, codes, counts = np.unique(table, return_inverse=True, return_counts=True)
            populations = dict(zip(map(int, ids), map(int, counts)))
            edges = {t: {} for t in populations}
            edge_counts = {t: {} for t in populations}
            nt = len(ids)
            for direction, (ptr, dst) in enumerate(((directions.fptr, directions.fdst),
                                                   (directions.pptr, directions.pdst))):
                dst = dst.numpy()
                src = np.repeat(np.arange(len(table), dtype=np.int32), np.diff(ptr.numpy()))
                groups = src.astype(np.int64) * nt + codes[dst]
                groups, degrees = distinct_pair_counts(groups, dst)
                typed = codes[groups // nt] * nt + groups % nt
                pairs, inverse = np.unique(typed, return_inverse=True)
                maxima = np.zeros(len(pairs), dtype=np.int64)
                np.maximum.at(maxima, inverse, degrees)
                totals = np.zeros(len(pairs), dtype=np.int64)
                np.add.at(totals, inverse, degrees)
                for pair, degree, total in zip(pairs.tolist(), maxima.tolist(), totals.tolist()):
                    key = (direction, int(ids[pair % nt]))
                    edges[int(ids[pair // nt])][key] = degree
                    edge_counts[int(ids[pair // nt])][key] = total
            if 'path_statistics' not in cache:
                cache['path_statistics'] = (populations,
                    {t: tuple(sorted(es.items())) for t, es in edges.items()})
            cache['path_edge_counts'] = {t: dict(sorted(es.items()))
                                         for t, es in edge_counts.items()}
        populations, edges = cache['path_statistics']
        if 'path_schema' not in cache:
            cache['path_schema'] = cls(populations, edges,
                                       edge_counts=cache['path_edge_counts'])
        return cache['path_schema']

    def weight_scale(self, parent_table, direction, dest_table):
        """Fixed-point scale of the stream's last edge only: WEIGHT_ONE / its mean fan-out, capped at WEIGHT_ONE."""
        if self.edge_counts is None:
            raise ValueError("stream_weight_policy='selectivity' requires cold-admitted "
                             'per-edge counts (PathSchema.admit)')
        pairs = int(self.edge_counts[parent_table][(direction, dest_table)])
        population = int(self.populations[parent_table])
        if pairs <= population:
            return WEIGHT_ONE
        return max(1, min(WEIGHT_ONE, WEIGHT_ONE * population // pairs))

    def plan(self, spec):
        key = (spec.path_depth, spec.max_path_streams, spec.frontier_rows,
               spec.stream_weight_policy)
        if key in self.plans:
            return self.plans[key]
        selective = spec.stream_weight_policy == 'selectivity'
        counts = {t: 0 for t in self.populations}
        for _ in range(spec.path_depth):
            counts = {t: sum(1 + counts[dest] for (_, dest), _ in self.edges[t])
                      for t in self.populations}
        roots, capacity, n_streams = {}, spec.frontier_rows, 1
        for root in self.populations:
            total = counts[root] + 1
            pool = ((), 2, root)
            selected = [dict(path=pool, parent=-1, direction=2, table=root, depth=0,
                             demand=self.populations[root], witnesses=self.populations[root])]
            parent_edges = {token for token, _ in self.edges[root] if token[0] == 0}
            if len(parent_edges) > 1 and (spec.path_depth < 1 or spec.max_path_streams <= len(parent_edges)):
                raise ValueError('path geometry cannot cover every entity parent plus the label pool')
            frontier = [dict(path=(), table=root, walks=1, id=-1)]
            for depth in range(1, spec.path_depth + 1):
                room = spec.max_path_streams - len(selected)
                if not room:
                    break
                def extensions():
                    for parent in frontier:
                        for (direction, dest), degree in self.edges[parent['table']]:
                            walks = parent['walks'] * degree
                            yield dict(path=parent['path'] + ((direction, dest),),
                                       parent=parent['id'], direction=direction, table=dest,
                                       depth=depth, demand=min(self.populations[dest], walks),
                                       walks=walks, witnesses=min(EXPLORATION_CAP, walks))
                mandatory = ([s for s in extensions() if (s['direction'], s['table']) in parent_edges]
                             if depth == 1 and len(parent_edges) > 1 else [])
                frontier = nsmallest(room, extensions(), key=lambda s: (-s['demand'], s['path']))
                if depth == 1 and len(parent_edges) > 1:
                    others = [s for s in frontier if (s['direction'], s['table']) not in parent_edges]
                    frontier = sorted(mandatory + others[:room-len(mandatory)],
                                      key=lambda s: (-s['demand'], s['path']))
                for s in frontier:
                    s['id'] = len(selected)
                    selected.append(s)
                if not frontier:
                    break
            selected[0]['id'] = 0
            used = set()
            for s in selected:
                parent_hash = root if s['parent'] < 0 else selected[s['parent']]['signature']
                sig = signature_id(parent_hash, s['direction'], s['table'])
                while sig in used:
                    sig = (sig + 1) & SIGN_MASK
                used.add(sig)
                s['signature'] = sig
                if selective and s['depth'] > 0:
                    parent_table = root if s['parent'] < 0 else selected[s['parent']]['table']
                    s['weight_scale'] = self.weight_scale(parent_table, s['direction'], s['table'])
                else:
                    s['weight_scale'] = WEIGHT_ONE
            root_capacity = max(spec.frontier_rows,
                min(EXPLORATION_CAP, 1 + sum(s['witnesses'] for s in selected)))
            roots[root] = dict(streams=selected, total=min(SIGN_MASK, total),
                               pruned=min(SIGN_MASK, total-len(selected)),
                               saturated=int(total > SIGN_MASK), capacity=root_capacity)
            n_streams = max(n_streams, len(selected))
            capacity = max(capacity, root_capacity)
        capacity = min(EXPLORATION_CAP, capacity)
        headers, records = [], []
        for root, plan in roots.items():
            ss = plan['streams']
            headers.extend((root, len(roots)*SCHEMA_HEADER + len(records), len(ss),
                            plan['total'], plan['pruned'], plan['saturated']))
            for s in sorted(ss, key=lambda s: (s['parent'], s['direction'], s['table'])):
                records.extend(schema_record(s, spec))
        result = SchemaPlan(roots, n_streams, capacity, tuple(headers + records))
        validate_weight_scales(result.flat, spec)
        self.plans[key] = result
        return result


def frontier_underfill(receipt):
    """Underfilled because discovery hit its row frontier (not DB exhaustion)."""
    return ((receipt[..., 6] == 3)
            & (receipt[..., PATH_CAP_DROPS] > receipt[..., PATH_STREAMS_PRUNED]))


def stream_histogram(receipt):
    """Host diagnostic only; accepts a single episode's existing receipt."""
    records = receipt[STREAM_OFFSET:boundary_offset(receipt)].reshape(-1, len(STREAM_FIELDS))
    result = [dict(zip(STREAM_FIELDS, map(int, r))) for r in records.tolist() if r[0] >= 0]
    for record in result:
        branch = record
        while branch['parent'] >= 0:
            branch = result[branch['parent']]
        record['history_key_table'] = (branch['table'] if record['same_entity']
                                       and branch['direction'] == 0 else -1)
    return result


def compose_paths(g, d, q, spec, m, settings=None, *, diagnostics=None,
                  truncate_seed_block=False):
    settings = context_settings(q, spec) if settings is None else settings
    B, L = len(q), max(spec.seq_len_set)
    visits = torch.full((B, spec.walk_count, spec.walk_steps), -1, dtype=torch.int64)
    candidates = torch.full((B, spec.candidate_count), -1, dtype=torch.int64)
    scores = torch.zeros_like(candidates)
    cells = torch.full((B, L), -1, dtype=torch.int64)
    roles = torch.zeros_like(cells)
    geometry = m.path_schema.plan(spec)
    capacity = int(m.task_capacity[0])
    provenance = spec._model_requires_provenance
    receipts = torch.zeros((B, receipt_size(geometry.n_streams, provenance)), dtype=torch.int64)
    boundary = boundary_offset(receipts)
    seed_receipts = []
    table, times, task = (x.tolist() for x in (g.row_table, g.row_time, g.row_task))
    cp, columns, available, labels, drops = (x.tolist() for x in
        (g.cell_ptr, g.cell_column, g.cell_available, g.cell_label, g.cell_drop))
    types, cardinality, targets, classes, entity = (x.tolist() for x in
        (m.types, m.cardinality, m.target, m.classes, m.entity))
    kp, keys = m.entity_key_ptr.tolist(), m.entity_keys.tolist()
    history_ptr, history_rows = m.history_ptr.tolist(), m.history_rows.tolist()
    population = dict(zip(m.table_ids.tolist(), m.caps.tolist()))
    queryable = dict(zip(m.table_ids.tolist(), m.stream_mask.tolist()))
    entity_keys = [set(keys[kp[r]:kp[r+1]]) for r in range(len(table))]
    neighbors = [[sorted(set(dst[int(ptr[r]):int(ptr[r+1])].tolist()), key=lambda c: (times[c], c)) for r in range(len(table))]
                 for ptr, dst in ((d.fptr, d.fdst), (d.pptr, d.pdst))]
    rw_neighbors = [[sorted([int(child) for child in dst[int(ptr[r]):int(ptr[r+1])].tolist()],
                            key=lambda child: (times[child], child)) for r in range(len(table))]
                    for ptr, dst in ((d.fptr, d.fdst), (d.pptr, d.pdst))]
    for b, (root, cut, target, key, strict) in enumerate(q.tolist()):
        requested_length = int(settings[b, 0])
        frontier = (min(capacity, max(spec.frontier_rows,
                        capacity * int(settings[b, 4]) // MAX_SEQ_LEN))
                    if spec.discovery_scales_with_length else capacity)
        mixed_extension = spec.extension_policy == 'typed_temporal_mix'
        length = (min(requested_length, spec.protected_core_len)
                  if spec.protected_core_len else requested_length)
        core_total = length
        limit = frontier - spec.label_pool_reserve
        episode_diagnostic = dict(episode=b, seed_row=root, cut=cut, target_column=target,
                                  L=length, status='planning', path_streams=[],
                                  frontier_rows=spec.frontier_rows,
                                  budget=dict(L=length, seed_hops=spec.seed_hops,
                                              path_depth=spec.path_depth,
                                              frontier_rows=spec.frontier_rows,
                                              max_path_streams=spec.max_path_streams))
        if diagnostics is not None:
            diagnostics.append(episode_diagnostic)
        if not queryable.get(table[root], 0):
            raise ValueError('query table is masked out for this task')
        temporal = spec.label_visibility != 'query_only'
        forecast = spec.task_kind == 'forecast'
        real = spec.real_training_cutoff is not None
        if real and (not strict or not 0 < cut <= spec.real_training_cutoff or task[root] != 0
                     or (times[root] != STATIC and
                         (times[root] >= spec.real_training_cutoff or times[root] > cut))):
            raise ValueError('real query crosses the training cutoff or uses a task table')
        def structural(r):
            return ((not real or task[r] == 0) and (r == root or task[r] == 0 or
                ((task[r] == 1 or forecast) and table[r] == table[root])))
        def row_ok(r):
            return structural(r) and (not temporal or r == root or times[r] == STATIC or
                                      times[r] < cut or (not strict and times[r] == cut))
        def drop(c, r):
            return drops[c] and (temporal or r == root)
        def label_ok(r):
            c = targets[r]
            return (r != root and row_ok(r) and table[r] == table[root] and c >= 0 and
                    classes[r] >= 0 and ((not temporal and not drop(c, r)) or
                    (not drops[c] and spec.label_visibility == 'history_available' and available[c] < cut)))
        def feature_ok(c, r):
            if labels[c] or columns[c] == target or drop(c, r) or not row_ok(r) or (temporal and available[c] > cut):
                return False
            return (r == root or spec.context_text_policy == 'keep' or
                (types[c] in (0, 2, 3) if spec.context_text_policy == 'numeric_only' else
                 types[c] != 1 or 0 <= cardinality[c] <= spec.text_cardinality_limit))
        focal = targets[root]
        if focal < 0 or columns[focal] != target:
            raise ValueError('exactly one focal target required')
        if forecast and entity[root] < 0:
            raise ValueError('forecast seed lacks declared entity')
        history_key = root if int(m.history_ptr[root+1]) > int(m.history_ptr[root]) or entity[root] < 0 else entity[root]
        root_keys = entity_keys[root]
        multi_parent = len(root_keys) > 1
        keyed_history = multi_parent or root_keys == {root}
        def role(r):
            if r == root:
                return 0
            if keyed_history:
                return 1 if root_keys & entity_keys[r] else (2 if table[r] == table[root] else 3)
            if entity[r] >= 0 and entity[r] == history_key:
                return 1
            return 2 if table[r] == table[root] else 3
        def row_cells(r, *, filler=False):
            return [c for c in range(cp[r], cp[r+1]) if (r == root and c == focal) or
                    feature_ok(c, r) or (not filler and c == targets[r] and label_ok(r))]
        payload = [row_cells(r) for r in range(len(table))]
        if forecast and not payload[entity[root]] and not (
                cp[entity[root]] == cp[entity[root]+1] and row_ok(entity[root])):
            raise ValueError('forecast seed lacks a visible declared entity')
        ego, ego_seen, ego_cap = [(root, 0)], {root}, 0
        for r, depth in ego:
            if depth >= spec.seed_hops:
                continue
            for direction in (0, 1):
                for child in (reversed(neighbors[direction][r]) if spec.canonical_recent else neighbors[direction][r]):
                    if child in ego_seen or not row_ok(child):
                        continue
                    if len(ego) >= spec.frontier_rows:
                        ego_cap += 1
                        continue
                    ego_seen.add(child)
                    ego.append((child, depth+1))
        schema = geometry.roots[table[root]]
        allowed = {(s['parent'], s['direction'], s['table']): s for s in schema['streams']}
        streams, lookup, states, seen = [], {}, [(root, -1, -1, 0, False)], set()
        def stream(parent, direction, dest_table):
            token = (parent, direction, dest_table)
            if token not in lookup:
                canonical_parent = -1 if parent < 0 else streams[parent]['canonical']
                declared = allowed.get((canonical_parent, direction, dest_table))
                if declared is None:
                    return -1
                sid = len(streams)
                lookup[token] = sid
                streams.append(dict(signature=declared['signature'], parent=parent, direction=direction, table=dest_table,
                    eligible=0, discovered=0, selected=0, time=0, cap=0, budget=0, charged=0, weight=0,
                    same_entity=0, canonical=declared['id'],
                    weight_scale=declared['weight_scale'],
                    branch=sid if parent < 0 else streams[parent]['branch'], items=[]))
            return lookup[token]
        frontier_capped = False
        def add(r, parent_state, sid, depth):
            nonlocal frontier_capped
            token = (sid, r)
            first = token not in seen
            if not first and depth >= spec.path_depth:
                return
            if first:
                if len(seen) >= limit-1:
                    streams[sid]['cap'] = 1
                    frontier_capped = True
                    return
                seen.add(token)
            if not row_ok(r):
                if first:
                    streams[sid]['time'] += 1
                return
            if len(states) >= limit:
                streams[sid]['cap'] = 1
                frontier_capped = True
                return
            index = len(states)
            same_entity = role(r) == 1
            if multi_parent:
                branch = streams[streams[sid]['branch']]
                branch_keys = {k for k in root_keys if branch['direction'] == 0
                               and table[k] == branch['table']}
                same_entity = r != root and bool(branch_keys & entity_keys[r])
            states.append((r, parent_state, sid, depth, same_entity))
            streams[sid]['same_entity'] |= int(same_entity)
            if first and payload[r]:
                streams[sid]['items'].append(index)
                streams[sid]['eligible'] += 1
                streams[sid]['discovered'] += 1
        cursor = 0
        while cursor < len(states) and not frontier_capped:
            r, parent_state, parent_sid, depth, _ = states[cursor]
            if depth < spec.path_depth:
                ancestors, at = set(), cursor
                while at >= 0:
                    ancestors.add(states[at][0]); at = states[at][1]
                for direction in (0, 1):
                    for child in (reversed(neighbors[direction][r]) if spec.canonical_recent else neighbors[direction][r]):
                        if child in ancestors or not structural(child) or (spec.canonical_recent and not row_ok(child)):
                            continue
                        sid = stream(parent_sid, direction, table[child])
                        if sid < 0:
                            continue
                        add(child, cursor, sid, depth+1)
                        if frontier_capped:
                            break
                    if frontier_capped:
                        break
            cursor += 1
        pool_sid = None
        if spec.label_pool_reserve:
            limit = frontier
            pool = [r for r in range(len(table) - 1, -1, -1)
                    if label_ok(r) and available[targets[r]] < cut]
        else:
            pool = [] if frontier_capped else [r for r in range(len(table)) if label_ok(r)]
        if pool:
            sid = stream(-1, 2, table[root])
            if sid >= 0:
                pool_sid = sid
                for r in pool:
                    add(r, 0, sid, spec.path_depth)
        selective = spec.stream_weight_policy == 'selectivity'
        for s in streams:
            s['weight'] = isqrt(s['eligible'] *
                                (s['weight_scale'] if selective else WEIGHT_ONE))
            s['items'].sort(key=lambda i: (mix(key ^ states[i][0]), states[i][0]))
        branch_weights = defaultdict(int)
        for s in streams:
            branch_weights[s['branch']] += (isqrt(s['eligible'] * WEIGHT_ONE)
                                            if selective else s['weight'])
        output, tags, ends, emitted = [], [], [], set()
        withheld = set()
        row_words, row_slots, cell_rows = [], {}, []
        kenc = 0
        def bundle(index):
            chain = []
            while index >= 0:
                chain.append(index); index = states[index][1]
            return list(reversed(chain))
        def admit(rows, filler=False, witnesses=None, *, charge_extension=False, hidden=None):
            nonlocal kenc
            new, new_tags, new_rows, local, label_count = [], [], [], set(), kenc
            for r in rows:
                for c in (row_cells(r, filler=True) if filler else payload[r]):
                    if c in emitted or c in local or c in withheld:
                        continue
                    is_label = c == targets[r] and r != root
                    if is_label and (label_count >= spec.k_enc_max or r == hidden):
                        continue
                    label_count += is_label
                    local.add(c); new.append(c); new_rows.append(r); new_tags.append(4 if filler else role(r))
            if len(output) + len(new) > length:
                return None
            for r in new_rows if provenance else ():
                if r not in row_slots:
                    index = witnesses.get(r) if witnesses is not None else None
                    if r == root:
                        sig, hop = 0, 0
                    elif index is not None:
                        _, _, sid, hop, _ = states[index]
                        sig = compact_signature(streams[sid]['signature'])
                    else:
                        sig = compact_signature(signature_id(table[root], 3 if filler else 0, table[r]))
                        hop = 65 if filler else 1
                        own = entity[root] if forecast else root
                        if spec.own_recent_first and not filler and own >= 0 and r != own and r in neighbors[1][own]:
                            parent_sig = signature_id(table[root], 0, table[own]) if forecast else table[root]
                            sig = compact_signature(signature_id(parent_sig, 1, table[r]))
                            hop = 2 if forecast else 1
                    degrees = [min(31, integer_log_bucket(sum(row_ok(c) for c in axis[r])))
                               for axis in neighbors]
                    row_slots[r] = len(row_words)
                    row_words.append(provenance_word(sig, hop,
                        time_delta_bucket(times[r], cut, query=r == root), degrees[1], degrees[0]))
                cell_rows.append(row_slots[r])
            output.extend(new); tags.extend(new_tags); emitted.update(new); kenc = label_count
            if hidden is not None and targets[hidden] >= 0 and targets[hidden] not in emitted:
                withheld.add(targets[hidden])
            if charge_extension and extension_group_cost is not None:
                for row in new_rows:
                    group = (table[row], extension_time_bucket(
                        times[row], cut, spec.extension_time_unit_seconds,
                        spec.time_strata))
                    extension_group_cost[group] += 1
            ends.extend([len(output)] * len(new))
            return len(new)
        identity = [root] + ([entity[root]] if forecast else [])
        identity_witnesses = {}
        for i, state in enumerate(states[1:], 1):
            if state[0] in identity and state[1] == 0 and streams[state[2]]['direction'] == 0:
                identity_witnesses.setdefault(state[0], i)
        if admit(identity, witnesses=identity_witnesses) is None:
            raise RuntimeError('protected query/forecast identity seed cells exceed L')
        seed_end = len(output)
        seed_receipts.append((seed_end, seed_end))
        extension_group_cost = defaultdict(int) if mixed_extension else None
        rw_scanned = rw_truncated = rw_graph_exhausted = 0
        positions = [0] * len(streams)
        ordered = sorted(range(len(streams)), key=lambda i: streams[i]['signature'])
        blocked = False
        def take(sid, recent=False):
            s = streams[sid]
            at = pick = positions[sid]
            if recent or spec.canonical_recent:
                pick = max(range(at, len(s['items'])),
                           key=lambda j: (times[states[s['items'][j]][0]], -states[s['items'][j]][0]))
            index = s['items'][pick]
            chain = bundle(index)
            charged = admit([states[i][0] for i in chain], witnesses={states[i][0]: i for i in chain})
            if charged is None:
                return False
            if pick != at:
                s['items'].insert(at, s['items'].pop(pick))
            positions[sid] += 1; s['selected'] += 1; s['charged'] += charged
            return True
        if multi_parent:
            parent_tables = {table[k] for k in root_keys}
            for sid in ordered:
                s = streams[sid]
                if s['parent'] < 0 and s['direction'] == 0 and s['table'] in parent_tables and s['eligible']:
                    if not take(sid):
                        blocked = True
                        break
        general = bool(spec.label_rich_first or spec.own_recent_first or spec.own_label_rows or spec.path_quota_rows or spec.path_quota_member_rows
                       or spec.path_quota_parent_rows or spec.path_quota_grandparent_rows
                       or spec.path_quota_grandchild_rows or spec.seed_tier_percent)
        scale = max(1, core_total // 1024)
        if spec.label_rich_first:
            own = entity[root] if forecast else root
            priority = [r for r in range(len(table)) if label_ok(r)
                        and available[targets[r]] < cut and targets[r] not in emitted]
            priority.sort(key=lambda r: (not (own >= 0 and entity[r] == own), -times[r], r))
            for r in priority[:spec.candidate_count]:
                if admit([r]) is None:
                    break
        if spec.own_recent_first or spec.label_rich_first:
            own = entity[root] if forecast else root
            eligible = []
            own_keys = ({root} | set(neighbors[0][root])) if spec.own_source_rank else {own}
            own_rows = sorted({r for key in own_keys if key >= 0 for r in neighbors[1][key]})
            rank_times = m.source_time.tolist() if spec.own_source_rank else times
            for r in own_rows:
                if r == root or not row_ok(r):
                    continue
                if task[r] and not label_ok(r):
                    continue
                if not payload[r]:
                    continue
                eligible.append(r)
            eligible.sort(key=lambda r: (-rank_times[r], r))
            if spec.own_recent_interleave:
                by_table = defaultdict(list)
                for r in eligible:
                    by_table[table[r]].append(r)
                priority = [by_table[t][k] for k in range(max(map(len, by_table.values()), default=0))
                            for t in sorted(by_table) if k < len(by_table[t])]
            else:
                first = {}
                for r in eligible:
                    first.setdefault(table[r], r)
                priority = [first[t] for t in sorted(first)]
                priority += [r for r in eligible if r not in set(priority)]
            exhausted_tables = set()
            if spec.own_recent_cell_balance:
                by_table = defaultdict(list)
                for r in eligible:
                    if not any(c in emitted for c in payload[r]):by_table[table[r]].append(r)
                own_positions={t:0 for t in by_table};served={t:0 for t in by_table}
                taken=0
                while taken<length:
                    available=[t for t in by_table if t not in exhausted_tables and own_positions[t]<len(by_table[t])]
                    if not available:break
                    t=min(available,key=lambda t:(served[t],t))
                    r=by_table[t][own_positions[t]];own_positions[t]+=1
                    charged=admit([r])
                    if charged is None:exhausted_tables.add(t)
                    else:served[t]+=charged;taken+=1
                priority=[]
            for r in priority:
                if table[r] in exhausted_tables:
                    continue
                if admit([r]) is None:
                    if spec.own_source_rank:
                        exhausted_tables.add(table[r])
                        continue
                    break
        elif spec.own_label_rows:
            own = entity[root] if forecast else -1
            if own >= 0:
                taken = 0
                for i in range(history_ptr[own + 1] - 1, history_ptr[own] - 1, -1):
                    if taken >= spec.own_label_rows * scale:
                        break
                    r = history_rows[i]
                    if not label_ok(r):
                        continue
                    if admit([r]) is None:
                        break
                    taken += 1
        else:
            own_taken = 0
            for sid in ordered:
                if blocked:
                    break
                s = streams[sid]
                if not (s['same_entity'] and s['table'] == table[root] and s['eligible']):
                    continue
                while positions[sid] < s['eligible'] and own_taken < OWN_HISTORY_ROWS:
                    if not take(sid):
                        blocked = True
                        break
                    own_taken += 1
        quota_classes = (spec.path_quota_parent_rows, spec.path_quota_rows, spec.path_quota_member_rows,
                         spec.path_quota_grandparent_rows, spec.path_quota_grandchild_rows)
        entity_table = table[entity[root]] if forecast else table[root]

        def entity_stream(sid):
            if not forecast:
                return sid < 0
            if sid < 0:
                return False
            s = streams[sid]
            return s['parent'] < 0 and s['direction'] == 0 and s['table'] == entity_table

        def own_event(sid):
            s = streams[sid]
            return s['direction'] == 1 and entity_stream(s['parent']) and s['table'] != table[root]

        def parent_path(sid):
            s = streams[sid]
            return s['direction'] == 0 and entity_stream(s['parent']) and s['table'] != entity_table

        def quota_class(sid):
            s = streams[sid]
            if s['direction'] == 2:
                return -1
            if own_event(sid):
                return 1
            if parent_path(sid):
                return 0
            if s['parent'] < 0:
                return -1
            if s['direction'] == 0 and own_event(s['parent']) and s['table'] != entity_table:
                return 2
            if s['direction'] == 0 and parent_path(s['parent']) and s['table'] != entity_table:
                return 3
            if (s['direction'] == 1 and own_event(s['parent']) and s['table'] != entity_table
                    and s['table'] != table[root]):
                return 4
            return -1

        def quota_rows(sid):
            cls = quota_class(sid)
            return 0 if cls < 0 else quota_classes[cls] * scale

        if any(quota_classes):
            def quota_key(sid):
                cls = quota_class(sid)
                group = streams[sid]['parent'] if cls in (2, 4) else sid
                near = pop = 0
                if cls in (1, 2, 4):
                    pop = population.get(streams[group]['table'], 0)
                    near = 0 if 2 * pop <= 3 * population.get(entity_table, 0) else 1
                return ((0, 1, 2, 2, 3)[cls], near, pop, streams[group]['signature'],
                        streams[sid]['signature'], sid)
            if spec.path_quota_order == 'density':
                order = sorted((sid for sid in range(len(streams)) if quota_rows(sid)), key=quota_key)
            else:
                order = ordered
            share = core_total * spec.path_quota_share // 100
            length = max(len(output), share) if spec.path_quota_share else core_total
            blocked = False
            for quota_round in range(max(quota_classes) * scale):
                if blocked:
                    break
                for sid in order:
                    if blocked:
                        break
                    if (quota_round < quota_rows(sid) and positions[sid] <= quota_round
                            and positions[sid] < streams[sid]['eligible'] and not take(sid, recent=True)):
                        blocked = True
            blocked, length = False, core_total

        def generate_walks():
            """Native generate_rw: bounded midpoint windows, deterministic draws."""
            witnesses, scanned_total, truncated_total, exhausted_total = {}, 0, 0, 0
            for walk in range(spec.walk_count):
                path = [root]
                for step in range(spec.walk_steps):
                    salt = key ^ mix(walk + 1) ^ mix(step + 0x10001)
                    window, degree, scanned = extension_window(
                        rw_neighbors, path[-1], spec.rw_scan_budget)
                    scanned_total += scanned
                    truncated_total += degree - scanned
                    legal = []
                    for direction, child in window:
                        if child in path or not row_ok(child):
                            continue
                        if mixed_extension:
                            bucket = extension_time_bucket(
                                times[child], cut, spec.extension_time_unit_seconds,
                                spec.time_strata)
                            draw_key = ((mix(salt ^ signature_id(table[child], direction, bucket + 17))
                                         & 0xffffffff00000000)
                                        | (mix(salt ^ mix(child)) & 0xffffffff))
                        else:
                            draw_key = mix(salt ^ mix(child))
                        legal.append((draw_key, child))
                    if not legal:
                        if degree == scanned:
                            exhausted_total += 1
                        break
                    _, chosen = min(legal)
                    path.append(chosen)
                    visits[b, walk, step] = chosen
                    previous = witnesses.get(chosen)
                    if previous is None or len(path) < len(previous):
                        witnesses[chosen] = list(path)
            return witnesses, scanned_total, truncated_total, exhausted_total

        def children_until(row, limit):
            """First reverse-CSR position of row whose child is later than limit."""
            kids = rw_neighbors[1][row]
            lo, hi = 0, len(kids)
            if limit == STATIC:
                return hi
            while lo < hi:
                mid = (lo + hi) // 2
                if times[kids[mid]] <= limit:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        def seed_ok(r):
            c = targets[r]
            return label_ok(r) and available[c] < cut and c not in emitted and c not in withheld

        def route_count(table_words, row):
            """Native route_count: open-address (row << 16 | count) hash, 3/4 load."""
            n = len(table_words)
            i = mix(row) % n
            while table_words[i] >= 0 and table_words[i] >> 16 != row:
                i = (i + 1) % n
            if table_words[i] < 0:
                if 4 * (route_state['distinct'] + 1) > 3 * n:
                    return
                table_words[i] = row << 16
                route_state['distinct'] += 1
            if table_words[i] & 0xffff < 0xffff:
                table_words[i] += 1

        def route_seeds(other, table_words):
            if not forecast:
                if table[other] == table[root] and seed_ok(other):
                    route_count(table_words, other)
                return
            kids = rw_neighbors[1][other]
            j, scanned, found = children_until(other, cut) - 1, 0, 0
            while j >= 0 and scanned < ROUTE_SCAN and found < ROUTE_TASKS:
                t = kids[j]
                if not (j + 1 < len(kids) and kids[j + 1] == t) and table[t] == table[root] and seed_ok(t):
                    route_count(table_words, t)
                    found += 1
                j -= 1
                scanned += 1

        def stratified(count, size):
            return [((2 * k + 1) * count) // (2 * size) for k in range(size)]

        route_state = {}

        def shared_routes():
            """Native shared_routes: stratified co-member samples into a count hash."""
            table_words = [-1] * (spec.walk_count * spec.walk_steps)
            route_state['distinct'] = 0
            own = entity[root] if forecast else root
            if own < 0:
                return table_words
            etable, fanout = table[own], spec.seed_route_fanout
            forward = rw_neighbors[0][own]
            for j, parent in enumerate(forward):
                if (j and forward[j - 1] == parent) or task[parent] or not row_ok(parent):
                    continue
                kids = rw_neighbors[1][parent]
                count = children_until(parent, cut)
                for k in stratified(count, min(count, fanout)):
                    other = kids[k]
                    if other != own and table[other] == etable and row_ok(other):
                        route_seeds(other, table_words)
            kids = rw_neighbors[1][own]
            k, scanned = children_until(own, cut) - 1, 0
            while k >= 0 and scanned < ROUTE_CHILDREN:
                event = kids[k]
                if not (k + 1 < len(kids) and kids[k + 1] == event) and not task[event] and row_ok(event):
                    members = rw_neighbors[0][event]
                    for j, member in enumerate(members):
                        if ((j and members[j - 1] == member) or member == own or table[member] == etable
                                or task[member] or not row_ok(member)):
                            continue
                        peers = rw_neighbors[1][member]
                        count = children_until(member, cut)
                        for x in stratified(count, min(count, fanout)):
                            peer = peers[x]
                            if (peer == event or table[peer] != table[event] or task[peer]
                                    or not row_ok(peer)):
                                continue
                            others = rw_neighbors[0][peer]
                            for y, other in enumerate(others):
                                if ((y and others[y - 1] == other) or other == own
                                        or table[other] != etable or not row_ok(other)):
                                    continue
                                route_seeds(other, table_words)
                k -= 1
                scanned += 1
            return table_words

        def torch_mix(x):
            """openrfm.sampler.retrieval.mix applied to an int64 tensor (wrapping int64 arithmetic)."""
            def wrap(v):
                v %= 1 << 64
                return v - (1 << 64) if v >= 1 << 63 else v
            x = wrap(x + 0x9e3779b97f4a7c15)
            x = wrap((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9)
            x = wrap((x ^ (x >> 27)) * 0x94d049bb133111eb)
            return x ^ (x >> 31)

        def program_routes():
            """Native program_routes: the context-program seed walks, exactly."""
            cap = spec.seed_hop_cap
            width = min(max(2 * cap, 64), 4096)
            mature = spec.seed_program_mature
            etable, final, paths = (table[entity[root]] if forecast else table[root]), [], 0
            for sid in ordered:
                if paths >= PROGRAM_PATHS:
                    break
                cls = quota_class(sid)
                if cls not in (0, 2):
                    continue
                paths += 1
                s = streams[sid]
                hops = [(0, etable)] if forecast else []
                if cls == 2:
                    c = streams[s['parent']]['table']
                    hops += [(1, c), (0, s['table']), (1, c), (0, etable)]
                else:
                    hops += [(0, s['table']), (1, etable)]
                if forecast:
                    hops.append((1, table[root]))
                frontier = [root]
                for h, (direction, hop_table) in enumerate(hops):
                    if not frontier:
                        break
                    last, emitted_rows = h == len(hops) - 1, []
                    for r in frontier:
                        edges = rw_neighbors[direction][r]
                        end = children_until(r, cut - 1) if (mature and direction == 1) else len(edges)
                        for off in range(min(end, width)):
                            x = edges[end - 1 - off]
                            if table[x] != hop_table or (not last and task[x] != 0):
                                continue
                            if times[x] != STATIC and times[x] >= cut:
                                continue
                            if mature and last and (targets[x] < 0 or available[targets[x]] >= cut):
                                continue
                            emitted_rows.append((torch_mix(key ^ x), len(emitted_rows), x))
                    kept = sorted(emitted_rows)[:cap]
                    frontier = [x for _, _, x in sorted(kept, key=lambda item: item[1])]
                final += frontier
            counts = Counter(final)
            return {r: n for r, n in counts.items() if seed_ok(r)}

        walked, routes_used = None, False
        ranked_seeds, seed_bundle = [], None
        if spec.seed_tier_percent:
            length = min(core_total, len(output) + core_total * spec.seed_tier_percent // 100)
            if spec.seed_rank == 'program':
                seed_counts = program_routes()
            elif spec.seed_rank == 'shared':
                words = shared_routes()
                visits[b].view(-1)[:] = torch.tensor(words, dtype=torch.int64)
                routes_used = True
                seed_counts = {w >> 16: w & 0xffff for w in words if w >= 0 and seed_ok(w >> 16)}
            else:
                walked = generate_walks()
                rw_scanned += walked[1]; rw_truncated += walked[2]; rw_graph_exhausted += walked[3]
                visited = Counter(r for r in visits[b].flatten().tolist() if r >= 0)
                seed_counts = {r: n for r, n in visited.items() if seed_ok(r)}
            if spec.seed_fill and pool_sid is not None:
                considered = 0
                for index in range(1, len(states)):
                    if considered >= SEED_POOL_FACTOR * spec.candidate_count:
                        break
                    if states[index][2] != pool_sid:
                        continue
                    considered += 1
                    r = states[index][0]
                    if seed_ok(r) and r not in seed_counts:
                        seed_counts[r] = 0
            ranked_seeds = sorted(seed_counts, key=lambda r: (-seed_counts[r], -times[r], r)
                                  )[:spec.candidate_count]
            if spec.seed_balance:
                in_class, keyed = Counter(), []
                for position, r in enumerate(ranked_seeds):
                    label_class = min(SEED_CLASS_CAP, classes[r])
                    keyed.append((in_class[label_class], label_class, position, r))
                    in_class[label_class] += 1
                ranked_seeds = [r for *_, r in sorted(keyed)]
            for index, r in enumerate(ranked_seeds):
                candidates[b, index], scores[b, index] = r, seed_counts[r]

            def seed_bundle(seed, limit_rows=None, policy=None):
                limit_rows = spec.seed_local_rows if limit_rows is None else limit_rows
                policy = spec.seed_local_policy if policy is None else policy
                rows, ts = [seed], times[seed]
                if not limit_rows:
                    return rows

                def local_ok(x):
                    return (x != root and task[x] == 0 and row_ok(x) and x not in rows
                            and (ts == STATIC or times[x] == STATIC or times[x] <= ts))
                if policy in ('entity', 'entity_largest'):
                    own_entity = entity[seed] if forecast else seed
                    if own_entity < 0:
                        return rows
                    if own_entity != seed and local_ok(own_entity):
                        rows.append(own_entity)
                    kids = rw_neighbors[1][own_entity]
                    hi = children_until(own_entity, ts)
                    child_tables = []
                    j, scanned = hi - 1, 0
                    while j >= 0 and scanned < SEED_LOCAL_SCAN:
                        y = kids[j]
                        if (not task[y] and row_ok(y) and table[y] not in [t for _, t in child_tables]
                                and len(child_tables) < SEED_LOCAL_TABLES):
                            child_tables.append((population.get(table[y], 0), table[y]))
                            child_tables.sort(key=(lambda item: (-item[0], item[1]))
                                              if policy == 'entity_largest' else None)
                        j -= 1
                        scanned += 1
                    for rnd in range(limit_rows):
                        if len(rows) > limit_rows:
                            break
                        for _, child_table in child_tables:
                            if len(rows) > limit_rows:
                                break
                            seen_rows, j, scanned = 0, hi - 1, 0
                            while j >= 0 and scanned < SEED_LOCAL_SCAN:
                                y = kids[j]
                                if (not (j + 1 < len(kids) and kids[j + 1] == y) and table[y] == child_table
                                        and not task[y] and row_ok(y)):
                                    if seen_rows >= rnd:
                                        if local_ok(y):
                                            rows.append(y)
                                        break
                                    seen_rows += 1
                                j -= 1
                                scanned += 1
                    return rows
                parents_of = rw_neighbors[0][seed]
                for j, x in enumerate(parents_of):
                    if len(rows) > limit_rows:
                        break
                    if j and parents_of[j - 1] == x:
                        continue
                    if local_ok(x):
                        rows.append(x)
                for x in rows[1:]:
                    if len(rows) > limit_rows:
                        break
                    kids = rw_neighbors[1][x]
                    j, scanned = children_until(x, ts) - 1, 0
                    while j >= 0 and scanned < SEED_LOCAL_SCAN and len(rows) <= limit_rows:
                        y = kids[j]
                        if not (j + 1 < len(kids) and kids[j + 1] == y) and local_ok(y):
                            rows.append(y)
                        j -= 1
                        scanned += 1
                return rows

            for r in ranked_seeds:
                if len(output) >= length:
                    break
                admit(seed_bundle(r), hidden=r if spec.seed_hide_labels else None)
            length = core_total
        if general:
            blocked = not spec.path_stream_fill
        for sid in ordered:
            if blocked:
                break
            if streams[sid]['eligible'] and not positions[sid] and not take(sid):
                blocked = True; break
        while not blocked:
            active = [i for i in ordered if positions[i] < streams[i]['eligible']]
            if not active:
                break
            chosen = active[0]
            for i in active[1:]:
                a, c = streams[i], streams[chosen]
                if a['charged'] * branch_weights[a['branch']] * c['weight'] < c['charged'] * branch_weights[c['branch']] * a['weight']:
                    chosen = i
            if not take(chosen):
                blocked = True
        cap_drops = min(SIGN_MASK, schema['pruned'] + sum(s['cap'] for s in streams))
        exhausted = not blocked and not cap_drops
        if exhausted and spec.filler_percent:
            for r in range(len(table)):
                if r != root and admit([r], filler=True) is None:
                    break
        hybrid = bool(spec.protected_core_len and requested_length > spec.protected_core_len)
        if hybrid:
            length, blocked = requested_length, False
            if mixed_extension and spec.peer_extension_percent:
                length = (len(output) + max(0, requested_length - spec.protected_core_len)
                          * spec.peer_extension_percent // 100)
                for r in ranked_seeds:
                    if len(output) >= length:
                        break
                    admit(seed_bundle(r, spec.peer_extension_rows, 'entity_largest'), charge_extension=True,
                          hidden=r if spec.seed_hide_labels else None)
                length = requested_length
            if mixed_extension:
                rw_budget = (len(output) + max(0, requested_length - spec.protected_core_len)
                             * spec.rw_extension_percent // 100)
                length = rw_budget
                witnesses_by_row = {}
                if spec.rw_extension_percent:
                    if walked is None:
                        if routes_used:
                            visits[b].fill_(-1)
                        walked = generate_walks()
                        rw_scanned += walked[1]; rw_truncated += walked[2]
                        rw_graph_exhausted += walked[3]
                    witnesses_by_row = walked[0]

                    counts = Counter(r for r in visits[b].flatten().tolist() if r >= 0)
                    ranked = sorted(counts, key=lambda r: (mix(key ^ r), r))[:spec.candidate_count]
                    for index, r in enumerate(ranked):
                        candidates[b, index], scores[b, index] = r, counts[r]

                    pending = set(ranked)
                    while pending:
                        r = min(
                            pending,
                            key=lambda row: (
                                extension_group_cost[(
                                    table[row],
                                    extension_time_bucket(
                                        times[row], cut,
                                        spec.extension_time_unit_seconds,
                                        spec.time_strata),
                                )],
                                extension_time_bucket(
                                    times[row], cut,
                                    spec.extension_time_unit_seconds,
                                    spec.time_strata),
                                mix(key ^ row),
                                row,
                            ),
                        )
                        pending.remove(r)
                        path = witnesses_by_row[r]
                        old_rows = len(row_words)
                        if admit(path, charge_extension=True) is None:
                            continue
                        if provenance:
                            sig = table[root]
                            for hop, child in enumerate(path[1:], 1):
                                direction = 0 if child in neighbors[0][path[hop - 1]] else 1
                                sig = signature_id(sig, direction, table[child])
                                slot = row_slots.get(child, -1)
                                if slot >= old_rows:
                                    row_words[slot] = (
                                        (row_words[slot] & ~((1 << 38) - 1))
                                        | compact_signature(sig) | (hop << 31)
                                    )

                length = requested_length
                full_blocked = False
                stream_attempted = [set(s['items'][:positions[sid]])
                                    for sid, s in enumerate(streams)]

                def mixed_stream_candidates(sid):
                    return [index for index in streams[sid]['items']
                            if index not in stream_attempted[sid]]

                def mixed_take(sid):
                    s = streams[sid]
                    available = mixed_stream_candidates(sid)
                    if not available:
                        return None
                    index = min(
                        available,
                        key=lambda item: (
                            extension_group_cost[(
                                table[states[item][0]],
                                extension_time_bucket(
                                    times[states[item][0]], cut,
                                    spec.extension_time_unit_seconds,
                                    spec.time_strata),
                            )],
                            extension_time_bucket(
                                times[states[item][0]], cut,
                                spec.extension_time_unit_seconds,
                                spec.time_strata),
                            mix(key ^ states[item][0]),
                            states[item][0],
                        ),
                    )
                    stream_attempted[sid].add(index)
                    chain = bundle(index)
                    charged = admit(
                        [states[i][0] for i in chain],
                        witnesses={states[i][0]: i for i in chain},
                        charge_extension=True,
                    )
                    if charged is None:
                        return False
                    s['selected'] += 1
                    s['charged'] += charged
                    return True

                while len(output) < length:
                    active = [sid for sid in ordered if mixed_stream_candidates(sid)]
                    if not active:
                        break
                    chosen = active[0]
                    for sid in active[1:]:
                        a, c = streams[sid], streams[chosen]
                        if (a['charged'] * branch_weights[a['branch']] * c['weight']
                                < c['charged'] * branch_weights[c['branch']] * a['weight']):
                            chosen = sid
                    if mixed_take(chosen) is False:
                        full_blocked = True

                blocked = full_blocked
                exhausted = False
            else:
                if walked is None:
                    if routes_used:
                        visits[b].fill_(-1)
                    walked = generate_walks()
                    rw_scanned += walked[1]; rw_truncated += walked[2]
                    rw_graph_exhausted += walked[3]
                witnesses_by_row = walked[0]
                counts = Counter(r for r in visits[b].flatten().tolist() if r >= 0)
                ranked = sorted(counts, key=lambda r: (-counts[r], r))[:spec.candidate_count]
                for index, r in enumerate(ranked):
                    candidates[b, index], scores[b, index] = r, counts[r]
                    old_rows = len(row_words)
                    path = witnesses_by_row[r]
                    if admit(path) is None:
                        blocked = True
                        continue
                    if provenance:
                        sig = table[root]
                        for hop, child in enumerate(path[1:], 1):
                            direction = 0 if child in neighbors[0][path[hop - 1]] else 1
                            sig = signature_id(sig, direction, table[child])
                            slot = row_slots.get(child, -1)
                            if slot >= old_rows:
                                row_words[slot] = (
                                    (row_words[slot] & ~((1 << 38) - 1))
                                    | compact_signature(sig) | (hop << 31)
                                )
                exhausted = False
        rec = receipts[b]
        rec[STREAM_OFFSET:boundary] = -1
        rec[boundary:] = -1
        for i, s in enumerate(streams):
            s['budget'] = s['eligible'] - s['selected']
            width = len(STREAM_FIELDS)
            rec[STREAM_OFFSET+i*width:STREAM_OFFSET+(i+1)*width] = torch.tensor([s[k] for k in STREAM_FIELDS])
        rows_selected = {bisect_right(cp, c) - 1 for c in output}
        peer_count = sum(r != root and role(r) == 2 and targets[r] in emitted for r in rows_selected)
        history_count = sum(role(r) == 1 and targets[r] in emitted for r in rows_selected)
        role_counts = [tags.count(i) for i in range(5)]
        rec[:22] = torch.tensor([length, len(output), len(rows_selected), peer_count, kenc, 1,
            0 if len(output) == length else (2 if blocked else 3 if cap_drops else 1), peer_count, history_count,
            sum(types[c] == 1 for c in output), role_counts[4], seed_end,
            seed_end + role_counts[1], len(output)-role_counts[4], 0, 0, 0,
            3 if hybrid else 2, 0, 0, 0, 0])
        rec[14:17] = torch.tensor([rw_scanned, rw_truncated, rw_graph_exhausted])
        rec[22:STREAM_OFFSET] = torch.tensor([len(streams), sum(s['eligible'] for s in streams),
            sum(s['discovered'] for s in streams), sum(s['selected'] for s in streams),
            sum(s['time'] for s in streams), cap_drops, length-len(output) if blocked else 0, int(exhausted),
            schema['total'], schema['pruned'], frontier, schema['saturated'], len(ego), ego_cap])
        cells[b, :len(output)] = torch.tensor(output, dtype=torch.int64)
        roles[b, :len(tags)] = torch.tensor(tags, dtype=torch.int64)
        rec[boundary:boundary+len(ends)] = torch.tensor(ends, dtype=torch.int64)
        if provenance:
            start = boundary + MAX_SEQ_LEN
            rec[start:start+len(row_words)] = torch.tensor(row_words, dtype=torch.int64)
            rec[start+MAX_SEQ_LEN:start+MAX_SEQ_LEN+len(cell_rows)] = torch.tensor(cell_rows, dtype=torch.int64)
        if diagnostics is not None:
            episode_diagnostic.update(dict(episode=b, seed_row=root, L=length, status='passed',
                seed_cells=seed_end, seed_cells_kept=seed_end, seed_cells_complete=True,
                seed_block_cells_dropped=0, path_streams=stream_histogram(rec),
                atomic_ends=sorted(set(ends)), packing_slack=int(rec[28]),
                schema_stream_count=schema['total'], path_streams_pruned=schema['pruned'],
                path_capacity=capacity, path_frontier=frontier, path_cap_drops=cap_drops,
                discovery_complete=not cap_drops,
                frontier_underfill=bool(spec.discovery_scales_with_length
                                        and frontier_underfill(rec)),
                seed_ego_rows=len(ego), seed_ego_cap_drops=ego_cap))
    result = (visits, candidates, scores, cells, roles, receipts)
    return (*result, torch.tensor(seed_receipts, dtype=torch.int64)) if truncate_seed_block else result
