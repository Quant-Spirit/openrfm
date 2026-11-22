"""Exact sparse attention for the col, feat and nbr axes, planned and executed on the device.

  feat  keys of q = cells of node(q)  U  cells of each parent node of q
  nbr   keys of q = cells k with node(q) in parents(k)
  col   keys of q = cells with the same (table, column)

Every key cell lies in exactly one tile, so dK/dV need no atomics (deterministic).
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

TILE = 16                  # queries per tile; tl.dot minimum
BLOCK_KEYS = 32
_BIG = torch.iinfo(torch.int64).max


class HalfPlan(NamedTuple):
    """One direction of one axis: tiles of group cells and their key ranges over `ent_cell` entries."""
    perm: torch.Tensor          # [B, L] sorted position -> cell
    ntiles: torch.Tensor        # [B]
    tile_start: torch.Tensor    # [B, L] sorted position of the tile's first cell
    tile_cnt: torch.Tensor      # [B, L]
    ent_cell: torch.Tensor      # [B, ENT] entry -> cell
    lo: torch.Tensor            # [B, L, R] entry range per tile and slot
    hi: torch.Tensor
    skip: torch.Tensor          # [B, ENT] uint8, entries to ignore (feat backward self-parent duplicates)


class AxisPlan(NamedTuple):
    axis: str
    fwd: HalfPlan
    bwd: HalfPlan
    skip_bwd: bool              # whether bwd.skip is meaningful

    def attend(self, q, k, v):
        """q, k, v: [B, H, L, D]; rows without any present key return zeros (like the masked dense path)."""
        return _attend(q, k, v, list(self.fwd), list(self.bwd), self.skip_bwd)[0]


def _tiles(new_group: torch.Tensor, nvalid: torch.Tensor):
    """Tiles of at most TILE consecutive sorted cells inside one group. new_group[b, i]: a group starts at i."""
    B, L = new_group.shape
    pos = torch.arange(L, device=new_group.device).expand(B, L)
    run_start = torch.cummax(torch.where(new_group, pos, 0), dim=1).values
    is_tile = ((pos - run_start) % TILE == 0) & (pos < nvalid[:, None])
    tile_id = is_tile.long().cumsum(1) - 1
    start = torch.full((B, L + 1), L, dtype=torch.long, device=pos.device)
    start.scatter_(1, torch.where(is_tile, tile_id, L), pos)
    start = start[:, :L]
    following = torch.cat([start[:, 1:], start.new_full((B, 1), L)], 1)
    count = (torch.minimum(following, nvalid[:, None]) - start).clamp_min(0)
    return is_tile.sum(1), start, count


def _search(sorted_keys: torch.Tensor, probe: torch.Tensor):
    shape = probe.shape
    flat = probe.reshape(shape[0], -1).contiguous()
    lo = torch.searchsorted(sorted_keys, flat, side='left')
    hi = torch.searchsorted(sorted_keys, flat, side='right')
    return lo.view(shape), hi.view(shape)


def _i32(*tensors):
    return [t.to(torch.int32).contiguous() for t in tensors]


def _half(perm, ntiles, start, count, ent_cell, lo, hi, skip=None):
    perm, ntiles, start, count, ent_cell, lo, hi = _i32(perm, ntiles, start, count, ent_cell, lo, hi)
    return HalfPlan(perm, ntiles, start, count, ent_cell, lo, hi, ntiles if skip is None else skip)


def build_node_plans(node: torch.Tensor, parents: torch.Tensor, valid: torch.Tensor) -> dict[str, AxisPlan]:
    """FEAT and NBR plans: both are ranges over the node-sorted cells and the parent-sorted (cell, slot) entries."""
    B, L = node.shape
    P = parents.shape[-1]
    sorted_node, perm = node.masked_fill(~valid, _BIG).sort(dim=1, stable=True)
    nvalid = valid.sum(1)
    perm3 = perm[..., None].expand(-1, -1, P)
    sorted_parents = parents.gather(1, perm3)
    # distinct non-negative parents of each cell (the dense mask is an OR over slots)
    parent_ok = parents >= 0
    for slot in range(1, P):
        parent_ok[..., slot] &= ~(parents[..., slot:slot + 1] == parents[..., :slot]).any(-1)
    sorted_ok = parent_ok.gather(1, perm3)
    new_group = torch.ones_like(valid)
    new_group[:, 1:] = (sorted_node[:, 1:] != sorted_node[:, :-1]) | (sorted_parents[:, 1:] != sorted_parents[:, :-1]).any(-1)
    ntiles, start, count = _tiles(new_group, nvalid)
    first = start.clamp_max(L - 1)
    tile_node = sorted_node.gather(1, first)
    tile_parents = sorted_parents.gather(1, first[..., None].expand(-1, -1, P))
    tile_ok = sorted_ok.gather(1, first[..., None].expand(-1, -1, P))
    own_lo, own_hi = _search(sorted_node, tile_node)
    par_lo, par_hi = _search(sorted_node, tile_parents.reshape(B, -1))
    par_lo, par_hi = par_lo.view(B, L, P), par_hi.view(B, L, P)

    # entries (cell, slot) keyed by the parent they point at; invalid cells and repeated parents are dropped
    entry_key = torch.where(parent_ok & valid[..., None], parents, _BIG).view(B, L * P)
    sorted_entry_key, entry_perm = entry_key.sort(dim=1, stable=True)
    entry_cell = torch.div(entry_perm, P, rounding_mode='floor')
    entry_self = (node[..., None] == parents).view(B, L * P).gather(1, entry_perm)
    child_lo, child_hi = _search(sorted_entry_key, tile_node)
    child_lo, child_hi = child_lo + L, child_hi + L
    ent_cell = torch.cat([perm, entry_cell], 1)
    ent_self = torch.cat([torch.zeros_like(perm), entry_self.long()], 1).to(torch.uint8).contiguous()

    parent_lo = torch.where(tile_ok, par_lo, 0)
    parent_hi = torch.where(tile_ok, par_hi, 0)
    # FEAT forward: own node plus parent nodes (a parent equal to the own node is already covered by the own range)
    feat_ok = tile_ok & (tile_parents != tile_node[..., None])
    feat_lo = torch.cat([own_lo[..., None], torch.where(feat_ok, par_lo, 0)], -1)
    feat_hi = torch.cat([own_hi[..., None], torch.where(feat_ok, par_hi, 0)], -1)
    tiles = (ntiles, start, count)
    fwd_feat = _half(perm, *tiles, ent_cell, feat_lo, feat_hi)
    fwd_nbr = _half(perm, *tiles, ent_cell, child_lo[..., None], child_hi[..., None])
    # transposed relations
    bwd_feat = _half(perm, *tiles, ent_cell, torch.stack([own_lo, child_lo], -1), torch.stack([own_hi, child_hi], -1),
                     skip=ent_self)
    bwd_nbr = _half(perm, *tiles, ent_cell, parent_lo, parent_hi)
    return {'feat': AxisPlan('feat', fwd_feat, bwd_feat, True), 'nbr': AxisPlan('nbr', fwd_nbr, bwd_nbr, False)}


def build_col_plan(table: torch.Tensor, col: torch.Tensor, valid: torch.Tensor) -> AxisPlan:
    B, L = table.shape
    key = (table << 32) + col
    sorted_key, perm = key.masked_fill(~valid, _BIG).sort(dim=1, stable=True)
    nvalid = valid.sum(1)
    new_group = torch.ones_like(valid)
    new_group[:, 1:] = sorted_key[:, 1:] != sorted_key[:, :-1]
    ntiles, start, count = _tiles(new_group, nvalid)
    tile_key = sorted_key.gather(1, start.clamp_max(L - 1))
    lo, hi = _search(sorted_key, tile_key)
    half = _half(perm, ntiles, start, count, perm, lo[..., None], hi[..., None])
    return AxisPlan('col', half, half, False)


def build_device_plan(batch: dict) -> dict[str, AxisPlan]:
    """Plans of the col, feat and nbr axes; padding cells are neither queries nor keys."""
    valid = ~batch['is_padding'].bool()
    plans = build_node_plans(batch['node_idxs'], batch['f2p_nbr_idxs'], valid)
    plans['col'] = build_col_plan(batch['table_name_idxs'], batch['col_name_idxs'], valid)
    return plans


@triton.jit
def _tile_rows(perm, ntiles, tile_start, tile_cnt, b, t, L, BQ: tl.constexpr):
    start = tl.load(tile_start + b * L + t)
    count = tl.load(tile_cnt + b * L + t)
    rows = tl.arange(0, BQ)
    rmask = rows < count
    cell = tl.load(perm + b * L + start + rows, mask=rmask, other=0).to(tl.int64)
    return cell, rmask


@triton.jit
def _fwd_kernel(Q, K, V, O, LSE, perm, ntiles, tile_start, tile_cnt, ent_cell, lo_ptr, hi_ptr,
                sq_b, sq_h, sq_s, sk_b, sk_h, sk_s, sv_b, sv_h, sv_s, scale, H, L, ENT,
                R: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr, D: tl.constexpr, IEEE: tl.constexpr):
    t = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    if t >= tl.load(ntiles + b):
        return
    cell, rmask = _tile_rows(perm, ntiles, tile_start, tile_cnt, b, t, L, BQ)
    dd = tl.arange(0, D)
    b64 = b.to(tl.int64)
    q = tl.load(Q + b64 * sq_b + h * sq_h + cell[:, None] * sq_s + dd[None, :], mask=rmask[:, None], other=0.0)
    m_i = tl.full([BQ], float('-inf'), tl.float32)
    l_i = tl.zeros([BQ], tl.float32)
    acc = tl.zeros([BQ, D], tl.float32)
    qk_scale = scale * 1.4426950408889634
    for r in tl.static_range(R):
        lo = tl.load(lo_ptr + (b * L + t) * R + r)
        hi = tl.load(hi_ptr + (b * L + t) * R + r)
        for e0 in range(lo, hi, BK):
            e = e0 + tl.arange(0, BK)
            em = e < hi
            kc = tl.load(ent_cell + b * ENT + e, mask=em, other=0).to(tl.int64)
            k = tl.load(K + b64 * sk_b + h * sk_h + kc[:, None] * sk_s + dd[None, :], mask=em[:, None], other=0.0)
            v = tl.load(V + b64 * sv_b + h * sv_h + kc[:, None] * sv_s + dd[None, :], mask=em[:, None], other=0.0)
            if IEEE:
                s = tl.dot(q, tl.trans(k), input_precision='ieee') * qk_scale
            else:
                s = tl.dot(q, tl.trans(k)) * qk_scale
            s = tl.where(em[None, :], s, float('-inf'))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float('-inf'), 0.0, m_new)
            p = tl.exp2(s - m_safe[:, None])
            alpha = tl.exp2(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, 1)
            if IEEE:
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision='ieee')
            else:
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
    out = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
    tl.store(O + (b64 * H + h) * L * D + cell[:, None] * D + dd[None, :], out.to(O.dtype.element_ty), mask=rmask[:, None])
    lse = tl.where(l_i == 0.0, float('inf'), m_i + tl.log2(tl.where(l_i == 0.0, 1.0, l_i)))
    tl.store(LSE + (b64 * H + h) * L + cell, lse, mask=rmask)


@triton.jit
def _dq_kernel(Q, K, V, DO, LSE, DELTA, DQ, perm, ntiles, tile_start, tile_cnt, ent_cell, lo_ptr, hi_ptr,
               sq_b, sq_h, sq_s, sk_b, sk_h, sk_s, sv_b, sv_h, sv_s, scale, H, L, ENT,
               R: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr, D: tl.constexpr, IEEE: tl.constexpr):
    t = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    if t >= tl.load(ntiles + b):
        return
    cell, rmask = _tile_rows(perm, ntiles, tile_start, tile_cnt, b, t, L, BQ)
    dd = tl.arange(0, D)
    b64 = b.to(tl.int64)
    q = tl.load(Q + b64 * sq_b + h * sq_h + cell[:, None] * sq_s + dd[None, :], mask=rmask[:, None], other=0.0)
    do = tl.load(DO + (b64 * H + h) * L * D + cell[:, None] * D + dd[None, :], mask=rmask[:, None], other=0.0)
    lse = tl.load(LSE + (b64 * H + h) * L + cell, mask=rmask, other=0.0)
    delta = tl.load(DELTA + (b64 * H + h) * L + cell, mask=rmask, other=0.0)
    dq = tl.zeros([BQ, D], tl.float32)
    qk_scale = scale * 1.4426950408889634
    for r in tl.static_range(R):
        lo = tl.load(lo_ptr + (b * L + t) * R + r)
        hi = tl.load(hi_ptr + (b * L + t) * R + r)
        for e0 in range(lo, hi, BK):
            e = e0 + tl.arange(0, BK)
            em = e < hi
            kc = tl.load(ent_cell + b * ENT + e, mask=em, other=0).to(tl.int64)
            k = tl.load(K + b64 * sk_b + h * sk_h + kc[:, None] * sk_s + dd[None, :], mask=em[:, None], other=0.0)
            v = tl.load(V + b64 * sv_b + h * sv_h + kc[:, None] * sv_s + dd[None, :], mask=em[:, None], other=0.0)
            if IEEE:
                s = tl.dot(q, tl.trans(k), input_precision='ieee') * qk_scale
                dp = tl.dot(do, tl.trans(v), input_precision='ieee')
            else:
                s = tl.dot(q, tl.trans(k)) * qk_scale
                dp = tl.dot(do, tl.trans(v))
            p = tl.where(em[None, :], tl.exp2(s - lse[:, None]), 0.0)
            ds = p * (dp - delta[:, None])
            if IEEE:
                dq += tl.dot(ds.to(k.dtype), k, input_precision='ieee')
            else:
                dq += tl.dot(ds.to(k.dtype), k)
    tl.store(DQ + (b64 * H + h) * L * D + cell[:, None] * D + dd[None, :], (dq * scale).to(DQ.dtype.element_ty),
             mask=rmask[:, None])


@triton.jit
def _dkdv_kernel(Q, K, V, DO, LSE, DELTA, DK, DV, perm, ntiles, tile_start, tile_cnt, ent_cell, lo_ptr, hi_ptr, skip_ptr,
                 sq_b, sq_h, sq_s, sk_b, sk_h, sk_s, sv_b, sv_h, sv_s, scale, H, L, ENT,
                 R: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr, D: tl.constexpr, SKIP: tl.constexpr,
                 IEEE: tl.constexpr):
    t = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    if t >= tl.load(ntiles + b):
        return
    cell, rmask = _tile_rows(perm, ntiles, tile_start, tile_cnt, b, t, L, BQ)
    dd = tl.arange(0, D)
    b64 = b.to(tl.int64)
    k = tl.load(K + b64 * sk_b + h * sk_h + cell[:, None] * sk_s + dd[None, :], mask=rmask[:, None], other=0.0)
    v = tl.load(V + b64 * sv_b + h * sv_h + cell[:, None] * sv_s + dd[None, :], mask=rmask[:, None], other=0.0)
    dk = tl.zeros([BQ, D], tl.float32)
    dv = tl.zeros([BQ, D], tl.float32)
    qk_scale = scale * 1.4426950408889634
    for r in tl.static_range(R):
        lo = tl.load(lo_ptr + (b * L + t) * R + r)
        hi = tl.load(hi_ptr + (b * L + t) * R + r)
        for e0 in range(lo, hi, BK):
            e = e0 + tl.arange(0, BK)
            em = e < hi
            if SKIP:
                em = em & (tl.load(skip_ptr + b * ENT + e, mask=em, other=1) == 0)
            qc = tl.load(ent_cell + b * ENT + e, mask=em, other=0).to(tl.int64)
            q = tl.load(Q + b64 * sq_b + h * sq_h + qc[:, None] * sq_s + dd[None, :], mask=em[:, None], other=0.0)
            do = tl.load(DO + (b64 * H + h) * L * D + qc[:, None] * D + dd[None, :], mask=em[:, None], other=0.0)
            lse = tl.load(LSE + (b64 * H + h) * L + qc, mask=em, other=0.0)
            delta = tl.load(DELTA + (b64 * H + h) * L + qc, mask=em, other=0.0)
            if IEEE:
                st = tl.dot(k, tl.trans(q), input_precision='ieee') * qk_scale
                dpt = tl.dot(v, tl.trans(do), input_precision='ieee')
            else:
                st = tl.dot(k, tl.trans(q)) * qk_scale
                dpt = tl.dot(v, tl.trans(do))
            pt = tl.where(em[None, :], tl.exp2(st - lse[None, :]), 0.0)
            dst = pt * (dpt - delta[None, :])
            if IEEE:
                dv += tl.dot(pt.to(do.dtype), do, input_precision='ieee')
                dk += tl.dot(dst.to(q.dtype), q, input_precision='ieee')
            else:
                dv += tl.dot(pt.to(do.dtype), do)
                dk += tl.dot(dst.to(q.dtype), q)
    base = (b64 * H + h) * L * D
    tl.store(DK + base + cell[:, None] * D + dd[None, :], (dk * scale).to(DK.dtype.element_ty), mask=rmask[:, None])
    tl.store(DV + base + cell[:, None] * D + dd[None, :], dv.to(DV.dtype.element_ty), mask=rmask[:, None])


def _strides(x):
    return x.stride(0), x.stride(1), x.stride(2)


def _forward_impl(q, k, v, half):
    B, H, L, D = q.shape
    o = torch.zeros((B, H, L, D), device=q.device, dtype=q.dtype)
    lse = torch.empty((B, H, L), device=q.device, dtype=torch.float32)
    perm, ntiles, start, count, ent, lo, hi, _ = half
    _fwd_kernel[(L, B * H)](q, k, v, o, lse, perm, ntiles, start, count, ent, lo, hi,
                            *_strides(q), *_strides(k), *_strides(v), D ** -0.5, H, L, ent.shape[1],
                            R=lo.shape[-1], BQ=TILE, BK=BLOCK_KEYS, D=D, IEEE=q.dtype == torch.float32, num_warps=2)
    return o, lse


def _backward_impl(do, q, k, v, o, lse, fwd, bwd, skip_bwd):
    B, H, L, D = q.shape
    do = do.contiguous()
    delta = (do.float() * o.float()).sum(-1)
    dq = torch.zeros((B, H, L, D), device=q.device, dtype=q.dtype)
    dk = torch.zeros_like(dq)
    dv = torch.zeros_like(dq)
    ieee = q.dtype == torch.float32
    scale = D ** -0.5
    perm, ntiles, start, count, ent, lo, hi, _ = fwd
    _dq_kernel[(L, B * H)](q, k, v, do, lse, delta, dq, perm, ntiles, start, count, ent, lo, hi,
                           *_strides(q), *_strides(k), *_strides(v), scale, H, L, ent.shape[1],
                           R=lo.shape[-1], BQ=TILE, BK=BLOCK_KEYS, D=D, IEEE=ieee, num_warps=2)
    perm, ntiles, start, count, ent, lo, hi, skip = bwd
    _dkdv_kernel[(L, B * H)](q, k, v, do, lse, delta, dk, dv, perm, ntiles, start, count, ent, lo, hi, skip,
                             *_strides(q), *_strides(k), *_strides(v), scale, H, L, ent.shape[1],
                             R=lo.shape[-1], BQ=TILE, BK=BLOCK_KEYS, D=D, SKIP=skip_bwd, IEEE=ieee, num_warps=2)
    return dq, dk, dv


# Custom ops keep the Triton launches opaque to torch.compile and CUDA-graph capture; plan tensors are plain arguments.
@torch.library.custom_op('openrfm::sparse_attend', mutates_args=())
def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, fwd: list[torch.Tensor], bwd: list[torch.Tensor],
            skip_bwd: bool) -> tuple[torch.Tensor, torch.Tensor]:
    return _forward_impl(q, k, v, HalfPlan(*fwd))


@_attend.register_fake
def _(q, k, v, fwd, bwd, skip_bwd):
    return torch.empty(q.shape, device=q.device, dtype=q.dtype), q.new_empty(q.shape[:3], dtype=torch.float32)


@torch.library.custom_op('openrfm::sparse_attend_backward', mutates_args=())
def _attend_backward(do: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                     lse: torch.Tensor, fwd: list[torch.Tensor], bwd: list[torch.Tensor], skip_bwd: bool
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _backward_impl(do, q, k, v, o, lse, HalfPlan(*fwd), HalfPlan(*bwd), skip_bwd)


@_attend_backward.register_fake
def _(do, q, k, v, o, lse, fwd, bwd, skip_bwd):
    return tuple(torch.empty(q.shape, device=q.device, dtype=q.dtype) for _ in range(3))


def _setup(ctx, inputs, output):
    q, k, v, fwd, bwd, skip_bwd = inputs
    ctx.save_for_backward(q, k, v, *output, *fwd, *bwd)
    ctx.skip_bwd = skip_bwd


def _backward(ctx, do, _dlse):
    q, k, v, o, lse, *plan = ctx.saved_tensors
    dq, dk, dv = _attend_backward(do, q, k, v, o, lse, plan[:8], plan[8:], ctx.skip_bwd)
    return dq, dk, dv, [None] * 8, [None] * 8, None


_attend.register_autograd(_backward, setup_context=_setup)
