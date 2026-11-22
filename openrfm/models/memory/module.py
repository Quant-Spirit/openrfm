"""The memory module: one learned cross-attention read over the frozen backbone's context states and the focal row's
temporal history.

Tokens for one request (a focal row at a cutoff):
  - every live cell of its backbone context at every depth: LN(state) plus a learned depth and source embedding;
  - every cell of the `rows` nearest visible rows of its temporal foreign-key neighbourhood (`hops` hops; a row is
    visible when its timestamp is <= the cutoff). Each cell goes through the backbone's frozen cell encoders; time
    enters through the backbone's context-time encoding of (row time, cutoff) plus a learned reach-class embedding.
`queries` learned queries, one head each, read all tokens; each query exports its softmax-weighted value and its
log-partition, 2*queries columns per request."""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from openrfm.models.memory.config import MemoryConfig
from openrfm.models.memory.history import context_children, dense_cells, encode_cells, reach

# History rows are encoded in fixed blocks of this many rows, so every GEMM has one shape and a row's encoding does
# not depend on which other rows are read with it.
BLOCK = 256
# Context role of a history token in the backbone's context-time encoding.
HISTORY_ROLE = 3


def pad_to(x, size):
    """`x` padded to `size` rows by repeating its last row."""
    return torch.cat((x, x[-1:].expand(size-len(x), *x.shape[1:]))) if len(x) < size else x


class MemoryModule(nn.Module):
    """Learned read over backbone context states and history cells; `history.py` describes the history world."""

    def __init__(self, config=MemoryConfig()):
        super().__init__()
        self.config = c = config
        d, Q, h = c.d_model, c.queries, c.head_width
        self.norm = nn.LayerNorm(d)
        self.depth = nn.Parameter(torch.randn(c.depths, d)*.02)
        self.source = nn.Parameter(torch.randn(2, d)*.02)
        self.reach = nn.Parameter(torch.randn(2*c.hops, d)*.02)
        self.query = nn.Parameter(torch.randn(Q, h))
        self.key = nn.Linear(d, Q*h)
        self.value = nn.Linear(d, Q)
        self.cell_norm = nn.LayerNorm(d)
        # The backbone is used, never trained, here; it is kept out of the module's parameters and state.
        self.__dict__['rt'] = None
        self._graph = (None, None, 0)

    def bind(self, rt):
        """Use the frozen backbone `rt`'s cell encoders and context-time encoding for history cells."""
        self.__dict__['rt'] = rt
        return self

    def graph(self, world):
        """(children index, widest row) of `world`, computed once per world."""
        if self._graph[0] is not world.parents:
            self._graph = (world.parents, context_children(world), int(world.ptr.diff().max()))
        return self._graph[1:]

    def history(self, world, rows, cutoffs):
        """[n, rows] history rows of each request, their reach class and presence (`history.reach`)."""
        children, _ = self.graph(world)
        return reach(world, children, rows, cutoffs, self.config.rows, self.config.hops)

    def row_cells(self, world, unique):
        """[U, W, d] LN(cell encoding) of each row's cells and [U, W] presence, W = the world's widest row."""
        _, width = self.graph(world)
        out, live = [], []
        for a in range(0, len(unique), BLOCK):
            ids = pad_to(unique[a:a+BLOCK], BLOCK)
            cells = dense_cells(world, ids, width).to(ids.device)
            k = min(BLOCK, len(unique)-a)
            out.append(self.cell_norm(encode_cells(self.rt, world, cells.flatten()).view(BLOCK, width, -1))[:k])
            live.append((cells >= 0)[:k])
        return torch.cat(out), torch.cat(live)

    def history_tokens(self, world, rows, cutoffs):
        """[n, R*W, d] history cell tokens and their [n, R*W] mask."""
        found, kind, keep = self.history(world, rows, cutoffs)
        unique, where = found.clamp_min(0).unique(return_inverse=True)
        with torch.no_grad():
            requests = torch.stack((torch.zeros_like(cutoffs), cutoffs), 1)
            time = self.rt.context_time_role(dict(
                context_role=torch.full_like(found, HISTORY_ROLE), timestamps=world.times[found.clamp_min(0)],
                sampler_requests=requests, is_padding=~keep)).float()
        x, present = self.row_cells(world, unique)
        tokens = self.norm(x[where]+(time+self.reach[kind])[:, :, None])+self.source[1]
        mask = present[where] & keep[:, :, None]
        return tokens.flatten(1, 2), mask.flatten(1)

    def read(self, states, live, history, kept):
        """[c, 2Q] softmax-weighted values and log-partitions of the Q queries for one chunk of requests."""
        c = self.config
        parts, masks = [], []
        if states is not None:
            parts.append((self.norm(states.float())+self.depth+self.source[0]).flatten(1, 2))
            masks.append(live[:, :, None].expand(-1, -1, states.shape[2]).flatten(1))
        if history is not None:
            parts.append(history)
            masks.append(kept)
        tokens, mask = torch.cat(parts, 1), torch.cat(masks, 1)
        # Per-head scores q_h . (W_h t + b_h) / sqrt(h), with the head's key map folded into its query.
        weight = self.key.weight.view(c.queries, c.head_width, -1)
        bias = self.key.bias.view(c.queries, c.head_width)
        a = torch.einsum('qh,qhd->qd', self.query, weight)
        s = ((tokens @ a.T)+(self.query*bias).sum(1))/math.sqrt(c.head_width)
        s = s.masked_fill(~mask[..., None], -torch.inf)
        lse = s.logsumexp(1)
        p = (s-lse[:, None]).exp().nan_to_num(0.)
        mean = (p*(tokens @ self.value.weight.T+self.value.bias)).sum(1)
        # A request with no token at all exports zeros.
        return torch.cat((mean, lse.masked_fill(~mask.any(1, keepdim=True), 0.)), 1)

    def columns(self, world, rows, cutoffs, context=None):
        """[n, 2Q] memory columns of focal rows `rows` at `cutoffs`.

        Requests are read in chunks padded to `config.chunk`, so a request's columns do not depend on its chunk.
        """
        size = self.config.chunk
        out = []
        for a in range(0, len(rows), size):
            b = slice(a, a+size)
            history, kept = (None, None) if world is None else self.history_tokens(
                world, pad_to(rows[b], size), pad_to(cutoffs[b], size))
            states = (None, None) if context is None else (pad_to(context[0][b], size), pad_to(context[1][b], size))
            if torch.is_grad_enabled():
                x = checkpoint(self.read, *states, history, kept, use_reentrant=False)
            else:
                x = self.read(*states, history, kept)
            out.append(x[:len(rows[b])])
        return torch.cat(out)
