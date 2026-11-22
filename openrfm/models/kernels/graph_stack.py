"""The sparse attention plan and the whole block stack as one CUDA-graphed forward and backward per shape class.

All classes share one memory pool, so a class's forward must be followed by its backward before any
other class is replayed.
"""
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from .sparse_attention import build_device_plan

BATCH_KEYS = ("node_idxs", "f2p_nbr_idxs", "table_name_idxs", "col_name_idxs", "is_padding", "is_targets",
              "is_task_nodes", "timestamps", "is_support_seed", "support_timeless", "support_label_available_at",
              "episode_donor_mask")


class _Class:
    """One captured shape class: static inputs, forward graph, backward graph."""

    def __init__(self, stack, x, tensors, spec, is_tensor, constants):
        self.stack, self.spec, self.is_tensor, self.constants = stack, spec, is_tensor, constants
        self.params = [p for p in stack.blocks.parameters() if p.requires_grad]
        self.blocks = torch.compile(self.run_blocks, dynamic=False) if stack.compile else self.run_blocks
        self.x = x.detach().clone().requires_grad_()
        self.tensors = [t.clone() for t in tensors]
        self.grad_out = torch.zeros_like(x)
        # A cached autocast cast can keep the warm-up stream's autograd history; capture makes its own casts.
        with torch.autocast('cuda', enabled=torch.is_autocast_enabled('cuda'),
                            dtype=torch.get_autocast_dtype('cuda'), cache_enabled=False):
            warmup = torch.cuda.Stream()
            warmup.wait_stream(torch.cuda.current_stream())
            with torch.enable_grad(), torch.cuda.stream(warmup):
                for _ in range(3):
                    out = self.run(self.x, *self.tensors)
                    torch.autograd.grad(out, (self.x, *self.params), self.grad_out, allow_unused=True)
            torch.cuda.current_stream().wait_stream(warmup)
            self.fwd, self.bwd = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.enable_grad():
                with torch.cuda.graph(self.fwd, pool=stack.pool):
                    self.out = self.run(self.x, *self.tensors)
                with torch.cuda.graph(self.bwd, pool=stack.pool):
                    grads = torch.autograd.grad(self.out, (self.x, *self.params), self.grad_out, allow_unused=True)
        self.dx, self.param_grads = grads[0], grads[1:]

    def run_blocks(self, x, plans, batch, rows):
        for block in self.stack.blocks:
            x = block(x, plans, batch, rows, True)
        return x

    def run(self, x, *tensors):
        tensor_iter, const_iter = iter(tensors), iter(self.constants)
        leaves = [next(tensor_iter) if is_tensor else next(const_iter) for is_tensor in self.is_tensor]
        batch, rows = tree_unflatten(leaves, self.spec)
        return self.blocks(x, build_device_plan(batch), batch, rows)

    def accumulate(self):
        live = [(p, g) for p, g in zip(self.params, self.param_grads) if g is not None]
        old = [(p.grad, g) for p, g in live if p.grad is not None]
        new = [(p, g) for p, g in live if p.grad is None]
        if old:
            torch._foreach_add_([a for a, _ in old], [g for _, g in old])
        if new:
            for (p, _), grad in zip(new, torch._foreach_mul([g for _, g in new], 1.0)):
                p.grad = grad


class _Replay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graphed, x, *tensors):
        graphed.x.copy_(x)
        for static, t in zip(graphed.tensors, tensors):
            static.copy_(t)
        graphed.fwd.replay()
        ctx.graphed = graphed
        return graphed.out.clone()

    @staticmethod
    def backward(ctx, grad):
        graphed = ctx.graphed
        graphed.grad_out.copy_(grad)
        graphed.bwd.replay()
        graphed.accumulate()
        return (None, graphed.dx.clone(), *[None] * len(graphed.tensors))


class GraphedStack:
    """Callable ``(x, batch, rows) -> x`` replaying the captured class of the input shapes (support axis on)."""

    def __init__(self, blocks, compile: bool = False, max_classes: int | None = None):
        self.blocks, self.compile, self.max_classes = blocks, compile, max_classes
        self.cache = {}
        self.pool = torch.cuda.graph_pool_handle()

    def __call__(self, x, batch, rows):
        batch = {k: batch[k] for k in BATCH_KEYS if k in batch}
        leaves, spec = tree_flatten((batch, rows))
        is_tensor = tuple(torch.is_tensor(v) for v in leaves)
        tensors = [v for v in leaves if torch.is_tensor(v)]
        constants = tuple(v for v in leaves if not torch.is_tensor(v))
        key = (tuple((t.shape, t.dtype) for t in [x, *tensors]), constants, str(spec))
        if key not in self.cache:
            if self.max_classes is not None and len(self.cache) >= self.max_classes:
                raise RuntimeError(f'graph_stack shape-class limit {self.max_classes} reached')
            self.cache[key] = _Class(self, x, tensors, spec, is_tensor, constants)
            torch.cuda.empty_cache()  # keep capture transients out of the allocator cache before the next class
        return _Replay.apply(self.cache[key], x, *tensors)
