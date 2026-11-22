"""Muon for the hidden matrices, AdamW for every other parameter, and the learning-rate schedule."""
import math

import torch


def zeropower_newton_schulz(update: torch.Tensor, steps: int) -> torch.Tensor:
    """Orthogonalize each trailing matrix of ``update`` by the quintic Newton-Schulz iteration (in bf16)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = update.bfloat16()
    transposed = update.size(-2) > update.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.mT
    return x.to(update.dtype)


class Muon(torch.optim.Optimizer):
    """Muon (orthogonalized Nesterov momentum) and AdamW with cautious weight decay, batched by shape or step."""

    def __init__(self, groups, *, momentum, nesterov, ns_steps, adam_betas, adam_eps):
        super().__init__(groups, dict(momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                                      adam_betas=adam_betas, adam_eps=adam_eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for (use_muon, _, _, shape, step), params in self._buckets(group).items():
                if use_muon:
                    self._muon(group, params, shape)
                else:
                    self._adamw(group, params, step)

    def _buckets(self, group):
        buckets = {}
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if group["use_muon"]:
                state.setdefault("muon_buf", torch.zeros_like(p))
                step = 0
            else:
                if "adam_step" not in state:
                    state.update(adam_step=0, adam_exp_avg=torch.zeros_like(p), adam_exp_avg_sq=torch.zeros_like(p))
                state["adam_step"] += 1
                step = state["adam_step"]
            key = (group["use_muon"], p.device, p.dtype, tuple(p.shape) if group["use_muon"] else (), step)
            buckets.setdefault(key, []).append(p)
        return buckets

    def _muon(self, group, params, shape):
        lr, wd, momentum = group["lr"], group["weight_decay"], group["momentum"]
        grads = [p.grad for p in params]
        buffers = [self.state[p]["muon_buf"] for p in params]
        torch._foreach_mul_(buffers, momentum)
        torch._foreach_add_(buffers, grads)
        updates = torch._foreach_add(grads, buffers, alpha=momentum) if group["nesterov"] else buffers
        update = zeropower_newton_schulz(torch.stack(tuple(updates)), group["ns_steps"])
        update.mul_(0.2 * float(max(shape[-2:])) ** 0.5)
        if wd > 0:
            same_sign = (update.sign() == torch.stack(params).sign()).to(update.dtype)
            torch._foreach_mul_(params, list((1.0 - lr * wd * same_sign).unbind()))
        torch._foreach_add_(params, list(update.unbind()), alpha=-lr)

    def _adamw(self, group, params, step):
        lr, wd = group["lr"], group["weight_decay"]
        beta1, beta2 = group["adam_betas"]
        grads = [p.grad for p in params]
        means = [self.state[p]["adam_exp_avg"] for p in params]
        variances = [self.state[p]["adam_exp_avg_sq"] for p in params]
        torch._foreach_mul_(means, beta1)
        torch._foreach_add_(means, grads, alpha=1 - beta1)
        torch._foreach_mul_(variances, beta2)
        torch._foreach_addcmul_(variances, grads, grads, value=1 - beta2)
        denoms = torch._foreach_sqrt(variances)
        torch._foreach_div_(denoms, (1 - beta2 ** step) ** 0.5)
        torch._foreach_add_(denoms, group["adam_eps"])
        if wd > 0:
            direction = torch.cat([value.reshape(-1) for value in torch._foreach_div(means, denoms)])
            flat = torch.cat([p.reshape(-1) for p in params])
            factors = 1.0 - lr * wd * (direction.sign() == flat.sign()).to(params[0].dtype)
            torch._foreach_mul_(params, [f.view_as(p) for f, p in zip(factors.split([p.numel() for p in params]), params)])
        torch._foreach_addcdiv_(params, means, denoms, value=-lr / (1 - beta1 ** step))


def is_hidden_matrix(name: str, parameter: torch.nn.Parameter) -> bool:
    """Muon parameters: 2-D weights of the backbone's hidden layers and of the column-slot head."""
    if parameter.ndim != 2 or name.endswith(".bias") or "emb" in name.lower() or "norm" in name.lower():
        return False
    return not name.startswith(("enc_dict.", "dec_dict.", "structural_features.", "rt_handoff."))


def build_optimizer(named, config) -> tuple[Muon, torch.optim.lr_scheduler.LambdaLR]:
    """Muon/AdamW over ``named`` (in order) with linear warmup, then cosine decay from ``decay_start_step``."""
    muon = [p for name, p in named if is_hidden_matrix(name, p)]
    aux = [p for name, p in named if not is_hidden_matrix(name, p)]
    o = config["optimizer"]
    optimizer = Muon([dict(params=muon, use_muon=True, lr=o["muon_lr"], weight_decay=o["weight_decay"]),
                      dict(params=aux, use_muon=False, lr=o["aux_lr"], weight_decay=o["weight_decay"])],
                     momentum=o["momentum"], nesterov=o["nesterov"], ns_steps=o["ns_steps"],
                     adam_betas=tuple(o["aux_betas"]), adam_eps=o["aux_eps"])
    total, warmup, start, floor = config["optimizer_steps"], o["warmup_steps"], o["decay_start_step"], o["min_lr_ratio"]

    def factor(step: int) -> float:
        if step < warmup:
            return min(1.0, float(step + 1) / float(warmup))
        progress = max(0.0, min(1.0, (step - start) / (total - start)))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
