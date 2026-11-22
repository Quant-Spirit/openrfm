"""Equal-share balance of the backbone gradient between the masked-cell objective and the consumer objective."""
import torch


class GradientBalance:
    """Equal-share balance of the backbone gradient between the masked-cell and consumer objectives."""
    # g = g_mask + ramp * (|g_mask| / |g_icl|) * g_icl with g_mask = g - g_icl, norms joint over every trainable
    # tensor except the handoff (whose gradient is untouched); g_icl is isolated by snapshots around the consumer pass.

    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.icl = self.snapshot = None

    def window_start(self) -> None:
        self.icl = self.snapshot = None

    def before_coupling(self) -> None:
        self.snapshot = [None if p.grad is None else p.grad.detach().clone() for p in self.parameters]

    def after_coupling(self) -> None:
        with torch.no_grad():
            if self.icl is None:
                self.icl = [torch.zeros_like(p, dtype=torch.float32) for p in self.parameters]
            for acc, p, before in zip(self.icl, self.parameters, self.snapshot):
                if p.grad is not None:
                    delta = p.grad.detach().float()
                    acc.add_(delta if before is None else delta - before.float())
        self.snapshot = None

    def finish(self, ramp: float) -> None:
        if self.icl is None or ramp <= 0:
            self.icl = None
            return
        with torch.no_grad():
            grads = [torch.zeros_like(p, dtype=torch.float32) if p.grad is None else p.grad.detach().float()
                     for p in self.parameters]
            masks = [g - i for g, i in zip(grads, self.icl)]
            n_mask = torch.sqrt(sum((m * m).sum() for m in masks))
            n_icl = torch.sqrt(sum((i * i).sum() for i in self.icl))
            coef = torch.where(n_icl > 0, float(ramp) * n_mask / (n_icl + 1e-30), torch.zeros_like(n_icl))
            for p, m, i in zip(self.parameters, masks, self.icl):
                if p.grad is not None:
                    p.grad.copy_((m + coef * i).to(p.grad.dtype))
        self.icl = None
