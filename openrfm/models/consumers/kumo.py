"""Frozen Kumo-Tabular consumer (package `structured-data-models`, weights `nvidia/Kumo-Tabular`)."""
import torch
import torch.nn.functional as F
from torch import nn

# Kumo reads at most this many columns.
MAX_COLUMNS = 500


def load_kumo(task_type, size='small', device='cuda'):
    """The pretrained, frozen `sdm.models.KumoTabular` ('classification' or 'regression')."""
    from sdm.models import KumoTabular
    model = KumoTabular(task=task_type, size=size, pretrained=True, device=device)
    return model.eval().requires_grad_(False)


def soft_clip(x, threshold=4.):
    """Identity inside +-threshold, logarithmic beyond it (the package's ClipSigma)."""
    over = (x.abs()-threshold).clamp_min(0)
    return torch.where(over > 0, x.sign()*(threshold+over.log1p()), x)


def standardized(x, k):
    """Columns of `x` standardized on its first `k` rows (the support) and soft-clipped; constant columns dropped."""
    x = x.float()
    mean, std = x[:k].mean(0), x[:k].std(0, unbiased=False)
    return soft_clip(((x-mean)/std.clamp_min(1e-6)).clamp(-100, 100))[:, std > 1e-6]


class KumoHead(nn.Module):
    """The frozen Kumo network as a differentiable training consumer."""

    def __init__(self, task_type, size='small', device='cuda'):
        super().__init__()
        self.task_type = task_type
        self.net = next(iter(load_kumo(task_type, size, device).models.values()))

    def join(self, x, k, seed):
        """Consumer input over support then query rows, standardized on the support, at most `MAX_COLUMNS` columns."""
        x = standardized(x, k)
        if x.shape[1] > MAX_COLUMNS:
            g = torch.Generator(device=x.device).manual_seed(seed)
            x = x[:, torch.randperm(x.shape[1], generator=g, device=x.device)[:MAX_COLUMNS]]
        return x

    def forward(self, x, y_support):
        """Query outputs: logits [Q, classes] for codes 0..C-1, or 999 quantiles of the support-standardized target."""
        if self.task_type == 'classification':
            y = y_support.long()
        else:
            y = (y_support.float()-y_support.float().mean())/y_support.float().std().clamp_min(1e-8)
        mask = torch.zeros((1, x.shape[1]), dtype=torch.bool, device=x.device)
        # The consumer runs in FP32 even under an enclosing autocast.
        with torch.autocast('cuda', enabled=False):
            out = self.net(x=x[None], y=y[None], categorical_mask=mask)[0].float()
        if self.task_type == 'classification':
            return out[:, :max(2, int(y.max())+1)]
        return out

    def loss(self, out, y_support, y_query):
        """Cross-entropy, or the pinball loss over the quantile grid of the support-standardized target."""
        if self.task_type == 'classification':
            return F.cross_entropy(out, y_query.long())
        mean, std = y_support.float().mean(), y_support.float().std().clamp_min(1e-8)
        target = ((y_query.float()-mean)/std)[:, None]
        levels = torch.arange(1, out.shape[1]+1, device=out.device, dtype=out.dtype)/(out.shape[1]+1)
        error = target-out
        return torch.maximum(levels*error, (levels-1)*error).mean()


class KumoPredictor:
    """In-context prediction with the package's recipe; a fixed device seed before construction fixes it."""

    def __init__(self, task_type, size='small', estimators=1, device='cuda'):
        self.task_type, self.estimators, self.device = task_type, estimators, device
        self.model = load_kumo(task_type, size, device)

    @torch.no_grad()
    def fit(self, x, y):
        """Fit on support features [K, D] and labels [K] (class codes, or real targets z-scored on the support)."""
        y = y.to(self.device)
        if self.task_type == 'classification':
            self.classes = max(int(y.max())+1, 2)
            y = y.long()[:, None]
        else:
            self.mean, self.std = y.float().mean(), y.float().std().clamp_min(1e-8)
            y = ((y.float()-self.mean)/self.std)[:, None]
        self.model.fit(x=x.float().to(self.device), y=y, num_estimators=self.estimators)
        return self

    @torch.no_grad()
    def predict(self, x):
        """Class log-probabilities [Q, classes], or the predictive quantiles [Q, 999] in the target's units."""
        from sdm import Stype
        output = self.model.predict(x.float().to(self.device))
        values = output.numerical.float()
        while values.ndim > 2 and values.shape[0] == 1:
            values = values.squeeze(0)
        if self.task_type == 'regression':
            return values*self.std+self.mean
        # The recipe shuffles the class columns per call; the output's column names carry the true class codes.
        codes = [int(name) for name in output.columns[Stype.numerical]]
        p = values.new_zeros((len(values), self.classes))
        p[:, codes] = values
        p = p.clamp_min(torch.finfo(p.dtype).tiny)
        return (p/p.sum(-1, keepdim=True)).log()

    def clear(self):
        """Release the support cache."""
        self.model.clear()
