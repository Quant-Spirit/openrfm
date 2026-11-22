"""Recipe R3: R2 plus a residual head trained per task on the support rows; the backbone, memory module and Kumo-Tabular
stay frozen.

The head h reads a row's exported features. Classification: logit of the positive class = Kumo logit + h(x).
Regression: point = Kumo point + s*h(x) in the transformed-target space, s the std of the transformed support labels of
the draw. Its last layer is zero-initialised, so before training R3 equals R2.

Training: episodes over the fit rows (the support rows outside a newest-cutoff holdout, mature before it). An episode
draws one of their cutoffs, at most `queries` rows at it as queries and at most `support` rows mature by it as the
in-context support (a random split when the rows have a single cutoff); frozen Kumo predicts the queries and the loss
of Kumo + h is cross-entropy (classification) or the pinball loss at the median or over all quantiles (regression).
AdamW (weight decay 0.01), gradient norm clipped at 1, 20 warm-up steps and a cosine schedule to 0.1x at step 600;
the checkpoint step (including step 0) is selected on the temporal holdout of the support set."""
import math

import numpy as np
import torch

from openrfm.infer import predict

WEIGHT_DECAY, CLIP, WARMUP, TOTAL, FLOOR = 0.01, 1.0, 20, 600, 0.1
HOLDOUT = 0.1
CHECKPOINTS = (5, 10, 25, 50, 100, 200, 400)


class ColumnSpline(torch.nn.Module):
    """Per column a zero-initialised affine map plus a linear spline on 9 knots over [-4, 4], added to the value."""

    def __init__(self, d):
        super().__init__()
        self.register_buffer('knots', torch.linspace(-4, 4, 9))
        self.affine = torch.nn.Parameter(torch.zeros(2, d))
        self.values = torch.nn.Parameter(torch.zeros(d, 9))

    def forward(self, x):
        c = x.clamp(self.knots[0], self.knots[-1])
        left = (torch.searchsorted(self.knots, c.contiguous(), right=True)-1).clamp(0, 7)
        w = (c-self.knots[left])/(self.knots[1]-self.knots[0])
        cols = torch.arange(x.shape[1], device=x.device).expand_as(left)
        return x+x*self.affine[0]+self.affine[1]+torch.lerp(self.values[cols, left], self.values[cols, left+1], w)


class ResidualHead(torch.nn.Module):
    """Columns standardized on the fit rows (constant ones dropped), ColumnSpline, Linear(d, hidden), ReLU and a
    zero-initialised Linear(hidden, 1): one scalar per row."""

    def __init__(self, x_fit, hidden=256):
        super().__init__()
        mean, std = x_fit.mean(0), x_fit.std(0, unbiased=False)
        live = std > 1e-6
        self.register_buffer('live', live)
        self.register_buffer('mean', mean[live])
        self.register_buffer('std', std[live])
        d = int(live.sum())
        self.cols = ColumnSpline(d)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
        torch.nn.init.zeros_(self.mlp[2].weight)
        torch.nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x):
        return self.mlp(self.cols((x[:, self.live]-self.mean)/self.std)).squeeze(-1)

    @torch.no_grad()
    def residual(self, x):
        """h of rows x (NumPy) as float64."""
        return torch.cat([self(torch.as_tensor(x[s:s+4096], dtype=torch.float32, device='cuda'))
                          for s in range(0, len(x), 4096)]).double().cpu().numpy()


def lr_mult(t):
    if t < WARMUP:
        return (t+1)/WARMUP
    p = min(1., (t-WARMUP)/(TOTAL-WARMUP))
    return FLOOR+(1-FLOOR)*0.5*(1+math.cos(math.pi*p))


def split_holdout(cut, is_val, horizon, frac=HOLDOUT):
    """(holdout, fit) masks over the support rows: the holdout is the newest cutoffs of the VAL rows (all rows when there
    are none) reaching `frac` of the rows, and the fit rows mature before its earliest cutoff. With a single cutoff the
    holdout is a seeded random `frac` of the pool and the fit rows are the rest."""
    pool = is_val if is_val.any() else np.ones(len(cut), bool)
    hold = np.zeros(len(cut), bool)
    if len(set(cut[pool].tolist())) == 1:
        n = min(int(pool.sum()), max(1, int(frac*len(cut))))
        hold[np.random.default_rng(0).choice(np.flatnonzero(pool), n, replace=False)] = True
        return hold, ~hold
    for c in sorted(set(cut[pool].tolist()), reverse=True):
        hold |= pool & (cut == c)
        if hold.sum() >= frac*len(cut):
            break
    return hold, ~hold & (cut+horizon <= cut[hold].min())


def transform(y, name):
    return np.log1p(y) if name == 'log1p' else np.sqrt(y) if name == 'sqrt' else y


def inverse(y, name):
    return np.expm1(y) if name == 'log1p' else np.square(np.maximum(y, 0)) if name == 'sqrt' else y


def base_outputs(kumo, sx, sy, qx, task_type):
    """Frozen Kumo on one episode: positive-class log-odds [Q] (classification) or the quantiles [Q, 999] of the
    support-scaled target (regression)."""
    torch.manual_seed(predict.CONSUMER_SEED)
    s, q = predict.standardize(torch.as_tensor(sx, device='cuda'), torch.as_tensor(qx, device='cuda'))
    dtype = torch.long if task_type == 'classification' else torch.float32
    kumo.fit(s, torch.as_tensor(sy, device='cuda', dtype=dtype))
    try:
        out = kumo.predict(q).float()
    finally:
        kumo.clear()
    return out[:, 1]-out[:, 0] if task_type == 'classification' else out


def holdout_scorer(kumo, sx, sy, y, fit_idx, hold_idx, task_type, cell):
    """Holdout metric function (higher is better) using frozen Kumo fitted on `fit_idx` only; None if unavailable."""
    if not len(hold_idx) or len(fit_idx) < 2 or np.ptp(y[fit_idx]) == 0:
        return None
    k = min(cell['K'], len(fit_idx))
    s_idx = fit_idx[predict.support_draw(y[fit_idx], k, 0, cell['selection'], task_type)]
    if len(s_idx) < 2 or np.ptp(y[s_idx]) == 0:
        return None
    torch.manual_seed(predict.CONSUMER_SEED)
    np.random.seed(predict.CONSUMER_SEED)
    s, q = predict.standardize(torch.as_tensor(sx[s_idx], device='cuda'), torch.as_tensor(sx[hold_idx], device='cuda'))
    chunk = 4096 if (task_type == 'classification' and k <= 16384) else 500
    if task_type == 'classification':
        kumo.fit(s, torch.as_tensor(y[s_idx], device='cuda', dtype=torch.long))
        parts = []
        try:
            for start in range(0, len(q), chunk):
                out = kumo.predict(q[start:start+chunk]).float()
                parts.append((out[:, 1]-out[:, 0]).double().cpu().numpy())
        finally:
            kumo.clear()
        base = np.concatenate(parts)
        hy = y[hold_idx]

        def score(h):
            logit = base+h.residual(sx[hold_idx])
            if len(np.unique(hy)) >= 2:
                return predict.metric(hy, 1/(1+np.exp(-np.clip(logit, -40, 40))), 'classification')
            return -float(np.mean(np.logaddexp(0, logit)-hy*logit))

        return score

    center, scale = float(np.median(y[s_idx])), max(float(np.std(y[s_idx])), 1e-6)
    kumo.fit(s, torch.as_tensor((y[s_idx]-center)/scale, device='cuda', dtype=torch.float32))
    parts = []
    try:
        for start in range(0, len(q), chunk):
            out = kumo.predict(q[start:start+chunk]).float()
            parts.append(out.median(-1).values.double().cpu().numpy())
    finally:
        kumo.clear()
    base = np.concatenate(parts)*scale+center
    hy_raw = np.asarray(sy[hold_idx], dtype=np.float64)

    def score(h):
        pred = inverse(base+scale*h.residual(sx[hold_idx]), cell['transform'])
        return -predict.metric(hy_raw, pred, 'regression')

    return score


def train(sx, sy, cut, is_val, horizon, task_type, cell, seed=0):
    """The residual head of a task, trained on episodes of the fit rows and selected on the temporal holdout."""
    spec = cell['head']
    hold, fit = split_holdout(cut, is_val, horizon)
    hold_idx, fit_idx = np.flatnonzero(hold), np.flatnonzero(fit)
    torch.manual_seed(seed)
    if not len(fit_idx):
        return ResidualHead(torch.as_tensor(sx, dtype=torch.float32, device='cuda'), spec.get('hidden', 256)).cuda().eval()
    if task_type == 'classification':
        y = np.asarray(sy, dtype=np.int64)
    else:
        y = transform(np.asarray(sy, dtype=np.float64), cell['transform'])
    kumo = predict.KumoPredictor(task_type, 'small')
    h = ResidualHead(torch.as_tensor(sx[fit_idx], dtype=torch.float32, device='cuda'), spec.get('hidden', 256)).cuda()
    params = list(h.parameters())
    opt = torch.optim.AdamW(params, lr=spec['lr'], weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
    rng = np.random.default_rng(seed)
    distinct = sorted(set(cut[fit].tolist()))
    one_cut = len(distinct) == 1 or not any((fit & (cut+horizon <= c)).any() for c in distinct)
    checkpoints = set(spec.get('checkpoints', CHECKPOINTS))
    max_steps = int(spec.get('steps', max(checkpoints)))
    checkpoints.add(max_steps)
    scorer = holdout_scorer(kumo, sx, sy, y, fit_idx, hold_idx, task_type, cell)
    best_score = scorer(h.eval()) if scorer is not None else None
    best_state = {k: v.detach().cpu().clone() for k, v in h.state_dict().items()}
    h.train()
    for step in range(1, max_steps+1):
        if one_cut:
            perm = rng.permutation(fit_idx)[:spec['queries']+spec['support']]
            q, s = perm[:spec['queries']], perm[spec['queries']:]
        else:
            c = distinct[int(rng.integers(len(distinct)))]
            qc, sc = np.flatnonzero(fit & (cut == c)), np.flatnonzero(fit & (cut+horizon <= c))
            q = qc if len(qc) <= spec['queries'] else rng.choice(qc, spec['queries'], replace=False)
            s = sc if len(sc) <= spec['support'] else rng.choice(sc, spec['support'], replace=False)
        if task_type == 'classification':
            q = q[np.isin(y[q], np.unique(y[s]))]
        if len(q) and len(s) >= 2 and np.ptp(y[s]) > 0:
            opt.zero_grad(set_to_none=True)
            r = h(torch.as_tensor(sx[q], dtype=torch.float32, device='cuda'))
            if task_type == 'classification':
                logit = base_outputs(kumo, sx[s], y[s], sx[q], task_type)+r
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.as_tensor(y[q], device='cuda').float())
            else:
                center, scale = float(np.median(y[s])), max(float(np.std(y[s])), 1e-6)
                base = base_outputs(kumo, sx[s], (y[s]-center)/scale, sx[q], task_type)
                target = torch.as_tensor((y[q]-center)/scale, dtype=torch.float32, device='cuda')[:, None]
                tau = torch.arange(1, base.shape[1]+1, device='cuda', dtype=torch.float32)/(base.shape[1]+1)
                if spec['loss'] == 'median':
                    mid = base.shape[1]//2
                    base, tau = base[:, mid:mid+1], tau[mid:mid+1]
                error = target-(base+r[:, None])
                loss = torch.maximum(tau*error, (tau-1)*error).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
            sched.step()
        if scorer is not None and step in checkpoints:
            score = scorer(h.eval())
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in h.state_dict().items()}
            h.train()
    if scorer is not None:
        h.load_state_dict(best_state)
    return h.eval()


def apply(h, draws, sx, sy, qx, cell, task_type):
    """The R2 draws [E, Q] corrected by the head and aggregated as in R2."""
    r = h.residual(qx)
    if task_type == 'classification':
        p = np.clip(draws, 1e-12, 1-1e-12)
        return predict.ensemble(1/(1+np.exp(-(np.log(p)-np.log1p(-p)+r))), task_type)
    y, k, out = transform(np.asarray(sy, dtype=np.float64), cell['transform']), min(cell['K'], len(sy)), []
    for draw, p in enumerate(draws):
        keep = predict.support_draw(sy, k, draw, cell['selection'], task_type, cell.get('disjoint_draws', False))
        out.append(inverse(transform(p, cell['transform'])+max(float(np.std(y[keep])), 1e-6)*r, cell['transform']))
    return predict.ensemble(np.asarray(out), task_type)
