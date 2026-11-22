"""The training loss (the frozen Kumo head's task loss on the backbone columns and the memory columns) and the
reducible-loss measurement that steers the world allocator."""
import torch

from openrfm.models.memory.context import signed_log
from openrfm.train.memory.programs import draw_live_programs, execute_world


def standardize(x, k):
    """Columns of `x` standardized on the first `k` rows; a constant column keeps scale 1."""
    mean, std = x[:k].mean(0), x[:k].std(0, unbiased=False)
    return (x-mean)/torch.where(std > 1e-8, std, torch.ones_like(std))


def transform_labels(target, k, kind):
    """Labels z-scored on the support (`slz`: after a signed log); columns constant on the support are dropped."""
    if kind == 'slz':
        target = signed_log(target)
    std = target[:k].std(0)
    return ((target-target[:k].mean(0))/std.clamp_min(1e-9))[:, std > 1e-9]


def ridge_query_loss(support_x, query_x, support_y, query_y, prefix_width, strength, eigenvalue_floor=1e-4):
    """Query loss of a ridge fit on the support. Labels are whitened on the support (one vote per retained principal
    direction); the feature blocks [:prefix_width] and [prefix_width:] are scaled to equal trace."""
    sx, qx, sy, qy = support_x.float(), query_x.float(), support_y.float(), query_y.float()
    mean = sy.mean(0)
    _, singular, vh = torch.linalg.svd(sy-mean, full_matrices=False)
    eigenvalues = singular.square()/len(sy)
    active = (eigenvalues > eigenvalues[0]*eigenvalue_floor) & (eigenvalues > 1e-12)
    whitening = vh[active].T/eigenvalues[active].sqrt()
    sy, qy = (sy-mean) @ whitening, (qy-mean) @ whitening
    center = sx.mean(0)
    sx, qx = sx-center, qx-center
    blocks, queries = [], []
    for start, stop in ((0, prefix_width), (prefix_width, sx.shape[1])):
        scale = (len(sx)/sx[:, start:stop].square().sum().clamp_min(1e-12)).sqrt()
        blocks.append(sx[:, start:stop]*scale)
        queries.append(qx[:, start:stop]*scale)
    sx, qx = torch.cat(blocks, 1), torch.cat(queries, 1)
    ridge = strength*sx.square().sum()/len(sx)
    if len(sx) < sx.shape[1]:
        chol = torch.linalg.cholesky(sx @ sx.T+ridge*torch.eye(len(sx), device=sx.device))
        prediction = (qx @ sx.T) @ torch.cholesky_solve(sy, chol)
    else:
        chol = torch.linalg.cholesky(sx.T @ sx+ridge*torch.eye(sx.shape[1], device=sx.device))
        prediction = qx @ torch.cholesky_solve(sx.T @ sy, chol)
    return float((prediction-qy).square().sum()/max(1, qy.numel()))


def binary_task(labels, kind):
    """A world task label as one regression target, the top class against the rest for classification."""
    return labels.float() if kind == 'regression' else (labels == labels.max()).float()


@torch.no_grad()
def reducible_loss(world, rows, cutoffs, labels, prefix, memory, k, kind, seed, groups):
    """Ridge query loss on a world's task label, on a support of `k` rows: backbone alone, backbone plus memory, and
    backbone plus exact program values (an oracle of what history holds). Reducible = what memory still loses
    against the better of the other two; None when the label is constant on the support."""
    y = transform_labels(binary_task(labels, kind)[:, None], k, 'slz' if kind == 'regression' else 'z')
    if not y.shape[1]:
        return None
    base = standardize(prefix, k).nan_to_num().float()
    programs = draw_live_programs(world, rows[:k], cutoffs[:k], 128, seed, scoped_share=.75, count_share=.5,
                                  groups=groups)
    oracle = transform_labels(execute_world(world, rows, cutoffs, programs, exclude_focal=True), k, 'slz').float()

    def with_base(right):
        both = torch.cat((base, right), 1)
        return ridge_query_loss(both[:k], both[k:], y[:k], y[k:], base.shape[1], 1.)
    alone = ridge_query_loss(base[:k], base[k:], y[:k], y[k:], 0, 1.)
    joint = with_base(standardize(memory, k))
    best = with_base(oracle) if oracle.shape[1] else alone
    return dict(backbone=alone, joint=joint, oracle=best, reducible=max(joint-min(best, alone), 0.))


def kumo_task_losses(consumers, prefix, features, tasks, k, seed):
    """Backpropagate the mean frozen-Kumo loss over `tasks` [(kind, y)]: the consumer reads [prefix, memory columns]
    standardized on the first `k` rows. Returns the per-task losses."""
    losses = []
    for i, (kind, y) in enumerate(tasks):
        consumer = consumers[kind]
        x = consumer.join(torch.cat((prefix, features.float()), 1), k, seed+i)
        loss = consumer.loss(consumer(x, y[:k]), y[:k], y[k:])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'nonfinite Kumo loss, task {i} {kind}')
        (loss/len(tasks)).backward(retain_graph=i < len(tasks)-1)
        losses.append(float(loss.detach()))
    return losses
