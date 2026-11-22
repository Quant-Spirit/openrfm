"""Per-micro-batch corruption of the masked-cell objective: extra masked cells and support sequences."""
import torch

from openrfm.sampler.retrieval import STATIC, UNKNOWN

NO_TIME = (-1, -(2**31))
SEM_NUMBER, SEM_TEXT, SEM_DATETIME, SEM_BOOLEAN, SEM_DISCRETE = range(5)


def maskable_cells(batch: dict) -> torch.Tensor:
    """Finite non-target number/datetime/boolean cells and valid class cells (text cells have no decoder)."""
    eligible = ~batch["is_padding"].bool() & ~batch["is_targets"].bool()
    sem_types = batch["sem_types"]
    numeric = torch.zeros_like(eligible)
    for sem_type, key in ((SEM_NUMBER, "number_values"), (SEM_DATETIME, "datetime_values"), (SEM_BOOLEAN, "boolean_values")):
        numeric |= (sem_types == sem_type) & torch.isfinite(batch[key].float()).all(dim=-1)
    labels = batch["class_value_idxs"].long()
    cardinalities = batch["discrete_num_classes"].long()
    discrete = (sem_types == SEM_DISCRETE) & (labels >= 0) & (cardinalities >= 2) & (labels < cardinalities)
    return eligible & (numeric | discrete)


def mask_extra_cells(batch: dict, p_max: float, generator: torch.Generator) -> dict:
    """Mask each maskable cell of context ``i`` with probability ``p_i ~ U(0, p_max)``."""
    masks = batch["masks"].bool()
    rates = torch.rand((masks.shape[0], 1), generator=generator, device=masks.device, dtype=torch.float32) * p_max
    draws = torch.rand(masks.shape, generator=generator, device=masks.device, dtype=torch.float32)
    return dict(batch, masks=masks | (maskable_cells(batch) & (draws < rates)))


def _focal_times(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    is_targets = batch["is_targets"].bool()
    target_pos = is_targets.float().argmax(dim=1)
    return target_pos, torch.gather(batch["timestamps"].long(), 1, target_pos[:, None]).squeeze(1)


def designate_support(batch: dict, support_fraction: float, *, horizon_safe: bool) -> dict:
    """Mark up to ``support_fraction`` of the sequences as support (targets visible, out of the loss)."""
    # With real focal times: the earliest sequences whose label is known before the latest query; otherwise a random
    # subset (global CUDA RNG), and the batch is marked ``support_timeless``.
    masks = batch["masks"]
    size = int(masks.shape[0])
    support = torch.zeros(size, dtype=torch.bool, device=masks.device)
    target_pos, row_time = _focal_times(batch)
    temporal = not bool(torch.isin(row_time, torch.tensor(NO_TIME, device=row_time.device)).all())
    n_support = min(int(size * support_fraction), size - 1)
    out = dict(batch)
    if n_support <= 0:
        return dict(out, is_support_seed=support)
    if temporal:
        eligible = batch["is_targets"].bool().any(dim=1)
        for sentinel in NO_TIME:
            eligible &= row_time != sentinel
        known_at = row_time
        if horizon_safe:
            available = torch.gather(batch["label_available_at"].long(), 1, target_pos[:, None]).squeeze(1)
            real = (available != UNKNOWN) & (available != STATIC) & (available >= row_time)
            horizon_safe = bool((real & eligible).any())
            known_at = torch.where(real, available, row_time)
        order = torch.argsort(torch.where(eligible, known_at, torch.full_like(row_time, torch.iinfo(torch.int64).max)),
                              stable=True)
        n_eligible = int(eligible.sum())
        n_support = min(n_support, max(n_eligible - 1, 0))
        if n_support <= 0:
            return dict(out, is_support_seed=support)
        candidate = order[:n_support]
        if horizon_safe:
            keep = candidate[known_at[candidate] < row_time[order[n_support:n_eligible]].max()]
        else:
            keep = candidate[row_time[candidate] < row_time[order[n_support]]]
        if keep.numel() == 0:
            return dict(out, is_support_seed=support)
        support[keep] = True
    else:
        support[torch.randperm(size, device=masks.device)[:n_support]] = True
        out["support_timeless"] = True
    out["masks"] = masks & ~support[:, None]
    out["is_support_seed"] = support
    if horizon_safe and temporal:
        out["support_label_available_at"] = known_at
    return out


def prepare_micro_batch(batch: dict, *, regression_tasks: torch.Tensor, step: int, seed: int, p_max: float,
                        support_fraction: float) -> dict:
    """Regression flags, extra masking (seeded by the optimizer step), value scrubbing and support designation."""
    target_pos = batch["is_targets"].float().argmax(dim=1)
    tasks = torch.gather(batch["dataset_idxs"].long(), 1, target_pos.unsqueeze(1)).squeeze(1)
    batch["is_regression_task"] = regression_tasks[tasks]
    generator = torch.Generator(device=batch["masks"].device)
    generator.manual_seed((seed * 1000003 + step + 1) % (2**63 - 1))
    batch = mask_extra_cells(batch, p_max, generator)
    for key in ("number_values", "boolean_values"):
        batch[key] = torch.nan_to_num(batch[key], nan=0.0, posinf=0.0, neginf=0.0)
    return designate_support(batch, support_fraction, horizon_safe=True)
