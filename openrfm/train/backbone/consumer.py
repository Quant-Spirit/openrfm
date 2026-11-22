"""The frozen TabICL v2 consumers of the handoff and the consumer loss of one episode."""
import hashlib
from pathlib import Path

import torch
from tabicl.model.attention import flash_attn3_toggle
from tabicl.model.inference import InferenceManager
from tabicl.model.inference_config import InferenceConfig
from tabicl.model.tabicl import TabICL

WINSOR, SKEW_THRESHOLD, ZERO_INFLATED_FRACTION, IQR_TO_SIGMA, IQR_FLOOR_RELATIVE = 8.0, 3.0, 0.3, 1.349, 1e-6


class GradInferenceManager(InferenceManager):
    """TabICL's inference manager, keeping the autograd graph when an input requires gradients."""

    def __call__(self, forward_fn, inputs, auto_batch=True, output_repeat=1):
        if torch.is_grad_enabled() and any(torch.is_tensor(v) and v.requires_grad for v in inputs.values()):
            prepared = self._prepare_inputs(inputs)
            with flash_attn3_toggle(False), torch.autocast(device_type=self.exe_device.type, dtype=torch.float16,
                                                           enabled=self.use_amp and self.exe_device.type == "cuda"):
                return forward_fn(**prepared)
        return super().__call__(forward_fn, inputs, auto_batch=auto_batch, output_repeat=output_repeat)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Consumer:
    """One frozen TabICL v2 model (classifier or regressor) used with an episode-local support cache."""

    def __init__(self, path: Path, sha256: str, regression: bool, device):
        if _sha256(path) != sha256:
            raise ValueError(f"{path} is not the pinned TabICL v2 checkpoint")
        self.regression = regression
        self.model = TabICL(max_classes=0, num_quantiles=999, recompute=True) if regression else TabICL(max_classes=10, recompute=True)
        state = torch.load(path, map_location="cpu", weights_only=False)
        missing, unexpected = self.model.load_state_dict(state.get("state_dict", state), strict=False)
        current = self.model.state_dict()
        if unexpected or any(not key.endswith(".bias") for key in missing):
            raise ValueError(f"{path} does not match TabICL v2: missing={missing} unexpected={unexpected}")
        for key in missing:
            current[key].zero_()
        self.model = self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        for module in self.model.modules():
            if isinstance(getattr(module, "inference_mgr", None), InferenceManager):
                module.inference_mgr.__class__ = GradInferenceManager
        self.config = InferenceConfig()
        for config in (self.config.COL_CONFIG, self.config.ROW_CONFIG, self.config.ICL_CONFIG):
            config.update({"use_fa3": False, "offload": "gpu", "use_async": False})

    def _labels(self, support_y: torch.Tensor):
        dtype = next(self.model.parameters()).dtype
        if not self.regression:
            return support_y.unsqueeze(0).to(dtype=dtype), None
        y = support_y.float()
        mean, std = y.mean(), y.std(unbiased=False).clamp_min(1e-8)
        if not bool(torch.isfinite(mean) & torch.isfinite(std)):
            # Finite targets whose variance overflows fp32: moments in fp64.
            wide = y.double()
            mean, std = wide.mean(), wide.std(unbiased=False).clamp_min(1e-8)
            return ((wide - mean) / std).float().unsqueeze(0).to(dtype), (mean.float(), std.float())
        return ((y - mean) / std).unsqueeze(0).to(dtype), (mean, std)

    def __call__(self, support_x, support_y, query_x) -> torch.Tensor:
        """Query logits (classifier) or ``[Q, 1]`` predicted means (regressor); gradients reach ``query_x`` only."""
        dtype = next(self.model.parameters()).dtype
        for config in (self.config.COL_CONFIG, self.config.ROW_CONFIG, self.config.ICL_CONFIG):
            config.update({"device": support_x.device})
        y_train, stats = self._labels(support_y)
        common = dict(return_logits=True, softmax_temperature=0.9, cache_mode="kv", inference_config=self.config)
        try:
            with torch.no_grad():
                self.model.forward_with_cache(X_train=support_x.detach().to(dtype).unsqueeze(0), y_train=y_train,
                                              X_test=None, use_cache=False, store_cache=True, **common)
            with torch.enable_grad():
                out = self.model.forward_with_cache(X_test=query_x.to(dtype).unsqueeze(0), use_cache=True,
                                                    store_cache=False, cache=self.model._cache, **common)
        finally:
            self.model._cache = None
        preds = out.squeeze(0)
        if not self.regression:
            return preds.float()
        mean = self.model.quantile_dist(preds).quantiles.mean(dim=-1).reshape(-1, 1)
        return (mean * stats[1].to(mean.device) + stats[0].to(mean.device)).float()


class TargetTransform:
    """Support-fitted robust regression transform (optional signed log, median centering, IQR scaling)."""

    def __init__(self, support: torch.Tensor):
        y = support.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        y = y[torch.isfinite(y)]
        non_negative = bool((y >= 0).all())
        zero_inflated = non_negative and float((y == 0).double().mean()) >= ZERO_INFLATED_FRACTION
        centered = y - y.mean()
        std = centered.square().mean().sqrt()
        skew = float((centered ** 3).mean() / std ** 3) if bool(std > 0) else 0.0
        self.log = non_negative and (skew > SKEW_THRESHOLD or zero_inflated)
        self.pre_scale = 1.0
        if self.log:
            magnitude = y[y != 0].abs()
            self.pre_scale = float(magnitude.median()) if magnitude.numel() else 1.0
        u = self._u(y)
        self.center = float(u.median())
        q25, q75 = torch.quantile(u, torch.tensor([0.25, 0.75], dtype=torch.float64)).tolist()
        self.scale = (q75 - q25) / IQR_TO_SIGMA
        floor = IQR_FLOOR_RELATIVE * max(float(u.abs().max()), 1e-300)
        if not self.scale > floor:
            spread = float(u.std(unbiased=False))
            self.scale = spread if spread > floor else 1.0

    def _u(self, y):
        y = y.to(torch.float64)
        return torch.sign(y / self.pre_scale) * torch.log1p((y / self.pre_scale).abs()) if self.log else y

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        return ((self._u(y) - self.center) / self.scale).to(device=y.device, dtype=torch.float32)


def group_loss(consumers: dict, classes, support_labels, query_labels, support_x, query_x) -> torch.Tensor:
    """Per-query consumer loss: cross-entropy over the support classes, or Huber on the robust-transformed target."""
    device = support_x.device
    if classes is not None:
        index = {cls: position for position, cls in enumerate(classes)}
        support_y = torch.tensor([index[int(label)] for label in support_labels], dtype=torch.long, device=device)
        query_y = torch.tensor([index[int(label)] for label in query_labels], dtype=torch.long, device=device)
        logits = consumers["classification"](support_x, support_y, query_x).float()
        per_query = torch.nn.functional.cross_entropy(logits, query_y, reduction="none")
    else:
        support_raw = torch.tensor(support_labels, dtype=torch.float64)
        query_raw = torch.tensor(query_labels, dtype=torch.float64)
        transform = TargetTransform(support_raw)
        prediction = consumers["regression"](support_x, transform(support_raw).to(device), query_x).float().reshape(-1)
        wanted = transform(query_raw).clamp(-WINSOR, WINSOR).to(prediction.device)
        finite = torch.isfinite(query_raw.to(prediction.device)) & torch.isfinite(wanted)
        per = torch.nn.functional.huber_loss(prediction, torch.where(finite, wanted, torch.zeros_like(wanted)),
                                             reduction="none", delta=1.0)
        per_query = torch.where(finite, per, torch.zeros_like(per))
        if not bool(finite.all()):
            per_query = per_query[finite]
    if not bool(torch.isfinite(per_query).all()):
        raise FloatingPointError("non-finite consumer loss")
    return per_query
