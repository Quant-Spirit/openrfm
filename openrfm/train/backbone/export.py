"""Released backbone file from a training checkpoint."""
from pathlib import Path

import torch

# Training-only tensors that the inference backbone does not build: an auxiliary projection and regression adapters
# whose output layers are exactly zero, and a dtype marker buffer.
UNUSED_PREFIXES = ("proto_proj.", "regression_regime_adapter.", "regression_operator_residual.",
                   "regression_operator_mixture_gate", "datetime_value_fp32_enabled")


def released(checkpoint: dict) -> dict:
    """The inference fields of a training checkpoint: the backbone weights and the handoff, with the handoff's
    typed-scalar value plane (zero) switched on."""
    handoff = dict(checkpoint["rt_handoff"])
    h3e = dict(handoff["h3e"])
    if handoff["mode"] != "h3e" or h3e["k"] != 4 or h3e.get("value_channel", False):
        raise ValueError("not an h3e training checkpoint")
    h3e.update(value_channel=True, typed_scalars=True)
    state = dict(handoff["state_dict"], value_plane=torch.zeros(handoff["d_model"], 2))
    weights = {k: v for k, v in checkpoint["state_dict"].items() if not k.startswith(UNUSED_PREFIXES)}
    return dict(schema="openrfm_backbone_v1", state_dict=weights, rt_handoff=dict(handoff, h3e=h3e, state_dict=state))


def export_backbone(source: Path, destination: Path) -> None:
    """Write ``destination``, the released backbone file of training checkpoint ``source``."""
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(released(checkpoint), destination)
