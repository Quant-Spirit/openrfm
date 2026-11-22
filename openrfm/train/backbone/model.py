"""The trained modules: the backbone (bf16), the column-slot head and the handoff (fp32), in the recipe's order."""
import torch

from openrfm.models import RelationalTransformer, RTConfig
from openrfm.models.backbone.config import AXES

from .coupling import preserved_rng
from .handoff import Handoff
from .init import initialize
from .objective import ColumnSlotHead

VALUE_TYPES = ("number", "text", "datetime", "boolean")


def backbone_parameter_names(rt: RelationalTransformer) -> list[str]:
    """Backbone parameter names in the recipe's order (the optimizer state and every gradient norm follow it)."""
    names = [f"enc_dict.{t}.{p}" for t in ("number", "text", "datetime", "col_name", "boolean") for p in ("weight", "bias")]
    names += ["enc_dict.discrete.weight"]
    names += [f"dec_dict.{t}.{p}" for t in VALUE_TYPES + ("discrete", "regression") for p in ("weight", "bias")]
    names += [f"norm_dict.{t}.weight" for t in ("number", "text", "datetime", "col_name", "boolean", "discrete")]
    names += [f"mask_embs.{t}" for t in ("boolean", "datetime", "number", "text", "discrete")]
    for b in range(len(rt.blocks)):
        names += [f"blocks.{b}.norms.{axis}.weight" for axis in AXES + ("ffn",)]
        for axis in AXES:
            names += [f"blocks.{b}.attns.{axis}.{p}" for p in ("q_norm_weight", "k_norm_weight", "wq.weight", "wk.weight",
                                                                 "wv.weight", "wo.weight")]
        names += [f"blocks.{b}.ffn.{w}.weight" for w in ("w1", "w2", "w3")]
        names += [f"blocks.{b}.entity_axis.{p}.weight" for p in ("norm", "q", "kv", "out")]
    names += ["norm_out.weight"]
    names += [f"structural_features.{e}.weight" for e in ("hop", "path", "time", "in_degree", "out_degree")]
    names += ["context_time_role.time_proj.weight", "context_time_role.role_embedding.weight"]
    names += [f"cell_time_residual.{p}" for p in ("residual_gate", "encoder.centres", "encoder.log_widths",
                                                  "encoder.proj.weight", "encoder.proj.bias", "projection.weight")]
    return names


def build_modules(config, device):
    """``(backbone, column-slot head, handoff, named trainable parameters in order)``."""
    rt = RelationalTransformer(RTConfig(), attention="sparse")
    initialize(rt, config["seed"])
    rt = rt.to(device).to(torch.bfloat16)
    rt.graph_stack(compile=True)
    with preserved_rng():
        head = ColumnSlotHead(rt.d_model, config["objective"]["column_slot_slots"]).to(device)
    c = config["coupling"]
    handoff = Handoff(rt.d_model, slots=c["handoff_slots"], taps=tuple(c["taps"]), k=c["k"], row_depth=c["row_depth"]).to(device)
    own = dict(rt.named_parameters())
    order = backbone_parameter_names(rt)
    if sorted(order) != sorted(own):
        raise RuntimeError("the backbone parameters differ from the recipe's parameter order")
    named = ([(name, own[name]) for name in order] + [(f"column_slot_aux_head.{n}", p) for n, p in head.named_parameters()]
             + [(f"rt_handoff.{n}", p) for n, p in handoff.named_parameters()])
    return rt, head, handoff, named
