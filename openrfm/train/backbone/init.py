"""Seeded initialization of a fresh backbone."""
import random

import numpy as np
import torch
from torch import nn

from openrfm.models import RelationalTransformer
from openrfm.models.backbone.config import AXES


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


TIME_FEATURES_SEED = 20260925


def _discarded(*shapes) -> None:
    """Draw the initialization of modules the recipe constructs but never trains (their values never matter)."""
    for kind, shape in shapes:
        layer = nn.Embedding(*shape) if kind == "embedding" else nn.Linear(*shape)
        del layer


def initialize(rt: RelationalTransformer, seed: int) -> None:
    """Draw every random initial value of ``rt`` from ``seed``."""
    # Order: input encoders, decoders, three regression adapters and a prototype projection (drawn, then discarded),
    # mask embeddings, each block's attention and feed-forward matrices, each entity axis, the structural features,
    # then the time features from their own seed. Zero-initialized weights stay zero.
    seed_everything(seed)
    with torch.no_grad():
        for name in ("number", "text", "datetime", "col_name", "boolean"):
            rt.enc_dict[name].reset_parameters()
        for name in ("number", "text", "datetime", "boolean"):
            rt.dec_dict[name].reset_parameters()
        rt.enc_dict["discrete"].reset_parameters()
        rt.dec_dict["discrete"].reset_parameters()
        rt.dec_dict["regression"].reset_parameters()
        _discarded(("embedding", (4, 2)), ("linear", (5, 16)), ("linear", (16, 1)))
        for name in ("number", "text", "datetime", "boolean", "discrete"):
            rt.mask_embs[name].copy_(torch.randn(rt.d_model))
        for block in rt.blocks:
            for axis in AXES:
                for layer in (block.attns[axis].wq, block.attns[axis].wk, block.attns[axis].wv, block.attns[axis].wo):
                    layer.reset_parameters()
            for layer in (block.ffn.w1, block.ffn.w2, block.ffn.w3):
                layer.reset_parameters()
        for block in rt.blocks:
            entity = block.entity_axis
            entity.q.reset_parameters()
            entity.kv.reset_parameters()
            entity.out.reset_parameters()
            entity.out.weight.zero_()
        _discarded(("linear", (rt.d_model, rt.d_model)), ("linear", (rt.d_model, rt.d_model)))
        structure = rt.structural_features
        for module in (structure.hop, structure.path, structure.time, structure.in_degree, structure.out_degree):
            module.reset_parameters()
            module.weight.zero_()
        # The time features draw from their own seed; the global stream continues unchanged after them.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(TIME_FEATURES_SEED)
            role = rt.context_time_role
            role.time_proj.reset_parameters()
            role.role_embedding.reset_parameters()
            role.time_proj.weight.zero_()
            role.role_embedding.weight.zero_()
            residual = rt.cell_time_residual
            residual.encoder.proj.reset_parameters()
            residual.projection.reset_parameters()
