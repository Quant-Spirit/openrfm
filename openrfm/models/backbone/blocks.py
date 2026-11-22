# Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
"""The SwiGLU feed-forward, the residual block and the block stack."""
import torch
import torch.nn.functional as F
from torch import nn

from .attention import MaskedAttention, SupportAxisAttention
from .config import AXES
from .structure import EntityAxis


class FFN(nn.Module):
    """SwiGLU feed-forward."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RelationalBlock(nn.Module):
    """Pre-norm residual block: the col, feat, nbr and support axes, the entity axis, then the FFN."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.norms = nn.ModuleDict({axis: nn.RMSNorm(d_model) for axis in AXES + ("ffn",)})
        self.attns = nn.ModuleDict({axis: SupportAxisAttention(d_model, num_heads) if axis == "support"
                                    else MaskedAttention(d_model, num_heads) for axis in AXES})
        self.ffn = FFN(d_model, d_ff)
        self.entity_axis = EntityAxis(d_model, num_heads)

    def forward(self, x: torch.Tensor, masks: dict, batch: dict, rows: tuple, support_axis_enabled: bool):
        for axis in AXES:
            if axis == "support":
                if support_axis_enabled:
                    x = x + self.attns[axis](self.norms[axis](x), batch)
            else:
                x = x + self.attns[axis](self.norms[axis](x), masks[axis])
        x = x + self.entity_axis(x, rows)
        return x + self.ffn(self.norms["ffn"](x))


def run_blocks(blocks, x: torch.Tensor, masks: dict, batch: dict, rows: tuple, support_axis_enabled: bool):
    """Apply the block stack in order."""
    for block in blocks:
        x = block(x, masks, batch, rows, support_axis_enabled)
    return x
