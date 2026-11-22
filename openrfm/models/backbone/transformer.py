# Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
"""The relational transformer: cell embedding, the block stack and the training readout."""
import torch
from torch import nn

from .attention import validate_support_timestamps
from .blocks import RelationalBlock, run_blocks
from .config import VALUE_TYPES, RTConfig
from .embedding import embed_cells, encode_cell_value, value_visibility
from .masks import dense_attention_masks
from .readouts import masked_cell_loss
from .structure import StructuralCellFeatures, entity_rows
from .time import ContextTimeRole, HorizonCellTimeResidual

ATTENTION = ("dense", "sparse")


class RelationalTransformer(nn.Module):
    """Transformer over the cells of a sampled relational context."""

    def __init__(self, config: RTConfig = RTConfig(), attention: str = "dense"):
        super().__init__()
        if attention not in ATTENTION:
            raise ValueError(f"attention must be one of {ATTENTION}")
        self.config, self.attention, self.graphed = config, attention, None
        d, d_text, classes = config.d_model, config.d_text, config.max_classes
        self.enc_dict = nn.ModuleDict({"number": nn.Linear(1, d), "text": nn.Linear(d_text, d),
                                       "datetime": nn.Linear(1, d), "col_name": nn.Linear(d_text, d),
                                       "boolean": nn.Linear(1, d), "discrete": nn.Embedding(classes, d)})
        self.dec_dict = nn.ModuleDict({"number": nn.Linear(d, 1), "text": nn.Linear(d, d_text),
                                       "datetime": nn.Linear(d, 1), "boolean": nn.Linear(d, 1),
                                       "discrete": nn.Linear(d, classes), "regression": nn.Linear(d, 1)})
        self.norm_dict = nn.ModuleDict({name: nn.RMSNorm(d) for name in
                                        ("number", "text", "datetime", "col_name", "boolean", "discrete")})
        self.mask_embs = nn.ParameterDict({name: nn.Parameter(torch.randn(d)) for name in VALUE_TYPES + ("discrete",)})
        self.blocks = nn.ModuleList(RelationalBlock(d, config.num_heads, config.d_ff)
                                    for _ in range(config.num_blocks))
        self.norm_out = nn.RMSNorm(d)
        self.structural_features = StructuralCellFeatures(d, config.path_buckets)
        self.context_time_role = ContextTimeRole(d)
        self.cell_time_residual = HorizonCellTimeResidual(d, config.cell_time_dim)

    @property
    def d_model(self) -> int:
        return self.config.d_model

    def encode_cell_value(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Encoded value ``[..., d_model]`` of cells of type ``name`` from their raw values ``[..., width]``."""
        return encode_cell_value(self, name, value)

    def value_visibility(self, batch: dict) -> torch.Tensor:
        """``[B, S]`` cells whose value the encoder reads."""
        return value_visibility(batch)

    def embed_cells(self, batch: dict) -> torch.Tensor:
        """Block-0 input ``[B, S, d_model]``; also stores the visibility in ``batch["value_visible_cells"]``."""
        return embed_cells(self, batch)

    def attention_masks(self, batch: dict) -> dict:
        """Per-axis masks of the col, feat and nbr axes: dense boolean tensors or sparse plans."""
        if self.attention == "sparse":
            from ..kernels.sparse_attention import build_device_plan
            return build_device_plan(batch)
        return dense_attention_masks(batch)

    def encode_hidden(self, batch: dict, *, support_axis_enabled: bool = True) -> torch.Tensor:
        """Contextual cell states ``[B, S, d_model]`` after the final norm."""
        if support_axis_enabled:
            validate_support_timestamps(batch)
        x = self.embed_cells(batch)
        rows = entity_rows(batch, 256 if self.attention == "sparse" else 0)
        if self.graphed is not None and support_axis_enabled:
            x = self.graphed(x, batch, rows)
        else:
            x = run_blocks(self.blocks, x, self.attention_masks(batch), batch, rows, support_axis_enabled)
        return self.norm_out(x)

    def forward(self, batch: dict, *, support_axis_enabled: bool = True) -> tuple[torch.Tensor, dict]:
        """Masked-cell training loss and outputs (per-type predictions and ``hidden``)."""
        x = self.encode_hidden(batch, support_axis_enabled=support_axis_enabled)
        return masked_cell_loss(self.dec_dict, x, batch)

    def graph_stack(self, compile: bool = False, max_classes: int | None = None) -> None:
        """Replay the block stack (with the support axis) as CUDA graphs; requires sparse attention."""
        if self.attention != "sparse":
            raise ValueError("graph_stack requires attention='sparse'")
        from ..kernels.graph_stack import GraphedStack
        self.graphed = GraphedStack(self.blocks, compile, max_classes)
