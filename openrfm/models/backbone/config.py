# Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
"""Dimensions of the relational transformer backbone."""
from dataclasses import dataclass

# Attention axes in execution order inside every block.
STRUCTURAL_AXES = ("col", "feat", "nbr")
AXES = STRUCTURAL_AXES + ("support",)
# Cell value types in `sem_types` order; DISCRETE is the class-label type.
VALUE_TYPES = ("number", "text", "datetime", "boolean")
DISCRETE = 4


@dataclass(frozen=True)
class RTConfig:
    d_model: int = 128
    d_text: int = 384
    num_blocks: int = 6
    num_heads: int = 4
    d_ff: int = 512
    # Fixed class slots shared by every discrete column; a column uses its first C slots.
    max_classes: int = 10
    path_buckets: int = 1024
    cell_time_dim: int = 16
