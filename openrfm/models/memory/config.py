"""Configuration of the memory module."""
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryConfig:
    queries: int = 64
    head_width: int = 32
    # History rows read per request.
    rows: int = 256
    # Hops of the temporal foreign-key neighbourhood the history covers.
    hops: int = 5
    d_model: int = 128
    # Backbone depths read from the context: the block-0 input, then each block's output.
    depths: int = 7
    # Requests per read; every read is padded to this size.
    chunk: int = 64
    # Depths of the target cell's residual stream passed to the consumer.
    target_depths: tuple = (3, 6)

    @property
    def export_width(self):
        return 2*self.queries
