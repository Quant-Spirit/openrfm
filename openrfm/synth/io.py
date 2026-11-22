"""Write relational tables with key and timestamp metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_parquet(
    path: Path, table, *, pkey: str | None, fkeys: Mapping[str, str], time_col: str | None
) -> None:
    import pyarrow.parquet as pq

    meta = {
        "pkey_col": json.dumps(pkey),
        "fkey_col_to_pkey_table": json.dumps(dict(fkeys)),
        "time_col": json.dumps(time_col),
    }
    table = table.replace_schema_metadata({k: v for k, v in meta.items()})
    pq.write_table(table, path)


def column_value_digest(column) -> str:
    """Hash the type tag, null markers and little-endian values in row order."""
    import pyarrow as pa
    import struct

    column = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    kind, digest = column.type, hashlib.sha256()
    values = column.to_pylist()
    if pa.types.is_int64(kind) or pa.types.is_timestamp(kind):
        _require(column.null_count == 0, "integer/timestamp anchor columns cannot hold nulls")
        digest.update(b"\x00")
        digest.update(np.asarray(column.cast(pa.int64()).to_numpy(), dtype="<i8").tobytes())
    elif pa.types.is_float64(kind):
        digest.update(b"\x01")
        for value in values:
            digest.update(b"\x00" if value is None else b"\x01" + struct.pack("<d", value))
    elif pa.types.is_boolean(kind):
        digest.update(b"\x02" + bytes(0 if v is None else 2 if v else 1 for v in values))
    elif pa.types.is_string(kind):
        digest.update(b"\x03")
        for value in values:
            if value is None:
                digest.update(b"\x00")
            else:
                raw = value.encode("utf-8")
                digest.update(b"\x01" + struct.pack("<Q", len(raw)) + raw)
    else:
        raise ValueError(f"no anchor digest for arrow type {kind}")
    return digest.hexdigest()
