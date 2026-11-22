"""Row times from a table's own time column, written as the archive's node-time sidecar (read by `policy.py`).

`rustler pre` dates a row through its foreign keys, so a row of a table with a time column but no dated foreign key is
archived without a time and is visible at every cutoff. The sidecar dates each such row at min(own time, earliest
referencing row), where a task row at cutoff c counts as c - 1 s: a referenced row exists by its first reference."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from openrfm.archive.policy import ARCHIVE_STATIC, POLICY, SIDECAR, TAG

NEVER = np.iinfo(np.int64).max


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def time_column(parquet):
    value = json.loads((pq.read_schema(parquet).metadata or {}).get(b'time_col', b'null').decode() or 'null')
    return value if isinstance(value, str) and value else None


def own_seconds(parquet, column):
    """int32 seconds of one time column, truncated toward zero as `rustler pre` does and clamped to the int32 range;
    nulls are the archive's no-time value."""
    values = pq.read_table(parquet, columns=[column]).column(column)
    if not pa.types.is_timestamp(values.type):
        values = pc.cast(values, pa.timestamp('ns'))
    unit = {'s': 1, 'ms': 10**3, 'us': 10**6, 'ns': 10**9}[values.type.unit]
    raw = np.asarray(values.cast(pa.int64()).fill_null(0).to_numpy(zero_copy_only=False), dtype=np.int64)
    out = np.clip(np.sign(raw)*(np.abs(raw)//unit), ARCHIVE_STATIC+1, np.iinfo(np.int32).max)
    out[~np.asarray(values.is_valid().to_numpy(zero_copy_only=False), dtype=bool)] = ARCHIVE_STATIC
    return out.astype(np.int32)


def earliest_reference(times, task_row, ptr, dst, offset, count):
    """Per row of one table, the earliest time of a row referencing it (task rows at time - 1 s); NEVER if none."""
    counts = np.diff(np.asarray(ptr[offset:offset+count+1], dtype=np.int64))
    earliest = np.full(count, NEVER, dtype=np.int64)
    begin, end = int(ptr[offset]), int(ptr[offset+count])
    if end == begin:
        return earliest
    child = np.asarray(dst[begin:end], dtype=np.int64)
    when = np.where(task_row[child], times[child]-1, times[child])
    when = np.where(times[child] == ARCHIVE_STATIC, NEVER, when)
    starts = np.flatnonzero(counts)
    earliest[starts] = np.minimum.reduceat(when, np.cumsum(counts)[starts]-counts[starts])
    return earliest


def build(archive, data):
    """Writes the sidecar of `archive`, converted from the RelBench-layout database `data`; none when every row of each
    table with a time column is already dated."""
    from rustler_archive import ArchiveV1
    archive = Path(archive)
    arrays = dict(ArchiveV1.load_arrays(str(archive), 'all-MiniLM-L12-v2', 384))
    archived = np.asarray(arrays['node_timestamps']).astype(np.int64)
    info = json.loads((archive/'table_info.json').read_text())
    times, task_row = archived.copy(), np.zeros(len(archived), dtype=bool)
    own, tables = {}, {}
    for key, v in sorted(info.items(), key=lambda kv: int(kv[1]['node_idx_offset'])):
        name, split = key.rsplit(':', 1)
        offset, count = int(v['node_idx_offset']), int(v['num_nodes'])
        if split != 'Db':
            task_row[offset:offset+count] = True
            continue
        parquet = Path(data)/'db'/f'{name}.parquet'
        column = time_column(parquet) if parquet.is_file() else None
        dated = archived[offset:offset+count] != ARCHIVE_STATIC
        if column is None or dated.all():
            continue
        row_time = np.where(dated, archived[offset:offset+count], own_seconds(parquet, column))
        times[offset:offset+count] = row_time
        own[name] = (row_time, ~dated)
        tables[name] = dict(offset=offset, num_nodes=count, time_col=column)
    if not tables:
        return
    ptr, dst = np.asarray(arrays['p2f_rowptr']), np.asarray(arrays['p2f_col_idx'])
    out = {}
    for name, (row_time, undated) in own.items():
        t = tables[name]
        floor = earliest_reference(times, task_row, ptr, dst, t['offset'], t['num_nodes'])
        redate = undated & (row_time != ARCHIVE_STATIC)
        final = row_time.copy()
        final[redate] = np.clip(np.minimum(row_time[redate], floor[redate]), ARCHIVE_STATIC+1, np.iinfo(np.int32).max)
        out[name] = final.astype(np.int32)
    np.savez(archive/SIDECAR, **out)
    policy = dict(tag=TAG, table_info_sha256=sha256(archive/'table_info.json'), sidecar_sha256=sha256(archive/SIDECAR),
                  tables=tables)
    (archive/POLICY).write_text(json.dumps(policy, indent=1, sort_keys=True)+'\n')
