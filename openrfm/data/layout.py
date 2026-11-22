"""The RelBench layout that archives are converted from: `<db>/db/<table>.parquet` and
`<db>/tasks/<task>/{train,val,test}.parquet`, each file carrying `pkey_col`, `fkey_col_to_pkey_table` and `time_col` as
Parquet key-value metadata. Primary keys are row positions and foreign keys point at them."""
import json

import pyarrow as pa
import pyarrow.parquet as pq


def write(table, path, pkey, fkeys, clock):
    """A Parquet file with RelBench's key-value metadata."""
    if not isinstance(table, pa.Table):
        table = pa.Table.from_pandas(table, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k.encode(): json.dumps(v).encode() for k, v in
            dict(pkey_col=pkey, fkey_col_to_pkey_table=fkeys, time_col=clock).items()}
    pq.write_table(table.replace_schema_metadata(meta), path, compression='zstd')
