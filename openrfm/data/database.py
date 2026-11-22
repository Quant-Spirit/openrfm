"""A relational database as OpenRFM reads it: tables, their primary keys, foreign keys and time columns, loaded from
data frames, a SQL database or a Neo4j graph and written in the RelBench layout."""
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from openrfm.data.layout import write


@dataclass
class Database:
    """Tables keyed by name. `pkeys[t]` is the primary-key column of table t (absent: no primary key), `fkeys[t]` maps a
    column of t to the table it references, and `time[t]` is the column that dates each row (absent: a static table)."""
    tables: dict[str, pd.DataFrame]
    pkeys: dict[str, str] = field(default_factory=dict)
    fkeys: dict[str, dict[str, str]] = field(default_factory=dict)
    time: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_frames(cls, tables, pkeys=None, fkeys=None, time=None):
        """A database of data frames. Time-zone aware columns become naive UTC; time columns that `time` does not name
        are inferred (see `infer_time`)."""
        tables = {name: naive_utc(frame) for name, frame in tables.items()}
        fkeys = {t: {c: cols[c] for c in tables[t].columns if c in cols} for t, cols in (fkeys or {}).items()}
        return cls(tables, dict(pkeys or {}), fkeys, infer_time(tables, time))

    @classmethod
    def from_sql(cls, url, tables=None, time=None):
        """A SQL database by URL, with the keys its constraints declare: `duckdb:///file.duckdb`, `sqlite:///file.db`,
        or any SQLAlchemy URL such as `postgresql://user:password@host/db` or `mysql+pymysql://...`."""
        from openrfm.data import sources
        return cls.from_frames(*sources.read_sql(url, tables), time=time)

    @classmethod
    def from_neo4j(cls, uri, auth=None, database=None, time=None):
        """A Neo4j graph: one table per node label (primary key `id`) and one per relationship type between two labels
        (primary key `id`, foreign keys `src` and `dst`); properties become columns."""
        from openrfm.data import sources
        return cls.from_frames(*sources.read_neo4j(uri, auth, database), time=time)

    def write(self, root, name):
        """Replaces `root/name/db/` and `root/name/keys/`: writes `root/name/db/<table>.parquet` with primary keys replaced by row positions and foreign keys by the
        positions they reference (null when the referenced row is absent). The original keys are kept in
        `root/name/keys/<table>.parquet`, row i holding the key of position i."""
        dest = Path(root)/name
        for folder in ('db', 'keys'):
            shutil.rmtree(dest/folder, ignore_errors=True)
        for table, frame in self.tables.items():
            frame = frame.reset_index(drop=True).copy()
            for column, parent in self.fkeys.get(table, {}).items():
                frame[column] = positions(frame[column], self.key_index(parent))
            pkey = self.pkeys.get(table)
            if pkey is not None:
                write(frame[[pkey]].rename(columns={pkey: 'id'}), dest/'keys'/f'{table}.parquet', None, {}, None)
                frame[pkey] = range(len(frame))
            write(frame, dest/'db'/f'{table}.parquet', pkey, self.fkeys.get(table, {}), self.time.get(table))
        return dest

    def key_index(self, table):
        """Original primary key -> row position of `table`."""
        keys = self.tables[table][self.pkeys[table]].reset_index(drop=True)
        if keys.duplicated().any():
            raise ValueError(f'{table}.{self.pkeys[table]} is not unique')
        return pd.Series(range(len(keys)), index=keys.values)


def positions(column, index):
    """Positions of the keys in `column` under `index`, as nullable integers."""
    return column.map(index).astype('Int64')


def naive_utc(frame):
    """The frame with time-zone aware columns in naive UTC, as RelBench stores them."""
    frame = frame.copy()
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.DatetimeTZDtype):
            frame[column] = frame[column].dt.tz_convert('UTC').dt.tz_localize(None)
    return frame


def infer_time(tables, time=None):
    """The time column of every table: `time[t]` when given (None for a static table), else its only datetime column.
    A table with several datetime columns and no entry in `time` is ambiguous and raises."""
    time, out = dict(time or {}), {}
    for name, frame in tables.items():
        if name in time:
            if time[name] is not None:
                out[name] = time[name]
            continue
        candidates = [c for c in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[c])]
        if len(candidates) > 1:
            raise ValueError(f'{name}: several datetime columns {candidates}; name the row time with time={{{name!r}: ...}}')
        if candidates:
            out[name] = candidates[0]
    return out
