"""A prediction task on a `Database`: per entity and time a label over the following window, split by time."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from openrfm.data.database import Database, naive_utc, positions
from openrfm.data.layout import write

SPLITS = ('train', 'val', 'test')


@dataclass
class Task:
    """Rows `id` (a primary key of the entity table), `timestamp` and `target` per split; the label of a row is known
    `horizon` after its timestamp."""
    name: str
    entity: str
    target: str
    task_type: str
    horizon: pd.Timedelta
    splits: dict[str, pd.DataFrame]

    @classmethod
    def from_frames(cls, name, entity, target, task_type, horizon, train, val, test):
        """A task from three frames with columns `id`, `timestamp` and `target` (test may omit the target). A
        classification target is binary (0 or 1); a regression target is a number."""
        if task_type not in ('classification', 'regression'):
            raise ValueError(f'task_type is classification or regression, not {task_type!r}')
        if task_type == 'classification' and not set(pd.unique(train[target])) <= {0, 1}:
            raise ValueError(f'{target}: classification is binary, with labels 0 and 1')
        splits = {s: naive_utc(f.assign(timestamp=pd.to_datetime(f['timestamp']))) for s, f in zip(SPLITS, (train, val, test))}
        return cls(name, entity, target, task_type, pd.Timedelta(horizon), splits)

    @classmethod
    def from_sql(cls, db: Database, name, entity, label, timestamps, val_start, test_start, horizon,
                 task_type='classification', target='label'):
        """A task whose labels a DuckDB query computes over the tables of `db` (original keys).

        The query reads the table `cases(id, timestamp, "end")`: every row of `entity` (that exists by then, when the
        table is dated) at every one of `timestamps`, with `"end"` = timestamp + horizon. It returns `id`, `timestamp`
        and `target`; for instance, no order in the next 30 days:

            SELECT c.id, c.timestamp, COUNT(o.order_id) = 0 AS label
            FROM cases c LEFT JOIN orders o
              ON o.customer_id = c.id AND o.order_time >= c.timestamp AND o.order_time < c."end"
            WHERE EXISTS (SELECT 1 FROM orders p WHERE p.customer_id = c.id AND p.order_time < c.timestamp)
            GROUP BY c.id, c.timestamp

        Rows before `val_start` form TRAIN, rows before `test_start` VAL and the rest TEST."""
        import duckdb

        horizon = pd.Timedelta(horizon)
        con = duckdb.connect()
        for table, frame in db.tables.items():
            con.register(table, frame)
        key, clock = db.pkeys[entity], db.time.get(entity)
        grid = pd.DataFrame({'timestamp': pd.to_datetime(list(timestamps))})
        con.register('grid', grid)
        born = f'WHERE e."{clock}" < g.timestamp' if clock else ''
        con.execute(f'CREATE TABLE cases AS SELECT e."{key}" AS id, g.timestamp, g.timestamp + INTERVAL '
                    f'{int(horizon.total_seconds())} SECOND AS "end" FROM "{entity}" e CROSS JOIN grid g {born}')
        rows = con.execute(label).df()
        if task_type == 'classification':
            rows[target] = rows[target].astype('int64')
        rows = rows[['id', 'timestamp', target]].sort_values(['timestamp', 'id'], kind='stable', ignore_index=True)
        val_start, test_start = pd.Timestamp(val_start), pd.Timestamp(test_start)
        parts = (rows[rows.timestamp < val_start], rows[(rows.timestamp >= val_start) & (rows.timestamp < test_start)],
                 rows[rows.timestamp >= test_start])
        return cls.from_frames(name, entity, target, task_type, horizon, *parts)

    def write(self, db: Database, root, db_name):
        """Writes `root/db_name/tasks/<name>/{train,val,test}.parquet` with entity keys as row positions and returns the
        task declaration that `openrfm predict --spec` reads."""
        index, key = db.key_index(self.entity), db.pkeys[self.entity]
        for split, frame in self.splits.items():
            frame = pd.DataFrame({'timestamp': frame['timestamp'].values, key: positions(frame['id'], index).values,
                                  **({self.target: frame[self.target].values} if self.target in frame else {})})
            frame = frame[frame[key].notna()]
            write(frame, Path(root)/db_name/'tasks'/self.name/f'{split}.parquet', None, {key: self.entity}, 'timestamp')
        return dict(db=db_name, table=self.name, target=self.target, task_type=self.task_type,
                    entity_table=self.entity, horizon_s=int(self.horizon.total_seconds()))
