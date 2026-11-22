# openrfm.data

`Database` and `Task` bring a database of one's own into the RelBench layout that `openrfm predict` reads.

| File | Content |
|---|---|
| `database.py` | `Database`: tables, primary keys, foreign keys and row times; `from_frames`, `from_sql`, `from_neo4j`, `write` |
| `sources.py` | Readers of SQL databases (DuckDB natively, others through SQLAlchemy) and Neo4j graphs |
| `task.py` | `Task`: labelled rows per split; `from_frames`, `from_sql`, `write` |
| `layout.py` | The Parquet writer with RelBench's key-value metadata |

`Database.write(root, name)` replaces primary keys by row positions and foreign keys by the positions they reference;
`root/name/keys/<table>.parquet` keeps the original keys, which `openrfm.predict` uses to label its output. Only
single-column keys are read; a foreign key that references another unique column of its parent is rewritten to the
parent's primary key.

A Neo4j graph becomes one table per node label (primary key `id`, the element id) and one per relationship type
(primary key `id`, foreign keys `src` and `dst`). A relationship type that joins several label pairs gets one table per
pair, `<type>_<source label>_<target label>`; a node with several labels is filed under its first one.

`openrfm import --source URL --data DATA --name NAME` writes a SQL database or a Neo4j graph to `DATA/NAME/` from the
command line; `--time TABLE=COLUMN` names a row time (`TABLE=` for a static table).
