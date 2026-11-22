"""Readers of external databases into (tables, primary keys, foreign keys). Only single-column keys are kept. A foreign
key that references a unique column other than its parent's primary key is rewritten to hold the referenced parent's
primary key."""
from collections import defaultdict

import pandas as pd


def read_sql(url, tables=None):
    """A SQL database: `duckdb:///<path>` natively (like SQLAlchemy URLs, `duckdb:////abs/path` for an absolute
    path), everything else through SQLAlchemy."""
    if url.startswith('duckdb:///'):
        return read_duckdb(url[len('duckdb:///'):], tables)
    from sqlalchemy import create_engine, inspect

    engine = create_engine(url)
    inspector = inspect(engine)
    names = tables or inspector.get_table_names()
    frames, pkeys, fkeys = {}, {}, {}
    with engine.connect() as connection:
        for name in names:
            frames[name] = pd.read_sql_table(name, connection)
            pk = inspector.get_pk_constraint(name).get('constrained_columns') or []
            if len(pk) == 1:
                pkeys[name] = pk[0]
            fkeys[name] = {fk['constrained_columns'][0]: (fk['referred_table'], fk['referred_columns'][0])
                           for fk in inspector.get_foreign_keys(name)
                           if len(fk['constrained_columns']) == 1 and fk['referred_table'] in names}
    return frames, pkeys, _to_primary(frames, fkeys, pkeys)


def read_duckdb(path, tables=None):
    """A DuckDB file and the keys of its PRIMARY KEY and FOREIGN KEY constraints."""
    import duckdb

    con = duckdb.connect(path, read_only=True)
    names = tables or [r[0] for r in con.execute('SELECT table_name FROM duckdb_tables() ORDER BY table_name').fetchall()]
    frames = {name: con.execute(f'SELECT * FROM "{name}"').df() for name in names}
    pkeys, fkeys = {}, defaultdict(dict)
    rows = con.execute("SELECT table_name, constraint_type, constraint_column_names, referenced_table, "
                       "referenced_column_names FROM duckdb_constraints() "
                       "WHERE constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')").fetchall()
    for table, kind, columns, parent, referenced in rows:
        if table in frames and len(columns) == 1:
            if kind == 'PRIMARY KEY':
                pkeys[table] = columns[0]
            elif parent in frames:
                fkeys[table][columns[0]] = (parent, referenced[0])
    con.close()
    return frames, pkeys, _to_primary(frames, dict(fkeys), pkeys)


def _to_primary(frames, fkeys, pkeys):
    """Foreign keys (column -> parent table) to parents with a single-column primary key. A key that references another
    unique column of its parent is rewritten in `frames` to the parent's primary key (null when no parent row has it)."""
    out = {}
    for table, cols in fkeys.items():
        out[table] = {}
        for column, (parent, referenced) in cols.items():
            if parent not in pkeys:
                continue
            if referenced != pkeys[parent]:
                lookup = frames[parent].set_index(referenced)[pkeys[parent]]
                if not lookup.index.is_unique:
                    raise ValueError(f'{table}.{column} references {parent}.{referenced}, which is not unique')
                frames[table][column] = frames[table][column].map(lookup)
            out[table][column] = parent
    return out


def read_neo4j(uri, auth=None, database=None):
    """Node labels and relationship types of a Neo4j graph as tables (see `Database.from_neo4j`). A node with several
    labels is filed under its first label; a relationship type joining several label pairs gets one table per pair,
    named `<type>_<source label>_<target label>`."""
    from neo4j import GraphDatabase

    nodes, edges = defaultdict(list), defaultdict(list)
    with GraphDatabase.driver(uri, auth=auth) as driver, driver.session(database=database) as session:
        for r in session.run('MATCH (n) RETURN elementId(n) AS id, labels(n)[0] AS label, properties(n) AS p'):
            nodes[r['label']].append({'id': r['id'], **_native(r['p'])})
        query = ('MATCH (a)-[r]->(b) RETURN elementId(r) AS id, type(r) AS type, elementId(a) AS src, labels(a)[0] AS sl, '
                 'elementId(b) AS dst, labels(b)[0] AS dl, properties(r) AS p')
        for r in session.run(query):
            edges[r['type'], r['sl'], r['dl']].append({'id': r['id'], 'src': r['src'], 'dst': r['dst'], **_native(r['p'])})
    pairs = defaultdict(int)
    for kind, _, _ in edges:
        pairs[kind] += 1
    frames = {label: pd.DataFrame(rows) for label, rows in nodes.items()}
    pkeys, fkeys = {label: 'id' for label in frames}, {}
    for (kind, source, target), rows in edges.items():
        name = kind if pairs[kind] == 1 else f'{kind}_{source}_{target}'
        frames[name], pkeys[name], fkeys[name] = pd.DataFrame(rows), 'id', {'src': source, 'dst': target}
    for frame in frames.values():
        for column in frame.columns:
            if frame[column].map(lambda v: hasattr(v, 'tzinfo')).any():
                frame[column] = pd.to_datetime(frame[column], utc=True)
    return frames, pkeys, fkeys


def _native(properties):
    """Neo4j property values as Python values: temporal values natively, lists and maps as strings."""
    out = {}
    for key, value in properties.items():
        if hasattr(value, 'to_native'):
            value = value.to_native()
        elif isinstance(value, (list, dict)):
            value = str(value)
        out[key] = value
    return out
