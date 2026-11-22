"""The `openrfm` command: download, import, convert, predict, generate and train."""
import argparse
import sys
from pathlib import Path


def convert(a):
    import rustler_archive
    from openrfm.infer import recipe
    if a.dbinfer:
        from openrfm.infer import dbinfer
        dbinfer.convert(a.db, a.dbinfer, a.data)
    rustler_archive.convert(a.db, str(a.data), str(a.out), archive_format=a.format, text_encoder=a.text_encoder,
                            text_encoder_device=a.device, allow_cpu=a.allow_cpu, time_cutoffs_file=a.time_cutoffs)
    if a.db in recipe.load('databases.yaml')['own_row_time']:
        from openrfm.archive import node_time
        node_time.build(a.out/a.db, a.data/a.db)


def predict(a):
    from openrfm.infer.run import run
    if (a.task is None) == (a.spec is None):
        raise SystemExit('openrfm predict: give exactly one of --task and --spec')
    run(a.task, a.recipe, a.data, a.work, checkpoints=a.checkpoints, spec=a.spec)


def import_(a):
    from openrfm.data import Database
    time = {k: (v or None) for k, v in (item.split('=', 1) for item in a.time)}
    if a.source.startswith(('neo4j', 'bolt')):
        auth = tuple(a.auth.split(':', 1)) if a.auth else None
        db = Database.from_neo4j(a.source, auth=auth, database=a.database, time=time)
    else:
        db = Database.from_sql(a.source, tables=a.tables, time=time)
    print(db.write(a.data, a.name))
    for table, frame in db.tables.items():
        print(f'  {table}: {len(frame)} rows, key {db.pkeys.get(table)}, references {db.fkeys.get(table, {})}, '
              f'time {db.time.get(table)}')


def download(a):
    from openrfm.infer.download import download
    print(download(a.db, a.data))


def generate(rest):
    from openrfm.synth import generate as synth
    sys.argv = ['openrfm generate', *rest]
    synth.main()


def train(rest):
    if not rest or rest[0] not in ('backbone', 'memory'):
        raise SystemExit('usage: openrfm train {backbone,memory} --config CONFIG ...')
    if rest[0] == 'backbone':
        from openrfm.train.backbone.__main__ import main
    else:
        from openrfm.train.memory.__main__ import main
    main(rest[1:])


def parser():
    p = argparse.ArgumentParser(prog='openrfm', description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('download', help='download a benchmark database and its tasks into DATA/<db>/ (RelBench layout)')
    d.add_argument('--db', required=True, help='e.g. rel-f1 or stackexchange-4db')
    d.add_argument('--data', type=Path, required=True)
    i = sub.add_parser('import', help='write a SQL database or a Neo4j graph into DATA/<name>/ (RelBench layout)')
    i.add_argument('--source', required=True, help='duckdb:///file.duckdb, sqlite:///file.db, postgresql://..., '
                   'mysql+pymysql://... or neo4j://host:7687 (bolt:// also)')
    i.add_argument('--data', type=Path, required=True)
    i.add_argument('--name', required=True, help='database name; it is written to DATA/<name>/')
    i.add_argument('--tables', nargs='+', help='SQL tables to read (default: all)')
    i.add_argument('--time', nargs='+', default=[], metavar='TABLE=COLUMN',
                   help='row-time column of a table; TABLE= marks a static table (default: its only datetime column)')
    i.add_argument('--auth', help='Neo4j USER:PASSWORD')
    i.add_argument('--database', help='Neo4j database (default: the server default)')
    c = sub.add_parser('convert', help='convert a database (a world, a RelBench or a 4DBInfer database) into an archive')
    c.add_argument('--db', required=True, help='database name; it is read from DATA/<db>/')
    c.add_argument('--data', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True, help='the archive is written to OUT/<db>/')
    c.add_argument('--format', choices=('v1', 'v2'), default='v1')
    c.add_argument('--text-encoder', default='all-MiniLM-L12-v2')
    c.add_argument('--device', default='auto', help='text-encoder device')
    c.add_argument('--allow-cpu', action='store_true', help='permit CPU text encoding')
    c.add_argument('--time-cutoffs', help='JSON map from database name to its validation start')
    c.add_argument('--dbinfer', type=Path,
                   help='extracted 4DBInfer download; DATA/<db>/ is first written from it in RelBench layout')
    q = sub.add_parser('predict', help='predict one task with a released recipe and score it')
    q.add_argument('--task', help='benchmark task, e.g. rel-f1/driver-dnf; its database is downloaded when absent')
    q.add_argument('--spec', type=Path, help='YAML file declaring a task of your own database DATA/<db>/')
    q.add_argument('--data', type=Path, required=True, help='directory holding the database <db>/ in RelBench layout')
    q.add_argument('--recipe', choices=('r1', 'r2', 'r3'), default='r2')
    q.add_argument('--work', type=Path, default=Path('openrfm-work'))
    q.add_argument('--checkpoints', type=Path, help='directory holding the checkpoint files (default: download)')
    sub.add_parser('generate', help='generate the synthetic corpus (openrfm.synth.generate)', add_help=False)
    sub.add_parser('train', help='train the backbone or the memory module', add_help=False)
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ('generate', 'train'):
        return (generate if argv[0] == 'generate' else train)(argv[1:])
    a = parser().parse_args(argv)
    {'download': download, 'import': import_, 'convert': convert, 'predict': predict}[a.command](a)


if __name__ == '__main__':
    main()
