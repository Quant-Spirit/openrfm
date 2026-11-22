"""The archive a recipe reads: the converted benchmark database (for the `repair` schema with its restored event
outcomes, `dbinfer.py`), or the converted output of the task's schema program."""
import hashlib
import json
import shutil
from pathlib import Path

from openrfm.infer import catalog, recipe


def convert(db, data, out, cutoffs, own_row_time=False):
    """`rustler pre` of `data/<db>` into `out/<db>`, then with `own_row_time` its node-time sidecar
    (`openrfm.archive.node_time`). Format v1 keeps numbers and timestamps in FP32. The archive is rebuilt whenever the
    tables or the cutoffs differ from those it was built from."""
    done = Path(out)/db/'CONVERTED'
    stamp = fingerprint(Path(data)/db/'db', cutoffs, own_row_time)
    if not (done.exists() and done.read_text() == stamp):
        shutil.rmtree(Path(out)/db, ignore_errors=True)
        import rustler_archive
        rustler_archive.convert(db, str(data), str(out), archive_format='v1', time_cutoffs_file=str(cutoffs))
        if own_row_time:
            from openrfm.archive import node_time
            node_time.build(Path(out)/db, Path(data)/db)
        done.write_text(stamp)
    return Path(out)/db


def fingerprint(tables, cutoffs, own_row_time):
    """Digest of the table files (name, size, modification time), the cutoffs and the node-time setting."""
    h = hashlib.sha256(Path(cutoffs).read_bytes()+bytes([own_row_time]))
    for path in sorted(Path(tables).glob('*.parquet')):
        st = path.stat()
        h.update(f'{path.name}:{st.st_size}:{st.st_mtime_ns};'.encode())
    return h.hexdigest()


def benchmark(task, data, out):
    """The archive of database `data/<db>` of `task` in `out/<db>`. Its normalization cutoff and own-row-time setting
    come from openrfm/configs/infer/databases.yaml for a benchmark database, else from the task (`cutoff`, by default the first
    VAL time; `own_row_time`)."""
    db, settings = task['db'], recipe.load('databases.yaml')
    if db in settings['cutoffs']:
        cutoff, own_row_time = settings['cutoffs'][db], db in settings['own_row_time']
    else:
        cutoff = str(task.get('cutoff') or catalog.first_time(data, task, 'val'))
        own_row_time = task.get('own_row_time', False)
    cutoffs = Path(out)/f'{db}-cutoffs.json'
    cutoffs.parent.mkdir(parents=True, exist_ok=True)
    cutoffs.write_text(json.dumps({db: cutoff}))
    return convert(db, data, out, cutoffs, own_row_time)


def with_history(data, db, out):
    """`out/<db>`: the database `data/<db>` with each table of `data/<db>/history` in place of its own."""
    source, dest = Path(data)/db, Path(out)/db
    (dest/'db').mkdir(parents=True, exist_ok=True)
    for path in (source/'db').glob('*.parquet'):
        restored = source/'history'/path.name
        link = dest/'db'/path.name
        if not link.is_symlink():
            link.symlink_to((restored if restored.exists() else path).resolve())
    for name in ('tasks', 'task_metadata.json'):
        if not (dest/name).is_symlink():
            (dest/name).symlink_to((source/name).resolve())
    return Path(out)


def archive(task, cell, proto, data, work):
    """(archive base, raw database directory) of `task` under the recipe cell's schema."""
    db, work = proto['db'], Path(work)
    if cell['schema'] == 'repair':
        if not (Path(data)/db/'history').is_dir():
            raise FileNotFoundError(f'{Path(data)/db}/history: the repair schema reads a 4DBInfer database written by openrfm download')
        data, work = with_history(data, db, work/'repair'/'data'), work/'repair'
    if cell['schema'] != 'program':
        return benchmark(proto, data, work/'archives'), Path(data)/db
    from openrfm.rewrite import program as rewrite
    out = work/'rewrite'/task.replace('/', '__')
    if not (out/'schema_manifest.json').exists():
        info = rewrite.task_info(task, Path(data)/db)
        rewrite.materialize(rewrite.validate(recipe.program(task), info), info, out)
    return convert(db, out/'data', out/'pre', out/'fit_time_cutoffs.json'), out/'data'/db
