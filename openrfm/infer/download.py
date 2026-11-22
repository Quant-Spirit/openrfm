"""`openrfm download`: a benchmark database and its tasks, checked by sha256 (openrfm/configs/infer/downloads.yaml), in the
RelBench layout `DATA/<db>/{db,tasks}` that `openrfm predict` reads: RelBench databases from the RelBench server,
4DBInfer databases from their official archives mirrored on the Hugging Face Hub."""
import fcntl
import shutil
import tarfile
import zipfile
from pathlib import Path

from openrfm.hub import sha256
from openrfm.infer import recipe


def fetch(url, path, digest):
    """`url` saved to `path` unless a file with sha256 `digest` is already there."""
    path = Path(path)
    if not (path.exists() and sha256(path) == digest):
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name+'.partial')
        import requests
        with requests.get(url, stream=True, timeout=60) as response, partial.open('wb') as out:
            response.raise_for_status()
            for chunk in response.iter_content(1 << 20):
                out.write(chunk)
        if sha256(partial) != digest:
            raise ValueError(f'{url} does not match its released sha256')
        partial.replace(path)
    return path


def relbench(db, data, out):
    """`db.zip` into OUT/<db>/db and each benchmark task zip into OUT/<db>/tasks/<task>, named after the zip (a zip's
    own folder name can differ, e.g. `user-rating-churn` in `rel-ratebeer/tasks/user-churn.zip`)."""
    source = recipe.load('downloads.yaml')['relbench']
    for name, digest in source['files'].items():
        if name.startswith(f'{db}/'):
            archive = fetch(source['url']+name, Path(data)/'raw'/name, digest)
            folder = Path(out)/db/('tasks/'+Path(name).stem if '/tasks/' in name else 'db')
            with zipfile.ZipFile(archive) as z:
                for member in z.infolist():
                    if not member.is_dir():
                        path = folder.joinpath(*Path(member.filename).parts[1:])
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(member) as src, path.open('wb') as dst:
                            shutil.copyfileobj(src, dst)


def dbinfer(db, data, out):
    """The 4DBInfer archive of `db`, from the Hugging Face mirror, extracted under DATA/raw/, then written in RelBench
    layout to OUT/<db>."""
    from huggingface_hub import hf_hub_download
    from openrfm.infer import dbinfer as layout
    source = recipe.load('downloads.yaml')['dbinfer']
    name, digest = source['files'][db]
    archive = hf_hub_download(source['repo'], name, repo_type='dataset', local_dir=Path(data)/'raw'/'4dbinfer')
    if sha256(archive) != digest:
        raise ValueError(f'{archive} does not match its released sha256')
    with tarfile.open(archive) as tar:
        tar.extractall(Path(data)/'raw', filter='data')
    layout.convert(db, Path(data)/'raw', out)


def download(db, data):
    """DATA/<db> of benchmark database `db`, downloaded unless it is already there. The downloaded files stay in
    DATA/raw/."""
    dest = Path(data)/db
    Path(data).mkdir(parents=True, exist_ok=True)
    with open(Path(data)/f'.{db}.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)          # one download of `db` at a time; the others then find DATA/<db>
        if not dest.exists():
            partial = Path(data)/'.partial'/db
            shutil.rmtree(partial, ignore_errors=True)
            (dbinfer if db in recipe.load('downloads.yaml')['dbinfer']['files'] else relbench)(db, data, partial.parent)
            partial.replace(dest)
    return dest
