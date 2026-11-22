"""The released pretraining corpus from the Hugging Face Hub, laid out as ``openrfm train backbone corpus`` writes it:
CORPUS/sources/<group>/<world>, CORPUS/<archives>/<alias> links with the task catalog, and CORPUS/consumers.

The dataset holds ``pretraining/worlds-NNNN.tar.zst`` shards of 250 aliases each, ``index.json`` (aliases and sha256 of
each shard), ``sources.json`` (the source world of each alias; several aliases can share one world) and the catalog."""
import json
import shutil
import subprocess
from pathlib import Path

from openrfm.hub import DATA_REPO, sha256

from .corpus import fetch_consumers


def _file(name: str, out: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(DATA_REPO, f"pretraining/{name}", repo_type="dataset", local_dir=out / "download"))


def fetch(out: Path, config: dict) -> dict:
    """Download and lay out the corpus in ``out``, one shard at a time (each archive is removed once extracted)."""
    index = json.loads(_file("index.json", out).read_text())
    source_of = json.loads(_file("sources.json", out).read_text())
    union, staging = out / config["archives"], out / "staging"
    union.mkdir(parents=True, exist_ok=True)
    for shard, entry in sorted(index.items()):
        aliases = entry["worlds"]
        if all((union / alias).is_symlink() for alias in aliases):
            continue
        archive = _file(Path(shard).name, out)
        if sha256(archive) != entry["sha256"]:
            raise ValueError(f"{shard} does not match its released sha256")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(staging)], check=True)
        for alias in aliases:
            source = out / source_of[alias]
            if source.exists():
                shutil.rmtree(staging / alias)
            else:
                source.parent.mkdir(parents=True, exist_ok=True)
                (staging / alias).rename(source)
            link = union / alias
            if not link.is_symlink():
                link.symlink_to(Path("..") / source_of[alias])
        archive.unlink()
    shutil.rmtree(staging, ignore_errors=True)
    for name in ("compact_task_catalog.json", "corpus_manifest.json", "replay_selection.json"):
        shutil.copyfile(_file(name, out), union / name)
    fetch_consumers(config, out)
    return dict(aliases=len(source_of), worlds=len(set(source_of.values())), shards=len(index))
