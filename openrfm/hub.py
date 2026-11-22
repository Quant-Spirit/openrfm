"""Released checkpoints (model repository) and data (dataset repository) on the Hugging Face Hub, verified by sha256."""
import hashlib
from pathlib import Path

REPO = 'zkchen/openrfm'
DATA_REPO = 'zkchen/openrfm'
CHECKPOINTS = {
    'backbone': ('openrfm_backbone.pt', '92a7f3d650872acd422d7d407df71b9e779b2fb1db3b98fe02408d9489938513'),
    'memory': ('openrfm_memory_v1.pt', '8e6310f5efe745038910afbe173a69af6397e483d38f0b73fd5cad6f083f7922'),
}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checkpoint(name, directory=None):
    """Local path of checkpoint `name` ('backbone' or 'memory'): from `directory`, else downloaded from the Hub."""
    file, digest = CHECKPOINTS[name]
    if directory is not None:
        path = Path(directory)/file
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(REPO, file))
    if sha256(path) != digest:
        raise ValueError(f'{path} does not match the sha256 of the released {name} checkpoint')
    return path
