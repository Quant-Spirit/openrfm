"""Build, register and load the native episode extension.

    python -m openrfm.sampler.native_build --build-root build/episodes --output build/episodes.json --arch 9.0
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys

import torch



def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_native_build(path, *, preflight_only=False, import_only=False):
    """Load a native build after checking its receipt against the current sources and GPU."""
    if import_only and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU native import requires CUDA_VISIBLE_DEVICES empty')
    receipt = json.loads(Path(path).read_text())
    library = Path(receipt['library'])
    if sha(library) != receipt['library_sha256']:
        raise ValueError('native library hash differs from build receipt')
    if receipt['sources'] != checkout_source_hashes():
        raise ValueError('native build was compiled from different kernel sources')
    device = receipt['device']
    if device == 'cuda':
        capability = receipt['capability']
        if (not isinstance(capability, list) or len(capability) != 2
                or any(type(value) is not int or value < 0 for value in capability)):
            raise ValueError('native CUDA receipt requires a two-integer capability')
    if preflight_only and not import_only:
        return device
    if not import_only:
        check_receipt_device(receipt)
    module = sys.modules.get('_openrfm_episodes')
    if module is None:
        spec = importlib.util.spec_from_file_location('_openrfm_episodes', library)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules['_openrfm_episodes'] = module
    elif Path(module.__file__).resolve() != library.resolve():
        raise RuntimeError('a different native extension is already loaded; start a clean process')
    if import_only:
        if not hasattr(module, 'rt_retrieve') or (device == 'cuda' and not module.has_cuda):
            raise RuntimeError('build does not contain the requested RT kernels')
        return device
    from openrfm.sampler import native as gpu_episodes_ext
    gpu_episodes_ext.admit_extension(module, via='receipt', receipt_path=path, receipt=receipt)
    from openrfm.sampler.episodes import configure_backend
    configure_backend('native')
    if not hasattr(module, 'rt_retrieve') or (device == 'cuda' and not module.has_cuda):
        raise RuntimeError('build does not contain the requested RT kernels')
    return device


def check_receipt_device(receipt):
    """Runtime device/capability gate shared by every receipt admission."""
    if receipt['device'] == 'cuda':
        capability = receipt['capability']
        if not torch.cuda.is_available():
            raise RuntimeError('native acceptance requires CUDA; no skip or CPU substitution')
        if list(torch.cuda.get_device_capability()) != capability:
            raise RuntimeError('GPU architecture differs from build receipt')


def checkout_source_hashes():
    """The kernel sources the acceptance receipt must certify."""
    from openrfm.sampler.native import KERNELS
    return {p.name: sha(p) for p in sorted(KERNELS.iterdir()) if p.suffix in ('.cpp','.cu','.h','.cuh')}

def build(args):
    if platform.system() != 'Linux':
        raise RuntimeError('native builds require Linux')
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA build requires a visible GPU')
        capability = list(torch.cuda.get_device_capability())
        if capability != [int(x) for x in args.arch.split('.')]:
            raise RuntimeError(f'expected sm_{args.arch}, got {capability}')
        if os.environ.get('TORCH_CUDA_ARCH_LIST') != args.arch:
            raise ValueError('TORCH_CUDA_ARCH_LIST must match explicit --arch')
    else:
        capability = None
    if args.library is not None:
        library = Path(args.library)
        if not library.is_file():
            raise FileNotFoundError(f'native library does not exist: {library}')
        library = str(library.resolve())
    else:
        from openrfm.sampler.episodes import configure_backend
        module = configure_backend('native', build=True, with_cuda=args.device == 'cuda',
                                   build_root=args.build_root, verbose=True)
        library = module.__file__
    write_new(args.output, dict(schema_version=1, device=args.device, capability=capability,
        machine=platform.machine(), torch=torch.__version__, cuda=torch.version.cuda,
        library=library, library_sha256=sha(library), sources=checkout_source_hashes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--library', help='register an already-built extension instead of rebuilding')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--arch', default='9.0')
    args = parser.parse_args()
    if (args.build_root is None) == (args.library is None):
        parser.error('provide exactly one of --build-root (rebuild) or --library (register an existing build)')
    if not re.fullmatch(r'[0-9]+\.[0-9]+', args.arch):
        parser.error('--arch must be a CUDA capability such as 9.0, 10.0 or 12.0')
    build(args)


if __name__ == '__main__':
    main()
