"""Private native build and dispatch support; the public API is openrfm.sampler.episodes."""
from __future__ import annotations

import hashlib
import importlib
import importlib.resources
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from types import ModuleType
from collections.abc import Mapping
from typing import Any, Literal

import torch

_MODE: Literal['auto', 'torch', 'native'] = 'auto'
_EXTENSION: ModuleType | None = None
# The kernel sources ship inside the package; JIT builds are cached in the user cache directory.
KERNELS = Path(str(importlib.resources.files('openrfm.sampler')/'kernels'))
CACHE = Path(os.environ.get('XDG_CACHE_HOME') or Path.home()/'.cache')/'openrfm/episodes'
_ADMISSION: dict[str, Any] | None = None
NATIVE_RECEIPT_ENV = 'OPENRFM_NATIVE_BUILD_RECEIPT'
_UNADMITTED_WARNED = False


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def admit_extension(module: ModuleType, *, via: Literal['receipt', 'build'],
                    receipt_path: str | Path | None = None,
                    receipt: Mapping[str, Any] | None = None) -> ModuleType:
    """Record the loaded extension as admitted; fail closed on any mismatch."""
    global _EXTENSION, _ADMISSION
    from openrfm.sampler.native_build import checkout_source_hashes
    library = getattr(module, '__file__', None)
    if not library:
        raise RuntimeError('native extension has no library file; cannot verify its build')
    library_sha256 = _file_sha256(library)
    sources = checkout_source_hashes()
    if via == 'receipt':
        if receipt is None:
            raise ValueError('receipt admission requires the build receipt')
        if library_sha256 != receipt.get('library_sha256'):
            raise RuntimeError('loaded native library hash differs from its build receipt')
        if receipt.get('sources') != sources:
            raise RuntimeError('native build was compiled from different kernel sources')
    elif via != 'build':
        raise ValueError('admission must be via receipt or build')
    _EXTENSION = module
    _ADMISSION = dict(module=module, via=via, library=str(Path(library).resolve()),
                      library_sha256=library_sha256, sources=sources,
                      receipt=None if receipt_path is None else str(Path(receipt_path).resolve()))
    return module


def admission() -> dict[str, Any] | None:
    """Public copy of the current admission record (no module object)."""
    return None if _ADMISSION is None else {k: v for k, v in _ADMISSION.items() if k != 'module'}


def _admit_from_environment(extension: ModuleType) -> ModuleType | None:
    """Admit an implicitly imported extension only through an explicit receipt."""
    global _UNADMITTED_WARNED
    path = os.environ.get(NATIVE_RECEIPT_ENV)
    if path:
        import json
        receipt = json.loads(Path(path).read_text())
        library = Path(receipt['library'])
        if Path(extension.__file__).resolve() != library.resolve():
            raise RuntimeError(f'{NATIVE_RECEIPT_ENV} certifies {library}, but '
                               f'{extension.__file__} is loaded; start a clean process')
        from openrfm.sampler.native_build import check_receipt_device
        check_receipt_device(receipt)
        return admit_extension(extension, via='receipt', receipt_path=path, receipt=receipt)
    message = (f'native extension {getattr(extension, "__file__", "?")} was imported without a '
               'build receipt; load it with openrfm.sampler.native_build.load_native_build(receipt) '
               f'or name the receipt in {NATIVE_RECEIPT_ENV}')
    if _MODE == 'native':
        raise RuntimeError(message)
    if not _UNADMITTED_WARNED:
        import warnings
        warnings.warn(message + '; auto mode uses the torch oracle instead', RuntimeWarning)
        _UNADMITTED_WARNED = True
    return None


def get_extension(device: torch.device | None = None) -> ModuleType | None:
    global _EXTENSION
    if _MODE == 'torch':
        return None
    if _EXTENSION is None:
        try:
            _EXTENSION = importlib.import_module('_openrfm_episodes')
        except ModuleNotFoundError as error:
            if error.name != '_openrfm_episodes':
                raise
    extension = _EXTENSION
    if extension is not None and (_ADMISSION is None or _ADMISSION['module'] is not extension):
        extension = _admit_from_environment(extension)
    if device is not None and device.type == 'cuda' and extension is not None and not extension.has_cuda:
        extension = None
    if _MODE == 'native' and extension is None:
        raise RuntimeError('native episodes backend unavailable; call configure_backend(build=True)')
    return extension


def _cpu_build(directory: Path, sources: list[str], verbose: bool) -> ModuleType:
    from torch.utils.cpp_extension import include_paths, library_paths
    compiler = os.environ.get('CXX') or shutil.which('clang++') or shutil.which('c++')
    if not compiler:
        raise RuntimeError('CPU extension needs a C++ compiler')
    includes = include_paths() + [sysconfig.get_path('include')]
    flags = ['-std=c++20', '-O2', '-fPIC', '-DTORCH_EXTENSION_NAME=_openrfm_episodes',
             f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}']
    for path in includes:
        flags.extend(['-isystem', path])
    objects = []
    for source in sources:
        obj = directory / (Path(source).stem + '.o')
        if not obj.exists():
            partial = obj.with_suffix(f'.{os.getpid()}.tmp.o')
            result = subprocess.run([compiler, *flags, '-c', source, '-o', str(partial)], capture_output=True, text=True)
            if verbose or result.returncode:
                print(result.stdout + result.stderr, file=sys.stderr)
            result.check_returncode()
            partial.replace(obj)
        objects.append(str(obj))
    library = directory / ('_openrfm_episodes' + sysconfig.get_config_var('EXT_SUFFIX'))
    if not library.exists():
        partial_library = library.with_suffix(f'.{os.getpid()}.tmp.so')
        link = [compiler, '-shared', *objects, '-o', str(partial_library)]
        if sys.platform == 'darwin':
            link += ['-undefined', 'dynamic_lookup']
        for path in library_paths():
            link += ['-L', path, f'-Wl,-rpath,{path}']
        subprocess.run(link + ['-lc10', '-ltorch_cpu', '-ltorch', '-ltorch_python'], check=True)
        partial_library.replace(library)
    spec = importlib.util.spec_from_file_location('_openrfm_episodes', library)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load {library}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules['_openrfm_episodes'] = module
    return module


def configure(mode: Literal['auto', 'torch', 'native'] = 'auto', *, build: bool = False,
              with_cuda: bool | None = None, build_root: str | Path | None = None,
              verbose: bool = False) -> ModuleType | None:
    global _MODE, _EXTENSION
    if mode not in ('auto', 'torch', 'native'):
        raise ValueError('backend must be auto, torch, or native')
    if build:
        from torch.utils.cpp_extension import CUDA_HOME, load
        cuda = torch.cuda.is_available() if with_cuda is None else with_cuda
        if cuda and (sys.platform != 'linux' or CUDA_HOME is None):
            raise RuntimeError('CUDA compilation requires the CUDA toolkit (nvcc)')
        folder = KERNELS
        sources = [str(folder / name) for name in ('episodes.cpp', 'rt_retrieval.cpp', 'rt_composition.cpp', 'rt_provenance.cpp', 'rt_alias.cpp')]
        if cuda:
            sources.append(str(folder / 'episodes_cuda.cu'))
            sources.append(str(folder / 'rt_retrieval_cuda.cu'))
            sources.append(str(folder / 'rt_provenance_cuda.cu'))
        signature = hashlib.sha256()
        signature.update(f'{torch.__version__}|{sys.version}|{cuda}|{sys.platform}|host-c++20|cuda-c++17|O2|v2'.encode())
        signature.update(str((os.environ.get('CXX'), os.environ.get('TORCH_CUDA_ARCH_LIST'))).encode())
        if cuda:
            signature.update(str([torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]).encode())
        for path in sorted(folder.glob('*')):
            if path.suffix in ('.h', '.cpp', '.cu', '.cuh'):
                signature.update(path.read_bytes())
        directory = Path(build_root or CACHE) / signature.hexdigest()[:20]
        directory.mkdir(parents=True, exist_ok=True)
        if shutil.which('ninja'):
            _EXTENSION = load(name='_openrfm_episodes', sources=sources, build_directory=str(directory),
                              extra_cflags=['-O2', '-std=c++20'] + (['-DWITH_CUDA'] if cuda else []),
                              extra_cuda_cflags=['-O2', '-std=c++17', '--fmad=false', '-DWITH_CUDA'],
                              with_cuda=cuda, verbose=verbose)
        elif cuda:
            raise RuntimeError('CUDA JIT build requires Ninja')
        else:
            _EXTENSION = _cpu_build(directory, sources, verbose)
        admit_extension(_EXTENSION, via='build')
    _MODE = mode
    return get_extension()


