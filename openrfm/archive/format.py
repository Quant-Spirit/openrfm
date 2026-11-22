"""RT archive: little-endian, individually hashed, mmap-able typed arrays; the manifest is written last."""
from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import weakref
import warnings
from pathlib import Path

import numpy as np

MANIFEST = 'archive_manifest.json'
ROOT_SUFFIX = '_archive_v2'
FLOATS = {'cell_numbers', 'cell_datetimes', 'cell_booleans', 'embedding_bits'}
INDICES = {
    'node_timestamps', 'node_table_name_idxs', 'f2p_rowptr', 'f2p_col_idx',
    'p2f_rowptr', 'p2f_col_idx', 'p2f_table_name_idxs', 'p2f_table_types',
    'p2f_timestamps', 'cell_nodes', 'cell_columns', 'cell_types',
    'cell_text_ids', 'cell_codes', 'cell_num_classes',
}
DTYPES = {'float32': '<f4', 'bf16': '<u2', 'uint8': 'u1', 'uint16': '<u2', 'int32': '<i4', 'int64': '<i8'}
CHUNK = 262144
_MAPPINGS = {}

_VERIFIED_SHA256 = {}
_VERIFIED_BOUNDS = {}
_VERIFIED_MANIFESTS = set()


def _file_identity(path):
    stat = os.stat(path)
    return (stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def verified_sha256(path, stamp=None):
    """``sha256_file`` memoized per process on the file's identity stamps."""
    key = stamp or _file_identity(path)
    digest = _VERIFIED_SHA256.get(key)
    if digest is None:
        digest = sha256_file(path)
        _VERIFIED_SHA256[key] = digest
    return digest


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def index_dtype(minimum, maximum):
    """Signed sentinels count: -1 cannot be silently wrapped into uint8."""
    if 0 <= minimum <= maximum <= 255:
        return 'uint8'
    if 0 <= minimum <= maximum <= 65535:
        return 'uint16'
    if -(2**31) <= minimum <= maximum < 2**31:
        return 'int32'
    if -(2**63) <= minimum <= maximum < 2**63:
        return 'int64'
    raise ValueError('index outside signed int64 address space')


def bounds(array):
    lo = hi = 0
    if array.size:
        lo, hi = int(array.min()), int(array.max())
    return lo, hi


def _verified_bounds(path, dtype, shape):
    key = (_file_identity(path), dtype, tuple(shape))
    value = _VERIFIED_BOUNDS.get(key)
    if value is None:
        value = bounds(_map(path, dtype, shape))
        _VERIFIED_BOUNDS[key] = value
    return value


def archive_format(root):
    root = archive_root(root)
    path = Path(root) / MANIFEST
    if not path.exists():
        if (Path(root).name.endswith(ROOT_SUFFIX) or Path(root).parent.name.endswith(ROOT_SUFFIX)
                or (Path(root) / 'cell_numbers.bin').exists() or (Path(root) / '.v2-writing').exists()):
            raise ValueError(f'uncommitted v2 archive: {root}')
        return 'v1'
    manifest = json.loads(path.read_text())
    if manifest.get('archive_format') != 'v2':
        raise ValueError(f'unsupported archive format: {path}')
    return 'v2'


def archive_root(root):
    """Converters publish a sibling root; native preprocessing writes DB/v2."""
    root = Path(root)
    if (root / 'v2').exists():
        if not (root / 'v2' / MANIFEST).is_file():
            raise ValueError(f'uncommitted direct v2 archive: {root}')
        return root / 'v2'
    return root


def read_manifest(root, *, verify=True, shared_caches=(), verify_workers=1):
    bundle = Path(root)
    root = archive_root(root)
    manifest = json.loads((root / MANIFEST).read_text())
    if shared_caches:
        assert_private_payloads(bundle, shared_caches)
    if verify and verify_workers>1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=verify_workers,thread_name_prefix='rt-admit') as pool:
            list(pool.map(verified_sha256,(path for path in root.rglob('*')
                                           if path.is_file() and path.name!=MANIFEST)))
    return validate_manifest(root, manifest, verify=verify)


def shared_cache_aliases(bundle, shared_caches):
    """Bundle files that share an inode with a file in a text-embedding cache dir."""
    bundle = Path(bundle)
    linked = {}
    for path in sorted(bundle.rglob('*')):
        if path.is_file() and not path.is_symlink():
            stat = path.stat()
            if stat.st_nlink > 1:
                linked.setdefault((stat.st_dev, stat.st_ino), []).append(path)
    if not linked:
        return []
    aliases = []
    for cache in shared_caches:
        cache = Path(cache)
        if not cache.is_dir():
            raise ValueError(f'text-embedding cache dir is missing: {cache}')
        for path in sorted(cache.rglob('*')):
            if path.is_file() and not path.is_symlink():
                stat = path.stat()
                for payload in linked.get((stat.st_dev, stat.st_ino), ()):
                    aliases.append(dict(payload=payload.relative_to(bundle).as_posix(),
                                        cache_file=str(path), nlink=stat.st_nlink))
    return aliases


def assert_private_payloads(bundle, shared_caches):
    """Fail closed when any bundle file shares its inode with a cache file."""
    aliases = shared_cache_aliases(bundle, shared_caches)
    if aliases:
        names = ', '.join(sorted({a['payload'] for a in aliases}))
        raise ValueError(f'bundle payload hard-linked into a shared text-embedding cache '
                         f'(a later pre rewrites it): {bundle}: {names}')


def _inventory(root):
    """{relative path: (dev, ino, size, mtime_ns, ctime_ns)} of every file, via scandir."""
    root = os.fspath(root)
    base = len(root) + 1
    files, pending = {}, [root]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry.path)
                elif not entry.is_dir():
                    s = entry.stat()
                    files[entry.path[base:]] = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    return files


def validate_manifest(root, manifest, *, verify=True):
    """Reuse structural verification only with an unchanged complete inventory."""
    root = Path(root)
    files = _inventory(root)
    fingerprint = hashlib.sha256(json.dumps(
        [manifest, sorted((name, *stamp) for name, stamp in files.items())],
        sort_keys=True, separators=(',', ':')).encode()).digest()
    if fingerprint not in _VERIFIED_MANIFESTS:
        _validate_manifest(root, manifest, verify=verify, files=files)
        if verify:
            _VERIFIED_MANIFESTS.add(fingerprint)
    return manifest


def _validate_manifest(root, manifest, *, verify=True, files=None):
    root = Path(root)
    files = _inventory(root) if files is None else files
    if manifest.get('archive_format') != 'v2' or manifest.get('byte_order') != 'little':
        raise ValueError('unsupported RT archive format/byte order')
    arrays = manifest.get('arrays', {})
    if set(arrays) != FLOATS | INDICES:
        raise ValueError('v2 archive has missing/unknown arrays')
    expected = {MANIFEST}
    for name, entry in arrays.items():
        dtype = entry.get('dtype')
        if dtype not in DTYPES or not (
                (name in {'cell_datetimes', 'cell_numbers'} and dtype == 'float32')
                or ((name in FLOATS) == (dtype == 'bf16') and dtype != 'float32')):
            raise ValueError(f'forbidden v2 dtype: {name}={dtype}')
        if entry.get('file') != name + '.bin':
            raise ValueError(f'invalid payload path: {name}')
        shape = entry.get('shape')
        if not isinstance(shape, list) or len(shape) != 1 or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f'invalid shape: {name}')
        path = root / entry['file']
        expected.add(entry['file'])
        size = math.prod(shape) * np.dtype(DTYPES[dtype]).itemsize
        if entry['file'] not in files:
            raise FileNotFoundError(str(path))
        if entry.get('nbytes') != size or files[entry['file']][2] != size:
            raise ValueError(f'payload size mismatch: {name}')
        if verify and verified_sha256(path, files[entry['file']]) != entry.get('sha256'):
            raise ValueError(f'payload hash mismatch: {name}')
        if name in INDICES:
            if type(entry.get('min')) is not int or type(entry.get('max')) is not int:
                raise ValueError(f'index bounds must be integers: {name}')
            if index_dtype(entry['min'], entry['max']) != dtype:
                raise ValueError(f'nonminimal index width: {name}')
            if verify and _verified_bounds(path, dtype, shape) != (entry['min'], entry['max']):
                raise ValueError(f'index bounds mismatch: {name}')
    for name, digest in manifest.get('metadata_sha256', {}).items():
        if Path(name).name != name or not name.endswith('.json') or name == MANIFEST:
            raise ValueError('invalid metadata path')
        expected.add(name)
        if verify and verified_sha256(root / name, files.get(name)) != digest:
            raise ValueError(f'metadata hash mismatch: {name}')
    preserved = manifest.get('preserved_files_sha256', {})
    if 'preserved_files_sha256' in manifest:
        if preserved != manifest.get('conversion_source', {}).get('files'):
            raise ValueError('preserved inventory disagrees with conversion source')
        for name, digest in preserved.items():
            path = Path(name)
            if (path.is_absolute() or '..' in path.parts or path.as_posix() != name
                    or name == MANIFEST or name in {k + '.bin' for k in arrays}):
                raise ValueError(f'invalid preserved path: {name}')
            expected.add(name)
            if name not in files:
                raise ValueError(f'missing preserved file: {name}')
            if verify and verified_sha256(root / name, files[name]) != digest:
                raise ValueError(f'preserved file hash mismatch: {name}')
    actual = set(files)
    if actual != expected:
        raise ValueError(f'unmanifested/missing v2 files: {actual ^ expected}')
    if 'table_info.json' not in expected or 'text_map.json' not in expected:
        raise ValueError('missing RT schema metadata')
    if type(manifest.get('num_nodes')) is not int or manifest['num_nodes'] <= 0:
        raise ValueError('invalid num_nodes')
    if manifest['arrays']['node_timestamps']['shape'] != [manifest['num_nodes']]:
        raise ValueError('node count mismatch')
    if type(manifest.get('d_text')) is not int or manifest['d_text'] <= 0:
        raise ValueError('invalid text dimension')
    if manifest['arrays']['embedding_bits']['shape'][0] % manifest['d_text']:
        raise ValueError('partial embedding row')
    n = manifest['num_nodes']
    m = arrays['cell_nodes']['shape'][0]
    if not m or any(arrays[k]['shape'] != [m] for k in arrays if k.startswith('cell_')):
        raise ValueError('unaligned/empty cell arrays')
    if arrays['node_table_name_idxs']['shape'] != [n]:
        raise ValueError('unaligned node table array')
    for direction in ('f2p', 'p2f'):
        ptr_entry = arrays[direction + '_rowptr']
        dst_entry = arrays[direction + '_col_idx']
        if ptr_entry['shape'] != [n+1] or dst_entry['min'] < 0 or dst_entry['max'] >= n:
            raise ValueError('invalid CSR shape/destinations')
        ptr = _map(root / ptr_entry['file'], ptr_entry['dtype'], ptr_entry['shape'])
        if int(ptr[0]) != 0 or int(ptr[-1]) != dst_entry['shape'][0]:
            raise ValueError('CSR endpoints disagree with destinations')
        for start in range(0, n, CHUNK):
            part = ptr[start:start+CHUNK+1]
            if np.any(part[1:] < part[:-1]):
                raise ValueError('nonmonotone CSR offsets')
    if any(arrays[k]['shape'] != arrays['p2f_col_idx']['shape'] for k in
           ('p2f_table_name_idxs', 'p2f_table_types', 'p2f_timestamps')):
        raise ValueError('unaligned p2f metadata')
    if arrays['cell_nodes']['min'] < 0 or arrays['cell_nodes']['max'] >= n:
        raise ValueError('cell row address out of bounds')
    if arrays['cell_types']['min'] < 0 or arrays['cell_types']['max'] > 4:
        raise ValueError('unknown semantic type')
    meta = json.loads((root / 'table_info.json').read_text())
    intervals = sorted((v['node_idx_offset'], v['num_nodes']) for v in meta.values())
    end = 0
    for offset, count in intervals:
        if type(offset) is not int or type(count) is not int or offset != end or count < 0:
            raise ValueError('table offsets do not partition archive rows')
        end += count
    if end != n:
        raise ValueError('table offsets disagree with archive node count')
    text_map = json.loads((root / 'text_map.json').read_text())
    if (not isinstance(text_map, dict) or set(text_map.values()) != set(range(len(text_map)))
            or arrays['embedding_bits']['shape'][0] != len(text_map)*manifest['d_text']):
        raise ValueError('embedding rows disagree with text vocabulary')
    return manifest


_LIBC = {}


def _posix_readonly_buffer(fd, size):
    """Descriptor-free mmap for Python < 3.13 (which lacks trackfd=False).

    The ctypes buffer owns munmap; NumPy and Torch retain that buffer through
    their base/owner references, including when only a tensor view survives.
    """
    import ctypes
    if os.name != 'posix':
        raise RuntimeError('descriptor-free archive residency requires POSIX mmap')
    if not _LIBC:
        libc = ctypes.CDLL(None, use_errno=True)
        map_region = libc.mmap
        map_region.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_long)
        map_region.restype = ctypes.c_void_p
        unmap = libc.munmap
        unmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        unmap.restype = ctypes.c_int
        _LIBC.update(map=map_region, unmap=unmap)
    map_region, unmap = _LIBC['map'], _LIBC['unmap']
    address = map_region(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
    if address == ctypes.c_void_p(-1).value:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    try:
        buffer = (ctypes.c_ubyte * size).from_address(address)
        finalizer = weakref.finalize(buffer, unmap, address, size)
        finalizer.atexit = False
    except BaseException:
        unmap(address, size)
        raise
    return buffer


def _map(path, dtype, shape):
    if not math.prod(shape):
        return np.empty(shape, dtype=DTYPES[dtype])
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            buffer = mmap.mmap(fd, 0, access=mmap.ACCESS_READ, trackfd=False)
        except TypeError:
            buffer = _posix_readonly_buffer(fd, os.fstat(fd).st_size)
    finally:
        os.close(fd)
    array = np.ndarray(tuple(shape), dtype=DTYPES[dtype], buffer=buffer)
    array.setflags(write=False)
    start = array.ctypes.data
    _MAPPINGS[start] = (start + array.nbytes, weakref.ref(
        array, lambda owner, start=start: _MAPPINGS.pop(start, None)
        if _MAPPINGS.get(start, (None, None))[1] is owner else None))
    return array


def load_arrays(root, embedding_model, d_text, *, verify_workers=1):
    import torch
    root = archive_root(root)
    manifest = read_manifest(root) if verify_workers==1 else read_manifest(root,verify_workers=verify_workers)
    if (manifest['embedding_model'], manifest['d_text']) != (embedding_model, d_text):
        raise ValueError('archive encoder configuration mismatch')
    result = {'num_nodes': manifest['num_nodes']}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', message='The given NumPy array is not writable.*', category=UserWarning)
        for name, entry in manifest['arrays'].items():
            tensor = torch.from_numpy(_map(Path(root) / entry['file'], entry['dtype'], entry['shape']))
            if entry['dtype'] == 'bf16' and name != 'embedding_bits':
                tensor = tensor.view(torch.bfloat16)
            result[name] = tensor
    return result


def compact_tensor(value):
    """Select width on CPU, before residency/device transfer."""
    import torch
    if value.dtype == torch.bool:
        return value.to(torch.uint8)
    if value.is_floating_point():
        return value.to(torch.bfloat16)
    if value.device.type != 'cpu':
        raise ValueError('compact tensors during CPU admission only')
    raw = value.numpy()
    return torch.from_numpy(raw.astype(DTYPES[index_dtype(*bounds(raw))], copy=False))


def widen_index(value):
    """int64 view of compact index storage (uint16 has no CUDA index kernels; a uint8 index reads as a mask)."""
    import torch
    if value.dtype == torch.uint16:
        return value.view(torch.int16).long().bitwise_and(65535)
    return value.long() if value.dtype in (torch.uint8, torch.int32) else value


def widen_object(value):
    """Shallow copy of a tensor-bearing object with every integer field widened."""
    import copy
    import torch
    result = copy.copy(value)
    for name, item in vars(value).items():
        if torch.is_tensor(item) and not item.is_floating_point() and item.dtype != torch.bool:
            object.__setattr__(result, name, widen_index(item))
    return result


def take(value, indices):
    """Gather narrow storage, then widen only the selected batch (undoing sign extension after the gather)."""
    import torch
    if value.dtype == torch.uint16:
        return value.view(torch.int16)[indices].long().bitwise_and(65535)
    result = value[indices]
    return result.long() if result.dtype in (torch.uint8, torch.int32) else result


