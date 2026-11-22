"""Sampler mixtures: a distribution over context-construction settings for training.

A component is drawn deterministically once per draw unit, never per optimizer step.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from itertools import product
from typing import Any, Mapping

LABEL_FREE_FIELDS = frozenset({
    'walk_count', 'walk_steps', 'candidate_count', 'seed_hops', 'path_depth', 'rw_extension_percent',
    'own_recent_first',
    'own_recent_interleave',
    'canonical_recent',
})
MIXTURE_FIELDS = LABEL_FREE_FIELDS | frozenset({
    'label_pool_reserve', 'own_label_rows', 'k_enc_max',
    'path_quota_rows', 'path_quota_member_rows', 'path_quota_share', 'path_quota_parent_rows',
    'path_quota_grandparent_rows', 'path_quota_grandchild_rows', 'path_quota_order',
    'seed_tier_percent', 'seed_local_rows', 'seed_balance', 'seed_rank', 'seed_fill', 'seed_hide_labels',
    'seed_local_policy', 'seed_route_fanout', 'path_stream_fill', 'seed_hop_cap', 'seed_program_mature',
    'peer_extension_percent', 'peer_extension_rows', 'extension_labels_visible',
})
COMPONENT_KEYS = frozenset({'name', 'weight', 'views', 'set', 'vary'})
VIEWS = ('forecast', 'all')
UNITS = ('group_window', 'task_block')
MAX_SUPPORT_PER_COMPONENT = 1024
MAX_COMPONENTS = 64
_MASK = 0xFFFFFFFFFFFFFFFF
_SALT = 0x51_6D_49_58_51_31_36_00


def _scalar(value):
    if type(value) not in (int, bool, str):
        raise ValueError(f'sampler mixture values must be int, bool or str, got {value!r}')
    return value


def canonical_mixture(value) -> tuple:
    """Parse a YAML/JSON mixture (list of component mappings) or its canonical tuple form."""
    if value in (None, (), []):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError('sampler_mixture must be a list of components')
    out = []
    for item in value:
        if isinstance(item, Mapping):
            unknown = set(item) - COMPONENT_KEYS
            if unknown:
                raise ValueError(f'unknown sampler mixture component keys: {sorted(unknown)}')
            fixed = item.get('set') or {}
            vary = item.get('vary') or {}
            if not isinstance(fixed, Mapping) or not isinstance(vary, Mapping):
                raise ValueError('sampler mixture set/vary must be mappings')
            out.append((item.get('name'), item.get('weight'), item.get('views', 'forecast'),
                        tuple(sorted((k, _scalar(v)) for k, v in fixed.items())),
                        tuple(sorted((k, tuple(_scalar(x) for x in vs)) for k, vs in vary.items()))))
        elif isinstance(item, (list, tuple)) and len(item) == 5:
            name, weight, views, fixed, vary = item
            out.append((name, weight, views,
                        tuple(sorted((k, _scalar(v)) for k, v in fixed)),
                        tuple(sorted((k, tuple(_scalar(x) for x in vs)) for k, vs in vary))))
        else:
            raise ValueError(f'invalid sampler mixture component {item!r}')
    return tuple(out)


def mixture_as_list(mixture) -> list[dict[str, Any]]:
    """The manifest/JSON form (round-trips through canonical_mixture)."""
    return [dict(name=name, weight=weight, views=views, set=dict(fixed),
                 vary={k: list(vs) for k, vs in vary}) for name, weight, views, fixed, vary in mixture]


def check_structure(spec) -> None:
    """Field-level checks (cheap; run on every construction)."""
    mixture, unit = spec.sampler_mixture, spec.sampler_mixture_unit
    if unit not in UNITS:
        raise ValueError(f'unknown sampler_mixture_unit {unit!r}')
    if not mixture:
        if unit != 'group_window':
            raise ValueError('sampler_mixture_unit requires a sampler_mixture')
        return
    if type(mixture) is not tuple or not 1 <= len(mixture) <= MAX_COMPONENTS:
        raise ValueError(f'sampler_mixture must hold 1..{MAX_COMPONENTS} components')
    if spec.peer_view != 'none':
        raise ValueError('sampler_mixture and peer_view are exclusive (a peer view is one mixture component)')
    if spec.composition_algorithm != 'path_stream_v1':
        raise ValueError('sampler_mixture requires path_stream_v1')
    names = [c[0] for c in mixture]
    if any(type(n) is not str or not n for n in names) or len(set(names)) != len(names):
        raise ValueError('sampler mixture components need distinct non-empty names')
    for name, weight, views, fixed, vary in mixture:
        if type(weight) is not int or weight < 1:
            raise ValueError(f'{name}: weight must be a positive integer')
        if views not in VIEWS:
            raise ValueError(f'{name}: views must be one of {VIEWS}')
        keys = [k for k, _ in fixed] + [k for k, _ in vary]
        if len(set(keys)) != len(keys):
            raise ValueError(f'{name}: a field is both set and varied')
        bad = set(keys) - MIXTURE_FIELDS
        if bad:
            raise ValueError(f'{name}: fields not switchable by a mixture: {sorted(bad)}')
        if views == 'all' and set(keys) - LABEL_FREE_FIELDS:
            raise ValueError(f'{name}: views=all components may only change {sorted(LABEL_FREE_FIELDS)}')
        size = 1
        for field, values in vary:
            if not values or len(set(values)) != len(values):
                raise ValueError(f'{name}.{field}: vary needs distinct values')
            size *= len(values)
        if size > MAX_SUPPORT_PER_COMPONENT:
            raise ValueError(f'{name}: support {size} exceeds {MAX_SUPPORT_PER_COMPONENT}')


def component_support(component) -> list[dict[str, Any]]:
    """Every concrete overlay of one component (fixed fields + one value per varied field)."""
    _name, _weight, _views, fixed, vary = component
    fields = [k for k, _ in vary]
    return [dict(fixed, **dict(zip(fields, combo))) for combo in product(*(vs for _, vs in vary))]


def plain(spec):
    """The spec without its mixture."""
    if not spec.sampler_mixture:
        return spec
    return replace(spec, sampler_mixture=(), sampler_mixture_unit='group_window')


@lru_cache(maxsize=256)
def _validated(base, mixture) -> bool:
    for component in mixture:
        for overlay in component_support(component):
            try:
                replace(base, **overlay)
            except ValueError as error:
                raise ValueError(f'sampler mixture component {component[0]!r} overlay {overlay}: {error}') from None
    return True


def check_support(spec) -> None:
    """Every support point of every component is a valid spec on this base (fail closed, cached)."""
    if spec.sampler_mixture:
        _validated(plain(spec), spec.sampler_mixture)


def eligible(spec) -> list[int]:
    """Component indices that may switch this (resolved) task view."""
    forecast = spec.task_kind == 'forecast' and spec.label_visibility == 'history_available'
    return [i for i, c in enumerate(spec.sampler_mixture) if c[2] == 'all' or forecast]


def splitmix64(x: int) -> int:
    x = (int(x) + 0x9E3779B97F4A7C15) & _MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK
    return x ^ (x >> 31)


def draw(spec, key: int) -> tuple[str, dict[str, Any]]:
    """(component name, overlay) for draw key ``key`` on a resolved task view (deterministic)."""
    mixture = spec.sampler_mixture
    choices = eligible(spec)
    if not choices:
        return '', {}
    base = splitmix64((int(key) & _MASK) ^ _SALT)
    total = sum(mixture[i][1] for i in choices)
    u = base % total
    for index in choices:
        u -= mixture[index][1]
        if u < 0:
            break
    name, _weight, _views, fixed, vary = mixture[index]
    overlay = dict(fixed)
    for j, (field, values) in enumerate(vary):
        overlay[field] = values[splitmix64(base ^ ((j + 1) * 0xD1B54A32D192ED03 & _MASK)) % len(values)]
    return name, overlay


def draw_key(unit: str, batch_idx: int, window: int, task_block_start: int, epoch: int) -> int:
    """The draw unit's key: a coherent group window, or one contiguous task block (per epoch)."""
    if unit == 'group_window':
        window = int(window or 0)
        return int(batch_idx) // window if window > 0 else int(batch_idx)
    return (int(task_block_start) * 0x100000001B3 + int(epoch)) & _MASK
