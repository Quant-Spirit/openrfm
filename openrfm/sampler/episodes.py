"""Python entry points of the relational context sampler: backend selection, cell order and nested RT contexts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .retrieval import RTGraph
    from .spec import RTRetrievalSpec


def _rt_retrieval_module():
    from . import retrieval
    return retrieval


def _rt_retrieval_spec_module():
    from . import spec
    return spec


def __getattr__(name):
    """Keep the public typed RT API available without eager optional imports."""
    if name == 'RTRetrievalSpec':
        return _rt_retrieval_spec_module().RTRetrievalSpec
    if name == 'RTGraph':
        return _rt_retrieval_module().RTGraph
    raise AttributeError(name)


CellOrder = Literal['retrieval', 'table_row_col', 'canonical_recent']


def parse_cell_order(value: str) -> CellOrder:
    if value == 'retrieval':
        return 'retrieval'
    if value == 'table_row_col':
        return 'table_row_col'
    if value == 'canonical_recent':
        return 'canonical_recent'
    raise ValueError("cell_order must be 'retrieval', 'table_row_col', or 'canonical_recent'")


def rt_cell_order_permutation(
    selected: Tensor,
    *,
    cell_order: CellOrder,
    cell_nodes: Tensor,
    row_table: Tensor,
    cell_columns: Tensor,
    seed_rows: Tensor,
    row_time: Tensor | None = None,
) -> Tensor | None:
    """Return a device-side cell permutation, or ``None`` for the default order."""

    cell_order = parse_cell_order(cell_order)
    if cell_order == 'retrieval':
        return None
    if selected.ndim != 2:
        raise ValueError('selected cells must have shape [B,L]')
    if seed_rows.shape != (selected.shape[0],):
        raise ValueError('seed_rows must have shape [B]')
    if any(value.ndim != 1 for value in (cell_nodes, row_table, cell_columns)):
        raise ValueError('cell_nodes, row_table, and cell_columns must be vectors')
    if any(value.device != selected.device for value in (
        cell_nodes, row_table, cell_columns, seed_rows
    )):
        raise ValueError('cell-order tensors must share the selected-cell device')

    padding = selected < 0
    safe_cells = selected.clamp_min(0)
    from openrfm.archive.format import take
    rows = take(cell_nodes, safe_cells)
    tables = take(row_table, rows)
    columns = take(cell_columns, safe_cells)
    seed_group = (rows != seed_rows[:, None]).to(torch.int64)
    permutation = torch.arange(
        selected.shape[1], device=selected.device, dtype=torch.int64
    ).expand_as(selected)
    keys = (columns, rows, tables, seed_group, padding.to(torch.int64))
    if cell_order == 'canonical_recent':
        if row_time is None or row_time.ndim != 1 or row_time.device != selected.device:
            raise ValueError('canonical_recent requires row_time on the selected-cell device')
        times = take(row_time, rows).to(torch.int64)
        keys = (columns, rows, tables, -times, seed_group, padding.to(torch.int64))
    for key in keys:
        ranks = torch.argsort(
            torch.gather(key, 1, permutation), dim=1, stable=True
        )
        permutation = torch.gather(permutation, 1, ranks)
    return permutation


def configure_backend(mode: Literal['auto', 'torch', 'native'] = 'auto', *,
                      build: bool = False, with_cuda: bool | None = None,
                      build_root: str | Path | None = None, verbose: bool = False) -> Any:
    """Select the sampler backend; an explicit build caches native code under profile_output."""
    from .native import configure
    return configure(mode, build=build, with_cuda=with_cuda, build_root=build_root, verbose=verbose)


def _sample_rt_contexts_unpaired(*args, **kwargs):
    """Lazy alias for `retrieval.sample_rt_contexts`."""
    return _rt_retrieval_module().sample_rt_contexts(*args, **kwargs)


@dataclass(frozen=True)
class NestedContext:
    """Lengths of the paired core-only and core-plus-extension contexts."""
    core_len: int
    ext_len: int

    def __post_init__(self) -> None:
        if type(self.core_len) is not int or self.core_len < 1:
            raise ValueError('nested_context.core_len must be a positive integer')
        if type(self.ext_len) is not int or self.ext_len < 1:
            raise ValueError('nested_context.ext_len must be a positive integer')
        if self.core_len > 2**63 - 1 - self.ext_len:
            raise ValueError('nested_context length overflow')

    @classmethod
    def parse(cls, value: NestedContext | Mapping[str, Any]) -> NestedContext:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError('nested_context must be a mapping with core_len and ext_len')
        unknown = set(value) - {'core_len', 'ext_len'}
        missing = {'core_len', 'ext_len'} - set(value)
        if unknown or missing:
            raise ValueError(
                f'nested_context fields invalid; missing={sorted(missing)}, unknown={sorted(unknown)}'
            )
        return cls(value['core_len'], value['ext_len'])

    @property
    def long_len(self) -> int:
        return self.core_len + self.ext_len


@dataclass(frozen=True)
class NestedRTContexts:
    """Raw paired retrievals plus the mask needed at the model boundary."""
    short: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    long: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    pairing_key: Tensor
    long_hidden_labels: Tensor
    spec: NestedContext


def _context_assert(value: Tensor, message: str) -> None:
    """Fail closed without introducing a CUDA host synchronization."""
    if value.is_cuda:
        torch._assert_async(value, message)
    elif not bool(value):
        raise RuntimeError(message)


def _nested_pairing_key(requests: Tensor) -> Tensor:
    """Stable int64 identity for the (seed row, per-row RNG key) pair."""
    return torch.bitwise_xor(requests[:, 0] * 1_000_003, requests[:, 3]).bitwise_and(2**63 - 1)


def nested_arm_settings(base: Tensor, nested: NestedContext) -> tuple[Tensor, Tensor]:
    """SHORT/LONG settings of one nested pair; only column 0, the admission budget, differs."""
    from openrfm.sampler.retrieval import SETTINGS_WIDTH
    if base.dtype != torch.int64 or base.dim() != 2 or base.shape[1] != SETTINGS_WIDTH:
        raise ValueError(f'settings must be int64 [B,{SETTINGS_WIDTH}]')
    short, long = base.clone(), base.clone()
    short[:, 0] = nested.core_len
    long[:, 0] = nested.long_len
    check_nested_arm_settings(short, long)
    return short, long


def check_nested_arm_settings(short: Tensor, long: Tensor) -> None:
    """Nesting invariant at the settings level: arms differ only in column 0."""
    from openrfm.sampler.retrieval import SETTINGS_WIDTH
    if short.shape != long.shape or short.dim() != 2 or short.shape[1] != SETTINGS_WIDTH:
        raise ValueError(f'nested arms need equal [B,{SETTINGS_WIDTH}] settings')
    _context_assert((short[:, 1:] == long[:, 1:]).all(),
                    'nested SHORT/LONG arms must share settings columns 1-4 '
                    '(column 4 drives the L-scaled discovery frontier)')


def check_nested_frontier(short_receipt: Tensor, long_receipt: Tensor) -> None:
    """Nesting invariant at the receipt level: one discovery frontier per pair."""
    from openrfm.sampler.path_streams import PATH_FRONTIER, is_path_receipt
    if not (is_path_receipt(short_receipt) and is_path_receipt(long_receipt)):
        return
    if short_receipt.shape[0] != long_receipt.shape[0]:
        raise ValueError('nested arms have different batch sizes')
    _context_assert((short_receipt[:, PATH_FRONTIER] == long_receipt[:, PATH_FRONTIER]).all(),
                    'nested SHORT/LONG arms used different discovery frontiers; '
                    'both arms must share settings column 4')


def sample_nested_rt_contexts(
    graph: RTGraph,
    requests: Tensor,
    spec: RTRetrievalSpec,
    nested_context: NestedContext | Mapping[str, Any],
    *,
    k_enc_fixed_from_core: bool = False,
    settings: Tensor | None = None,
    directions=None, workspace=None, composition=None,
) -> NestedRTContexts:
    """Build paired SHORT/LONG contexts; SHORT is the exact prefix of LONG."""
    context_settings = _rt_retrieval_module().context_settings

    nested = NestedContext.parse(nested_context)
    if type(k_enc_fixed_from_core) is not bool:
        raise ValueError('k_enc_fixed_from_core must be Boolean')
    if nested.long_len > max(spec.seq_len_set):
        raise ValueError(
            f'nested context length {nested.long_len} exceeds allocated sampler capacity '
            f'{max(spec.seq_len_set)}'
        )
    base = context_settings(requests, spec) if settings is None else settings
    from openrfm.sampler.retrieval import SETTINGS_WIDTH
    if (base.dtype != torch.int64 or base.shape != (len(requests), SETTINGS_WIDTH)
            or base.device != requests.device or not base.is_contiguous()):
        raise ValueError(f'settings must be contiguous int64 [B,{SETTINGS_WIDTH}] on the request device')
    short_settings, long_settings = nested_arm_settings(base, nested)

    if spec.version in (2, 3):
        long_raw=tuple(t.clone() for t in _sample_rt_contexts_unpaired(
            graph,requests,spec,settings=long_settings,
            directions=directions,workspace=workspace,composition=composition))
        short=tuple(_sample_rt_contexts_unpaired(graph,requests,spec,settings=short_settings,
            directions=directions,workspace=workspace,composition=composition))
        check_nested_frontier(short[5], long_raw[5])
        short_cells,long_cells=short[3],long_raw[3]
        from openrfm.sampler.path_streams import boundary_offset
        boundary=boundary_offset(long_raw[5])
        ends=long_raw[5][:,boundary:boundary+long_cells.shape[1]]
        core_end = nested.core_len
        if spec.protected_core_len == nested.core_len:
            core_end = (short_cells >= 0).sum(dim=1, keepdim=True)
        expected=long_cells.masked_fill((ends<0)|(ends>core_end),-1)
        _context_assert((short_cells==expected).all(), 'path-stream atomic prefix nesting failed')
        if spec.protected_core_len == nested.core_len:
            _context_assert(
                ((short[4] == long_raw[4]) | (short_cells < 0)).all(),
                'protected-core roles changed in LONG',
            )
    else:
        short = tuple(_sample_rt_contexts_unpaired(graph, requests, spec, settings=short_settings))
        long_raw = tuple(_sample_rt_contexts_unpaired(graph, requests, spec, settings=long_settings))
        if len(short) != 6 or len(long_raw) != 6:
            raise RuntimeError('RT backend returned an unknown nested-context ABI')
        short_cells, long_cells = short[3], long_raw[3]
        short_roles, long_roles = short[4], long_raw[4]
        _context_assert(
            ((short_cells[:, :nested.core_len] == long_cells[:, :nested.core_len]).all()
             & (short_roles[:, :nested.core_len] == long_roles[:, :nested.core_len]).all()),
            'nested-context LONG prefix differs from the SHORT core',
        )
        _context_assert((short[5][:, 0] == nested.core_len).all(),
                        'nested-context SHORT receipt has the wrong allocated L')
        _context_assert((long_raw[5][:, 0] == nested.long_len).all(),
                        'nested-context LONG receipt has the wrong allocated L')

    hidden = torch.zeros_like(long_cells, dtype=torch.bool)
    long_receipt = long_raw[5].clone()
    if k_enc_fixed_from_core:
        positions = torch.arange(long_cells.shape[1], device=long_cells.device)
        outside = positions.unsqueeze(0) >= nested.core_len
        if spec.version in (2, 3):
            outside = short_cells < 0
        present = long_cells >= 0
        safe_cells = long_cells.clamp_min(0)
        hidden = outside & present & graph.cell_label[safe_cells].bool()
        long_receipt[:, 4] -= hidden.sum(1)
        _context_assert((long_receipt[:, 4] >= 0).all(),
                        'nested-context label accounting became negative')
        _context_assert((long_receipt[:, 4] == short[5][:, 4]).all(),
                        'nested-context K_enc differs between SHORT and LONG')

    long = (*long_raw[:5], long_receipt)
    return NestedRTContexts(
        short=short,
        long=long,
        pairing_key=_nested_pairing_key(requests),
        long_hidden_labels=hidden,
        spec=nested,
    )


def sample_rt_contexts(
    graph: RTGraph,
    requests: Tensor,
    spec: RTRetrievalSpec,
    *,
    settings: Tensor | None = None,
    nested_context: NestedContext | Mapping[str, Any] | None = None,
    k_enc_fixed_from_core: bool = False,
    directions=None, workspace=None, composition=None,
):
    """RT retrieval with an opt-in paired nested-context result."""
    if nested_context is None:
        if k_enc_fixed_from_core:
            raise ValueError('k_enc_fixed_from_core requires nested_context')
        return _sample_rt_contexts_unpaired(graph, requests, spec, settings=settings, directions=directions,workspace=workspace,composition=composition)
    return sample_nested_rt_contexts(
        graph,
        requests,
        spec,
        nested_context,
        k_enc_fixed_from_core=k_enc_fixed_from_core,
        settings=settings,
        directions=directions,workspace=workspace,composition=composition,
    )
