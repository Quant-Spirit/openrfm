"""Versioned RT retrieval policy. Environment variables never select this policy."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal, Mapping
import hashlib
import json

BudgetStage = Literal['seed', 'own_history', 'peers', 'filler']


MAX_SEQ_LEN = 16384


PEER_VIEWS = {
    'r2l0pk': dict(path_quota_rows=8, path_quota_member_rows=16, seed_tier_percent=90, seed_local_rows=0,
                   seed_rank='program', seed_fill=False, seed_hide_labels=False, path_stream_fill=False,
                   candidate_count=512, k_enc_max=1024),
    'r2l1pk': dict(path_quota_rows=8, path_quota_member_rows=16, seed_tier_percent=90, seed_local_rows=4,
                   seed_local_policy='entity', seed_rank='program', seed_fill=False, seed_hide_labels=False,
                   path_stream_fill=False, candidate_count=512, k_enc_max=1024),
    'r2l0pk_ext': dict(path_quota_rows=8, path_quota_member_rows=16, seed_tier_percent=90, seed_local_rows=0,
                       seed_rank='program', seed_fill=False, seed_hide_labels=False, path_stream_fill=False,
                       candidate_count=512, k_enc_max=1024, peer_extension_percent=50, peer_extension_rows=6),
    'r2l0pk_ext_lab': dict(path_quota_rows=8, path_quota_member_rows=16, seed_tier_percent=90, seed_local_rows=0,
                           seed_rank='program', seed_fill=False, seed_hide_labels=False, path_stream_fill=False,
                           candidate_count=512, k_enc_max=1024, peer_extension_percent=50, peer_extension_rows=6,
                           extension_labels_visible=True),
}
for _name in ('r2l0pk_ext', 'r2l0pk_ext_lab'):
    PEER_VIEWS[_name + '_c1k'] = dict(PEER_VIEWS[_name], candidate_count=1024, walk_count=64)
del _name
PEER_VIEWS['r2l0pk_ext_lab_c1k_r2'] = dict(PEER_VIEWS['r2l0pk_ext_lab_c1k'], peer_extension_rows=2)
PEER_VIEWS['r2l0pk_ext_lab_c1k_r2p75'] = dict(PEER_VIEWS['r2l0pk_ext_lab_c1k'], peer_extension_rows=2,
                                              peer_extension_percent=75)
PEER_VIEWS['r2l0pk_ext_lab_c1k_mat'] = dict(PEER_VIEWS['r2l0pk_ext_lab_c1k'], seed_program_mature=True,
                                            seed_hop_cap=2048)


@dataclass(frozen=True)
class RTRetrievalSpec:
    version: int = 2
    walk_count: int = 64
    walk_steps: int = 8
    candidate_count: int = 128
    seq_len_set: tuple[int, ...] = (1024,)
    local_cell_budget_set: tuple[int, ...] = (64, 128, 256)
    bfs_width_set: tuple[int, ...] = (16, 32, 64, 128)
    recency_set: tuple[int, ...] = (0, 1)
    label_visibility: Literal['strict_drop', 'history_available', 'query_only'] = 'strict_drop'
    label_horizon_seconds: int | None = None
    real_training_cutoff: int | None = None
    support_contract: Literal['existing_donor_mask'] = 'existing_donor_mask'
    traversal: Literal['two_stage'] = 'two_stage'
    max_hops: int = 0
    max_resident_dbs: int = 1
    reserve_bytes: int = 8 * 1024**3
    measure: bool = False
    frontier_hops: int = 4
    frontier_rows: int = 4096
    walk_probes: int = 8
    k_enc_max: int = 256
    budget_order: tuple[BudgetStage, ...] = ('seed', 'own_history', 'peers', 'filler')
    task_kind: Literal['autocomplete', 'forecast'] = 'autocomplete'
    entity_table: str | None = None
    entity_parent_tables: tuple[str, ...] = ()
    seed_hops: int = 2
    path_depth: int = 6
    max_path_streams: int = 4096
    child_table_cap: int = 32
    child_table_caps: tuple[tuple[str, int], ...] = ()
    drop_columns: tuple[tuple[str, str], ...] = ()
    context_text_policy: Literal['keep', 'numeric_only', 'low_cardinality'] = 'low_cardinality'
    text_cardinality_limit: int = 32
    peer_labels_per_1024: int = 8
    time_strata: int = 4
    filler_percent: int = 20
    regression_label_bins: tuple[float, ...] = ()
    seed_block_policy: Literal['fail_closed', 'truncate_priority'] = 'fail_closed'
    name_neutralization_vectorized: bool = False
    min_live_fill_fraction: float = 0.0
    require_nonseed_roles: bool = False
    max_resident_task_views: int = 128
    coherent_group_window: int = 0
    composition_algorithm: Literal['path_stream_v1'] = 'path_stream_v1'
    protected_core_len: int = 0
    extension_policy: Literal['rw_visit', 'typed_temporal_mix'] = 'rw_visit'
    rw_extension_percent: int = 25
    extension_time_unit_seconds: int = 86400
    rw_scan_budget: int = 4096
    archive_residency_device: Literal['cuda', 'cpu'] = 'cuda'
    archive_cell_order: Literal['retrieval', 'table_row_col'] = 'retrieval'
    discovery_scales_with_length: bool = False
    stream_weight_policy: Literal['sqrt_eligible', 'selectivity'] = 'sqrt_eligible'
    label_pool_reserve: int = 0
    own_label_rows: int = 0
    own_recent_first: bool = False
    # Label-rich order: matured same-task rows first, own entity before peers.
    label_rich_first: bool = False
    own_recent_interleave: bool = False
    own_source_rank: bool = False
    own_recent_cell_balance: bool = False
    canonical_recent: bool = False
    path_quota_rows: int = 0
    path_quota_member_rows: int = 0
    path_quota_share: int = 0
    seed_tier_percent: int = 0
    seed_local_rows: int = 0
    seed_balance: bool = False
    seed_rank: Literal['walk', 'shared', 'program'] = 'walk'
    seed_fill: bool = True
    seed_hide_labels: bool = False
    seed_local_policy: Literal['parents', 'entity'] = 'parents'
    seed_route_fanout: int = 32
    path_stream_fill: bool = True
    path_quota_parent_rows: int = 0
    path_quota_grandparent_rows: int = 0
    path_quota_grandchild_rows: int = 0
    path_quota_order: Literal['signature', 'density'] = 'signature'
    seed_hop_cap: int = 1024
    seed_program_mature: bool = False
    peer_extension_percent: int = 0
    peer_extension_rows: int = 0
    extension_labels_visible: bool = False
    peer_view: Literal['none', 'r2l0pk', 'r2l1pk', 'r2l0pk_ext', 'r2l0pk_ext_lab',
                       'r2l0pk_ext_c1k', 'r2l0pk_ext_lab_c1k', 'r2l0pk_ext_lab_c1k_r2',
                       'r2l0pk_ext_lab_c1k_r2p75', 'r2l0pk_ext_lab_c1k_mat'] = 'none'
    peer_view_percent: int = 0
    sampler_mixture: tuple = ()
    sampler_mixture_unit: Literal['group_window', 'task_block'] = 'group_window'
    training_seed_order: Literal['table', 'cutoff_interleaved', 'cutoff_hashed', 'cutoff_stratified'] = 'table'
    forecast_batch_min_cutoffs: int = 0
    support_label_horizon_safe: bool = False
    _model_requires_provenance: bool = False

    def peer_view_spec(self) -> 'RTRetrievalSpec':
        """This policy with its peer-view overlay applied (and no nested peer view)."""
        from dataclasses import replace
        if self.peer_view == 'none':
            raise ValueError('no peer view is configured')
        return replace(self, **PEER_VIEWS[self.peer_view], peer_view='none', peer_view_percent=0)

    def for_model(self, model):
        from dataclasses import replace
        required = bool(getattr(model, 'requires_composition_provenance', False))
        if required and self.version not in (2, 3):
            raise ValueError('model requires composition provenance from a v2/v3 composer')
        return replace(self, _model_requires_provenance=required)

    def __post_init__(self):
        if self.archive_cell_order not in ('retrieval', 'table_row_col'):
            raise ValueError('archive_cell_order must be retrieval or table_row_col')
        if self.archive_residency_device not in ('cuda', 'cpu'):
            raise ValueError('archive_residency_device must be cuda or cpu')
        if self.training_seed_order not in ('table', 'cutoff_interleaved', 'cutoff_hashed', 'cutoff_stratified'):
            raise ValueError('training_seed_order must be table|cutoff_interleaved|cutoff_hashed|cutoff_stratified, '
                             f'got {self.training_seed_order!r}')
        if type(self.forecast_batch_min_cutoffs) is not int or self.forecast_batch_min_cutoffs < 0:
            raise ValueError('forecast_batch_min_cutoffs must be a nonnegative integer')
        if self.forecast_batch_min_cutoffs and self.training_seed_order != 'cutoff_stratified':
            raise ValueError('forecast_batch_min_cutoffs needs training_seed_order cutoff_stratified (the only order '
                             'with a provable per-batch cutoff floor)')
        if type(self.support_label_horizon_safe) is not bool:
            raise ValueError('support_label_horizon_safe must be Boolean')
        from openrfm.sampler import mixture as _mixture
        object.__setattr__(self, 'sampler_mixture', _mixture.canonical_mixture(self.sampler_mixture))
        _mixture.check_structure(self)
        if type(self._model_requires_provenance) is not bool:
            raise ValueError('model provenance capability must be Boolean')
        if self.peer_view not in ('none', *PEER_VIEWS):
            raise ValueError(f'unknown peer_view {self.peer_view!r}')
        if type(self.peer_view_percent) is not int or not 0 <= self.peer_view_percent <= 100:
            raise ValueError('peer_view_percent must be an integer in [0, 100]')
        if (self.peer_view == 'none') != (self.peer_view_percent == 0):
            raise ValueError('peer_view and peer_view_percent > 0 go together')
        if self.peer_view != 'none':
            if self.composition_algorithm != 'path_stream_v1' or self.seed_tier_percent:
                raise ValueError('a peer view overlays a plain path_stream_v1 policy (no seed tier of its own)')
        if (not isinstance(self.entity_parent_tables, tuple)
                or any(not isinstance(t, str) or not t for t in self.entity_parent_tables)
                or len(set(self.entity_parent_tables)) != len(self.entity_parent_tables)):
            raise ValueError('entity_parent_tables must be distinct table names')
        if self.composition_algorithm != 'path_stream_v1':
            raise ValueError('unknown composition algorithm')
        if type(self.discovery_scales_with_length) is not bool:
            raise ValueError('discovery_scales_with_length must be Boolean')
        if self.stream_weight_policy not in ('sqrt_eligible', 'selectivity'):
            raise ValueError('unknown stream_weight_policy')
        if ((self.discovery_scales_with_length
             or self.stream_weight_policy != 'sqrt_eligible')
                and self.composition_algorithm != 'path_stream_v1'):
            raise ValueError('discovery/stream-weight policies require path_stream_v1')
        for name, lo, hi in (('label_pool_reserve', 0, self.frontier_rows // 2
                                 if type(self.frontier_rows) is int else 0),
                             ('seed_tier_percent', 0, 90), ('seed_local_rows', 0, 64),
                             ('path_quota_rows', 0, 64), ('path_quota_member_rows', 0, 64),
                             ('path_quota_share', 0, 100), ('own_label_rows', 0, 256),
                             ('seed_route_fanout', 1, 4096), ('path_quota_parent_rows', 0, 64),
                             ('path_quota_grandparent_rows', 0, 64), ('path_quota_grandchild_rows', 0, 64),
                             ('seed_hop_cap', 1, 2048), ('peer_extension_percent', 0, 100),
                             ('peer_extension_rows', 0, 64)):
            if type(getattr(self, name)) is not int or not lo <= getattr(self, name) <= hi:
                raise ValueError(f'{name} must be an integer in [{lo}, {hi}]')
        if self.seed_route_fanout != 32 and self.seed_rank != 'shared':
            raise ValueError("seed_route_fanout requires seed_rank='shared'")
        for name in ('seed_balance', 'seed_fill', 'seed_hide_labels', 'path_stream_fill', 'own_recent_first', 'label_rich_first', 'own_recent_interleave', 'own_source_rank', 'own_recent_cell_balance', 'canonical_recent'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be Boolean')
        if type(self.extension_labels_visible) is not bool:
            raise ValueError('extension_labels_visible must be Boolean')
        if self.extension_labels_visible and not self.peer_extension_percent:
            raise ValueError('extension_labels_visible requires the peer extension')
        if self.peer_extension_rows and not self.peer_extension_percent:
            raise ValueError('peer_extension_rows requires peer_extension_percent > 0')
        if self.peer_extension_percent and not (
                self.protected_core_len and self.extension_policy == 'typed_temporal_mix'
                and self.seed_tier_percent
                and self.peer_extension_percent + self.rw_extension_percent <= 100):
            raise ValueError('peer_extension_percent requires a protected core, the typed_temporal_mix '
                             'extension, the seed tier and peer + rw extension percents <= 100')
        if type(self.seed_program_mature) is not bool:
            raise ValueError('seed_program_mature must be Boolean')
        if self.seed_program_mature and self.seed_rank != 'program':
            raise ValueError("seed_program_mature requires seed_rank='program'")
        if self.seed_hop_cap != 1024 and self.seed_rank != 'program':
            raise ValueError("seed_hop_cap requires seed_rank='program'")
        if self.seed_rank not in ('walk', 'shared', 'program') or self.seed_local_policy not in ('parents', 'entity'):
            raise ValueError('unknown seed_rank / seed_local_policy')
        if not self.seed_tier_percent and (self.seed_local_rows or self.seed_balance
                                           or self.seed_rank != 'walk' or not self.seed_fill
                                           or self.seed_hide_labels
                                           or self.seed_local_policy != 'parents'):
            raise ValueError('seed_local_rows/seed_balance/seed_rank/seed_fill/seed_hide_labels/'
                             'seed_local_policy require seed_tier_percent > 0')
        any_quota = bool(self.path_quota_rows or self.path_quota_member_rows or self.path_quota_parent_rows
                         or self.path_quota_grandparent_rows or self.path_quota_grandchild_rows)
        if self.path_quota_order not in ('signature', 'density'):
            raise ValueError('unknown path_quota_order')
        if self.path_quota_order != 'signature' and not any_quota:
            raise ValueError('path_quota_order requires a path quota')
        if self.own_recent_interleave and not self.own_recent_first:
            raise ValueError('own_recent_interleave requires own_recent_first')
        if self.own_source_rank and (not self.own_recent_interleave or self.version not in (2, 3)):
            raise ValueError('own_source_rank requires interleaved directed composition')
        if self.own_recent_cell_balance and not self.own_source_rank:
            raise ValueError('own_recent_cell_balance requires own_source_rank')
        if not self.path_stream_fill and not (self.own_recent_first or self.own_label_rows or any_quota or self.seed_tier_percent):
            raise ValueError('path_stream_fill=False requires own labels, a path quota or the seed tier')
        if self.path_quota_share and not any_quota:
            raise ValueError('path_quota_share requires a path quota')
        if ((self.canonical_recent or self.own_recent_first or self.label_rich_first or self.label_pool_reserve or self.seed_tier_percent or any_quota or self.own_label_rows)
                and self.composition_algorithm != 'path_stream_v1'):
            raise ValueError('canonical_recent/label_pool_reserve/own labels/seed tier/path quota require path_stream_v1')
        if self.seed_tier_percent and not 1 <= self.walk_steps <= 64:
            raise ValueError('the seed tier requires walk_steps <= 64')
        if type(self.protected_core_len) is not int or self.protected_core_len < 0:
            raise ValueError('protected_core_len must be a nonnegative integer')
        if self.extension_policy not in ('rw_visit', 'typed_temporal_mix'):
            raise ValueError('unknown extension_policy')
        if type(self.rw_extension_percent) is not int or not 0 <= self.rw_extension_percent <= 100:
            raise ValueError('rw_extension_percent must be an integer in [0, 100]')
        if (type(self.extension_time_unit_seconds) is not int
                or not 1 <= self.extension_time_unit_seconds < 2**31 - 1):
            raise ValueError('extension_time_unit_seconds must lie in [1, INT_MAX-1]')
        if type(self.rw_scan_budget) is not int or not 1 <= self.rw_scan_budget < 2**31 - 1:
            raise ValueError('rw_scan_budget must lie in [1, INT_MAX-1]')
        if self.real_training_cutoff is not None:
            if (type(self.real_training_cutoff) is not int or not 0 < self.real_training_cutoff < 2**31-1
                    or self.label_visibility != 'history_available' or self.label_horizon_seconds != 0
                    or self.task_kind != 'autocomplete' or self.version not in (2, 3)):
                raise ValueError('real training requires history_available autocomplete, horizon=0 and a train cutoff')
        if type(self.coherent_group_window) is not int or self.coherent_group_window < 0 or self.coherent_group_window % 2:
            raise ValueError('coherent_group_window must be zero or a positive even integer')
        if type(self.max_resident_task_views) is not int or self.max_resident_task_views < 0:
            raise ValueError('invalid max_resident_task_views')
        if self.max_resident_task_views == 0:
            raise ValueError('max_resident_task_views must be positive')
        if self.max_resident_dbs == 0:
            raise ValueError('max_resident_dbs must be positive')
        if self.seed_block_policy not in ('fail_closed', 'truncate_priority'):
            raise ValueError('unknown seed_block_policy')
        if type(self.name_neutralization_vectorized) is not bool:
            raise ValueError('name_neutralization_vectorized must be Boolean')
        if (type(self.min_live_fill_fraction) not in (int, float)
                or not 0 <= self.min_live_fill_fraction <= 1):
            raise ValueError('min_live_fill_fraction must be in [0, 1]')
        if type(self.require_nonseed_roles) is not bool:
            raise ValueError('require_nonseed_roles must be Boolean')
        if self.seed_block_policy == 'truncate_priority' and self.version == 2:
            object.__setattr__(self, 'version', 3)
        if (self.version == 3) != (self.seed_block_policy == 'truncate_priority'):
            raise ValueError('retrieval v3 requires truncate_priority; v1/v2 require fail_closed')
        if self.protected_core_len and (
                self.version != 3 or self.composition_algorithm != 'path_stream_v1'
                or self.protected_core_len > max(self.seq_len_set)
                or not 1 <= self.walk_steps <= 64):
            raise ValueError('protected RW requires v3 path_stream, core <= capacity and walk_steps <= 64')
        if self.extension_policy == 'typed_temporal_mix' and (
                self.version != 3 or self.composition_algorithm != 'path_stream_v1'
                or self.protected_core_len <= 0):
            raise ValueError('typed_temporal_mix requires a positive v3 protected path-stream core')
        if self.rw_scan_budget != 4096 and not self.protected_core_len:
            raise ValueError('rw_scan_budget is only an opt-in protected-core setting')
        if self.task_kind not in ('autocomplete','forecast') or (self.task_kind == 'forecast' and not self.entity_table):
            raise ValueError('forecast composition requires an explicit entity_table')
        if self.context_text_policy not in ('keep','numeric_only','low_cardinality'):
            raise ValueError('unknown context_text_policy')
        for name,lo,hi in (('seed_hops',0,64),('path_depth',0,64),
                          ('max_path_streams',1,4096),('child_table_cap',1,16384),
                          ('text_cardinality_limit',0,2**31-1),('peer_labels_per_1024',1,1024),
                          ('time_strata',1,32),('filler_percent',0,100 if self.label_visibility=='query_only' or self.real_training_cutoff is not None else 20)):
            if type(getattr(self,name)) is not int or not lo <= getattr(self,name) <= hi:
                raise ValueError(f'invalid {name}')
        if self.path_depth < self.seed_hops:
            raise ValueError('path_depth must be >= seed_hops')
        if (len(set(self.drop_columns)) != len(self.drop_columns)
                or any(not isinstance(p, tuple) or len(p) != 2 or any(not isinstance(x, str) or not x for x in p)
                       for p in self.drop_columns)):
            raise ValueError('drop_columns must be distinct (table, column) string pairs')
        if (len({n for n,_ in self.child_table_caps}) != len(self.child_table_caps)
                or any(not isinstance(n,str) or type(c) is not int or not 1<=c<=16384 for n,c in self.child_table_caps)):
            raise ValueError('invalid child_table_caps')
        import math
        if any(not math.isfinite(v) for v in self.regression_label_bins) or any(a>=b for a,b in zip(self.regression_label_bins,self.regression_label_bins[1:])):
            raise ValueError('regression_label_bins must be finite and strictly increasing')
        for name in ('version', 'walk_count', 'walk_steps', 'candidate_count',
                     'max_hops', 'max_resident_dbs', 'reserve_bytes', 'frontier_hops',
                     'frontier_rows', 'walk_probes', 'k_enc_max'):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name in
                    ('version', 'walk_count', 'walk_steps', 'candidate_count',
                     'frontier_hops', 'frontier_rows', 'walk_probes') else 0):
                raise ValueError(f'{name} has invalid integer value {value!r}')
        for name in ('seq_len_set', 'local_cell_budget_set', 'bfs_width_set', 'recency_set'):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values or len(set(values)) != len(values):
                raise ValueError(f'{name} must be a nonempty tuple of distinct integers')
            if any(type(x) is not int or x < (0 if name == 'recency_set' else 1) for x in values):
                raise ValueError(f'invalid {name}')
        if self.version not in (1, 2, 3) or set(self.recency_set) - {0, 1}:
            raise ValueError('unsupported version or recency policy')
        if self.label_visibility not in ('strict_drop', 'history_available', 'query_only'):
            raise ValueError('invalid label visibility')
        if self.label_visibility == 'query_only' and (self.task_kind != 'autocomplete'
                or self.entity_table is not None or self.label_horizon_seconds is not None
                or self.version not in (2, 3)):
            raise ValueError('query_only requires autocomplete composition without a label horizon v2/v3')
        if self.label_horizon_seconds is not None and (type(self.label_horizon_seconds) is not int or not 0 <= self.label_horizon_seconds <= 2**63-1):
            raise ValueError('label horizon must be a nonnegative integer or null')
        if self.label_visibility == 'history_available' and self.label_horizon_seconds is None:
            raise ValueError('history_available requires an explicit label_horizon_seconds')
        if self.support_contract != 'existing_donor_mask' or self.traversal != 'two_stage':
            raise ValueError('unsupported support/traversal contract')
        if type(self.measure) is not bool:
            raise ValueError('measure must be Boolean')
        if self.version in (2, 3):
            if max(self.local_cell_budget_set)**2*max(self.seq_len_set) >= 2**62:
                raise ValueError('composition ell scaling overflow')
            if max(*self.seq_len_set,*self.local_cell_budget_set,self.k_enc_max) >= 2**31:
                raise ValueError('v2 int32 cell budget overflow')
            if self.walk_count*self.walk_steps > 1024 or self.candidate_count > self.walk_count*self.walk_steps:
                raise ValueError('v2 segmented score workspace requires C <= W*K <= 1024')
            if self.frontier_rows < self.candidate_count+1 or max(self.bfs_width_set) > 256:
                raise ValueError('v2 requires frontier_rows >= C+1 and width <= 256')
            if self.frontier_rows > 16384 or max(self.frontier_hops,self.max_hops,self.walk_probes) > 64:
                raise ValueError('v2 requires frontier_rows <= 16384 and hops/probes <= 64')
        if max(self.seq_len_set) > MAX_SEQ_LEN:
            raise ValueError(f'seq_len_set max {max(self.seq_len_set)} exceeds MAX_SEQ_LEN {MAX_SEQ_LEN}')
        if (max(self.seq_len_set) > (2**63 - 1) // max(256, self.candidate_count+1)
                or self.candidate_count >= 2**31 or self.walk_count * self.walk_steps > 2**31 - 1):
            raise ValueError('retrieval workspace address overflow')
        _mixture.check_support(self)

    @classmethod
    def parse(cls, value: Mapping) -> 'RTRetrievalSpec':
        unknown = set(value) - {f.name for f in fields(cls) if not f.name.startswith('_')}
        if unknown:
            raise ValueError(f'unknown RT retrieval fields: {sorted(unknown)}')
        value = dict(value)
        for name in ('budget_order','regression_label_bins','entity_parent_tables'):
            if name in value:
                value[name]=tuple(value[name])
        if 'child_table_caps' in value:
            caps=value['child_table_caps']
            value['child_table_caps']=tuple(sorted(caps.items())) if isinstance(caps,dict) else tuple(tuple(x) for x in caps)
        if 'drop_columns' in value:
            drops=value['drop_columns']
            value['drop_columns']=(tuple(sorted((t,c) for t,cs in drops.items() for c in cs)) if isinstance(drops,dict)
                                   else tuple(tuple(x) for x in drops))
        for name in ('seq_len_set', 'local_cell_budget_set', 'bfs_width_set', 'recency_set'):
            if name in value:
                value[name] = tuple(value[name])
        return cls(**value)

    @classmethod
    def load(cls, path: str | Path) -> 'RTRetrievalSpec':
        import yaml
        value = yaml.safe_load(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError('retrieval config must be a mapping')
        return cls.parse(value)

    def as_dict(self):
        result = asdict(self)
        if self.archive_residency_device == 'cuda':
            del result['archive_residency_device']
        if self.archive_cell_order == 'retrieval':
            del result['archive_cell_order']
        del result['_model_requires_provenance']
        if not self.protected_core_len:
            del result['protected_core_len']
        if not self.entity_parent_tables:
            del result['entity_parent_tables']
        if not self.drop_columns:
            del result['drop_columns']
        if self.real_training_cutoff is None:
            del result['real_training_cutoff']
        for name, default in (('max_resident_task_views', 128), ('coherent_group_window', 0)):
            if result[name] == default:
                del result[name]
        if self.seed_block_policy == 'fail_closed':
            del result['seed_block_policy']
        if (self.protected_core_len == 0
                and self.extension_policy == 'rw_visit'
                and self.rw_extension_percent == 25
                and self.extension_time_unit_seconds == 86400
                and self.rw_scan_budget == 4096):
            del result['extension_policy']
            del result['rw_extension_percent']
            del result['extension_time_unit_seconds']
            del result['rw_scan_budget']
        elif self.extension_policy == 'typed_temporal_mix':
            result['extension_policy'] = self.extension_policy
            result['rw_extension_percent'] = self.rw_extension_percent
            result['extension_time_unit_seconds'] = self.extension_time_unit_seconds
        if not self.name_neutralization_vectorized:
            del result['name_neutralization_vectorized']
        if self.min_live_fill_fraction == 0:
            del result['min_live_fill_fraction']
        if not self.require_nonseed_roles:
            del result['require_nonseed_roles']
        if not self.discovery_scales_with_length:
            del result['discovery_scales_with_length']
        if self.stream_weight_policy == 'sqrt_eligible':
            del result['stream_weight_policy']
        for name, default in (('label_pool_reserve', 0), ('own_label_rows', 0),
                              ('own_recent_first', False),
                              ('label_rich_first', False),
                              ('own_recent_interleave', False),
                              ('own_source_rank', False),
                              ('own_recent_cell_balance', False),
                              ('canonical_recent', False),
                              ('path_quota_rows', 0), ('path_quota_member_rows', 0),
                              ('path_quota_share', 0), ('seed_tier_percent', 0),
                              ('seed_local_rows', 0), ('seed_balance', False),
                              ('seed_rank', 'walk'), ('seed_fill', True),
                              ('seed_hide_labels', False), ('seed_local_policy', 'parents'),
                              ('seed_route_fanout', 32), ('path_stream_fill', True),
                              ('path_quota_parent_rows', 0), ('path_quota_grandparent_rows', 0),
                              ('path_quota_grandchild_rows', 0), ('path_quota_order', 'signature'),
                              ('seed_hop_cap', 1024), ('peer_view', 'none'), ('peer_view_percent', 0),
                              ('peer_extension_percent', 0), ('peer_extension_rows', 0),
                              ('extension_labels_visible', False), ('seed_program_mature', False),
                              ('training_seed_order', 'table'), ('forecast_batch_min_cutoffs', 0),
                              ('support_label_horizon_safe', False)):
            if result[name] == default:
                del result[name]
        del result['sampler_mixture']
        del result['sampler_mixture_unit']
        if self.sampler_mixture:
            from openrfm.sampler.mixture import mixture_as_list
            result['sampler_mixture'] = mixture_as_list(self.sampler_mixture)
            result['sampler_mixture_unit'] = self.sampler_mixture_unit
        return result

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True,
                             separators=(',', ':')).encode()).hexdigest()
