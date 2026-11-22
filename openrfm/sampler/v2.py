"""Two-axis retrieval: independent CPU oracle and owned native workspaces."""
from dataclasses import dataclass
from math import prod
import numpy as np
import torch

from openrfm.sampler.retrieval import STATIC
from openrfm.archive.format import take


class CompositionPreflightError(RuntimeError):
    """CPU diagnostic receipt; raised before launching native composition."""
    def __init__(self,receipt):
        import json
        self.receipt=receipt
        super().__init__('composition preflight failure: '+json.dumps(receipt,sort_keys=True))


def check_composition_receipt(receipt, spec=None):
    """Reject every non-success composition receipt, without a host sync on CUDA."""
    from openrfm.sampler.retrieval import _assert
    code=receipt[:,6]
    _assert(((code!=-6)&(code!=-7)).all(),
        'composition path capacity/collision failure (cap); not DB exhaustion')
    _assert(((code!=-1)&(code!=-5)).all(),
        'composition seed budget failure: complete or protected seed cells exceed L (codes 1/5)')
    _assert((code!=-2).all(),
        'composition frontier failure: ego row plan exceeds frontier_rows (code 2)')
    _assert(((code!=-3)&(code!=-4)).all(),
        'composition identity failure: missing focal target or forecast identity (codes 3/4)')
    _assert((code!=-9).all(), 'query table is masked out for this task')
    _assert((code!=-10).all(),
        'composition stream weight scale outside [1, 1<<20] in the resident schema (code 10)')
    _assert((code>=0).all(), 'composition invalid admitted schema or witness (internal ABI failure)')


@dataclass
class DirectedCSR:
    fptr: torch.Tensor
    fdst: torch.Tensor
    ftime: torch.Tensor
    pptr: torch.Tensor
    pdst: torch.Tensor
    ptime: torch.Tensor

    def tensors(self):
        return list(vars(self).values())

    @classmethod
    def admit(cls, fptr, fdst, pptr, pdst, row_time, *, device, cpu_sort=False):
        """Cold admission, int32 only after explicit address overflow checks."""
        n = len(row_time)
        if max(n, len(fdst), len(pdst)) >= 2**31:
            raise ValueError('v2 CSR int32 address overflow')
        result = []
        for ptr, dst in ((fptr,fdst),(pptr,pdst)):
            ptr, dst = ptr.long().cpu(), dst.long().cpu()
            if (len(ptr) != n+1 or ptr[0] != 0 or ptr[-1] != len(dst)
                    or (ptr.diff()<0).any() or ((dst<0)|(dst>=n)).any()):
                raise ValueError('invalid directed CSR')
            times = row_time.cpu()[dst]
            if cpu_sort:
                # One stable lexicographic pass gives the same order as the three Torch sorts.
                src = np.repeat(np.arange(n), np.diff(ptr.numpy()))
                order = torch.from_numpy(np.lexsort((dst.numpy(), times.numpy(), src)))
            else:
                src = torch.repeat_interleave(torch.arange(n),ptr.diff())
                order = torch.argsort(dst,stable=True)
                order = order[torch.argsort(times[order],stable=True)]
                order = order[torch.argsort(src[order],stable=True)]
            result.extend((ptr.int().to(device),dst[order].int().to(device),times[order].to(device)))
        return cls(*result)


def oracle(graph, directions, requests, spec, settings=None, composition=None,
           *, truncate_seed_block=False):
    """Torch reference composer; follows the spec's seed_block_policy like the native path."""
    from openrfm.sampler.composition import compose_oracle
    if requests.is_cuda:
        raise ValueError('v2 oracle is CPU-only')
    if composition is None:
        raise ValueError('v2 requires admitted composition metadata')
    return compose_oracle(
        graph,directions,requests,spec,composition,settings,
        truncate_seed_block=truncate_seed_block,
    )


def check_reserve(needed, device, reserve):
    if torch.device(device).type == 'cuda':
        free,_ = torch.cuda.mem_get_info(device)
        if needed + reserve > free:
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(device)
        if needed + reserve > free:
            raise RuntimeError(f'v2 admission needs {needed} bytes plus {reserve} reserve; free={free}')


def seed_work_size(spec):
    """Per-episode entries of the program seed-ranking workspace (native seed_work_size)."""
    return (5 + 16) * spec.seed_hop_cap if spec.seed_rank == 'program' and spec.seed_tier_percent else 1


class Workspace:
    @staticmethod
    def _allocation(graph, batch_size, spec, composition):
        if composition is None:
            raise ValueError('v2 requires admitted composition metadata')
        from openrfm.sampler.path_streams import receipt_size
        geometry = composition.path_schema.plan(spec)
        B,L,F,C=batch_size,max(spec.seq_len_set),geometry.capacity,spec.candidate_count
        S=geometry.n_streams
        H=1
        while H<4*F:H*=2
        E=1
        while E<4*L:E*=2
        T=len(composition.caps)
        G=T*(spec.time_strata+1 if spec.extension_policy == 'typed_temporal_mix' else 1)
        priority=spec.seed_block_policy=='truncate_priority'
        shapes=((B,spec.walk_count,spec.walk_steps),(B,C),(B,C),(B,L),(B,L),(B,receipt_size(S, spec._model_requires_provenance)),
                (B,F,7),(B,G),(B,H),(B,E),(B,E),(B,S,20),(len(geometry.flat),),(B,S),(B,seed_work_size(spec)))
        if max(len(graph.row_table),len(graph.cell_column),*(prod(shape)+255 for shape in shapes))>=2**31:
            raise ValueError('v2 int32 workspace address overflow')
        return geometry, shapes, (C,L,F,H,E,T,G,priority,S)

    @staticmethod
    def estimate(graph, directions, batch_size, spec, composition=None):
        """Return the exact persistent buffer bytes allocated by ``__init__``."""
        del directions
        _geometry, shapes, _policy_geometry = Workspace._allocation(
            graph, batch_size, spec, composition)
        return sum(prod(shape)*8 for shape in shapes)

    def __init__(self, graph, directions, batch_size, spec, composition=None, *, pool_rows=None):
        if spec.own_source_rank != (getattr(composition, 'source_time', None) is not None):
            raise ValueError('native source rank: composition/spec clock binding differs')
        self.spec,self.graph,self.directions,self.composition=spec,graph,directions,composition
        self.pool_rows = pool_rows
        self.recent_rows = None
        if pool_rows is not None and (pool_rows.ndim != 1 or pool_rows.dtype != torch.int32
                or pool_rows.device != graph.row_table.device or not pool_rows.is_contiguous()):
            raise ValueError("inference label index must be contiguous resident int32 rows")
        geometry, shapes, policy_geometry = self._allocation(
            graph, batch_size, spec, composition)
        C,L,F,H,E,T,G,priority,S = policy_geometry
        self.path_geometry = geometry
        check_reserve(self.estimate(graph,directions,batch_size,spec,composition),
                      graph.edge_ptr.device,spec.reserve_bytes)
        from openrfm.sampler.path_streams import validate_weight_scales
        validate_weight_scales(geometry.flat, spec)
        admitted = int(composition.root_capacity.max())
        if graph.edge_ptr.device.type == 'cuda' and admitted > F:
            raise ValueError(
                f'composition admitted with root capacity {admitted} exceeds this spec\'s '
                f'plan capacity {F}; re-admit CompositionData under the workspace spec')
        self.buffers=[torch.empty(shape,dtype=torch.int64,device=graph.edge_ptr.device) for shape in shapes]
        self.buffers[12].copy_(torch.tensor(geometry.flat,dtype=torch.int64))
        self.policy=[spec.walk_count,spec.walk_steps,C,L,F,H,E,T,('strict_drop','history_available','query_only').index(spec.label_visibility),
            spec.k_enc_max,spec.path_depth,('keep','numeric_only','low_cardinality').index(spec.context_text_policy),
            spec.text_cardinality_limit,int(spec.task_kind=='forecast'),composition.anchor_table,
            spec.peer_labels_per_1024,spec.time_strata,spec.filler_percent,spec.walk_probes,int(priority),len(graph.row_table),
            S,spec.seed_hops,spec.frontier_rows,int(spec._model_requires_provenance),spec.protected_core_len,
            int(spec.extension_policy == 'typed_temporal_mix'),spec.rw_extension_percent,
            spec.extension_time_unit_seconds,spec.rw_scan_budget,
            int(spec.discovery_scales_with_length),
            int(spec.stream_weight_policy == 'selectivity'),
            spec.label_pool_reserve, spec.seed_tier_percent, spec.seed_local_rows,
            int(spec.seed_balance), spec.path_quota_rows, spec.path_quota_member_rows,
            spec.path_quota_share, spec.own_label_rows, ('walk', 'shared', 'program').index(spec.seed_rank),
            int(spec.seed_fill), int(spec.seed_hide_labels),
            int(spec.seed_local_policy == 'entity'), spec.seed_route_fanout,
            int(spec.path_stream_fill), spec.path_quota_parent_rows, spec.path_quota_grandparent_rows,
            spec.path_quota_grandchild_rows, int(spec.path_quota_order == 'density'), spec.seed_hop_cap,
            spec.peer_extension_percent, spec.peer_extension_rows, int(spec.seed_program_mature),
            4 if spec.label_rich_first else int(spec.own_recent_first)+int(spec.own_recent_interleave)+int(spec.own_recent_cell_balance), int(spec.canonical_recent)]
        if spec.real_training_cutoff is not None:
            self.policy[8] = 3

    @property
    def resident_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.buffers)

    def preflight(self,requests,settings,*,graph=None,composition=None,context=None):
        """Debug-only CPU composition check; caller must stay outside capture."""
        from openrfm.sampler.composition import compose_oracle
        graph=self.graph if graph is None else graph
        composition=self.composition if composition is None else composition
        g,d,m=(graph.to('cpu'),DirectedCSR(*(x.cpu() for x in self.directions.tensors())),
                 composition.to('cpu'))
        q,chosen=requests.cpu(),settings.cpu()
        diagnostics=[]
        try:
            compose_oracle(g,d,q,self.spec,m,chosen,diagnostics=diagnostics)
        except (ValueError,RuntimeError,IndexError) as error:
            detail=dict(context or {},**(diagnostics[-1] if diagnostics else {}))
            detail.update(status='failed',reason=str(error))
            raise CompositionPreflightError(detail) from error
        self.last_preflight_receipts=[dict(context or {},**record) for record in diagnostics]
        return self.last_preflight_receipts

    def sample(self, requests, settings, *, graph=None, composition=None):
        from openrfm.sampler.native import get_extension
        from openrfm.sampler.retrieval import _assert
        extension=get_extension(requests.device)
        if not hasattr(extension,'rt_compose_v2'):
            raise RuntimeError('v2 composition kernels missing; rebuild on the remote host')
        p=self.spec
        graph=self.graph if graph is None else graph
        composition=self.composition if composition is None else composition
        if composition.path_schema is not self.composition.path_schema:
            raise ValueError('task composition does not share the resident database geometry')
        if p.real_training_cutoff is not None:
            times = take(graph.row_time, requests[:, 0])
            _assert(((requests[:, 1] > 0) & (requests[:, 1] <= p.real_training_cutoff)
                & (requests[:, 4] == 1) & ((times == STATIC) | (times <= requests[:, 1]))
                & ((times == STATIC) | (times < p.real_training_cutoff))
                & (take(graph.row_task, requests[:, 0]) == 0)).all(),
                'real query crosses training cutoff or uses a task table')
        _assert(((requests[:,0]>=0)&(requests[:,0]<len(graph.row_table))).all(),'v2 invalid seed row')
        from openrfm.sampler.retrieval import SETTINGS_WIDTH
        if settings.dim()!=2 or settings.shape!=(len(requests),SETTINGS_WIDTH):
            raise ValueError(f'v2 settings must be [B,{SETTINGS_WIDTH}]')
        _assert(((settings[:,:3]>0).all()&(settings[:,0]<=max(p.seq_len_set)).all()
            &(settings[:,1]<=max(p.local_cell_budget_set)).all()&(settings[:,2]<=max(p.bfs_width_set)).all()
            &(settings[:,4]>0).all()&(settings[:,4]<=max(p.seq_len_set)).all()
            &((settings[:,3]==0)|(settings[:,3]==1)).all()),'v2 invalid context settings')
        metadata=composition.tensors()
        compose=extension.rt_compose_v2
        if self.pool_rows is not None:
            if not hasattr(extension,"rt_compose_v2_indexed"):
                raise RuntimeError("inference label-index kernel missing; rebuild on the remote host")
            compose=extension.rt_compose_v2_indexed
            if len(metadata)==19:metadata=metadata+[torch.empty(0,dtype=torch.int64,device=self.pool_rows.device)]
            metadata=metadata+[self.pool_rows]
        if self.recent_rows is not None:
            if self.pool_rows is None or not hasattr(extension,'rt_compose_v2_recent'):
                raise RuntimeError('inference recency index needs a label index and rebuilt native kernel')
            compose=extension.rt_compose_v2_recent
            metadata=metadata+[self.recent_rows]
        result=compose(graph.tensors(),self.directions.tensors(),metadata,
            requests,settings,self.buffers,self.policy)
        result=tuple(t[:len(requests)] for t in result)
        from openrfm.sampler.path_streams import has_row_provenance
        if has_row_provenance(result[5]) != p._model_requires_provenance:
            raise RuntimeError('native composer provenance ABI mismatch; rebuild on the remote host')
        if getattr(self,'last_preflight_receipts',None):
            codes=result[5][:,6].cpu()
            failure=codes<0
            failed=failure.nonzero().flatten()
            if len(failed):
                index=int(failed[0])
                detail=dict(self.last_preflight_receipts[index])
                detail.update(status='failed',phase='native_composition',error_code=int(codes[index]),
                    native_receipt=result[5][index].cpu().tolist())
                try:check_composition_receipt(result[5][index:index+1].cpu(),p)
                except ValueError as error:detail['reason']=str(error)
                raise CompositionPreflightError(detail)
        else:
            check_composition_receipt(result[5], p)
        return result


