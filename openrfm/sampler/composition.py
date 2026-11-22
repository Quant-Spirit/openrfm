"""Typed seed/history/peer/filler composition; independent CPU oracle metadata."""
from dataclasses import dataclass, replace
import numpy as np
import torch
from openrfm.archive.format import widen_object
from openrfm.sampler.retrieval import STATIC, UNKNOWN


def _task_assert(value, message):
    """Preserve cold ValueErrors while keeping CUDA task binding asynchronous."""
    if value.is_cuda:
        torch._assert_async(value, message)
    elif not bool(value):
        raise ValueError(message)


@dataclass
class CompositionData:
    types: torch.Tensor
    cardinality: torch.Tensor
    target: torch.Tensor
    classes: torch.Tensor
    entity: torch.Tensor
    history_ptr: torch.Tensor
    history_rows: torch.Tensor
    caps: torch.Tensor
    group_ptr: torch.Tensor
    group_edges: torch.Tensor
    group_tables: torch.Tensor
    group_rows: torch.Tensor
    group_times: torch.Tensor
    entity_key_ptr: torch.Tensor
    entity_keys: torch.Tensor
    table_ids: torch.Tensor
    root_capacity: torch.Tensor
    stream_mask: torch.Tensor
    task_capacity: torch.Tensor
    anchor_table: int = -1
    path_schema: object = None
    source_time: torch.Tensor = None

    def tensors(self):
        return [v for v in vars(self).values() if isinstance(v,torch.Tensor)]

    def to(self,device):
        return replace(self, **{k: v.to(device) for k, v in vars(self).items()
                               if isinstance(v, torch.Tensor)})

    @classmethod
    def admit(cls,g,d,payload,spec,table_names=None,*,schema_cache=None):
        """Admit the DB structure, then bind one task."""
        shared = cls.admit_shared(g, d, payload, spec, table_names,
                                  schema_cache=schema_cache)
        return shared.bind_task(g, d, payload, spec)

    @classmethod
    def admit_shared(cls,g,d,payload,spec,table_names=None,*,schema_cache=None,ordered_directed=False,cpu_sort=False):
        """Cold CPU metadata whose tensors and path geometry are database-wide."""
        if g.row_table.is_cuda:
            raise ValueError('composition metadata must be admitted before graph residency')
        g = widen_object(g)
        required={'cell_types','cell_text_ids','cell_codes','cell_numbers'}
        if required-payload.keys():
            raise ValueError(f'composition needs explicit cell metadata: {sorted(required-payload.keys())}')
        n,m=len(g.row_table),len(g.cell_column)
        types=payload['cell_types'].long()
        if (any(payload[k].shape!=(m,) for k in required)
                or ((types<0)|(types>4)).any() or (g.row_table<0).any()):
            raise ValueError('unaligned composition cell metadata')
        cache = {} if schema_cache is None else schema_cache
        if 'text_cardinality' not in cache:
            from openrfm.sampler.admission import distinct_pair_counts
            card=torch.full((m,),-1,dtype=torch.int64)
            text=types==1
            if text.any():
                cols,counts=distinct_pair_counts(g.cell_column[text].numpy(),
                                                payload['cell_text_ids'][text].numpy())
                cols,counts=torch.from_numpy(cols),torch.from_numpy(counts)
                card[text]=counts[torch.searchsorted(cols,g.cell_column[text])]
            cache['text_cardinality'] = card
        card = cache['text_cardinality']
        if card.dtype in (torch.uint8, torch.uint16):
            card = card.int()
        table_names=table_names or {}
        anchor=-1
        if spec.entity_table is not None:
            if spec.entity_table not in table_names:
                raise ValueError(f'unknown composition entity_table: {spec.entity_table}')
            anchor=int(table_names[spec.entity_table])
        entity=torch.full((n,),-1,dtype=torch.int64)
        query_only=spec.label_visibility=='query_only'
        parent_tables = spec.entity_parent_tables
        unknown = set(parent_tables) - table_names.keys()
        if unknown:
            raise ValueError(f'unknown composition entity parents: {sorted(unknown)}')
        if query_only:
            single=d.fptr.long().diff()==1
            entity[single]=d.fdst.long()[d.fptr[:-1].long()[single]]
        if anchor>=0 and (not parent_tables or spec.task_kind == 'forecast'):
            src=torch.repeat_interleave(torch.arange(n),d.fptr.long().diff())
            chosen=torch.nonzero(g.row_table[d.fdst.long()]==anchor).flatten()
            # A row with several foreign keys into the anchor table takes its last one in CSR order.
            last=torch.full((n,),-1,dtype=torch.int64).scatter_reduce(0,src[chosen],chosen,'amax')
            entity=torch.where(last>=0,d.fdst.long()[last.clamp_min(0)],entity)
            if spec.real_training_cutoff is not None:
                roots = g.row_table == anchor
                entity[roots] = torch.arange(n)[roots]
        key_src = torch.empty(0, dtype=torch.int64)
        keys = torch.empty(0, dtype=torch.int64)
        if parent_tables or query_only:
            key_src = torch.repeat_interleave(torch.arange(n), d.fptr.long().diff())
            keys = d.fdst.long()
            if parent_tables:
                declared = torch.tensor([table_names[t] for t in parent_tables], dtype=torch.int64)
                chosen = torch.isin(g.row_table[keys], declared)
                roots = torch.arange(n)[torch.isin(g.row_table, declared)]
                key_src = torch.cat((key_src[chosen], roots))
                keys = torch.cat((keys[chosen], roots))
            pairs = torch.unique(key_src * n + keys, sorted=True)
            key_src, keys = pairs // n, pairs % n
        key_counts = torch.bincount(key_src, minlength=n)
        kp = torch.cat((torch.zeros(1, dtype=torch.int64), key_counts.cumsum(0)))
        if parent_tables:
            single = key_counts == 1
            entity[single] = keys[kp[:-1][single]]
        table_ids=torch.unique(g.row_table,sorted=True)
        tables=len(table_ids)
        if tables<=0 or tables>=2**31:
            raise ValueError('composition table address overflow')
        caps=torch.bincount(torch.searchsorted(table_ids,g.row_table),minlength=tables)
        src=torch.repeat_interleave(torch.arange(n),d.pptr.long().diff());dst=d.pdst.long()
        allowed=g.row_task[dst]==0
        src,dst=src[allowed],dst[allowed]
        tab=g.row_table[dst];times=g.row_time[dst]
        if ordered_directed:
            group=src*tables+torch.searchsorted(table_ids,tab)
            order=(torch.from_numpy(np.argsort(group.numpy(),kind='stable'))
                   if cpu_sort else torch.argsort(group,stable=True))
        else:
            order=torch.argsort(dst,stable=True)
            for key in (times,tab,src):
                order=order[torch.argsort(key[order],stable=True)]
        src,dst,tab,times=(v[order] for v in (src,dst,tab,times))
        first=torch.ones(len(dst),dtype=torch.bool)
        first[1:]=(src[1:]!=src[:-1])|(tab[1:]!=tab[:-1])
        begins=first.nonzero().flatten()
        gp=torch.cat((torch.zeros(1,dtype=torch.int64),torch.bincount(src[first],minlength=n).cumsum(0)))
        ge=torch.cat((begins,torch.tensor([len(dst)])))
        from openrfm.sampler.path_streams import PathSchema
        schema = PathSchema.admit(g, d, schema_cache=cache)
        geometry = schema.plan(spec)
        root_capacity=torch.tensor([geometry.roots[int(t)]['capacity'] for t in table_ids],dtype=torch.int64)
        target=torch.full((n,),-1,dtype=torch.int64)
        classes=torch.full((n,),-1,dtype=torch.int64)
        hp=torch.zeros(n+1,dtype=torch.int64)
        hr=torch.empty(0,dtype=torch.int64)
        stream_mask=torch.ones(len(table_ids),dtype=torch.int64)
        task_capacity=torch.tensor([geometry.capacity],dtype=torch.int64)
        source_time = None
        if spec.own_source_rank:
            source_time = payload.get('own_source_time')
            if (not torch.is_tensor(source_time) or source_time.device.type != 'cpu'
                    or source_time.dtype not in (torch.int32, torch.int64)
                    or source_time.shape != (n,)
                    or ((source_time > g.row_time) | ((source_time < 0) & (source_time != STATIC))).any()):
                raise ValueError('own source rank requires admitted exact source clocks')
            source_time = source_time.contiguous()
        return cls(types,card,target,classes,entity,hp,hr,caps,gp,ge,
            torch.searchsorted(table_ids,tab[first]),dst,times,kp,keys,
            table_ids,root_capacity,stream_mask,task_capacity,anchor,schema,source_time)

    def bind_task(self,g,d,payload,spec):
        """Bind label visibility and stream eligibility on the resident device."""
        g, self = widen_object(g), widen_object(self)
        n,m=len(g.row_table),len(g.cell_column)
        device=g.row_table.device
        labels=g.cell_label.bool()
        label_cells=labels.nonzero(as_tuple=False).flatten()
        label_rows=torch.searchsorted(g.cell_ptr[1:].long(),label_cells,right=True)
        counts=torch.zeros(n,dtype=torch.int64,device=device)
        counts.scatter_add_(0,label_rows,torch.ones_like(label_rows,dtype=torch.int64))
        _task_assert((counts<=1).all(), 'composition requires one declared target column per row')
        target=torch.full((n,),-1,dtype=torch.int64,device=device)
        target.scatter_reduce_(0,label_rows,label_cells,reduce='amax',include_self=True)
        classes=torch.full((n,),-1,dtype=torch.int64,device=device)
        types=self.types;codes=payload['cell_codes']
        label_types=types[label_cells]
        discrete=label_types==4
        label_classes=torch.where(discrete,codes[label_cells].long(),
                                  torch.full_like(label_cells,-1,dtype=torch.int64))
        boolean=label_types==3
        if 'cell_booleans' not in payload:
            _task_assert(~boolean.any(), 'Boolean labels require explicit cell_booleans')
        else:
            values=payload['cell_booleans'][label_cells]
            if payload['cell_booleans'].shape!=(m,):
                raise ValueError('invalid Boolean target values: unaligned payload')
            _task_assert((~boolean|torch.isfinite(values)).all(),
                         'invalid Boolean target values: expected finite vector')
            lo=torch.where(boolean,values,torch.full_like(values,float('inf'))).min()
            hi=torch.where(boolean,values,torch.full_like(values,float('-inf'))).max()
            _task_assert((~boolean|(values==lo)|(values==hi)).all(),
                         'invalid Boolean target values: more than two levels')
            label_classes=torch.where(boolean,(values!=lo).long(),label_classes)
        numbers=payload['cell_numbers'][label_cells]
        numeric=(label_types==0)&torch.isfinite(numbers)
        bins=torch.tensor(spec.regression_label_bins,dtype=torch.float64,device=device)
        label_classes=torch.where(numeric,torch.bucketize(numbers.double(),bins,right=True),
                                 label_classes)
        classes.scatter_reduce_(0,label_rows,label_classes,reduce='amax',include_self=True)

        key_counts=self.entity_key_ptr[1:].long()-self.entity_key_ptr[:-1].long()
        if self.anchor_table>=0 and (not spec.entity_parent_tables or spec.task_kind=='forecast'):
            src=torch.repeat_interleave(torch.arange(n,device=device),d.fptr.long().diff())
            chosen=g.row_table[d.fdst.long()]==self.anchor_table
            count=torch.bincount(src[chosen],minlength=n)
            _task_assert(~(((g.row_task!=0)&(target>=0)&(count!=1)).any()),
                         'task rows require exactly one declared entity FK')
            if spec.real_training_cutoff is not None:
                roots=g.row_table==self.anchor_table
                _task_assert(~(((target>=0)&~roots&(count!=1)).any()),
                             'real target rows require exactly one declared entity FK')
                _task_assert(~(((target>=0)&(self.entity<0)).any()),
                             'real target row lacks its declared entity key')
        if spec.entity_parent_tables:
            _task_assert(~(((target>=0)&(key_counts==0)).any()),
                         'target row lacks its declared entity keys')

        own=((self.entity>=0) if (spec.label_visibility=='query_only' or
             spec.real_training_cutoff is not None) else
             (self.entity>=0)&(target>=0)&(g.row_task!=0))
        hr=torch.arange(n,device=device)[own]
        order=torch.argsort(hr,stable=True)
        order=order[torch.argsort(g.row_time[hr[order]],stable=True)]
        order=order[torch.argsort(self.entity[hr[order]],stable=True)]
        hr=hr[order]
        hp=torch.cat((torch.zeros(1,dtype=torch.int64,device=device),
                      torch.bincount(self.entity[hr],minlength=n).cumsum(0)))

        mask=torch.zeros_like(self.table_ids,dtype=torch.int64)
        table_slots=torch.searchsorted(self.table_ids,g.row_table[label_rows])
        mask.scatter_reduce_(0,table_slots,torch.ones_like(label_rows,dtype=torch.int64),
                             reduce='amax',include_self=True)
        task_capacity=torch.where(mask.bool(),self.root_capacity.long(),0).max().reshape(1)
        if device.type=='cpu' and self.types.dtype in (torch.uint8,torch.uint16,torch.int32):
            from openrfm.archive.format import compact_tensor
            target,classes,hp,hr,mask,task_capacity=(compact_tensor(x) for x in
                (target,classes,hp,hr,mask,task_capacity))
        return replace(self,target=target,classes=classes,history_ptr=hp,history_rows=hr,
                       stream_mask=mask,task_capacity=task_capacity)


def compose_oracle(g,d,q,spec,m,settings=None,truncate_seed_block=False,*,diagnostics=None):
    """The independent path-stream oracle; roles never allocate cells."""
    from openrfm.sampler.path_streams import compose_paths
    return compose_paths(g,d,q,spec,m,settings,diagnostics=diagnostics,
                         truncate_seed_block=truncate_seed_block)


def forecast_identity_mask(g,m,roots,spec,target_column,cuts=None):
    """Which query rows the composer can admit (native ``setup()`` identity codes -3/-4); bool over ``roots``.

    A forecast row without an entity payload cell visible at the cut aborts the native sampler, so callers exclude it.
    """
    g,m=widen_object(g),widen_object(m)
    device=g.row_table.device
    roots=torch.as_tensor(roots,dtype=torch.int64,device=device)
    cut=(g.row_time.long()[roots] if cuts is None else
         torch.as_tensor(cuts,dtype=torch.int64,device=device))
    target=m.target.long()
    focal=target[roots]
    ok=(focal>=0)&(g.cell_column.long()[focal.clamp_min(0)]==int(target_column))
    if spec.task_kind!='forecast':
        return ok
    history=('strict_drop','history_available','query_only').index(spec.label_visibility)
    n=len(g.row_table)
    entity=m.entity.long()[roots]
    ok&=entity>=0
    e=entity.clamp_min(0)
    table,task,row_time=g.row_table.long(),g.row_task.long(),g.row_time.long()
    same_table=table[e]==table[roots]
    row_ok=(task[e]==0)|((task[e]!=0)&same_table)
    if history!=2:
        row_ok&=(row_time[e]==STATIC)|(row_time[e]<cut)
    cols,label,drop=g.cell_column.long(),g.cell_label.bool(),g.cell_drop.bool()
    feature=~label&(cols!=int(target_column))
    if history!=2:
        feature&=~drop
    text=('keep','numeric_only','low_cardinality').index(spec.context_text_policy)
    if text:
        kind=m.types.long()
        feature&=((kind==0)|(kind==2)|(kind==3)) if text==1 else (
            (kind!=1)|((m.cardinality.long()>=0)&(m.cardinality.long()<=spec.text_cardinality_limit)))
    cell_rows=torch.repeat_interleave(torch.arange(n,device=device),g.cell_ptr.long().diff())
    available=torch.full_like(g.cell_available.long(),STATIC) if history==2 else g.cell_available.long()
    earliest=torch.full((n,),UNKNOWN,dtype=torch.int64,device=device)
    earliest.scatter_reduce_(0,cell_rows[feature],available[feature],reduce='amin',include_self=True)
    visible=row_ok&(earliest[e]!=UNKNOWN)&(earliest[e]<=cut)
    et=target[e];cell=et.clamp_min(0)
    label_ok=row_ok&same_table&(et>=0)&(m.classes.long()[e]>=0)
    if history==1:
        label_ok&=~drop[cell]&(g.cell_available.long()[cell]<cut)
    elif history!=2:
        label_ok&=False
    # A null-only entity row has no archived cells but still defines a query.
    empty=row_ok&(g.cell_ptr.long()[e]==g.cell_ptr.long()[e+1])
    return ok&((e==roots)|visible|label_ok|empty)


