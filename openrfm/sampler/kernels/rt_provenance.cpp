#include <torch/extension.h>
#include <climits>
#include "rt_retrieval.h"

#ifdef WITH_CUDA
Tensors rt_provenance_cuda(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,int64_t);
#endif

Tensors rt_provenance(Tensor node,Tensor parents,Tensor padding,Tensor target,Tensor tables,
                      Tensor timestamps,Tensor requests,Tensor row_table){
    TORCH_CHECK(node.is_cuda(),"provenance native path requires CUDA");
    auto device=node.device();
    TORCH_CHECK(node.dim()==2&&parents.dim()==3&&parents.size(0)==node.size(0)&&parents.size(1)==node.size(1),"provenance cell shapes");
    for(auto x:{node,parents,tables,timestamps,requests})TORCH_CHECK(x.is_contiguous()&&x.scalar_type()==at::kLong&&x.device()==device,"provenance int64 inputs");
    for(auto x:{padding,target})TORCH_CHECK(x.is_contiguous()&&x.scalar_type()==at::kBool&&x.device()==device&&x.sizes()==node.sizes(),"provenance Boolean inputs");
    TORCH_CHECK(tables.sizes()==node.sizes()&&timestamps.sizes()==node.sizes(),"provenance aligned cells");
    TORCH_CHECK(requests.dim()==2&&requests.size(0)==node.size(0)&&requests.size(1)==5,"provenance requests");
    TORCH_CHECK(row_table.is_contiguous()&&rt_index_type(row_table)&&row_table.device()==device,"provenance row table");
    int64_t B=node.size(0),S=node.size(1),P=parents.size(2);
    TORCH_CHECK(B>0&&S>0&&P>0&&B*S*P<INT_MAX&&S*P<(1LL<<28),"provenance dimensions");
#ifdef WITH_CUDA
    return rt_provenance_cuda(node,parents,padding,target,tables,timestamps,requests,row_table,row_table.numel());
#else
    TORCH_CHECK(false,"provenance CUDA kernel missing; rebuild remotely");
#endif
}
