#include <torch/extension.h>
#include <climits>
#include "rt_retrieval.h"

Tensors rt_retrieve(Tensors graph,Tensor q,Tensor settings,int64_t W,int64_t K,
                    int64_t C,int64_t L,int64_t hops,int64_t history,int64_t legacy) {
    TORCH_CHECK(graph.size()==11,"RT graph needs 11 tensors");
    for(auto& x:graph)TORCH_CHECK(x.is_contiguous()&&x.dim()==1&&rt_index_type(x)&&x.device()==q.device(),"invalid RT graph tensor");
    TORCH_CHECK(q.dim()==2&&q.size(1)==5&&q.scalar_type()==at::kLong&&q.is_contiguous(),"RT requests must be int64 [B,5]");
    TORCH_CHECK(settings.dim()==2&&settings.size(0)==q.size(0)&&settings.size(1)==4&&settings.scalar_type()==at::kLong&&settings.device()==q.device()&&settings.is_contiguous(),"RT settings must be int64 [B,4]");
    TORCH_CHECK(W>0&&K>0&&W<=INT_MAX/K&&C>0&&C<INT_MAX&&L>0&&hops>=0,"invalid RT retrieval dimensions");
    TORCH_CHECK(L<=INT64_MAX/256&&q.size(0)<=INT64_MAX/(L*256),"RT workspace overflow");
    TORCH_CHECK(L<=INT64_MAX/(C+1),"RT generation counter overflow");
    auto N=graph[3].numel(),M=graph[7].numel();
    TORCH_CHECK(N>0&&M>0&&graph[0].numel()==N+1&&graph[6].numel()==N+1,"invalid RT row arrays");
    for(int i:{4,5})TORCH_CHECK(graph[i].numel()==N,"unaligned RT rows");
    for(int i:{8,9,10})TORCH_CHECK(graph[i].numel()==M,"unaligned RT cells");
    TORCH_CHECK(graph[1].numel()==graph[2].numel(),"unaligned RT edges");
    RTPolicy p{W,K,C,L,hops,history,legacy};
#ifdef WITH_CUDA
    if(q.is_cuda())return rt_retrieve_cuda(graph,q,settings,p);
#endif
    TORCH_CHECK(!q.is_cuda(),"RT native extension was built without CUDA");
    RTGraph g{rt_index(graph[0]),rt_index(graph[1]),rt_index(graph[2]),rt_index(graph[3]),rt_index(graph[4]),rt_index(graph[5]),rt_index(graph[6]),rt_index(graph[7]),rt_index(graph[8]),rt_index(graph[9]),rt_index(graph[10])};
    int64_t B=q.size(0),S=W*K,H=1;while(H<8*L+2*C)H*=2;
    auto visits=at::full({B,W,K},-1,q.options()),candidates=at::full({B,C},-1,q.options()),scores=at::zeros({B,C},q.options());
    auto cells=at::full({B,L},-1,q.options()),roles=at::zeros({B,L},q.options()),receipt=at::zeros({B,7},q.options());
    auto queue=at::empty({B,2*L,4},q.options()),hash=at::full({B,5*H},-1,q.options());
    for(int64_t b=0;b<B;++b){auto r=q.data_ptr<int64_t>()+5*b;auto s=settings.data_ptr<int64_t>()+4*b;
        TORCH_CHECK(r[0]>=0&&r[0]<N&&s[0]>0&&s[0]<=L&&s[1]>0&&s[2]>0&&(s[3]==0||s[3]==1),"invalid RT request/settings");
        for(int64_t w=0;w<W;++w)rt_walk(g,r,p,w,visits.data_ptr<int64_t>()+b*S+w*K);
    }
    auto sorted=std::get<0>(at::sort(visits.reshape({B,S}),1));
    for(int64_t b=0;b<B;++b){
        for(int64_t i=0;i<S;++i)rt_rank(g,sorted.data_ptr<int64_t>()+b*S,S,i,settings.data_ptr<int64_t>()[4*b+3],C,candidates.data_ptr<int64_t>()+b*C,scores.data_ptr<int64_t>()+b*C);
        rt_bfs(g,q.data_ptr<int64_t>()+5*b,settings.data_ptr<int64_t>()+4*b,p,candidates.data_ptr<int64_t>()+b*C,cells.data_ptr<int64_t>()+b*L,roles.data_ptr<int64_t>()+b*L,receipt.data_ptr<int64_t>()+b*7,queue.data_ptr<int64_t>()+b*L*8,hash.data_ptr<int64_t>()+b*H*5,H);
    }
    return {visits,candidates,scores,cells,roles,receipt};
}
