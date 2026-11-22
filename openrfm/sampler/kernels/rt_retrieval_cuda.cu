#include "rt_retrieval.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "rt_composition_cuda.cuh"

namespace {
__global__ void walks(RTGraph g,const int64_t* q,RTPolicy p,int64_t B,int64_t* out){
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<B*p.W)rt_walk(g,q+5*(i/p.W),p,i%p.W,out+i*p.K);
}
__global__ void ranks(RTGraph g,const int64_t* visits,const int64_t* settings,
        RTPolicy p,int64_t B,int64_t* out,int64_t* scores){
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,S=p.W*p.K,b=i/S;
    if(b<B)rt_rank(g,visits+b*S,S,i%S,settings[4*b+3],p.C,out+b*p.C,scores+b*p.C);
}
__global__ void bfs(RTGraph g,const int64_t* q,const int64_t* settings,RTPolicy p,
        int64_t B,const int64_t* candidates,int64_t* cells,int64_t* roles,
        int64_t* receipt,int64_t* queue,int64_t* hash,int64_t H){
    int64_t b=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(b<B)rt_bfs(g,q+5*b,settings+4*b,p,candidates+b*p.C,cells+b*p.L,
        roles+b*p.L,receipt+7*b,queue+b*p.L*8,hash+b*H*5,H);
}
}
Tensors rt_retrieve_cuda(Tensors graph,Tensor q,Tensor settings,RTPolicy p){
    c10::cuda::CUDAGuard guard(q.device());auto stream=at::cuda::getCurrentCUDAStream();
    RTGraph g{rt_index(graph[0]),rt_index(graph[1]),rt_index(graph[2]),rt_index(graph[3]),rt_index(graph[4]),rt_index(graph[5]),rt_index(graph[6]),rt_index(graph[7]),rt_index(graph[8]),rt_index(graph[9]),rt_index(graph[10])};
    int64_t B=q.size(0),S=p.W*p.K,H=1;while(H<8*p.L+2*p.C)H*=2;
    auto visits=at::full({B,p.W,p.K},-1,q.options()),candidates=at::full({B,p.C},-1,q.options()),scores=at::zeros({B,p.C},q.options());
    auto cells=at::full({B,p.L},-1,q.options()),roles=at::zeros({B,p.L},q.options()),receipt=at::zeros({B,7},q.options());
    auto queue=at::empty({B,2*p.L,4},q.options()),hash=at::full({B,5*H},-1,q.options());
    if(B){
        walks<<<(B*p.W+127)/128,128,0,stream>>>(g,q.data_ptr<int64_t>(),p,B,visits.data_ptr<int64_t>());C10_CUDA_KERNEL_LAUNCH_CHECK();
        auto sorted=std::get<0>(at::sort(visits.reshape({B,S}),1));
        ranks<<<(B*S+127)/128,128,0,stream>>>(g,sorted.data_ptr<int64_t>(),settings.data_ptr<int64_t>(),p,B,candidates.data_ptr<int64_t>(),scores.data_ptr<int64_t>());C10_CUDA_KERNEL_LAUNCH_CHECK();
        bfs<<<(B+31)/32,32,0,stream>>>(g,q.data_ptr<int64_t>(),settings.data_ptr<int64_t>(),p,B,candidates.data_ptr<int64_t>(),cells.data_ptr<int64_t>(),roles.data_ptr<int64_t>(),receipt.data_ptr<int64_t>(),queue.data_ptr<int64_t>(),hash.data_ptr<int64_t>(),H);C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {visits,candidates,scores,cells,roles,receipt};
}
