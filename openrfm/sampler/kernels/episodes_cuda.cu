#include "episodes.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

namespace {
struct Range { explicit Range(const char* name) { nvtxRangePushA(name); } ~Range(){nvtxRangePop();} };
}

namespace {
template<typename scalar_t>
__global__ void gather_embeddings_kernel(const scalar_t* matrix,const int64_t* indices,scalar_t* output,int64_t n,int64_t vocabulary,int64_t width) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n*width)return;
    int64_t row=indices[i/width];CUDA_KERNEL_ASSERT(row>=0 && row<vocabulary);
    output[i]=matrix[row*width+i%width];
}
}
Tensor gather_embeddings_cuda(Tensor matrix,Tensor indices) {
    c10::cuda::CUDAGuard guard(indices.device());Range range("episodes::gather_embeddings");
    TORCH_CHECK(matrix.device()==indices.device() || (matrix.device().is_cpu() && matrix.is_pinned()),
                "UVA gather requires pinned host text or text on the index device");
    auto shape=indices.sizes().vec();shape.push_back(matrix.size(1));
    auto output=at::empty(shape,matrix.options().device(indices.device()));
    auto stream=at::cuda::getCurrentCUDAStream();void* mapped=matrix.data_ptr();
    if(matrix.device().is_cpu()) {
        C10_CUDA_CHECK(cudaHostGetDevicePointer(&mapped,matrix.data_ptr(),0));
    }
    int64_t cells=indices.numel()*matrix.size(1);
    if(cells) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,matrix.scalar_type(),"gather_embeddings_cuda",[&] {
            gather_embeddings_kernel<<<(cells+255)/256,256,0,stream>>>(static_cast<const scalar_t*>(mapped),indices.data_ptr<int64_t>(),output.data_ptr<scalar_t>(),indices.numel(),matrix.size(0),matrix.size(1));
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if(matrix.device().is_cpu()) {

            TORCH_CHECK(at::cuda::CachingHostAllocator_recordEvent(matrix.data_ptr(),matrix.storage().data_ptr().get_context(),stream),
                        "UVA matrix must use the Torch pinned-host allocator");
        }
    }
    return output;
}
