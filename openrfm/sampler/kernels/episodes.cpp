#include <torch/extension.h>
#include "episodes.h"

namespace py = pybind11;

#ifdef WITH_CUDA
Tensor gather_embeddings_cuda(Tensor matrix, Tensor indices);
#endif
Tensor gather_embeddings(Tensor matrix, Tensor indices) {

    TORCH_CHECK(matrix.dim()==2 && matrix.is_contiguous(), "embedding matrix must be contiguous [V,D]");
    TORCH_CHECK(indices.scalar_type()==at::kLong && indices.is_contiguous(), "embedding indices must be contiguous int64");
    if(indices.is_cuda()) {
#ifdef WITH_CUDA
        return gather_embeddings_cuda(matrix, indices);
#else
        TORCH_CHECK(false, "CUDA embedding gather unavailable");
#endif
    }
    TORCH_CHECK(matrix.device()==indices.device(), "embedding/index device mismatch");
    auto shape=indices.sizes().vec();shape.push_back(matrix.size(1));
    return matrix.index_select(0,indices.reshape({-1})).reshape(shape);
}

#include "rt_retrieval.h"
Tensors rt_compose_v2(Tensors,Tensors,Tensors,Tensor,Tensor,Tensors,std::vector<int64_t>);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("rt_retrieve", &rt_retrieve);
    m.def("rt_compose_v2", &rt_compose_v2);
    m.def("rt_compose_v2_indexed", &rt_compose_v2, py::call_guard<py::gil_scoped_release>());
    m.def("rt_compose_v2_recent", &rt_compose_v2, py::call_guard<py::gil_scoped_release>());
    m.def("gather_embeddings",&gather_embeddings);
    m.def("rt_provenance",&rt_provenance);
    m.def("rt_alias_slots",&rt_alias_slots,py::call_guard<py::gil_scoped_release>());
#ifdef WITH_CUDA
    m.attr("has_cuda")=true;
#else
    m.attr("has_cuda")=false;
#endif
}
