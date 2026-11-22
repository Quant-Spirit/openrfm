#pragma once
#include <ATen/ATen.h>
#include <cstdint>
#include <vector>
using at::Tensor;
using Tensors = std::vector<Tensor>;

#ifdef __CUDACC__
#define EP_HD __host__ __device__
#else
#define EP_HD
#endif

Tensors rt_provenance(Tensor node,Tensor parents,Tensor padding,Tensor target,Tensor tables,Tensor timestamps,Tensor requests,Tensor row_table);
std::tuple<Tensor,Tensor> rt_alias_slots(Tensor receipt,Tensor task_kind,Tensor seeds,int64_t alias_count);
