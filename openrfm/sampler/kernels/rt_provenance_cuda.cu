#include "rt_retrieval.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
typedef unsigned long long u64;
constexpr int THREADS = 512, DEPTH = 6, FAR = 65;
constexpr int64_t EMPTY = -1;

__device__ inline u64 path_hash(u64 prefix, int direction, int64_t table) {
    u64 h = rt_mix(prefix ^ rt_mix(u64(direction + 1)) ^ rt_mix(u64(table + 17))) & 0x7fffffffffffffffULL;
    return 1 + h % 0x7fffffffULL;
}

__device__ inline int slot_insert(int64_t* keys, int H, int64_t key) {
    int h = int(rt_mix(u64(key)) & u64(H - 1));
    for (;;) {
        int64_t old = int64_t(atomicCAS((u64*)&keys[h], u64(EMPTY), u64(key)));
        if (old == EMPTY || old == key) return h;
        h = (h + 1) & (H - 1);
    }
}
__device__ inline int slot_find(const int64_t* keys, int H, int64_t key) {
    int h = int(rt_mix(u64(key)) & u64(H - 1));
    for (;; h = (h + 1) & (H - 1)) {
        int64_t k = keys[h];
        if (k == key) return h;
        if (k == EMPTY) return -1;
    }
}
__device__ inline int floor_log2(u64 x) { return 63 - __clzll(x); }

__global__ void provenance(const int64_t* node, const int64_t* parents, const bool* padding, const bool* target,
        const int64_t* tables, const int64_t* stamps, const int64_t* requests, RTIndex row_table, int64_t N,
        int64_t S, int64_t P, int H, int HW, int64_t* words,
        int* state_all, int* edge_all, int16_t* hop_out, int32_t* sig_out, int8_t* td_out, int8_t* din_out, int8_t* dout_out) {
    const int64_t b = blockIdx.x;
    int64_t* keys = words + b * (3 * int64_t(H) + HW);
    u64* sig = (u64*)keys + H;
    u64* cand = sig + H;
    u64* wit = cand + H;
    int* dist = state_all + b * 6 * H;
    int* dnext = dist + H; int* present = dnext + H; int* outdeg = present + H; int* indeg = outdeg + H;
    int* edges = edge_all + b * (S + S * P);
    int* cell_slot = edges; int* edge_slot = edges + S;
    const int64_t* nd = node + b * S; const int64_t* par = parents + b * S * P; const bool* pad = padding + b * S;
    const bool* tg = target + b * S; const int64_t* tb = tables + b * S; const int64_t* ts = stamps + b * S;
    const int tid = threadIdx.x;
    __shared__ int root_cell, root_slot;
    for (int i = tid; i < H; i += THREADS) { keys[i] = EMPTY; dist[i] = FAR; present[i] = 0; outdeg[i] = 0; indeg[i] = 0; sig[i] = 0; }
    for (int i = tid; i < HW; i += THREADS) wit[i] = 0;
    if (tid == 0) root_cell = int(S);
    __syncthreads();
    for (int c = tid; c < S; c += THREADS) {
        int64_t row = nd[c] < 0 ? 0 : (nd[c] >= N ? N - 1 : nd[c]);
        cell_slot[c] = slot_insert(keys, H, row);
        if (tg[c] && !pad[c]) atomicMin(&root_cell, c);
    }
    __syncthreads();
    if (tid == 0) { if (root_cell == S) root_cell = 0; root_slot = cell_slot[root_cell]; }
    for (int c = tid; c < S; c += THREADS) if (!pad[c]) present[cell_slot[c]] = 1;
    __syncthreads();
    const int64_t root_table = tb[root_cell];

    for (int64_t i = tid; i < S * P; i += THREADS) {
        int64_t c = i / P, p = par[i];
        int slot = -1;
        if (!pad[c] && p >= 0 && p < N) { slot = slot_find(keys, H, p); if (slot >= 0 && !present[slot]) slot = -1; }
        edge_slot[i] = slot;
        if (slot >= 0 && slot != cell_slot[c]) {
            u64 key = (u64(cell_slot[c]) << 32 | u64(slot)) + 1;
            int h = int(rt_mix(key) & u64(HW - 1));
            for (;;) {
                u64 old = atomicCAS(&wit[h], u64(0), key);
                if (old == 0) { atomicAdd(&outdeg[cell_slot[c]], 1); atomicAdd(&indeg[slot], 1); break; }
                if (old == key) break;
                h = (h + 1) & (HW - 1);
            }
        }
    }
    if (tid == 0) dist[root_slot] = 0;
    __syncthreads();
    for (int round = 0; round < DEPTH; ++round) {
        for (int i = tid; i < H; i += THREADS) { dnext[i] = dist[i]; cand[i] = ~u64(0); }
        __syncthreads();
        for (int64_t i = tid; i < S * P; i += THREADS) {
            int v = edge_slot[i]; if (v < 0) continue;
            int u = cell_slot[i / P];
            atomicMin(&dnext[u], dist[v] + 1);
            atomicMin(&dnext[v], dist[u] + 1);
        }
        __syncthreads();
        for (int64_t i = tid; i < S * P; i += THREADS) {
            int v = edge_slot[i]; if (v < 0) continue;
            int64_t c = i / P; int u = cell_slot[c];
            int pd = dist[v], source = dist[u];
            if (pd < FAR && source == FAR && pd + 1 == dnext[u]) {
                u64 prefix = sig[v] ? sig[v] : u64(root_table);
                atomicMin(&cand[u], path_hash(prefix, 1, tb[c]));
            }
            if (source < FAR && pd == FAR && source + 1 == dnext[v]) {
                u64 prefix = sig[u] ? sig[u] : u64(root_table);
                atomicMin(&cand[v], path_hash(prefix, 0, row_table[par[i]]));
            }
        }
        __syncthreads();
        for (int i = tid; i < H; i += THREADS) { if (cand[i] != ~u64(0)) sig[i] = cand[i]; dist[i] = dnext[i]; }
        __syncthreads();
    }
    const int64_t cut = requests[b * 5 + 1];
    for (int c = tid; c < S; c += THREADS) {
        int64_t o = b * S + c; int u = cell_slot[c];
        if (pad[c]) { hop_out[o] = 0; sig_out[o] = 0; td_out[o] = 0; din_out[o] = 0; dout_out[o] = 0; continue; }
        int hops = dist[u] > FAR ? FAR : dist[u];
        hop_out[o] = int16_t(hops);
        sig_out[o] = hops == FAR ? int32_t(1 + tb[c] % 0x7fffffffLL) : int32_t(sig[u]);
        int64_t delta = ts[c] - cut, a = delta < 0 ? -delta : delta;
        int log = int(fmin(floor(log2((double)(a < 1 ? 1 : a))), 63.0)) + 1;
        int td = delta == 0 ? 0 : (delta > 0 ? log : -log);
        td_out[o] = int8_t((ts[c] == -(1LL << 62) || ts[c] == -(1LL << 31)) ? -128 : td);
        int di = floor_log2(u64(1 + indeg[u])), dd = floor_log2(u64(1 + outdeg[u]));
        din_out[o] = int8_t(di > 31 ? 31 : di); dout_out[o] = int8_t(dd > 31 ? 31 : dd);
    }
}
}

Tensors rt_provenance_cuda(Tensor node, Tensor parents, Tensor padding, Tensor target, Tensor tables,
                           Tensor stamps, Tensor requests, Tensor row_table, int64_t N) {
    c10::cuda::CUDAGuard guard(node.device()); auto stream = at::cuda::getCurrentCUDAStream();
    int64_t B = node.size(0), S = node.size(1), P = parents.size(2);
    int H = 1; while (H < 2 * S) H *= 2;
    int HW = 1; while (HW < 2 * S * P) HW *= 2;
    auto i64 = node.options(), i32 = node.options().dtype(at::kInt);
    auto words = at::empty({B, 3 * H + HW}, i64);
    auto state = at::empty({B, 6 * H}, i32), edges = at::empty({B, S + S * P}, i32);
    auto hop = at::empty({B, S}, node.options().dtype(at::kShort)), sig = at::empty({B, S}, i32);
    auto td = at::empty({B, S}, node.options().dtype(at::kChar)), din = at::empty_like(td), dout = at::empty_like(td);
    int64_t* w = words.data_ptr<int64_t>();
    provenance<<<B, THREADS, 0, stream>>>(node.data_ptr<int64_t>(), parents.data_ptr<int64_t>(), padding.data_ptr<bool>(),
        target.data_ptr<bool>(), tables.data_ptr<int64_t>(), stamps.data_ptr<int64_t>(), requests.data_ptr<int64_t>(),
        rt_index(row_table), N, S, P, H, HW, w,
        state.data_ptr<int>(), edges.data_ptr<int>(), hop.data_ptr<int16_t>(), sig.data_ptr<int32_t>(),
        td.data_ptr<int8_t>(), din.data_ptr<int8_t>(), dout.data_ptr<int8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {hop, sig, td, din, dout};
}
