#include <torch/extension.h>
#include <ATen/CPUGeneratorImpl.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using at::Tensor;

namespace {

const uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

std::array<uint8_t, 32> sha256(std::vector<uint8_t> message) {
    uint64_t bits = uint64_t(message.size()) * 8;
    message.push_back(0x80);
    while (message.size() % 64 != 56) message.push_back(0);
    for (int shift = 56; shift >= 0; shift -= 8) message.push_back(uint8_t(bits >> shift));
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    for (size_t block = 0; block < message.size(); block += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            const uint8_t* p = &message[block + 4 * i];
            w[i] = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
        }
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h[0], b=h[1], c=h[2], d=h[3], e=h[4], f=h[5], g=h[6], hh=h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    std::array<uint8_t, 32> out;
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++) out[4 * i + j] = uint8_t(h[i] >> (24 - 8 * j));
    return out;
}

void put_i64(std::vector<uint8_t>& out, int64_t value) {
    uint8_t raw[8];
    std::memcpy(raw, &value, 8);
    out.insert(out.end(), raw, raw + 8);
}

}

std::tuple<Tensor, Tensor> rt_alias_slots(Tensor receipt, Tensor task_kind, Tensor seeds, int64_t alias_count) {
    TORCH_CHECK(receipt.dim() == 3 && receipt.size(2) == 5 && receipt.scalar_type() == at::kLong
                && receipt.device().is_cpu() && receipt.is_contiguous(), "alias receipt must be contiguous CPU int64 [B,L,5]");
    TORCH_CHECK(task_kind.dim() == 1 && task_kind.scalar_type() == at::kLong && task_kind.is_contiguous()
                && task_kind.device().is_cpu(), "task kinds must be contiguous CPU int64");
    int64_t B = receipt.size(0), L = receipt.size(1);
    TORCH_CHECK(seeds.dim() == 1 && seeds.size(0) == B && seeds.scalar_type() == at::kLong && seeds.is_contiguous()
                && seeds.device().is_cpu(), "one int64 seed per episode");
    Tensor alias_slots = at::zeros({B, L}, receipt.options());
    Tensor synthetic = at::zeros({B}, receipt.options().dtype(at::kBool));
    const int64_t* rec = receipt.data_ptr<int64_t>();
    const int64_t* kinds = task_kind.data_ptr<int64_t>();
    const int64_t* seed = seeds.data_ptr<int64_t>();
    int64_t* slots_out = alias_slots.data_ptr<int64_t>();
    bool* synthetic_out = synthetic.data_ptr<bool>();
    for (int64_t episode = 0; episode < B; episode++) {
        const int64_t* base = rec + episode * L * 5;
        int64_t rows = 0;
        while (rows < L && base[rows * 5 + 4] != 0) rows++;
        std::string name = "episode " + std::to_string(episode);
        TORCH_CHECK(rows > 0, name + " has no valid dataset identity");
        int64_t task = base[0];
        for (int64_t r = 1; r < rows; r++)
            TORCH_CHECK(base[r * 5] == task, name + " must contain exactly one dataset/task identity");
        TORCH_CHECK(task >= 0, name + " has invalid dataset/task identity " + std::to_string(task));
        int64_t kind = task < task_kind.size(0) ? kinds[task] : 0;
        TORCH_CHECK(kind != 0, "task " + std::to_string(task) + " has no recognized explicit name policy");
        int64_t focal_table = 0, focal_column = 0, focal_count = 0;
        for (int64_t r = 0; r < rows; r++)
            if (base[r * 5 + 3] != 0) { focal_table = base[r * 5 + 1]; focal_column = base[r * 5 + 2]; focal_count++; }
        TORCH_CHECK(focal_count == 1, name + " must contain exactly one focal target; found " + std::to_string(focal_count));
        if (kind == 1) continue;
        std::vector<std::pair<int64_t, int64_t>> physical(rows);
        for (int64_t r = 0; r < rows; r++) {
            physical[r] = {base[r * 5 + 1], base[r * 5 + 2]};
            TORCH_CHECK(physical[r].first >= 0 && physical[r].second >= 0, name + " has invalid physical column identity");
        }
        std::vector<std::pair<int64_t, int64_t>> unique(physical);
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        std::vector<std::pair<int64_t, int64_t>> pairs;
        for (auto& p : unique) if (!(p.first == focal_table && p.second == focal_column)) pairs.push_back(p);
        TORCH_CHECK(int64_t(pairs.size()) <= alias_count, name + " needs " + std::to_string(pairs.size())
                    + " opaque aliases, but the bank has " + std::to_string(alias_count));
        std::vector<uint8_t> message;
        put_i64(message, seed[episode]);
        put_i64(message, task);
        put_i64(message, int64_t(pairs.size()));
        for (auto& p : pairs) { put_i64(message, p.first); put_i64(message, p.second); }
        auto digest = sha256(std::move(message));
        uint64_t key = 0;
        for (int i = 7; i >= 0; i--) key = (key << 8) | digest[i];
        key &= (uint64_t(1) << 63) - 1;
        at::Generator generator = at::detail::createCPUGenerator();
        generator.set_current_seed(key);
        Tensor order = at::randperm(alias_count, generator, receipt.options());
        const int64_t* perm = order.data_ptr<int64_t>();
        for (int64_t r = 0; r < rows; r++) {
            auto it = std::lower_bound(pairs.begin(), pairs.end(), physical[r]);
            bool is_focal = !(it != pairs.end() && *it == physical[r]);
            slots_out[episode * L + r] = is_focal ? 0 : perm[it - pairs.begin()] + 1;
        }
        synthetic_out[episode] = true;
    }
    return {alias_slots, synthetic};
}
