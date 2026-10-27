// src/kernels/cuda/q8_sm60.cuh - a Q8_0 decode GEMV for Pascal GP100 (compute capability 6.0), opt-in with
// STRATA_Q8_SM60=1.  Included by native_mmvq.cu, and by a standalone parity/speed check, so both run this code.
//
// Reads the packed planes of STRATA_Q8_PACKED (the SAME int8 values row-major, n_in bytes per row, then one fp16 scale
// per 32 of them) for every Q8_0 dense matrix and the head, whatever their shape.  Why the packed kernel above is slow
// on a GP100 die (measured on PH402, 1050 MHz: ~137 GB/s for the dense projections, ~160 for the head, against
// 220-270 for this layout): one workgroup per row, eight bytes per thread per pass and a __syncthreads() per row - a
// 2560-wide row is 80 blocks, so the reduction costs more than the row - and the window's activations read as single
// ints, because a q8_1 block is 36 bytes and its values start at byte 4.
//
// Here:
//   * a workgroup first copies the window's activations into shared memory as an int8 plane per column and a float
//     scale per 32 (once; the workgroup then loops over rows), so every activation read is a 16-byte shared load;
//   * TPR threads own a row (a whole warp for n_in >= 2048, a half or a quarter for narrower rows) and read its
//     weights 16 bytes at a time through the read-only cache; the dot of 4 ints is STRATA_DP4A (vmad on sm_60);
//   * the sum is a shuffle within those TPR lanes - no shared memory, no barrier per row.
// The per-32 products are added in another order than the exact kernels', so outputs are not bitwise theirs (float
// rounding only; the int8 dots are exact).
#pragma once

namespace q8sm60 {

constexpr int THREADS = 256;
constexpr std::size_t MAX_SMEM = 48 * 1024;   // a Pascal workgroup's static + dynamic limit

inline std::size_t smem_bytes(int ncols, int n_in) {
    return std::size_t(ncols) * std::size_t(n_in) + std::size_t(ncols) * std::size_t(n_in / 32) * sizeof(float);
}

template<int NCOLS, int TPR>
__launch_bounds__(THREADS)
__global__ void kernel(const int8_t* __restrict__ qs, const half* __restrict__ dpl, const Q81Block* __restrict__ x,
                       float* __restrict__ y, int n_in, int n_out) {
    extern __shared__ int4 q8sm60_smem[];
    int8_t* sx = reinterpret_cast<int8_t*>(q8sm60_smem);                         // [NCOLS][n_in]
    float* sd = reinterpret_cast<float*>(sx + std::size_t(NCOLS) * n_in);       // [NCOLS][n_in / 32]
    const int nb = n_in / 32;
    for (int b = int(threadIdx.x); b < NCOLS * nb; b += THREADS) {               // column j = b / nb, block b % nb
        const Q81Block* xb = x + b;
        const int* src = reinterpret_cast<const int*>(xb->qs);
        int* dst = reinterpret_cast<int*>(sx + std::size_t(b) * 32);
#pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = src[i];
        sd[b] = __low2float(xb->ds);
    }
    __syncthreads();

    constexpr int RPB = THREADS / TPR;
    const int lane = int(threadIdx.x) % TPR;
    const int rloc = int(threadIdx.x) / TPR;
    const int nch = n_in / 16;                                                   // 16-byte chunks, two per block
    const int groups = (n_out + RPB - 1) / RPB;
    for (int g = int(blockIdx.x); g < groups; g += int(gridDim.x)) {
        const int row = g * RPB + rloc;
        float acc[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) acc[j] = 0.0f;
        if (row < n_out) {
            const int4* w = reinterpret_cast<const int4*>(qs + std::size_t(row) * n_in);
            const half* d = dpl + std::size_t(row) * nb;
            for (int c = lane; c < nch; c += TPR) {
                const int4 wv = __ldg(w + c);
                const float dw = __half2float(d[c >> 1]);
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) {
                    const int4 xv = reinterpret_cast<const int4*>(sx + std::size_t(j) * n_in)[c];
                    int s = STRATA_DP4A(wv.x, xv.x, 0);
                    s = STRATA_DP4A(wv.y, xv.y, s);
                    s = STRATA_DP4A(wv.z, xv.z, s);
                    s = STRATA_DP4A(wv.w, xv.w, s);
                    acc[j] += dw * sd[j * nb + (c >> 1)] * float(s);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {                                       // every lane takes part, row or not
            float v = acc[j];
#pragma unroll
            for (int o = TPR / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, TPR);
            if (lane == 0 && row < n_out) y[std::size_t(j) * n_out + row] = v;
        }
    }
}

// One workgroup per SM would leave too few rows in flight on a 48-SM die; four per SM is what the shapes of
// Qwen3.8-Flash-Next measured best (STRATA_Q8_SM60_BPS overrides it).
inline int blocks_per_sm() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_Q8_SM60_BPS");
        const int n = e ? std::atoi(e) : 0;
        return n > 0 ? n : 4;
    }();
    return v;
}

inline int sm_count() {
    int dev = 0, n = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
        return 48;
    return n > 0 ? n : 48;
}

template<int NCOLS, int TPR>
void launch_one(const int8_t* qs, const half* d, const Q81Block* x, float* y, int n_in, int n_out, cudaStream_t s) {
    constexpr int RPB = THREADS / TPR;
    const int groups = (n_out + RPB - 1) / RPB;
    const int grid = (std::min)(groups, sm_count() * blocks_per_sm());
    kernel<NCOLS, TPR><<<unsigned(grid), THREADS, smem_bytes(NCOLS, n_in), s>>>(qs, d, x, y, n_in, n_out);
}

template<int NCOLS>
void launch_cols(const int8_t* qs, const half* d, const Q81Block* x, float* y, int n_in, int n_out, cudaStream_t s) {
    if (n_in >= 2048) launch_one<NCOLS, 32>(qs, d, x, y, n_in, n_out, s);
    else if (n_in >= 1024) launch_one<NCOLS, 16>(qs, d, x, y, n_in, n_out, s);
    else launch_one<NCOLS, 8>(qs, d, x, y, n_in, n_out, s);
}

// false: a shape this kernel does not take (the caller runs its usual kernel)
inline bool launch(const int8_t* qs, const half* d, const Q81Block* x, float* y, int n_in, int n_out, int ncols,
                   cudaStream_t s) {
    if (n_in % 32 != 0 || n_in < 128 || ncols < 1 || ncols > 8 || smem_bytes(ncols, n_in) > MAX_SMEM) return false;
    switch (ncols) {
        case 1: launch_cols<1>(qs, d, x, y, n_in, n_out, s); break;
        case 2: launch_cols<2>(qs, d, x, y, n_in, n_out, s); break;
        case 3: launch_cols<3>(qs, d, x, y, n_in, n_out, s); break;
        case 4: launch_cols<4>(qs, d, x, y, n_in, n_out, s); break;
        case 5: launch_cols<5>(qs, d, x, y, n_in, n_out, s); break;
        case 6: launch_cols<6>(qs, d, x, y, n_in, n_out, s); break;
        case 7: launch_cols<7>(qs, d, x, y, n_in, n_out, s); break;
        default: launch_cols<8>(qs, d, x, y, n_in, n_out, s); break;
    }
    return true;
}

// ---- STRATA_Q8_SM60_H2=1 (with STRATA_Q8_SM60=1): the same planes, the dot in fp16 (HFMA2) instead of vmad ----
//
// GP100 runs HFMA2 at full rate and has no __dp4a: the vmad dot above costs 16 vmads per 16 weights per column, and a
// verify window of 4-5 columns makes the kernel issue-bound (measured in the engine: ~156 GB/s on the dense Q8_0).
// Here both sides become fp16 with no rounding: an int8 weight q is (0x6400 | (q ^ 0x80)) - 1152, an int8 activation
// a staged in shared memory as (0x4400 | (a ^ 0x80)) - 4.5 = a / 256 (exact; the 2^-8 keeps a product <= 63 so 16 of
// them per half lane stay far below the fp16 limit).  Each lane takes a whole 32-weight block: one weight conversion
// shared by every column, 16 HFMA2 per column, then one flush to fp32 with the two scales.  Only the products and that
// 16-term fp16 sum round (~2^-11 relative), so outputs are close to, not equal to, the exact kernels'.
// A row too wide for its columns' fp16 plane (ssm_out / attn_output, 6144 wide, at 4+ columns) is split in two K
// halves (blockIdx.y) that atomicAdd into a zeroed y: two addends onto 0.0f, so the result does not depend on order.

inline std::size_t smem_bytes_h2(int ncols, int k_len) {
    return std::size_t(ncols) * std::size_t(k_len) * sizeof(half) + std::size_t(ncols) * std::size_t(k_len / 32) * sizeof(float);
}

__device__ __forceinline__ half2 h2_bits(unsigned u) { return *reinterpret_cast<half2*>(&u); }

template<int NCOLS, int TPR>
__launch_bounds__(THREADS)
__global__ void kernel_h2(const int8_t* __restrict__ qs, const half* __restrict__ dpl, const Q81Block* __restrict__ x,
                          float* __restrict__ y, int n_in, int n_out, int k_len) {
    extern __shared__ int4 q8sm60_smem[];
    half2* sx = reinterpret_cast<half2*>(q8sm60_smem);   // [NCOLS][4][k_len / 32] 16-byte chunks, chunk-major (no bank conflicts)
    float* sd = reinterpret_cast<float*>(sx + std::size_t(NCOLS) * (k_len / 2)); // [NCOLS][k_len / 32]
    const int nb = n_in / 32;
    const int kb0 = int(blockIdx.y) * (k_len / 32);                             // first block of this K tile
    const int nbt = k_len / 32;
    const half2 a_off = __floats2half2_rn(4.5f, 4.5f);
    for (int b = int(threadIdx.x); b < NCOLS * nbt; b += THREADS) {
        const int j = b / nbt, bb = b % nbt;
        const Q81Block* xb = x + std::size_t(j) * nb + kb0 + bb;
        const int* src = reinterpret_cast<const int*>(xb->qs);
        int4* dst = reinterpret_cast<int4*>(sx + std::size_t(j) * (k_len / 2));   // chunk q of block bb at q * nbt + bb
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            half2 h[4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const unsigned u = unsigned(src[2 * q + i]) ^ 0x80808080u;
                h[2 * i] = __hsub2(h2_bits(__byte_perm(u, 0x44444444u, 0x4140)), a_off);
                h[2 * i + 1] = __hsub2(h2_bits(__byte_perm(u, 0x44444444u, 0x4342)), a_off);
            }
            dst[q * nbt + bb] = *reinterpret_cast<const int4*>(h);
        }
        sd[j * nbt + bb] = __low2float(xb->ds) * 256.0f;
    }
    __syncthreads();

    constexpr int RPB = THREADS / TPR;
    const int lane = int(threadIdx.x) % TPR;
    const int rloc = int(threadIdx.x) / TPR;
    const int groups = (n_out + RPB - 1) / RPB;
    const half2 w_off = __floats2half2_rn(1152.0f, 1152.0f);
    for (int g = int(blockIdx.x); g < groups; g += int(gridDim.x)) {
        const int row = g * RPB + rloc;
        float acc[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) acc[j] = 0.0f;
        if (row < n_out) {
            const int4* w = reinterpret_cast<const int4*>(qs + std::size_t(row) * n_in) + 2 * kb0;
            const half* d = dpl + std::size_t(row) * nb + kb0;
            for (int bb = lane; bb < nbt; bb += TPR) {
                const int4 w0 = __ldg(w + 2 * bb), w1 = __ldg(w + 2 * bb + 1);
                const float dw = __half2float(d[bb]);
                const unsigned wi[8] = {unsigned(w0.x), unsigned(w0.y), unsigned(w0.z), unsigned(w0.w),
                                        unsigned(w1.x), unsigned(w1.y), unsigned(w1.z), unsigned(w1.w)};
                half2 wq[16];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const unsigned u = wi[i] ^ 0x80808080u;
                    wq[2 * i] = __hsub2(h2_bits(__byte_perm(u, 0x64646464u, 0x4140)), w_off);
                    wq[2 * i + 1] = __hsub2(h2_bits(__byte_perm(u, 0x64646464u, 0x4342)), w_off);
                }
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) {
                    const int4* xv = reinterpret_cast<const int4*>(sx + std::size_t(j) * (k_len / 2)) + bb;
                    half2 s = __float2half2_rn(0.0f);
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int4 v = xv[q * nbt];
                        s = __hfma2(wq[4 * q + 0], h2_bits(unsigned(v.x)), s);
                        s = __hfma2(wq[4 * q + 1], h2_bits(unsigned(v.y)), s);
                        s = __hfma2(wq[4 * q + 2], h2_bits(unsigned(v.z)), s);
                        s = __hfma2(wq[4 * q + 3], h2_bits(unsigned(v.w)), s);
                    }
                    acc[j] += dw * sd[j * nbt + bb] * (__low2float(s) + __high2float(s));
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            float v = acc[j];
#pragma unroll
            for (int o = TPR / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, TPR);
            if (lane == 0 && row < n_out) {
                if (gridDim.y == 1) y[std::size_t(j) * n_out + row] = v;
                else atomicAdd(y + std::size_t(j) * n_out + row, v);
            }
        }
    }
}

template<int NCOLS, int TPR>
void launch_one_h2(const int8_t* qs, const half* d, const Q81Block* x, float* y, int n_in, int n_out, int ktiles,
                   cudaStream_t s) {
    constexpr int RPB = THREADS / TPR;
    const int groups = (n_out + RPB - 1) / RPB;
    const int grid = (std::min)(groups, sm_count() * blocks_per_sm());
    const int k_len = n_in / ktiles;
    if (ktiles > 1) cudaMemsetAsync(y, 0, std::size_t(NCOLS) * n_out * sizeof(float), s);
    kernel_h2<NCOLS, TPR><<<dim3(unsigned(grid), unsigned(ktiles)), THREADS, smem_bytes_h2(NCOLS, k_len), s>>>(
        qs, d, x, y, n_in, n_out, k_len);
}

template<int NCOLS>
void launch_cols_h2(const int8_t* qs, const half* d, const Q81Block* x, float* y, int n_in, int n_out, int ktiles,
                    cudaStream_t s) {
    const int nbt = n_in / ktiles / 32;   // blocks per lane-group: a TPR that divides it keeps every lane busy
    if (nbt % 32 == 0 || nbt >= 192) launch_one_h2<NCOLS, 32>(qs, d, x, y, n_in, n_out, ktiles, s);
    else if (nbt % 16 == 0 || nbt >= 48) launch_one_h2<NCOLS, 16>(qs, d, x, y, n_in, n_out, ktiles, s);
    else if (nbt % 8 == 0 || nbt >= 24) launch_one_h2<NCOLS, 8>(qs, d, x, y, n_in, n_out, ktiles, s);
    else launch_one_h2<NCOLS, 4>(qs, d, x, y, n_in, n_out, ktiles, s);
}

inline bool h2_on() {
    static const bool v = [] {
        const char* e = std::getenv("STRATA_Q8_SM60_H2");
        const bool yes = e && e[0] == '1' && e[1] == '\0';
        if (yes) std::fprintf(stderr, "strata q8 sm60: the Q8_0 dot runs in fp16 (STRATA_Q8_SM60_H2=1)\n");
        return yes;
    }();
    return v;
}

// false: a shape this kernel does not take (the caller runs the vmad kernel)
inline bool launch_h2(const int8_t* qs, const half* d, const Q81Block* x, float* y, int n_in, int n_out, int ncols,
                      cudaStream_t s) {
    if (n_in % 64 != 0 || n_in < 128 || ncols < 1 || ncols > 8) return false;
    // Only where it measured faster than the vmad kernel (tools/q8_sm60_h2_check.cu on a P100): 4+ columns; 3 columns on
    // a 2560-wide matrix with many rows; a 6144-wide row (ssm_out / attn_output) only from 5 columns, where its fp16
    // plane no longer fits one tile anyway and the K split pays for itself.
    if (ncols < 3 || (ncols == 3 && (n_in >= 4096 || n_out < 2048)) || (n_in >= 4096 && ncols < 5)) return false;
    const int ktiles = smem_bytes_h2(ncols, n_in) > MAX_SMEM ? 2 : 1;
    if (smem_bytes_h2(ncols, n_in / ktiles) > MAX_SMEM) return false;
    switch (ncols) {
        case 1: launch_cols_h2<1>(qs, d, x, y, n_in, n_out, ktiles, s); break;
        case 2: launch_cols_h2<2>(qs, d, x, y, n_in, n_out, ktiles, s); break;
        case 3: launch_cols_h2<3>(qs, d, x, y, n_in, n_out, ktiles, s); break;
        case 4: launch_cols_h2<4>(qs, d, x, y, n_in, n_out, ktiles, s); break;
        case 5: launch_cols_h2<5>(qs, d, x, y, n_in, n_out, ktiles, s); break;
        case 6: launch_cols_h2<6>(qs, d, x, y, n_in, n_out, ktiles, s); break;
        case 7: launch_cols_h2<7>(qs, d, x, y, n_in, n_out, ktiles, s); break;
        default: launch_cols_h2<8>(qs, d, x, y, n_in, n_out, ktiles, s); break;
    }
    return true;
}

// GGUF Q8_0 blocks (34 bytes, 2-byte aligned) -> the packed planes, on the device (the head's 0.6 GiB stays off the host)
__global__ void pack_kernel(const Q80Block* __restrict__ blocks, int8_t* __restrict__ qs, half* __restrict__ d,
                            std::size_t count) {
    const std::size_t b = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= count) return;
    const Q80Block& src = blocks[b];
    const uint16_t* in = reinterpret_cast<const uint16_t*>(src.qs);
    uint16_t* out = reinterpret_cast<uint16_t*>(qs + b * 32);
#pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = in[i];
    d[b] = src.d;
}

} // namespace q8sm60
