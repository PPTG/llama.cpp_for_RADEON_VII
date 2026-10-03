#pragma once

// FlashAttention for token generation with a q8_0 K/V cache, head size 512 and 8 Q heads per KV head (Gemma 4 global
// layers), for AMD GPUs with 64 wide waves (GCN/CDNA). Each wave computes all 8 Q heads for 64 KV rows at a time:
//  - KQ: one lane per K row. Q is quantized to int8 per 32 values (the CPU path does the same for a q8_0 K) and dotted
//    with the q8_0 quants by v_dot4. The K rows are staged in LDS by coalesced loads (2 q8_0 blocks per step).
//  - VKQ: one lane per 8 values of the V row, read directly from memory (coalesced), p of the row broadcast by readlane.
// Every K/V byte is loaded once and never converted/read by several warps as in the tile kernel.
// The partial results of the waves of a block are combined in LDS, the blocks by flash_attn_combine_results_split.

#include "common.cuh"
#include "fattn-common.cuh"

#if defined(GGML_USE_HIP)

static constexpr int fa_q8w_D      = 512; // head size of K and V
static constexpr int fa_q8w_ncols  = 8;   // Q heads per block (GQA ratio or a divisor of it)
static constexpr int fa_q8w_nwaves = 4;   // waves per block
static constexpr int fa_q8w_rows   = 64;  // KV rows per wave and step
static constexpr int fa_q8w_occ    = 2;   // blocks per CU (VGPRs <= 128)
static constexpr int fa_q8w_pair   = 17;  // dwords of 2 q8_0 blocks (68 bytes)

static_assert(sizeof(block_q8_0) == 34, "bad block_q8_0");

// 4-byte aligned loads, merged to dwordx2/dwordx4
static __device__ __forceinline__ uint4 fa_q8w_load4(const char * p) {
    const uint32_t * q = (const uint32_t *) p;
    return make_uint4(q[0], q[1], q[2], q[3]);
}

static __device__ __forceinline__ uint2 fa_q8w_load2(const char * p) {
    const uint32_t * q = (const uint32_t *) p;
    return make_uint2(q[0], q[1]);
}

// value that is the same in all lanes, kept in an SGPR
static __device__ __forceinline__ float fa_q8w_uniform(const float x) {
    return __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(x)));
}

// LDS written by other lanes of the same wave is visible after this
static __device__ __forceinline__ void fa_q8w_wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// 4 int8 (bytes 2*i, 2*i+1 of v for i = 0, 1) -> 2 half2, exact
static __device__ __forceinline__ void fa_q8w_i8x4_to_h2(const uint32_t v, half2 & lo, half2 & hi) {
    const uint32_t x = v ^ 0x80808080u; // q + 128
    const uint32_t a = __builtin_amdgcn_perm(0x64646464u, x, 0x04010400u); // (1024 + u0, 1024 + u1)
    const uint32_t b = __builtin_amdgcn_perm(0x64646464u, x, 0x04030402u); // (1024 + u2, 1024 + u3)
    const half2 bias = make_half2(-1152.0f, -1152.0f);
    lo = __hadd2(*((const half2 *) &a), bias);
    hi = __hadd2(*((const half2 *) &b), bias);
}

template <bool use_logit_softcap>
__launch_bounds__(fa_q8w_nwaves*64, fa_q8w_occ)
static __global__ void flash_attn_q8_wave(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float logit_softcap,
        const int ne01, const int ne02, const int ne11, const int ne12,
        const int nb01, const int nb02, const int64_t nb03,
        const int nb11, const int nb12, const int64_t nb13,
        const int nb21, const int nb22, const int64_t nb23,
        const int nb31, const int64_t nb33, const int ne33) {
    constexpr int D      = fa_q8w_D;
    constexpr int NC     = fa_q8w_ncols;
    constexpr int NW     = fa_q8w_nwaves;
    constexpr int NR     = fa_q8w_rows;
    constexpr int PAIR   = fa_q8w_pair;
    constexpr int NBLK   = D/QK8_0;   // 16 q8_0 blocks per row
    constexpr int NPAIR  = NBLK/2;    // 8 steps of 2 blocks

    static_assert(NR == 64, "one lane per KV row");

    // Q quantized to int8: [block][column][32], its scales [block][column]
    __shared__ int   Qq_s[NBLK*NC*QK8_0/4];
    __shared__ float Qd_s[NBLK*NC];
    // K rows of 2 q8_0 blocks per wave: [row][17 dwords]; reused to combine the waves at the end
    __shared__ int   K_s[NW][NR*PAIR];

    const int lane = threadIdx.x;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.y); // uniform: k0, row counts and addresses in SGPRs
    const int tid  = wave*64 + lane;

    const int j        = blockIdx.x;                 // Q token
    const int ngroups  = ne02/NC;                    // groups of NC Q heads per sequence
    const int sequence = blockIdx.z / ngroups;
    const int head0    = (blockIdx.z - sequence*ngroups)*NC;
    const int gqa      = ne02/ne12;

    const char * K_h = K + nb13*sequence + int64_t(nb12)*(head0/gqa);
    const char * V_h = V + nb23*sequence + int64_t(nb22)*(head0/gqa);
    const half * maskh = mask ? (const half *) (mask + nb33*(sequence % ne33) + int64_t(nb31)*j) : nullptr;

    // quantize Q * scale to int8, 16 values per thread, 2 threads per q8_0 block
    static_assert(NC*D/16 == NW*64, "bad Q quantization");
    {
        const int c    = tid / (D/16);
        const int part = tid % (D/16);
        const float * Qf = (const float *) (Q + nb03*sequence + int64_t(nb02)*(head0 + c) + int64_t(nb01)*j) + 16*part;

        float x[16];
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            const float4 t = *((const float4 *) (Qf + i));
            x[i + 0] = t.x*scale;
            x[i + 1] = t.y*scale;
            x[i + 2] = t.z*scale;
            x[i + 3] = t.w*scale;
        }
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            amax = fmaxf(amax, fabsf(x[i]));
        }
        amax = fmaxf(amax, __shfl_xor(amax, 1, 64));

        const float d  = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f/d : 0.0f;

        int q[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q0 = __float2int_rn(x[4*i + 0]*id);
            const int q1 = __float2int_rn(x[4*i + 1]*id);
            const int q2 = __float2int_rn(x[4*i + 2]*id);
            const int q3 = __float2int_rn(x[4*i + 3]*id);
            q[i] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
        }
        const int b = part/2;
        *((int4 *) &Qq_s[(b*NC + c)*(QK8_0/4) + 4*(part % 2)]) = make_int4(q[0], q[1], q[2], q[3]);
        if (part % 2 == 0) {
            Qd_s[b*NC + c] = __half2float(__float2half(d)); // same scale as a q8_0 block
        }
    }
    __syncthreads();

    // VKQ lane layout: 8 values of the q8_0 block pair pv, taken from 3 dwords of the pair (see the V loop)
    const int pv  = lane / 8;
    const int sub = lane % 8;
    // unsigned offsets: SGPR base + 32 bit VGPR offset addressing
    const uint32_t offAB = 68*pv + 4*(2*sub + 1);      // 2 dwords with the quants
    const uint32_t offC  = 68*pv + (sub < 4 ? 0 : 32); // dword with the scale (and for sub 3 the quants 0, 1)
    const uint32_t mB = sub == 3 ? 0x0000FFFFu : 0xFFFFFFFFu;
    const int shC = sub < 4 ? 0 : 16;

    float KQ_max[NC];
    float KQ_sum[NC];
    half2 VKQ[NC][4];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        KQ_max[c] = -FLT_MAX/2.0f;
        KQ_sum[c] = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            VKQ[c][k] = make_half2(0.0f, 0.0f);
        }
    }

    int * Ks = K_s[wave];

    // each wave computes a contiguous range of rows (same number for all waves), in steps of up to 64 rows
    const int nwaves_all = gridDim.y*NW;
    const int rows_wave  = (ne11 + nwaves_all - 1) / nwaves_all;
    const int k_begin    = (blockIdx.y*NW + wave)*rows_wave;
    const int k_end      = min(ne11, k_begin + rows_wave);

    for (int k0 = k_begin; k0 < k_end; k0 += NR) {
        const int nrows = min(NR, k_end - k0); // rows >= nrows use the data of the last row and get p = 0

        // KQ: lane = K row. Staging loads: 4 dwordx4 (16 rows, 4 lanes per row) + 1 dword (lane = row) per pair
        const char * K_c = K_h + int64_t(k0)*nb11;
        int offK[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            offK[i] = min(16*i + lane/4, nrows - 1)*nb11 + 16*(lane % 4);
        }
        const int offK16 = min(lane, nrows - 1)*nb11 + 64;

        uint4    st[4];
        uint32_t st16;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            st[i] = fa_q8w_load4(K_c + offK[i]);
        }
        st16 = *((const uint32_t *) (K_c + offK16));

        float KQ[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            KQ[c] = 0.0f;
        }

#pragma unroll 1
        for (int p = 0; p < NPAIR; ++p) {
            fa_q8w_wave_sync(); // the previous pair was read by all lanes
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int * dstw = Ks + (16*i + lane/4)*PAIR + 4*(lane % 4);
                dstw[0] = st[i].x;
                dstw[1] = st[i].y;
                dstw[2] = st[i].z;
                dstw[3] = st[i].w;
            }
            Ks[lane*PAIR + 16] = st16;
            fa_q8w_wave_sync();

            if (p + 1 < NPAIR) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    st[i] = fa_q8w_load4(K_c + offK[i] + 68*(p + 1));
                }
                st16 = *((const uint32_t *) (K_c + offK16 + 68*(p + 1)));
            }

            uint32_t w[PAIR];
#pragma unroll
            for (int k = 0; k < PAIR; ++k) {
                w[k] = Ks[lane*PAIR + k];
            }

            // block 2p: scale in bytes 0-1, quants in bytes 2-33; block 2p+1: scale in bytes 34-35, quants 36-67
            int kq[2][8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                kq[0][k] = __builtin_amdgcn_alignbyte(w[k + 1], w[k], 2);
                kq[1][k] = w[9 + k];
            }
            const uint16_t d0 = w[0] & 0xFFFF;
            const uint16_t d1 = w[8] >> 16;
            const float dk[2] = {__half2float(*((const half *) &d0)), __half2float(*((const half *) &d1))};

#pragma unroll
            for (int bb = 0; bb < 2; ++bb) {
                const int b = 2*p + bb;
                const float4 qd0 = *((const float4 *) &Qd_s[b*NC + 0]);
                const float4 qd1 = *((const float4 *) &Qd_s[b*NC + 4]);
                const float qd[NC] = {qd0.x, qd0.y, qd0.z, qd0.w, qd1.x, qd1.y, qd1.z, qd1.w};
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const int4 qa = *((const int4 *) &Qq_s[(b*NC + c)*(QK8_0/4) + 0]);
                    const int4 qb = *((const int4 *) &Qq_s[(b*NC + c)*(QK8_0/4) + 4]);
                    int sumi = 0;
                    sumi = ggml_cuda_dp4a(kq[bb][0], qa.x, sumi);
                    sumi = ggml_cuda_dp4a(kq[bb][1], qa.y, sumi);
                    sumi = ggml_cuda_dp4a(kq[bb][2], qa.z, sumi);
                    sumi = ggml_cuda_dp4a(kq[bb][3], qa.w, sumi);
                    sumi = ggml_cuda_dp4a(kq[bb][4], qb.x, sumi);
                    sumi = ggml_cuda_dp4a(kq[bb][5], qb.y, sumi);
                    sumi = ggml_cuda_dp4a(kq[bb][6], qb.z, sumi);
                    sumi = ggml_cuda_dp4a(kq[bb][7], qb.w, sumi);
                    KQ[c] += float(sumi) * (dk[bb]*qd[c]);
                }
            }
        }

        // softcap, mask, online softmax. p of the rows as half2 (p, p) in LDS [row][column], read as a broadcast
        const bool  row_ok = lane < nrows;
        const float mk     = row_ok && maskh ? __half2float(maskh[k0 + lane]) : 0.0f;
        fa_q8w_wave_sync(); // all lanes have read the K rows
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float x = KQ[c];
            if constexpr (use_logit_softcap) {
                x = logit_softcap*tanhf(x);
            }
            x = row_ok ? x + mk : -INFINITY;

            const float max_new   = fa_q8w_uniform(fmaxf(KQ_max[c], warp_reduce_max<64>(x) + FATTN_KQ_MAX_OFFSET));
            const float max_scale = fa_q8w_uniform(expf(KQ_max[c] - max_new));
            KQ_max[c] = max_new;

            const float pc = expf(x - max_new);
            KQ_sum[c] = fa_q8w_uniform(KQ_sum[c]*max_scale + warp_reduce_sum<64>(pc));

            const half2 ms2 = make_half2(max_scale, max_scale);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                VKQ[c][k] *= ms2;
            }
            ((half2 *) Ks)[lane*NC + c] = make_half2(pc, pc);
        }
        fa_q8w_wave_sync();

        // VKQ: lane = 8 values of the V row. Rows in groups of 4, the next group is loaded while computing the current
        const char * V_c = V_h + int64_t(k0)*nb21;
        uint2    nab[4];
        uint32_t nc[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const char * row = V_c + int64_t(min(i, nrows - 1))*nb21;
            nab[i] = fa_q8w_load2(row + offAB);
            nc[i]  = *((const uint32_t *) (row + offC));
        }
        for (int r0 = 0; r0 < nrows; r0 += 4) {
            uint2    ab[4];
            uint32_t cc[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ab[i] = nab[i];
                cc[i] = nc[i];
            }
            if (r0 + 4 < nrows) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const char * row = V_c + int64_t(min(r0 + 4 + i, nrows - 1))*nb21;
                    nab[i] = fa_q8w_load2(row + offAB);
                    nc[i]  = *((const uint32_t *) (row + offC));
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // quants: dword A, dword B (sub 3: B bytes 0-1 and C bytes 2-3), scale: C bytes 0-1 or 2-3
                const uint32_t q2 = (ab[i].y & mB) | (cc[i] & ~mB);
                const uint16_t dv = (cc[i] >> shC) & 0xFFFF;
                const half2 d2 = __half2half2(*((const half *) &dv));

                half2 v[4];
                fa_q8w_i8x4_to_h2(ab[i].x, v[0], v[1]);
                fa_q8w_i8x4_to_h2(q2,      v[2], v[3]);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    v[k] = __hmul2(v[k], d2); // half(q*d) as in the conversion to f16
                }
                const uint4 p0 = ((const uint4 *) Ks)[2*(r0 + i) + 0];
                const uint4 p1 = ((const uint4 *) Ks)[2*(r0 + i) + 1];
                const uint32_t pr[NC] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const half2 p2 = *((const half2 *) &pr[c]);
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        VKQ[c][k] = __hfma2(v[k], p2, VKQ[c][k]);
                    }
                }
            }
        }
    }

    // combine the waves of the block: waves [s, 2s) hand over to waves [0, s)
    half2 * cbuf = (half2 *) &K_s[0][0];
    float * cmeta = (float *) (cbuf + (NW/2)*NC*4*64);
    static_assert(((NW/2)*NC*4*64*sizeof(half2) + (NW/2)*2*NC*sizeof(float)) <= sizeof(K_s), "combine buffer too small");
#pragma unroll
    for (int s = NW/2; s >= 1; s /= 2) {
        __syncthreads();
        if (wave >= s && wave < 2*s) {
            const int wb = wave - s;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    cbuf[((wb*NC + c)*4 + k)*64 + lane] = VKQ[c][k];
                }
                if (lane == 0) {
                    cmeta[(wb*2 + 0)*NC + c] = KQ_max[c];
                    cmeta[(wb*2 + 1)*NC + c] = KQ_sum[c];
                }
            }
        }
        __syncthreads();
        if (wave < s) {
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const float max_b = cmeta[(wave*2 + 0)*NC + c];
                const float sum_b = cmeta[(wave*2 + 1)*NC + c];
                const float max_n = fmaxf(KQ_max[c], max_b);
                const float sa    = expf(KQ_max[c] - max_n);
                const float sb    = expf(max_b     - max_n);
                KQ_max[c] = max_n;
                KQ_sum[c] = KQ_sum[c]*sa + sum_b*sb;
                const half2 sa2 = make_half2(sa, sa);
                const half2 sb2 = make_half2(sb, sb);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    VKQ[c][k] = __hfma2(VKQ[c][k], sa2, cbuf[((wave*NC + c)*4 + k)*64 + lane]*sb2);
                }
            }
        }
    }

    if (wave != 0) {
        return;
    }

    // attention sinks, only in the first of the parallel blocks
    if (sinks && blockIdx.y == 0) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float sink  = ((const float *) sinks)[head0 + c];
            const float max_n = fmaxf(KQ_max[c], sink);
            const float ms    = expf(KQ_max[c] - max_n);
            KQ_max[c] = max_n;
            KQ_sum[c] = KQ_sum[c]*ms + expf(sink - max_n);
            const half2 ms2 = make_half2(ms, ms);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                VKQ[c][k] *= ms2;
            }
        }
    }

    // dims of the 8 values of this lane (see the V loop)
    const int blk = 2*pv + (sub >= 4);
    int dims[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        dims[e] = QK8_0*blk + (sub < 3 ? 2 + 8*sub + e : (sub == 3 ? (e < 6 ? 26 + e : e - 6) : 8*(sub - 4) + e));
    }

#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const int j_dst_unrolled = ((sequence*ne01 + j)*ne02 + head0 + c)*gridDim.y + blockIdx.y;
        const float s = gridDim.y == 1 ? 1.0f/KQ_sum[c] : 1.0f;
        float * dst_c = dst + int64_t(j_dst_unrolled)*D;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 t = __half22float2(VKQ[c][k]);
            dst_c[dims[2*k + 0]] = t.x*s;
            dst_c[dims[2*k + 1]] = t.y*s;
        }
        if (gridDim.y != 1 && lane == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[c], KQ_sum[c]);
        }
    }
}

static inline bool ggml_cuda_fattn_q8_wave_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_FA_Q8_WAVE");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

// KV rows per wave the context is split into (1..64): fewer rows = more blocks for short contexts
static inline int ggml_cuda_fattn_q8_wave_rows() {
    static const int rows = [] {
        const char * env = getenv("GGML_CUDA_FA_Q8_WAVE_ROWS");
        return env ? std::max(1, std::min(fa_q8w_rows, atoi(env))) : fa_q8w_rows;
    }();
    return rows;
}

// true if flash_attn_q8_wave can compute dst (token generation, q8_0 K/V, head size 512, GQA multiple of 8, 64 wide waves)
static inline bool ggml_cuda_fattn_q8_wave_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    if (!ggml_cuda_fattn_q8_wave_enabled()) {
        return false;
    }
    if (ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != 64) {
        return false;
    }
    if (K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q8_0 || K->ne[0] != fa_q8w_D || V->ne[0] != fa_q8w_D) {
        return false;
    }
    if (Q->ne[1] != 1 || Q->ne[2] % K->ne[2] != 0 || (Q->ne[2]/K->ne[2]) % fa_q8w_ncols != 0 || K->ne[3] != Q->ne[3]) {
        return false;
    }
    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    if (max_bias != 0.0f) {
        return false;
    }
    // dword loads of K/V, float4 loads of Q
    const bool kv_ok = (uintptr_t) K->data % 4 == 0 && (uintptr_t) V->data % 4 == 0 &&
        K->nb[1] % 4 == 0 && K->nb[2] % 4 == 0 && K->nb[3] % 4 == 0 &&
        V->nb[1] % 4 == 0 && V->nb[2] % 4 == 0 && V->nb[3] % 4 == 0;
    const bool q_ok = (uintptr_t) Q->data % 16 == 0 && Q->nb[1] % 16 == 0 && Q->nb[2] % 16 == 0 && Q->nb[3] % 16 == 0;
    // 32 bit offsets inside a step of 64 rows and for the Q/mask strides
    const bool size_ok = int64_t(K->nb[1])*fa_q8w_rows < INT32_MAX && int64_t(V->nb[1])*fa_q8w_rows < INT32_MAX &&
        K->nb[2] < INT32_MAX && V->nb[2] < INT32_MAX && Q->nb[1] < INT32_MAX && Q->nb[2] < INT32_MAX;
    return kv_ok && q_ok && size_ok;
}

static inline void ggml_cuda_flash_attn_ext_q8_wave(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_ASSERT(Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    float scale         = 1.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    cudaStream_t stream = ctx.stream();

    auto kernel = logit_softcap == 0.0f ? flash_attn_q8_wave<false> : flash_attn_q8_wave<true>;

    int nblocks_sm = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks_sm, kernel, fa_q8w_nwaves*64, 0));
    nblocks_sm = std::max(nblocks_sm, 1);

    const int ntiles  = Q->ne[1] * (Q->ne[2]/fa_q8w_ncols) * Q->ne[3];
    const int rows    = fa_q8w_nwaves*ggml_cuda_fattn_q8_wave_rows();
    const int nchunks = (K->ne[1] + rows - 1) / rows;
    const int nparts  = std::max(1, std::min(nchunks, nblocks_sm*nsm / ntiles));

    ggml_cuda_pool_alloc<float>  dst_tmp(ctx.pool());
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(ctx.pool());
    if (nparts > 1) {
        dst_tmp.alloc(size_t(nparts)*ggml_nelements(dst));
        dst_tmp_meta.alloc(size_t(nparts)*ggml_nrows(dst));
    }

    const dim3 block_dim(64, fa_q8w_nwaves, 1);
    const dim3 blocks_num(Q->ne[1], nparts, (Q->ne[2]/fa_q8w_ncols)*Q->ne[3]);
    kernel<<<blocks_num, block_dim, 0, stream>>>(
        (const char *) Q->data, (const char *) K->data, (const char *) V->data,
        mask  ? (const char *) mask->data  : nullptr,
        sinks ? (const char *) sinks->data : nullptr,
        nparts > 1 ? dst_tmp.ptr : (float *) dst->data, dst_tmp_meta.ptr,
        scale, logit_softcap,
        Q->ne[1], Q->ne[2], K->ne[1], K->ne[2],
        Q->nb[1], Q->nb[2], Q->nb[3],
        K->nb[1], K->nb[2], K->nb[3],
        V->nb[1], V->nb[2], V->nb[3],
        mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1);
    CUDA_CHECK(cudaGetLastError());

    if (nparts > 1) {
        constexpr int DV = fa_q8w_D;
        const dim3 block_dim_combine(64, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1]*(DV/64), Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = nparts*(sizeof(float2) + sizeof(float));
        flash_attn_combine_results_split<DV><<<blocks_num_combine, block_dim_combine, nbytes_shared_combine, stream>>>(
            dst_tmp.ptr, dst_tmp_meta.ptr, (float *) dst->data, nparts);
        CUDA_CHECK(cudaGetLastError());
    }
}

#endif // defined(GGML_USE_HIP)
