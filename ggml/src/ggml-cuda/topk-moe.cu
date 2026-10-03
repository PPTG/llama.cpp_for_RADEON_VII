#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "topk-moe.cuh"

#include <cmath>
#include <initializer_list>

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
    bool use_sigmoid;
    bool use_sqrt_softplus;
    bool with_norm;
    bool delayed_softmax;
};

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sqrt_softplus_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? sqrtf(vals[i] > 20.0f ? vals[i] : logf(1.0f + expf(vals[i]))) : -INFINITY;
    }
}

/*
    This kernel does the following:
    1. optionally softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
template <int n_experts, bool has_bias>
__launch_bounds__(TOPK_MOE_ROWS_PER_BLOCK * WARP_SIZE, 1)
__global__ void topk_moe_cuda(const float *         logits,
                              float *               weights,
                              int32_t *             ids,
                              float *               bias,
                              const int             n_rows,
                              const int             n_expert_used,
                              const float           clamp_val,
                              const float           scale_val,
                              const topk_moe_config config) {
#if defined(GGML_USE_MUSA)
    // MUSA: every warp of a partially filled block must reach the barrier below.
    const int row = MIN(blockIdx.x * blockDim.y + threadIdx.y, n_rows - 1);
#else
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
#endif // defined(GGML_USE_MUSA)
    if (row >= n_rows) {
        return;
    }

    logits += n_experts * row;
    weights += n_expert_used * row;
    ids += n_experts * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

    // Initialize all slots to -INFINITY
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    // Weights and IDs can alias logits, so wait until every row in the block reads its logits.
    __syncthreads();

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
           sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else if (config.use_sqrt_softplus) {
           sqrt_softplus_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else {
           softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
    // NaN comparisons always return false, which would cause the same expert to be
    // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
    // -INFINITY sentinel used after each selection round.
    // More relevant for the cuBLAS path. See https://github.com/ggml-org/llama.cpp/issues/19659
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    [[maybe_unused]] float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert = i + threadIdx.x;
            selection_wt[i / WARP_SIZE] =
                (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + bias[expert] : -INFINITY;
        }
    }

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ggml_cuda_pdl_lc();
    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const float val_s  = __shfl_xor_sync(0xFFFFFFFF, max_val_s, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val    = val;
                    max_val_s  = val_s;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                selection_wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        } else {
#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        }

        if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            ids[k] = max_expert;
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        wt_sum              = warp_reduce_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

// Softmax gating without bias, one row per block of n_threads: warp 0 computes the softmax exactly as topk_moe_cuda,
// then every thread ranks its experts against all others (value desc, index asc) in shared memory instead of
// n_expert_used serial argmax rounds. The weight sum is accumulated in the same lane and order, so the result is
// bit-identical to topk_moe_cuda. Called by all threads of the block; logits may be in shared memory.
template <int n_experts, int n_threads>
static __device__ __forceinline__ void topk_moe_rank_block(const float * logits,
                                                           float *       weights,
                                                           int32_t *     ids,
                                                           const int     n_expert_used,
                                                           const float   clamp_val,
                                                           const float   scale_val,
                                                           const bool    with_norm) {
    static_assert(n_experts % WARP_SIZE == 0 && n_experts % n_threads == 0, "bad topk_moe_rank config");
    constexpr int experts_per_thread = n_experts / WARP_SIZE;

    __shared__ __align__(16) float s_wt[n_experts]; // read as float4
    __shared__ float s_sel[n_experts];
    __shared__ int   s_selid[n_experts];

    const int tid = threadIdx.x;

    if (tid < WARP_SIZE) {
        float wt[experts_per_thread];
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            wt[i] = logits[i * WARP_SIZE + tid];
        }
        softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, tid);
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            s_wt[i * WARP_SIZE + tid] = __isnanf(wt[i]) ? -FLT_MAX : wt[i];
        }
    }
    // also orders the logits reads before the writes: weights and ids can alias logits
    __syncthreads();
    ggml_cuda_pdl_lc();

#pragma unroll
    for (int e = tid; e < n_experts; e += n_threads) {
        const float v    = s_wt[e];
        int         rank = 0;
#pragma unroll 8
        for (int j = 0; j < n_experts; j += 4) {
            const float4 w = *(const float4 *) &s_wt[j];
            rank += (w.x > v || (w.x == v && j + 0 < e));
            rank += (w.y > v || (w.y == v && j + 1 < e));
            rank += (w.z > v || (w.z == v && j + 2 < e));
            rank += (w.w > v || (w.w == v && j + 3 < e));
        }
        if (rank < n_expert_used) {
            ids[rank]     = e;
            s_sel[rank]   = v;
            s_selid[rank] = e;
        }
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        float inv_sum = 1.0f;
        if (with_norm) {
            float wt_sum = 0.f;
            for (int k = 0; k < n_expert_used; k++) {
                if ((s_selid[k] & (WARP_SIZE - 1)) == tid) {
                    wt_sum += s_sel[k];
                }
            }
            wt_sum  = warp_reduce_sum(wt_sum);
            wt_sum  = max(wt_sum, clamp_val);
            inv_sum = 1.0f / wt_sum;
        }
        for (int k = tid; k < n_expert_used; k += WARP_SIZE) {
            weights[k] = (with_norm ? s_sel[k] * inv_sum : s_sel[k]) * scale_val;
        }
    }
}

template <int n_experts, int n_threads>
__launch_bounds__(n_threads, 1)
__global__ void topk_moe_rank_cuda(const float * logits,
                                   float *       weights,
                                   int32_t *     ids,
                                   const int     n_expert_used,
                                   const float   clamp_val,
                                   const float   scale_val,
                                   const bool    with_norm) {
    const int row = blockIdx.x;
    ggml_cuda_pdl_sync();
    topk_moe_rank_block<n_experts, n_threads>(logits + n_experts * row, weights + n_expert_used * row,
                                              ids + n_experts * row, n_expert_used, clamp_val, scale_val, with_norm);
}

// Router mul_mat (F32 weight [ncols, n_experts], one token) and the rank top-k in one kernel: each wave computes the
// logit of one expert, the last block to finish (atomic counter) runs topk_moe_rank_block on all logits.
// The logits are summed in another order than mul_mat_vec_f, the top-k on them is the same code as above.
template <int n_experts, int n_threads>
__launch_bounds__(n_threads, 1)
__global__ void router_topk_moe_cuda(const float *  x,
                                     const float *  w,
                                     const int      ncols,
                                     const int64_t  stride_w,
                                     float *        logits_tmp,
                                     unsigned int * counter,
                                     float *        weights,
                                     int32_t *      ids,
                                     const int      n_expert_used,
                                     const float    clamp_val,
                                     const float    scale_val,
                                     const bool     with_norm) {
    constexpr int ws     = ggml_cuda_get_physical_warp_size();
    constexpr int nwaves = n_threads / ws;
    static_assert(n_threads % ws == 0 && n_experts % nwaves == 0, "bad router_topk_moe config");

    const int tid  = threadIdx.x;
    const int lane = tid % ws;
    const int e    = blockIdx.x*nwaves + tid / ws;

    const float4 * x4 = (const float4 *) x;
    const float4 * w4 = (const float4 *) (w + e*stride_w);
    float sum = 0.0f;
    for (int c = lane; c < ncols/4; c += ws) {
        const float4 a = x4[c];
        const float4 b = w4[c];
        sum += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    }
    sum = warp_reduce_sum<ws>(sum);
    if (lane == 0) {
        logits_tmp[e] = sum;
    }

    // the last block takes over: the logits of all blocks are written (fence before the counter increment)
    __shared__ bool s_last;
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        s_last = atomicAdd(counter, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();
    if (tid == 0) {
        *counter = 0; // for the next launch (same stream)
    }

    __shared__ float s_logits[n_experts];
    for (int i = tid; i < n_experts; i += n_threads) {
        s_logits[i] = ((volatile const float *) logits_tmp)[i]; // written by other CUs: not from L1
    }
    __syncthreads();
    topk_moe_rank_block<n_experts, n_threads>(s_logits, weights, ids, n_expert_used, clamp_val, scale_val, with_norm);
}

// GGML_CUDA_TOPK_MOE_RANK=0/1: rank-based top-k for softmax gating without bias (topk_moe_rank_cuda). Default on.
static bool ggml_cuda_topk_moe_rank_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_TOPK_MOE_RANK");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

template <int n_experts>
static void launch_topk_moe_rank_cuda(ggml_backend_cuda_context & ctx,
                                      const float *               logits,
                                      float *                     weights,
                                      int32_t *                   ids,
                                      const int                   n_rows,
                                      const int                   n_expert_used,
                                      const float                 clamp_val,
                                      const float                 scale_val,
                                      const bool                  with_norm) {
    constexpr int n_threads = n_experts < 256 ? n_experts : 256;
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(dim3(n_rows, 1, 1), dim3(n_threads, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(topk_moe_rank_cuda<n_experts, n_threads>, launch_params,
        logits, weights, ids, n_expert_used, clamp_val, scale_val, with_norm);
}

// true if the rank kernel handled the op
static bool try_topk_moe_rank_cuda(ggml_backend_cuda_context & ctx,
                                   const float *               logits,
                                   float *                     weights,
                                   int32_t *                   ids,
                                   const int                   n_rows,
                                   const int                   n_expert,
                                   const int                   n_expert_used,
                                   const float                 clamp_val,
                                   const float                 scale_val,
                                   const topk_moe_config       config) {
    if (config.use_sigmoid || config.use_sqrt_softplus || config.delayed_softmax || !ggml_cuda_topk_moe_rank_enabled()) {
        return false;
    }
    switch (n_expert) {
        case 32:
            launch_topk_moe_rank_cuda<32>(ctx, logits, weights, ids, n_rows, n_expert_used, clamp_val, scale_val, config.with_norm);
            return true;
        case 64:
            launch_topk_moe_rank_cuda<64>(ctx, logits, weights, ids, n_rows, n_expert_used, clamp_val, scale_val, config.with_norm);
            return true;
        case 128:
            launch_topk_moe_rank_cuda<128>(ctx, logits, weights, ids, n_rows, n_expert_used, clamp_val, scale_val, config.with_norm);
            return true;
        case 256:
            launch_topk_moe_rank_cuda<256>(ctx, logits, weights, ids, n_rows, n_expert_used, clamp_val, scale_val, config.with_norm);
            return true;
        case 512:
            launch_topk_moe_rank_cuda<512>(ctx, logits, weights, ids, n_rows, n_expert_used, clamp_val, scale_val, config.with_norm);
            return true;
        default:
            return false;
    }
}

template<bool has_bias>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 float *                     bias,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used,
                                 const float                 clamp_val,
                                 const float                 scale_val,
                                 const topk_moe_config       config) {
    GGML_ASSERT(!(config.with_norm && config.delayed_softmax) &&
                "delayed softmax is not supported with weight normalization");
    const int    rows_per_block = TOPK_MOE_ROWS_PER_BLOCK;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);
    cudaStream_t stream = ctx.stream();
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    switch (n_expert) {
        case 1:
            ggml_cuda_kernel_launch(topk_moe_cuda<1, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 2:
            ggml_cuda_kernel_launch(topk_moe_cuda<2, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 4:
            ggml_cuda_kernel_launch(topk_moe_cuda<4, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 8:
            ggml_cuda_kernel_launch(topk_moe_cuda<8, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 16:
            ggml_cuda_kernel_launch(topk_moe_cuda<16, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 32:
            ggml_cuda_kernel_launch(topk_moe_cuda<32, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 64:
            ggml_cuda_kernel_launch(topk_moe_cuda<64, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 128:
            ggml_cuda_kernel_launch(topk_moe_cuda<128, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 256:
            ggml_cuda_kernel_launch(topk_moe_cuda<256, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 288: // StepFun 3.7
            ggml_cuda_kernel_launch(topk_moe_cuda<288, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 512:
            ggml_cuda_kernel_launch(topk_moe_cuda<512, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 576:
            ggml_cuda_kernel_launch(topk_moe_cuda<576, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        default:
            GGML_ASSERT(false && "fatal error");
            break;
    }
}

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    float *       bias_d    = bias ? (float *) bias->data : nullptr;

    float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const int n_expert_used = weights->ne[1];

    const bool with_norm = clamp != nullptr;

    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, 0);
    }

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = with_norm;
    config.delayed_softmax   = args.delayed_softmax;

    if (bias) {
        launch_topk_moe_cuda<true>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    } else if (!try_topk_moe_rank_cuda(ctx, logits_d, weights_d, ids_d, n_rows, n_experts, n_expert_used, clamp_val,
                                       scale_val, config)) {
        launch_topk_moe_cuda<false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    }
}

// scratch of the fused router top-k: the block counter (zero between launches), then the logits
static constexpr size_t router_topk_scratch_size = 256 + 512*sizeof(float);

bool ggml_cuda_router_topk_moe_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * bias,
                                         const ggml_cuda_topk_moe_args & args) {
    const ggml_tensor * w = mm->src[0];
    const ggml_tensor * x = mm->src[1];
    const int n_experts = mm->ne[0];
    if (bias || args.sigmoid || args.sqrt_softplus || args.delayed_softmax || !ggml_cuda_topk_moe_rank_enabled()) {
        return false;
    }
    if (n_experts != 128 && n_experts != 256 && n_experts != 512) {
        return false;
    }
    if (mm->op != GGML_OP_MUL_MAT || w->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 ||
            ggml_nrows(x) != 1 || ggml_nrows(w) != n_experts || w->ne[2] != 1 || w->ne[3] != 1 ||
            w->nb[0] != sizeof(float) || x->nb[0] != sizeof(float) || w->ne[0] % 4 != 0 || w->nb[1] % 16 != 0 ||
            (uintptr_t) w->data % 16 != 0 || (uintptr_t) x->data % 16 != 0) {
        return false;
    }
    if (ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != 64) {
        return false; // the kernel is compiled for the physical wave size, only tested with 64
    }
    if (ctx.router_topk_scratch == nullptr) {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture_status));
        if (capture_status != cudaStreamCaptureStatusNone) {
            return false;
        }
        CUDA_CHECK(cudaMalloc(&ctx.router_topk_scratch, router_topk_scratch_size));
        CUDA_CHECK(cudaMemset(ctx.router_topk_scratch, 0, router_topk_scratch_size));
    }
    return true;
}

template <int n_experts>
static void launch_router_topk_moe_cuda(ggml_backend_cuda_context & ctx, const float * x, const float * w, const int ncols,
                                        const int64_t stride_w, float * weights, int32_t * ids, const int n_expert_used,
                                        const float clamp_val, const float scale_val, const bool with_norm) {
    constexpr int n_threads = n_experts < 256 ? n_experts : 256;
    constexpr int nwaves    = n_threads / 64;
    unsigned int * counter = (unsigned int *) ctx.router_topk_scratch;
    float *        logits  = (float *) ((char *) ctx.router_topk_scratch + 256);
    router_topk_moe_cuda<n_experts, n_threads><<<n_experts/nwaves, n_threads, 0, ctx.stream()>>>(
        x, w, ncols, stride_w, logits, counter, weights, ids, n_expert_used, clamp_val, scale_val, with_norm);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_router_topk_moe(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * weights,
                                  ggml_tensor * ids, const ggml_tensor * clamp, const ggml_tensor * scale) {
    const ggml_tensor * w = mm->src[0];
    const ggml_tensor * x = mm->src[1];
    const int n_experts     = mm->ne[0];
    const int n_expert_used = weights->ne[1];
    GGML_ASSERT(weights->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const bool  with_norm = clamp != nullptr;
    const float clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    const float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    const float * x_d = (const float *) x->data;
    const float * w_d = (const float *) w->data;
    const int64_t stride_w = w->nb[1] / sizeof(float);
    float *   weights_d = (float *) weights->data;
    int32_t * ids_d     = (int32_t *) ids->data;

    switch (n_experts) {
        case 128:
            launch_router_topk_moe_cuda<128>(ctx, x_d, w_d, w->ne[0], stride_w, weights_d, ids_d, n_expert_used, clamp_val, scale_val, with_norm);
            break;
        case 256:
            launch_router_topk_moe_cuda<256>(ctx, x_d, w_d, w->ne[0], stride_w, weights_d, ids_d, n_expert_used, clamp_val, scale_val, with_norm);
            break;
        case 512:
            launch_router_topk_moe_cuda<512>(ctx, x_d, w_d, w->ne[0], stride_w, weights_d, ids_d, n_expert_used, clamp_val, scale_val, with_norm);
            break;
        default:
            GGML_ABORT("unsupported n_experts");
    }
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids) {
    // must match an instantiation of launch_topk_moe_cuda: a power of 2 up to 512,
    // or one of the non-power-of-2 expert counts of supported models
    const int n_expert = ids->nb[1] / ids->nb[0];
    if (((n_expert & (n_expert - 1)) != 0 || n_expert > 512) && n_expert != 288 && n_expert != 576) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        const ggml_tensor * softmax  = gating_op;
        float               scale    = 1.0f;
        float               max_bias = 0.0f;

        memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(softmax->src[0])) {
            return false;
        }

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        ggml_unary_op op = ggml_get_unary_op(gating_op);

        if (op != GGML_UNARY_OP_SIGMOID && op != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }
    }

    return true;
}
