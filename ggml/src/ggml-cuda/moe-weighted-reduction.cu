#include "moe-weighted-reduction.cuh"

// expert_scale: per (token, expert used) scale, or nullptr.
// scale_vec + ids: per expert scale looked up with the expert ids, used instead of expert_scale when not nullptr.
static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used,
                                                  const float * __restrict__ scale_vec,
                                                  const int32_t * __restrict__ ids,
                                                  const int64_t ids_stride) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    auto get_scale = [&](const int expert, const uint64_t row) {
        if (scale_vec != nullptr) {
            return scale_vec[ids[token * ids_stride + expert]];
        }
        return expert_scale != nullptr ? expert_scale[row] : 1.0f;
    };

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = get_scale(0, first_row);
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = get_scale(expert, row);
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream,
                                          const float *   scale_vec  = nullptr,
                                          const int32_t * ids        = nullptr,
                                          int64_t         ids_stride = 0) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used, scale_vec, ids, ids_stride);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_moe_weighted_reduction_ids(ggml_backend_cuda_context & ctx,
                                             const ggml_tensor *         experts,
                                             const ggml_tensor *         scale_vec,
                                             const ggml_tensor *         ids,
                                             const ggml_tensor *         weights,
                                             ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_vec->type == GGML_TYPE_F32 && ggml_is_contiguous(scale_vec));
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    GGML_ASSERT(ids->ne[0] >= n_expert_used && ids->ne[1] == n_tokens);

    launch_moe_weighted_reduction((const float *) experts->data, nullptr, (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, ctx.stream(),
                                  (const float *) scale_vec->data, (const int32_t *) ids->data, ids->nb[1] / sizeof(int32_t));
    CUDA_CHECK(cudaGetLastError());
}
