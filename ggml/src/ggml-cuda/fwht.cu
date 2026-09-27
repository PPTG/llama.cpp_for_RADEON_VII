#include "common.cuh"
#include "convert.cuh"
#include "fwht.cuh"

// In-register transform of one row of N values, value i*warp_size + lane is in reg[i].
template <int N>
static __device__ __forceinline__ void fwht_row(float * reg, const int lane) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static constexpr int el_w = N / warp_size;

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

template <int N, typename T>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const T * src, float * dst, const int64_t n_rows, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = ggml_cuda_cast<float>(src[i * warp_size + lane]) * scale;
    }

    fwht_row<N>(reg, lane);

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = reg[i];
    }
}

// fwht -> set_rows into a q8_0 cache: row r of the transform is part r % rows_per_idx of set_rows row
// r / rows_per_idx. Same math as fwht_cuda + k_set_rows_quant (quantize_f32_q8_0_block).
template <int N, typename T, typename idx_t>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_set_rows_q8_0_cuda(const T * src, char * dst, const idx_t * idx, const int64_t n_rows, const float scale,
                                        const int rows_per_idx, const int64_t s_idx, const int64_t nb_dst1) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size % QK8_0 == 0, "bad warp size");

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = ggml_cuda_cast<float>(src[i * warp_size + lane]) * scale;
    }

    fwht_row<N>(reg, lane);

    const int64_t dst_row = idx[(r / rows_per_idx) * s_idx];
    block_q8_0 * y = (block_q8_0 *) (dst + dst_row * nb_dst1) + (r % rows_per_idx) * (N / QK8_0);

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const float amax = warp_reduce_max<QK8_0>(fabsf(reg[i]));
        const float d    = amax / ((1 << 7) - 1);
        const float id   = d ? 1.0f/d : 0.0f;

        const int e = i * warp_size + lane;
        y[e / QK8_0].qs[e % QK8_0] = roundf(reg[i]*id);
        if (e % QK8_0 == 0) {
            y[e / QK8_0].d = d;
        }
    }
}

// fwht -> q8_1 for MMVQ: the transformed rows are the contiguous src1 of the next mul_mat (ne10 values per row,
// q8_1 rows padded to ne10_padded). Same math as fwht_cuda + quantize_q8_1.
template <int N, typename T>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_quantize_q8_1_cuda(const T * src, void * vy, const int64_t n_rows, const float scale,
                                        const int64_t ne10, const int64_t ne10_padded) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size % QK8_1 == 0, "bad warp size");

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = ggml_cuda_cast<float>(src[i * warp_size + lane]) * scale;
    }

    fwht_row<N>(reg, lane);

    const int64_t i10 = (r * N) % ne10;
    const int64_t i11 = (r * N) / ne10;
    block_q8_1 * y = (block_q8_1 *) vy + (i11 * ne10_padded + i10) / QK8_1;

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const float xi   = reg[i];
        const float amax = warp_reduce_max<QK8_1>(fabsf(xi));
        const float sum  = warp_reduce_sum<QK8_1>(xi);
        const float d    = amax / 127.0f;

        const int e = i * warp_size + lane;
        y[e / QK8_1].qs[e % QK8_1] = amax == 0.0f ? 0 : roundf(xi / d);
        if (e % QK8_1 == 0) {
            y[e / QK8_1].ds = make_half2(d, sum);
        }
    }
}

template <typename T>
static bool ggml_cuda_op_fwht_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    const int     n    = src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const T *     src_d = (const T *) src->data;
    float *       dst_d = (float *) dst->data;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;

    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;

    cudaStream_t                         stream = ctx.stream();
    dim3                                 grid_dims(num_blocks, 1, 1);
    dim3                                 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    const float scale = 1 / sqrtf(n);

    switch (n) {
        case 64:
            ggml_cuda_kernel_launch(fwht_cuda<64, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 128:
            ggml_cuda_kernel_launch(fwht_cuda<128, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 256:
            ggml_cuda_kernel_launch(fwht_cuda<256, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 512:
            ggml_cuda_kernel_launch(fwht_cuda<512, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op) {
    const struct ggml_tensor * a = op->src[0];
    const struct ggml_tensor * b = op->src[1];

    return op->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
           a->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           (b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16) && ggml_is_contiguous(b) && ggml_is_contiguous(op) &&
           ggml_are_same_shape(b, op);
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    switch (src->type) {
        case GGML_TYPE_F32:
            return ggml_cuda_op_fwht_impl<float>(ctx, src, dst);
        case GGML_TYPE_F16:
            return ggml_cuda_op_fwht_impl<half>(ctx, src, dst);
        default:
            return false;
    }
}

// fwht (a hadamard mul_mat) followed by RESHAPE/VIEW nodes and then consumer: returns the tensor the consumer reads,
// if it is a plain contiguous view of the whole fwht output, else nullptr.
static const ggml_tensor * ggml_cuda_fwht_view_of(const ggml_tensor * fwht, const ggml_tensor * t) {
    const ggml_tensor * v = t;
    while (v != fwht) {
        if ((v->op != GGML_OP_RESHAPE && v->op != GGML_OP_VIEW) || v->view_offs != 0) {
            return nullptr;
        }
        v = v->src[0];
    }
    if (t->data != fwht->data || ggml_nelements(t) != ggml_nelements(fwht) || !ggml_is_contiguous(t)) {
        return nullptr;
    }
    return t;
}

static bool ggml_cuda_fwht_fusable(const ggml_tensor * fwht) {
    if (!ggml_cuda_op_mul_mat_use_fwht(fwht)) {
        return false;
    }
    const int n = fwht->ne[0];
    return n == 64 || n == 128 || n == 256 || n == 512;
}

bool ggml_cuda_fwht_set_rows_supported(const ggml_tensor * fwht, const ggml_tensor * set_rows) {
    if (!ggml_cuda_fwht_fusable(fwht) || set_rows->op != GGML_OP_SET_ROWS || set_rows->type != GGML_TYPE_Q8_0) {
        return false;
    }
    const ggml_tensor * src = ggml_cuda_fwht_view_of(fwht, set_rows->src[0]);
    const ggml_tensor * idx = set_rows->src[1];
    if (src == nullptr || src->type != GGML_TYPE_F32 || src->ne[2] != 1 || src->ne[3] != 1 ||
            src->ne[0] % fwht->ne[0] != 0 || src->ne[0] != set_rows->ne[0]) {
        return false;
    }
    if ((idx->type != GGML_TYPE_I64 && idx->type != GGML_TYPE_I32) || idx->ne[0] != src->ne[1] || ggml_nrows(idx) != 1) {
        return false;
    }
    return set_rows->ne[2] == 1 && set_rows->ne[3] == 1 && set_rows->nb[0] == sizeof(block_q8_0);
}

template <typename T, typename idx_t>
static void ggml_cuda_op_fwht_set_rows_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * fwht, ggml_tensor * set_rows) {
    const ggml_tensor * src  = fwht->src[1];
    const ggml_tensor * idx  = set_rows->src[1];
    const int           n    = fwht->ne[0];
    const int64_t       rows = ggml_nrows(src);

    const T *     src_d = (const T *) src->data;
    const idx_t * idx_d = (const idx_t *) idx->data;
    char *        dst_d = (char *) set_rows->data;

    const int     rows_per_idx = set_rows->src[0]->ne[0] / n;
    const int64_t s_idx        = idx->nb[0] / sizeof(idx_t);
    const int64_t nb_dst1      = set_rows->nb[1];

    const int warp_size      = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;

    const dim3 grid_dims((rows + rows_per_block - 1) / rows_per_block, 1, 1);
    const dim3 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    const float scale = 1 / sqrtf(n);

    switch (n) {
        case 64:
            ggml_cuda_kernel_launch(fwht_set_rows_q8_0_cuda<64, T, idx_t>, launch_params, src_d, dst_d, idx_d, rows, scale, rows_per_idx, s_idx, nb_dst1);
            break;
        case 128:
            ggml_cuda_kernel_launch(fwht_set_rows_q8_0_cuda<128, T, idx_t>, launch_params, src_d, dst_d, idx_d, rows, scale, rows_per_idx, s_idx, nb_dst1);
            break;
        case 256:
            ggml_cuda_kernel_launch(fwht_set_rows_q8_0_cuda<256, T, idx_t>, launch_params, src_d, dst_d, idx_d, rows, scale, rows_per_idx, s_idx, nb_dst1);
            break;
        case 512:
            ggml_cuda_kernel_launch(fwht_set_rows_q8_0_cuda<512, T, idx_t>, launch_params, src_d, dst_d, idx_d, rows, scale, rows_per_idx, s_idx, nb_dst1);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_fwht_set_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * fwht, ggml_tensor * set_rows) {
    GGML_ASSERT(ggml_cuda_fwht_set_rows_supported(fwht, set_rows));
    const bool src_f16 = fwht->src[1]->type == GGML_TYPE_F16;
    const bool idx_i64 = set_rows->src[1]->type == GGML_TYPE_I64;
    if (src_f16) {
        idx_i64 ? ggml_cuda_op_fwht_set_rows_impl<half, int64_t>(ctx, fwht, set_rows)
                : ggml_cuda_op_fwht_set_rows_impl<half, int32_t>(ctx, fwht, set_rows);
    } else {
        idx_i64 ? ggml_cuda_op_fwht_set_rows_impl<float, int64_t>(ctx, fwht, set_rows)
                : ggml_cuda_op_fwht_set_rows_impl<float, int32_t>(ctx, fwht, set_rows);
    }
}

bool ggml_cuda_fwht_quantize_q8_1_supported(const ggml_tensor * fwht, const ggml_tensor * src1) {
    if (!ggml_cuda_fwht_fusable(fwht) || ggml_cuda_fwht_view_of(fwht, src1) == nullptr || src1->type != GGML_TYPE_F32) {
        return false;
    }
    return src1->ne[0] % fwht->ne[0] == 0;
}

template <typename T>
static void ggml_cuda_fwht_quantize_q8_1_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * fwht, const ggml_tensor * src1,
                                              void * q8, const int64_t ne10_padded) {
    const ggml_tensor * src  = fwht->src[1];
    const int           n    = fwht->ne[0];
    const int64_t       rows = ggml_nrows(src);
    const int64_t       ne10 = src1->ne[0];

    const T * src_d = (const T *) src->data;

    const int warp_size      = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;

    const dim3 grid_dims((rows + rows_per_block - 1) / rows_per_block, 1, 1);
    const dim3 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    const float scale = 1 / sqrtf(n);

    switch (n) {
        case 64:
            ggml_cuda_kernel_launch(fwht_quantize_q8_1_cuda<64, T>, launch_params, src_d, q8, rows, scale, ne10, ne10_padded);
            break;
        case 128:
            ggml_cuda_kernel_launch(fwht_quantize_q8_1_cuda<128, T>, launch_params, src_d, q8, rows, scale, ne10, ne10_padded);
            break;
        case 256:
            ggml_cuda_kernel_launch(fwht_quantize_q8_1_cuda<256, T>, launch_params, src_d, q8, rows, scale, ne10, ne10_padded);
            break;
        case 512:
            ggml_cuda_kernel_launch(fwht_quantize_q8_1_cuda<512, T>, launch_params, src_d, q8, rows, scale, ne10, ne10_padded);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_cuda_fwht_quantize_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * fwht, const ggml_tensor * src1, void * q8,
                                  const int64_t ne10_padded) {
    GGML_ASSERT(ggml_cuda_fwht_quantize_q8_1_supported(fwht, src1));
    // the padding of each q8_1 row is not written, clear it once
    if (ne10_padded != src1->ne[0]) {
        CUDA_CHECK(cudaMemsetAsync(q8, 0, ggml_nrows(src1) * ne10_padded * sizeof(block_q8_1) / QK8_1, ctx.stream()));
    }
    if (fwht->src[1]->type == GGML_TYPE_F16) {
        ggml_cuda_fwht_quantize_q8_1_impl<half>(ctx, fwht, src1, q8, ne10_padded);
    } else {
        ggml_cuda_fwht_quantize_q8_1_impl<float>(ctx, fwht, src1, q8, ne10_padded);
    }
}
