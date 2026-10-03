#include "norm.cuh"
#include <cstdint>

template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float2 mean_var = make_float2(0.0f, 0.0f);

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (x[col] - mean) * inv_std;
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum + 32);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

template <int block_size, bool do_multiply = false, bool do_add = false, bool do_scale = false>
static __global__ void rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float   scale_out            = 1.0f,
                                    const float * post_scale           = nullptr) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
    static_assert(!do_scale || !do_add, "fusing scale is not supported with add");

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            const float v     = scale * x[col] * mul[mul_col] + add[add_col];
            // optional multiply by a scalar tensor after the add (e.g. Gemma 4 layer_output_scale)
            dst[col]          = post_scale != nullptr ? v * post_scale[0] : v;
        } else if constexpr (do_multiply && do_scale) {
            // same op order as rms_norm -> scale -> mul
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col]          = (scale_out * (scale * x[col])) * mul[mul_col];
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col];
        } else if constexpr (do_scale) {
            dst[col] = scale_out * (scale * x[col]);
        } else {
            dst[col] = scale * x[col];
        }
    }
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size>
static __global__ void l2_norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    ggml_cuda_pdl_lc();

    // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col];
    }
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, block_dims.x > WARP_SIZE ? 2 * 32 * sizeof(float): 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

// RMS norm rows shorter than this use 256 threads per block, longer rows 1024.
// GGML_CUDA_NORM_BLOCK_THRESHOLD=<n> overrides it (e.g. 4096 to use 256 threads for n_embd 2816).
static bool ggml_cuda_rms_norm_small_block(const int ncols) {
    static const int threshold = [] {
        const char * env = getenv("GGML_CUDA_NORM_BLOCK_THRESHOLD");
        return env ? atoi(env) : 1024;
    }();
    return ncols < threshold;
}

template <bool do_scale = false>
static void rms_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream,
        const float scale_out = 1.0f) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ggml_cuda_rms_norm_small_block(ncols)) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<256, false, false, do_scale>, launch_params,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), scale_out, nullptr);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<1024, false, false, do_scale>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), scale_out, nullptr);
    }
}

static void rms_norm_mul_f32_cuda(const float *  x,
                                  const float *  mul,
                                  const float *  add,
                                  float *        dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream,
                                  const float    scale_out = 1.0f,
                                  const float *  post_scale = nullptr) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    GGML_ASSERT(scale_out == 1.0f || (mul != nullptr && add == nullptr));
    GGML_ASSERT(post_scale == nullptr || (mul != nullptr && add != nullptr));
    if (mul == nullptr) {
        rms_norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (add == nullptr) {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        if (ggml_cuda_rms_norm_small_block(ncols)) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(scale_out != 1.0f ? rms_norm_f32<256, true, false, true> : rms_norm_f32<256, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), scale_out, nullptr);
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(scale_out != 1.0f ? rms_norm_f32<1024, true, false, true> : rms_norm_f32<1024, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), scale_out, nullptr);
        }
    } else {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        if (ggml_cuda_rms_norm_small_block(ncols)) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims,block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, true, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed, 1.0f, post_scale);
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed, 1.0f, post_scale);
        }
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

static void l2_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<1024>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    float *             dst_d  = (float *) scale_tensor->data;
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    float scale;
    memcpy(&scale, (const float *) scale_tensor->op_params + 0, sizeof(float));

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda<true>(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream, scale);
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float * src0_d = (const float *) rms_norm_src->data;
    const float * mul_d = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_d = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ 0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_scale_mul_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor, ggml_tensor * mul_tensor) {
    const ggml_tensor * rms_norm_src = dst->src[0];
    float eps = 0.0f;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    float scale_out;
    memcpy(&scale_out, (const float *) scale_tensor->op_params + 0, sizeof(float));

    const ggml_tensor * mul_src = mul_tensor->src[0] == scale_tensor ? mul_tensor->src[1] : mul_tensor->src[0];
    GGML_ASSERT(mul_tensor->src[0] == scale_tensor || mul_tensor->src[1] == scale_tensor);

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);

    rms_norm_mul_f32_cuda((const float *) rms_norm_src->data, (const float *) mul_src->data, nullptr, (float *) mul_tensor->data,
                          rms_norm_src->ne[0], rms_norm_src->ne[1], rms_norm_src->ne[2], rms_norm_src->ne[3],
                          rms_norm_src->nb[1] / ts0, rms_norm_src->nb[2] / ts0, rms_norm_src->nb[3] / ts0,
                          mul_src->nb[1] / ts_mul, mul_src->nb[2] / ts_mul, mul_src->nb[3] / ts_mul,
                          mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
                          0, 0, 0,
                          0, 0, 0, 0,
                          eps, ctx.stream(), scale_out);
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor,
                                     ggml_tensor *               post_scale_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float *       src0_d  = (const float *) rms_norm_src->data;
    const float *       mul_d   = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d   = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_d   = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const float *       add_d   = nullptr;
    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_d   = (float *) add_tensor->src[1]->data;
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_d   = (float *) add_tensor->src[0]->data;
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    // with a post scale the output is the post scale mul node
    float *      dst_d  = (float *) (post_scale_tensor ? post_scale_tensor->data : add_tensor->data);
    const float * post_scale_d = nullptr;
    if (post_scale_tensor) {
        const ggml_tensor * scalar = post_scale_tensor->src[0] == add_tensor ? post_scale_tensor->src[1] : post_scale_tensor->src[0];
        GGML_ASSERT(scalar->type == GGML_TYPE_F32 && ggml_nelements(scalar) == 1);
        post_scale_d = (const float *) scalar->data;
    }
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d,add_d,dst_d,
                          ne00,ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream, 1.0f, post_scale_d);
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    l2_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

// Up to 3 rms_norm -> [scale ->] mul chains on the same input: the sum of squares is computed once.
// Each output uses the same formula as the single fused kernels above, so the results are the same.
struct ggml_cuda_rms_norm_multi_args {
    const float * w[3]         = {nullptr, nullptr, nullptr};
    float *       dst[3]       = {nullptr, nullptr, nullptr};
    void *        q8[3]        = {nullptr, nullptr, nullptr}; // optional q8_1 copies for MMVQ
    float         scale_out[3] = {1.0f, 1.0f, 1.0f};
    bool          has_scale[3] = {false, false, false};
    int           n            = 0;
    int           ncols_q8     = 0; // padded row size of the q8_1 copies
};

// Quantize one value per thread to q8_1 like quantize_q8_1 (same reductions, same result). All threads of the warp
// must call it; consecutive threads hold consecutive columns, col % QK8_1 == lane % QK8_1. Columns >= ncols_padded are
// not stored.
static __device__ __forceinline__ void rms_norm_store_q8_1(const float xi, void * vy, const int64_t row, const int col,
                                                           const int ncols_padded) {
    float amax = fabsf(xi);
    float sum  = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    if (col >= ncols_padded) {
        return;
    }
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    block_q8_1 * y = (block_q8_1 *) vy + row*(ncols_padded/QK8_1) + col/QK8_1;
    y->qs[col % QK8_1] = q;
    if (col % QK8_1 == 0) {
        y->ds = make_half2(d, sum);
    }
}

// rms_norm -> mul (weight row of ncols) that also writes the q8_1 copy of the result for MMVQ.
// The norm and the product are computed as in rms_norm_f32<block_size, true>, so dst is the same.
template <int block_size>
static __global__ void rms_norm_mul_q8_1_f32(const float * x, const float * w, float * dst, void * q8, const int ncols,
                                             const int64_t stride_row, const float eps, const int ncols_padded) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    x   += row*stride_row;
    dst += (int64_t) row*ncols;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    // all threads run every step (uniform warp reductions), columns >= ncols are the zero padding
    for (int col0 = 0; col0 < ncols_padded; col0 += block_size) {
        const int col = col0 + tid;
        float v = 0.0f;
        if (col < ncols) {
            v = scale * x[col] * w[col];
            dst[col] = v;
        }
        rms_norm_store_q8_1(v, q8, row, col, ncols_padded);
    }
}

template <int block_size, bool with_q8>
static __global__ void rms_norm_multi_f32(const float * x, const int ncols, const int64_t stride_row, const float eps,
                                          const ggml_cuda_rms_norm_multi_args args) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    x += row*stride_row;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    if constexpr (!with_q8) {
        for (int col = tid; col < ncols; col += block_size) {
            const float xi = x[col];
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                if (k < args.n) {
                    const float v = args.has_scale[k] ? (args.scale_out[k] * (scale * xi)) * args.w[k][col] : scale * xi * args.w[k][col];
                    args.dst[k][(int64_t) row*ncols + col] = v;
                }
            }
        }
    } else {
        // all threads run every step (uniform warp reductions), columns >= ncols are the zero padding
        for (int col0 = 0; col0 < args.ncols_q8; col0 += block_size) {
            const int   col = col0 + tid;
            const float xi  = col < ncols ? x[col] : 0.0f;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                if (k < args.n) {
                    float v = 0.0f;
                    if (col < ncols) {
                        v = args.has_scale[k] ? (args.scale_out[k] * (scale * xi)) * args.w[k][col] : scale * xi * args.w[k][col];
                        args.dst[k][(int64_t) row*ncols + col] = v;
                    }
                    if (args.q8[k] != nullptr) {
                        rms_norm_store_q8_1(v, args.q8[k], row, col, args.ncols_q8);
                    }
                }
            }
        }
    }
}

void ggml_cuda_op_rms_norm_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const * norms,
                                 const ggml_tensor * const * scales, const ggml_tensor * const * muls, const int n,
                                 void * const * q8) {
    GGML_ASSERT(n >= 1 && n <= 3);
    const ggml_tensor * src = norms[0]->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32 && src->nb[0] == sizeof(float) && ggml_is_contiguous_rows(src));
    GGML_ASSERT(src->ne[2] == 1 && src->ne[3] == 1);

    float eps;
    memcpy(&eps, norms[0]->op_params, sizeof(float));

    ggml_cuda_rms_norm_multi_args args;
    args.n = n;
    for (int k = 0; k < n; ++k) {
        const ggml_tensor * prev = scales[k] ? scales[k] : norms[k];
        const ggml_tensor * w    = muls[k]->src[0] == prev ? muls[k]->src[1] : muls[k]->src[0];
        args.w[k]   = (const float *) w->data;
        args.dst[k] = (float *) muls[k]->data;
        if (scales[k]) {
            args.has_scale[k] = true;
            memcpy(&args.scale_out[k], scales[k]->op_params, sizeof(float));
        }
        if (q8 != nullptr && q8[k] != nullptr) {
            GGML_ASSERT(ggml_is_contiguous(muls[k]));
            args.q8[k] = q8[k];
        }
    }
    const bool with_q8 = args.q8[0] != nullptr || args.q8[1] != nullptr || args.q8[2] != nullptr;
    args.ncols_q8 = GGML_PAD(src->ne[0], MATRIX_ROW_PADDING);

    const int     ncols      = src->ne[0];
    const int     nrows      = src->ne[1];
    const int64_t stride_row = src->nb[1] / sizeof(float);
    cudaStream_t  stream     = ctx.stream();

    // same block size as the single kernels, so the reduction order is the same
    const float * x = (const float *) src->data;
    if (ggml_cuda_rms_norm_small_block(ncols)) {
        if (with_q8) {
            rms_norm_multi_f32<256, true><<<nrows, 256, 32 * sizeof(float), stream>>>(x, ncols, stride_row, eps, args);
        } else {
            rms_norm_multi_f32<256, false><<<nrows, 256, 32 * sizeof(float), stream>>>(x, ncols, stride_row, eps, args);
        }
    } else {
        if (with_q8) {
            rms_norm_multi_f32<1024, true><<<nrows, 1024, 32 * sizeof(float), stream>>>(x, ncols, stride_row, eps, args);
        } else {
            rms_norm_multi_f32<1024, false><<<nrows, 1024, 32 * sizeof(float), stream>>>(x, ncols, stride_row, eps, args);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_rms_norm_mul_q8_1_supported(const ggml_tensor * rms_norm, const ggml_tensor * mul) {
    const ggml_tensor * src = rms_norm->src[0];
    const ggml_tensor * w   = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    return src->type == GGML_TYPE_F32 && src->nb[0] == sizeof(float) && ggml_is_contiguous_rows(src) &&
        src->ne[2] == 1 && src->ne[3] == 1 && src->ne[1] <= 65535 &&
        w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && w->ne[0] == src->ne[0] && ggml_nrows(w) == 1 &&
        mul->type == GGML_TYPE_F32 && ggml_is_contiguous(mul) && ggml_are_same_shape(mul, src);
}

void ggml_cuda_op_rms_norm_mul_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, ggml_tensor * mul,
                                    void * q8) {
    GGML_ASSERT(ggml_cuda_rms_norm_mul_q8_1_supported(rms_norm, mul));
    const ggml_tensor * src = rms_norm->src[0];
    const ggml_tensor * w   = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    float eps;
    memcpy(&eps, rms_norm->op_params, sizeof(float));

    const int     ncols        = src->ne[0];
    const int     nrows        = src->ne[1];
    const int64_t stride_row   = src->nb[1] / sizeof(float);
    const int     ncols_padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    cudaStream_t  stream       = ctx.stream();

    const float * x   = (const float *) src->data;
    const float * wd  = (const float *) w->data;
    float *       dst = (float *) mul->data;
    // same block size as rms_norm_f32, so the reduction order is the same
    if (ggml_cuda_rms_norm_small_block(ncols)) {
        rms_norm_mul_q8_1_f32<256><<<nrows, 256, 32 * sizeof(float), stream>>>(x, wd, dst, q8, ncols, stride_row, eps, ncols_padded);
    } else {
        rms_norm_mul_q8_1_f32<1024><<<nrows, 1024, 32 * sizeof(float), stream>>>(x, wd, dst, q8, ncols, stride_row, eps, ncols_padded);
    }
    CUDA_CHECK(cudaGetLastError());
}
