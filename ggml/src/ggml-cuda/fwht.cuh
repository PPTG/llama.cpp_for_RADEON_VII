#pragma once

#include "common.cuh"

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

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op);

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);

// fwht -> [reshape/view ->] set_rows into a q8_0 KV cache in one kernel.
bool ggml_cuda_fwht_set_rows_supported(const ggml_tensor * fwht, const ggml_tensor * set_rows);
void ggml_cuda_op_fwht_set_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * fwht, ggml_tensor * set_rows);

// fwht -> [reshape/view ->] mul_mat: transform and quantize src1 (a view of the fwht output) to q8_1 for MMVQ.
bool ggml_cuda_fwht_quantize_q8_1_supported(const ggml_tensor * fwht, const ggml_tensor * src1);
void ggml_cuda_fwht_quantize_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * fwht, const ggml_tensor * src1, void * q8,
                                  int64_t ne10_padded);
