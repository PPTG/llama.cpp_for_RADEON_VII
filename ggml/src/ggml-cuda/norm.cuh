#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_scale_mul_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor,
                                     ggml_tensor *               post_scale_tensor = nullptr);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Up to 3 rms_norm -> [scale ->] mul chains on the same input in one kernel.
// scales[k] may be nullptr, muls[k] is the output of chain k.
// q8: optional, per chain nullptr or a buffer for the q8_1 copy of the output (rows padded to MATRIX_ROW_PADDING).
void ggml_cuda_op_rms_norm_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const * norms,
                                 const ggml_tensor * const * scales, const ggml_tensor * const * muls, int n,
                                 void * const * q8 = nullptr);

// rms_norm -> mul by a weight row that also writes the q8_1 copy of the result (rows padded to MATRIX_ROW_PADDING)
bool ggml_cuda_rms_norm_mul_q8_1_supported(const ggml_tensor * rms_norm, const ggml_tensor * mul);
void ggml_cuda_op_rms_norm_mul_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, ggml_tensor * mul,
                                    void * q8);
