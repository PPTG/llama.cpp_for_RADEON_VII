#include "common.cuh"

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
