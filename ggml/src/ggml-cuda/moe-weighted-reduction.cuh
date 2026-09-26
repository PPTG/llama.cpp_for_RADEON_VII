#include "common.cuh"

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst);

// Same as above, but the per expert scale is scale_vec[ids[token, expert]] (replaces reshape -> repeat -> get_rows).
void ggml_cuda_op_moe_weighted_reduction_ids(ggml_backend_cuda_context & ctx,
                                             const ggml_tensor *         experts,
                                             const ggml_tensor *         scale_vec,
                                             const ggml_tensor *         ids,
                                             const ggml_tensor *         weights,
                                             ggml_tensor *               dst);
