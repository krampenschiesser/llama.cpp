#include "common.cuh"

struct ggml_cuda_topk_qsa_match {
    const ggml_tensor * scores   = nullptr;
    const ggml_tensor * cell_blk = nullptr;
    const ggml_tensor * mask     = nullptr;
    ggml_tensor *       top_k    = nullptr;
};

bool ggml_cuda_match_topk_qsa(const ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_qsa_match & match);

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_topk_qsa(ggml_backend_cuda_context & ctx, const ggml_tensor * scores, const ggml_tensor * cell_blk,
                           const ggml_tensor * mask, ggml_tensor * top_k);
