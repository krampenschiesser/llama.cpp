#pragma once

void ggml_cuda_launch_mm_ids_zero_skipped(
        float * dst, const int32_t * ids, int64_t ne0, int64_t n_tokens, int64_t n_expert_used, int64_t ids_stride,
        int64_t nb0, int64_t nb1, int64_t nb2, cudaStream_t stream);

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);
