#include "amax-scale.cuh"

// per-row power-of-two rescale factor used by the MoE down projection:
//   e = exp(ln2 * ceil(clamp(log(amax / 1024) * log2e, 0, 127)))
// one block per row: reduce max|cur|, then let thread 0 compute the factor
static __global__ void k_amax_scale_f32(
        const float * __restrict__ cur,
        float * __restrict__ scale,
        const int64_t n_embd) {
    __shared__ float shared[CUDA_AMAX_SCALE_BLOCK_SIZE/WARP_SIZE];

    const int64_t row = blockIdx.x;
    const float * x = cur + row*n_embd;

    float amax = 0.0f;
    for (int64_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        amax = fmaxf(amax, fabsf(x[i]));
    }

    amax = block_reduce<block_reduce_method::MAX, CUDA_AMAX_SCALE_BLOCK_SIZE>(amax, shared);

    if (threadIdx.x == 0) {
        constexpr float k_log2_e = 1.4426950408889634f; // 1 / ln(2)
        constexpr float k_ln_2   = 0.6931471805599453f;
        // leave x32 headroom for the F16 sums of 32 values in Q8 activation blocks
        float e = logf(amax*(1.0f/1024.0f))*k_log2_e;
        e = ceilf(e);
        e = fminf(fmaxf(e, 0.0f), 127.0f);
        scale[row] = expf(e*k_ln_2);
    }
}

void ggml_cuda_op_amax_scale(ggml_backend_cuda_context & ctx, const ggml_tensor * cur, ggml_tensor * scale) {
    GGML_ASSERT(cur->type   == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(cur));
    GGML_ASSERT(ggml_is_contiguous(scale));
    GGML_ASSERT(scale->ne[0] == 1 && scale->ne[1] == cur->ne[1] &&
                scale->ne[2] == cur->ne[2] && scale->ne[3] == cur->ne[3]);

    const int64_t n_embd = cur->ne[0];
    const int64_t n_rows = ggml_nrows(cur);

    k_amax_scale_f32<<<n_rows, CUDA_AMAX_SCALE_BLOCK_SIZE, 0, ctx.stream()>>>(
            (const float *) cur->data, (float *) scale->data, n_embd);
    CUDA_CHECK(cudaGetLastError());
}
