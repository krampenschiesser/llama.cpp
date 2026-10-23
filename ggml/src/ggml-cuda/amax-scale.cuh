#pragma once

#include "common.cuh"

#define CUDA_AMAX_SCALE_BLOCK_SIZE 256

// fused ABS -> POOL_1D(MAX) -> SCALE -> LOG -> SCALE -> CEIL -> CLAMP -> SCALE -> EXP
void ggml_cuda_op_amax_scale(ggml_backend_cuda_context & ctx, const ggml_tensor * cur, ggml_tensor * scale);
