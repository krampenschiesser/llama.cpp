#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}

#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}

// match qwen4exp's QSA indexer: get_rows -> ... -> add(f16 mask) -> top_k
bool ggml_cuda_match_topk_qsa(const ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_qsa_match & match) {
    static const std::initializer_list<enum ggml_op> ops = {
        GGML_OP_GET_ROWS, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_CPY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_TOP_K
    };

    const int n_ops = (int) ops.size();
    if (node_idx + n_ops > cgraph->n_nodes) {
        return false;
    }
    for (int j = 0; j < n_ops; ++j) {
        const ggml_tensor * node = cgraph->nodes[node_idx + j];
        if (node->op != ops.begin()[j] || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            (j < n_ops - 1 && (node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0)) {
            return false;
        }
    }
    if (!ggml_check_edges(cgraph, node_idx,
            { { 1, 0, 0 }, { 2, 0, 1 }, { 4, 0, 3 }, { 5, 0, 2 }, { 5, 1, 4 }, { 6, 0, 5 } })) {
        return false;
    }

    // elided nodes must be single-use (cpy counts its own src[1] self-reference)
    for (int j = 0; j < n_ops - 1; ++j) {
        const ggml_tensor * node = cgraph->nodes[node_idx + j];
        const int32_t want = node->op == GGML_OP_CPY ? 2 : 1;
        if (ggml_node_get_use_count(cgraph, node_idx + j) != want) {
            return false;
        }
    }

    const ggml_tensor * get_rows = cgraph->nodes[node_idx + 0];
    const ggml_tensor * add      = cgraph->nodes[node_idx + 5];
    ggml_tensor *       top_k    = cgraph->nodes[node_idx + 6];

    const ggml_tensor * scores   = get_rows->src[0]; // [n_tps, n_blocks, n_stream]
    const ggml_tensor * cell_blk = get_rows->src[1]; // [n_kv, n_stream]
    const ggml_tensor * expanded = add->src[0];      // [n_kv, n_tps, n_stream]

    // raw mask: follow the reshape/cpy chain back to the materialized f16 input
    const ggml_tensor * mask = add->src[1];
    while (mask && (mask->op == GGML_OP_RESHAPE || mask->op == GGML_OP_CPY)) {
        mask = mask->src[0];
    }
    if (!mask || mask->type != GGML_TYPE_F16) {
        return false;
    }

    if (scores->type != GGML_TYPE_F32 || cell_blk->type != GGML_TYPE_I32 || top_k->type != GGML_TYPE_I32) {
        return false;
    }
    if (!ggml_is_contiguous(scores) || !ggml_is_contiguous(cell_blk) || !ggml_is_contiguous(mask) ||
        !ggml_is_contiguous(expanded) || !ggml_is_contiguous(top_k)) {
        return false;
    }

    const int64_t n_tps    = scores->ne[0];
    const int64_t n_blocks = scores->ne[1];
    const int64_t n_stream = scores->ne[2];
    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t width    = top_k->ne[0];

    // pin the indexer layout the kernel addressing assumes
    if (scores->ne[3] != 1 || cell_blk->ne[1] != n_stream || ggml_nrows(cell_blk) != n_stream ||
        ggml_nelements(mask) != n_kv * n_tps * n_stream || expanded->ne[0] != n_kv || expanded->ne[1] != n_tps ||
        expanded->ne[2] != n_stream || top_k->ne[1] != n_tps || top_k->ne[2] != n_stream || top_k->ne[3] != 1 ||
        n_blocks <= 0 || n_kv <= 0 || width <= 0 || width > n_kv) {
        return false;
    }

    // small k is faster with the unfused top_k path
    if (width <= 256) {
        return false;
    }

    match.scores   = scores;
    match.cell_blk = cell_blk;
    match.mask     = mask;
    match.top_k    = top_k;
    return true;
}

static __device__ __forceinline__ uint32_t topk_qsa_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

// one block per output row; radix-select the top width cells by gathered key
template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void topk_qsa_kernel(
        const float * __restrict__ scores,
        const int32_t * __restrict__ cell_blk,
        const ggml_half * __restrict__ mask,
        int32_t * __restrict__ top_k,
        int n_kv,
        int width,
        int n_tps,
        int n_blocks,
        int n_stream) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int t   = row % n_tps;
    const int s   = row / n_tps;

    const int32_t *  row_cell   = cell_blk + (size_t) s * n_kv;
    const ggml_half * row_mask  = mask + ((size_t) s * n_tps + t) * n_kv;
    const float *    row_scores = scores + (size_t) s * n_blocks * n_tps;
    int32_t *        row_out    = top_k + (size_t) row * width;

    __shared__ int histogram[NBINS];
    __shared__ int s_bucket;
    __shared__ int s_above;
    __shared__ int out_count;

    uint32_t prefix  = 0;
    int      desired = width;

    // four 8-bit passes, most significant bucket first
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        for (int i = tid; i < NBINS; i += BLOCK_SIZE) {
            histogram[i] = 0;
        }
        __syncthreads();

        const uint32_t hi_mask   = shift + RADIX_BITS >= 32 ? 0u : 0xFFFFFFFFu << (shift + RADIX_BITS);
        const uint32_t prefix_hi = prefix & hi_mask;

        for (int i = tid; i < n_kv; i += BLOCK_SIZE) {
            const int block = row_cell[i];
            const float v = row_scores[(size_t) block * n_tps + t] + __half2float(row_mask[i]);
            const uint32_t key = topk_qsa_float_to_ordered(v);
            if ((key & hi_mask) == prefix_hi) {
                atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
            }
        }
        __syncthreads();

        if (tid == 0) {
            int acc = 0;
            int bin = 0;
            for (int b = NBINS - 1; b >= 0; --b) {
                const int count = histogram[b];
                if (acc + count >= desired) {
                    bin = b;
                    break;
                }
                acc += count;
            }
            s_bucket = bin;
            s_above  = acc;
        }
        __syncthreads();

        prefix |= (uint32_t) s_bucket << shift;
        desired -= s_above;
        __syncthreads();
    }

    const uint32_t threshold = prefix;

    if (tid == 0) {
        out_count = 0;
    }
    __syncthreads();

    for (int i = tid; i < n_kv; i += BLOCK_SIZE) {
        const int block = row_cell[i];
        const float v = row_scores[(size_t) block * n_tps + t] + __half2float(row_mask[i]);
        if (topk_qsa_float_to_ordered(v) > threshold) {
            const int pos = atomicAdd(&out_count, 1);
            if (pos < width) {
                row_out[pos] = i;
            }
        }
    }
    __syncthreads();

    // ties fill the remaining slots; all strictly-greater cells are already placed
    for (int i = tid; i < n_kv; i += BLOCK_SIZE) {
        const int block = row_cell[i];
        const float v = row_scores[(size_t) block * n_tps + t] + __half2float(row_mask[i]);
        if (topk_qsa_float_to_ordered(v) == threshold) {
            const int pos = atomicAdd(&out_count, 1);
            if (pos < width) {
                row_out[pos] = i;
            }
        }
    }
}

void ggml_cuda_op_topk_qsa(ggml_backend_cuda_context & ctx, const ggml_tensor * scores, const ggml_tensor * cell_blk,
                           const ggml_tensor * mask, ggml_tensor * top_k) {
    GGML_ASSERT(scores->type == GGML_TYPE_F32 && ggml_is_contiguous(scores));
    GGML_ASSERT(cell_blk->type == GGML_TYPE_I32 && ggml_is_contiguous(cell_blk));
    GGML_ASSERT(mask->type == GGML_TYPE_F16 && ggml_is_contiguous(mask));
    GGML_ASSERT(top_k->type == GGML_TYPE_I32 && ggml_is_contiguous(top_k));

    const int n_tps    = scores->ne[0];
    const int n_blocks = scores->ne[1];
    const int n_stream = scores->ne[2];
    const int n_kv     = cell_blk->ne[0];
    const int width    = top_k->ne[0];
    const int nrows    = n_tps * n_stream;

    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;

    topk_qsa_kernel<BLOCK_SIZE, RADIX_BITS><<<nrows, BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) scores->data, (const int32_t *) cell_blk->data, (const ggml_half *) mask->data,
        (int32_t *) top_k->data, n_kv, width, n_tps, n_blocks, n_stream);
}
