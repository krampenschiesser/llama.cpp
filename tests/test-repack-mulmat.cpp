// Integration test for the CPU_REPACK weight buffer. The same quantized bytes are loaded into a
// CPU_REPACK tensor (the one-time repack runs in set_tensor, like the model load path) and into a
// plain CPU tensor, the two identical graphs are computed on the CPU backend and the outputs are
// compared. Covers MUL_MAT with 2D weights and MUL_MAT_ID with 3D expert weights.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

// ggml_backend_cpu_repack_buffer_type is declared in ggml/src/ggml-cpu/repack.h; only the symbol
// is needed here, like the kernel prototypes in test-repack-gemv.cpp
ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

// failed cases; main exits non-zero on any failures
static int n_failed = 0;

// deterministic PRNG, every case reproduces after a rerun
static uint32_t rng_state = 0x12345678u;

static float rand_f32(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float) (rng_state >> 8) * (1.0f / 8388608.0f) - 1.0f; // [-1, 1)
}

static void fill_random(std::vector<float> & v) {
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = rand_f32();
    }
}

static ggml_context * new_ctx(void) {
    struct ggml_init_params ip = { 2 * 1024 * 1024, nullptr, true };
    return ggml_init(ip);
}

// quantized bytes of nrows rows of k values; the same array is loaded into both weight tensors
static std::vector<uint8_t> quantize_weights(ggml_type type, int64_t nrows, int64_t k) {
    std::vector<float> src((size_t) (nrows * k));
    fill_random(src);

    std::vector<uint8_t> out(ggml_row_size(type, k) * (size_t) nrows);
    std::vector<float> imatrix;
    const float * im = nullptr;
    if (ggml_quantize_requires_imatrix(type)) {
        imatrix.assign(src.size(), 1.0f);
        im = imatrix.data();
    }
    ggml_quantize_chunk(type, src.data(), out.data(), 0, nrows, k, im);
    return out;
}

// true if this build selects a repack trait for the type, i.e. the CPU_REPACK tensor gets an extra
static bool has_repack_trait(ggml_type type) {
    ggml_context * ctx_w = new_ctx();
    ggml_tensor * w = ggml_new_tensor_2d(ctx_w, type, 256, 8);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, ggml_backend_cpu_repack_buffer_type());
    const bool has = buf_w != nullptr && w->extra != nullptr;
    if (buf_w != nullptr) {
        ggml_backend_buffer_free(buf_w);
    }
    ggml_free(ctx_w);
    return has;
}

// weight tensor in buft, everything else in the plain CPU buffer, both graphs are built the same
static bool compute_mul_mat(ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                            ggml_type type, int64_t k, int64_t n, int64_t nr,
                            const std::vector<uint8_t> & w_q, const std::vector<float> & act,
                            std::vector<float> & out) {
    ggml_context * ctx_w = new_ctx();
    ggml_context * ctx   = new_ctx();

    ggml_tensor * w   = ggml_new_tensor_2d(ctx_w, type, k, n);
    ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, nr);
    ggml_tensor * dst = ggml_mul_mat(ctx, w, a);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    ggml_backend_buffer_t buf   = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());

    bool ok = buf_w != nullptr && buf != nullptr;
    if (ok) {
        ggml_backend_tensor_set(w, w_q.data(), 0, ggml_nbytes(w));
        ggml_backend_tensor_set(a, act.data(), 0, ggml_nbytes(a));
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        out.resize(ggml_nelements(dst));
        ggml_backend_tensor_get(dst, out.data(), 0, ggml_nbytes(dst));
    }

    if (buf != nullptr) {
        ggml_backend_buffer_free(buf);
    }
    if (buf_w != nullptr) {
        ggml_backend_buffer_free(buf_w);
    }
    ggml_free(ctx);
    ggml_free(ctx_w);
    return ok;
}

// same as compute_mul_mat, but src0 is [k, n, n_experts], src1 is [k, n_ids, n_tokens] and ids is
// [n_ids, n_tokens] as in llama.cpp
// partial_write feeds the weights one expert plane at a time, like the MoE cache does
static bool compute_mul_mat_id(ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                               ggml_type type, int64_t k, int64_t n, int64_t n_experts,
                               int64_t n_ids, int64_t n_tokens,
                               const std::vector<uint8_t> & w_q, const std::vector<float> & act,
                               const std::vector<int32_t> & ids_data, std::vector<float> & out,
                               bool partial_write = false) {
    ggml_context * ctx_w = new_ctx();
    ggml_context * ctx   = new_ctx();

    ggml_tensor * w   = ggml_new_tensor_3d(ctx_w, type, k, n, n_experts);
    ggml_tensor * a   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_ids, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_ids, n_tokens);
    ggml_tensor * dst = ggml_mul_mat_id(ctx, w, a, ids);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    ggml_backend_buffer_t buf   = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());

    bool ok = buf_w != nullptr && buf != nullptr;
    if (ok) {
        if (partial_write) {
            // one plane is one expert: plain-layout bytes into a packed offset
            const size_t plane = ggml_row_size(type, k) * (size_t) n;
            GGML_ASSERT(w->nb[2] >= plane);
            for (int64_t e = 0; e < n_experts; ++e) {
                ggml_backend_tensor_set(w, w_q.data() + e * plane, (size_t) e * w->nb[2], plane);
            }
        } else {
            ggml_backend_tensor_set(w, w_q.data(), 0, ggml_nbytes(w));
        }
        ggml_backend_tensor_set(a, act.data(), 0, ggml_nbytes(a));
        ggml_backend_tensor_set(ids, ids_data.data(), 0, ggml_nbytes(ids));
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        out.resize(ggml_nelements(dst));
        ggml_backend_tensor_get(dst, out.data(), 0, ggml_nbytes(dst));
    }

    if (buf != nullptr) {
        ggml_backend_buffer_free(buf);
    }
    if (buf_w != nullptr) {
        ggml_backend_buffer_free(buf_w);
    }
    ggml_free(ctx);
    ggml_free(ctx_w);
    return ok;
}

// updates pointers to max_err and rms_err, prints the verdict
static void compare_and_report(const char * desc, const std::vector<float> & ref, const std::vector<float> & out) {
    float max_err = 0.0f;
    float max_ref = 0.0f;
    double sum_sq = 0.0;
    bool ok = ref.size() == out.size() && !ref.empty();

    const int64_t n = (int64_t) (ref.size() < out.size() ? ref.size() : out.size());
    for (int64_t i = 0; i < n; ++i) {
        const float err = fabsf(ref[i] - out[i]);
        if (!(err >= 0.0f)) {
            ok = false; // NaN
        }
        if (err > max_err) {
            max_err = err;
        }
        if (fabsf(ref[i]) > max_ref) {
            max_ref = fabsf(ref[i]);
        }
        sum_sq += (double) err * err;
    }
    const float rms_err = n > 0 ? (float) sqrt(sum_sq / n) : 0.0f;

    // both paths run the same int8 dot products, only the accumulation order and the activation
    // quantization block size can differ
    const float tol = 1e-4f * fmaxf(1.0f, max_ref);
    const bool  pass = ok && max_err <= tol;
    if (!pass) {
        ++n_failed;
    }
    printf("%s %s : max_err = %.6g rms_err = %.6g max |ref| = %.6g\n",
           pass ? "OK  " : "FAIL", desc, max_err, rms_err, max_ref);
}

static void test_mul_mat(ggml_backend_t backend, ggml_backend_buffer_type_t buft_repack,
                         ggml_type type, int64_t k, int64_t n, int64_t nr,
                         const std::vector<uint8_t> & w_q) {
    std::vector<float> act((size_t) (k * nr));
    fill_random(act);

    std::vector<float> ref;
    std::vector<float> out;
    const bool ok_ref = compute_mul_mat(backend, ggml_backend_cpu_buffer_type(), type, k, n, nr, w_q, act, ref);
    const bool ok_out = compute_mul_mat(backend, buft_repack,                    type, k, n, nr, w_q, act, out);

    char desc[160];
    snprintf(desc, sizeof(desc), "MUL_MAT     %-6s K = %4lld N = %4lld nr = %2lld",
             ggml_type_name(type), (long long) k, (long long) n, (long long) nr);
    if (!ok_ref || !ok_out) {
        ++n_failed;
        printf("FAIL %s : graph compute failed (plain %d, repack %d)\n", desc, (int) ok_ref, (int) ok_out);
        return;
    }
    compare_and_report(desc, ref, out);
}

static void test_mul_mat_id(ggml_backend_t backend, ggml_backend_buffer_type_t buft_repack,
                            ggml_type type, int64_t k, int64_t n, int64_t n_experts,
                            int64_t n_ids, int64_t n_tokens, const std::vector<uint8_t> & w_q,
                            const std::vector<int32_t> & ids_data, const char * tag) {
    std::vector<float> act((size_t) (k * n_ids * n_tokens));
    fill_random(act);

    std::vector<float> ref;
    std::vector<float> out;
    const bool ok_ref = compute_mul_mat_id(backend, ggml_backend_cpu_buffer_type(), type, k, n, n_experts,
                                           n_ids, n_tokens, w_q, act, ids_data, ref);
    const bool ok_out = compute_mul_mat_id(backend, buft_repack,                    type, k, n, n_experts,
                                           n_ids, n_tokens, w_q, act, ids_data, out);

    char desc[192];
    snprintf(desc, sizeof(desc), "MUL_MAT_ID  %-6s %s K = %4lld N = %4lld n_exp = %lld ids = %lld tok = %lld",
             ggml_type_name(type), tag, (long long) k, (long long) n, (long long) n_experts,
             (long long) n_ids, (long long) n_tokens);
    if (!ok_ref || !ok_out) {
        ++n_failed;
        printf("FAIL %s : graph compute failed (plain %d, repack %d)\n", desc, (int) ok_ref, (int) ok_out);
        return;
    }
    compare_and_report(desc, ref, out);
}

int main(void) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        printf("failed to initialize the CPU backend\n");
        return 1;
    }
    // the chunk split and the gemm/gemv row split depend on the thread count
    ggml_backend_cpu_set_n_threads(backend, 16);

    ggml_backend_buffer_type_t buft_repack = ggml_backend_cpu_repack_buffer_type();

    const ggml_type types[] = {
        GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q2_K,
        GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_S, GGML_TYPE_MXFP4, GGML_TYPE_Q8_0,
    };

    const int64_t k_dense[]     = { 256, 512, 1024 };
    const int64_t n_dense[]     = { 8, 24, 32, 64, 128 };
    const int64_t nr_dense[]    = { 1, 2, 3, 4, 5, 7, 8, 12, 16, 17 };

    // many 8-aligned chunks per thread, so a chunk start or size that is not 16-aligned is common
    const int64_t n_dense_big[]  = { 1024, 1408 };
    const int64_t nr_dense_big[] = { 1, 4, 5, 7, 8, 16, 17 };

    const int64_t k_id[]      = { 256, 1024 };
    const int64_t n_id[]      = { 64, 512, 1408 };
    const int64_t nexps_id[]  = { 4, 8 };
    const int64_t nids_id[]   = { 1, 2, 4 };
    const int64_t ntoks_id[]  = { 1, 3, 8, 16, 64 };

    for (size_t it = 0; it < sizeof(types)/sizeof(types[0]); ++it) {
        const ggml_type type = types[it];
        if (!has_repack_trait(type)) {
            printf("SKIP %s : no repack trait in this build\n", ggml_type_name(type));
            continue;
        }

        for (size_t ik = 0; ik < sizeof(k_dense)/sizeof(k_dense[0]); ++ik) {
            for (size_t in = 0; in < sizeof(n_dense)/sizeof(n_dense[0]); ++in) {
                const std::vector<uint8_t> w_q = quantize_weights(type, n_dense[in], k_dense[ik]);
                for (size_t ir = 0; ir < sizeof(nr_dense)/sizeof(nr_dense[0]); ++ir) {
                    test_mul_mat(backend, buft_repack, type, k_dense[ik], n_dense[in], nr_dense[ir], w_q);
                }
            }
        }

        for (size_t in = 0; in < sizeof(n_dense_big)/sizeof(n_dense_big[0]); ++in) {
            const std::vector<uint8_t> w_q = quantize_weights(type, n_dense_big[in], 1024);
            for (size_t ir = 0; ir < sizeof(nr_dense_big)/sizeof(nr_dense_big[0]); ++ir) {
                test_mul_mat(backend, buft_repack, type, 1024, n_dense_big[in], nr_dense_big[ir], w_q);
            }
        }

        for (size_t ik = 0; ik < sizeof(k_id)/sizeof(k_id[0]); ++ik) {
            for (size_t in = 0; in < sizeof(n_id)/sizeof(n_id[0]); ++in) {
                for (size_t ie = 0; ie < sizeof(nexps_id)/sizeof(nexps_id[0]); ++ie) {
                    const int64_t k   = k_id[ik];
                    const int64_t n   = n_id[in];
                    const int64_t n_e = nexps_id[ie];
                    const std::vector<uint8_t> w_q = quantize_weights(type, n * n_e, k);

                    for (size_t ii = 0; ii < sizeof(nids_id)/sizeof(nids_id[0]); ++ii) {
                        for (size_t itk = 0; itk < sizeof(ntoks_id)/sizeof(ntoks_id[0]); ++itk) {
                            const int64_t n_ids    = nids_id[ii];
                            const int64_t n_tokens = ntoks_id[itk];
                            std::vector<int32_t> ids_data((size_t) (n_ids * n_tokens));
                            for (size_t i = 0; i < ids_data.size(); ++i) {
                                ids_data[i] = (int32_t) (rng_state % (uint32_t) n_e);
                                rng_state = rng_state * 1664525u + 1013904223u;
                            }
                            test_mul_mat_id(backend, buft_repack, type, k, n, n_e, n_ids, n_tokens, w_q, ids_data, "rand");
                        }
                    }
                }
            }
        }

        // expert planes written one at a time, like the MoE cache loads them
        {
            const int64_t k = 1024, n = 512, n_e = 8;
            const std::vector<uint8_t> w_q = quantize_weights(type, n * n_e, k);

            for (int64_t n_ids : { 1, 4 }) {
                std::vector<int32_t> ids_data((size_t) (n_ids * 8));
                for (size_t i = 0; i < ids_data.size(); ++i) {
                    ids_data[i] = (int32_t) (rng_state % (uint32_t) n_e);
                    rng_state = rng_state * 1664525u + 1013904223u;
                }

                std::vector<float> act((size_t) (k * n_ids * 8));
                fill_random(act);
                std::vector<float> ref;
                std::vector<float> out;
                const bool ok_ref = compute_mul_mat_id(backend, ggml_backend_cpu_buffer_type(), type, k, n, n_e,
                                                       n_ids, 8, w_q, act, ids_data, ref);
                const bool ok_out = compute_mul_mat_id(backend, buft_repack, type, k, n, n_e,
                                                       n_ids, 8, w_q, act, ids_data, out, true);

                char desc[192];
                snprintf(desc, sizeof(desc), "MUL_MAT_ID  %-6s planes K = %4lld N = %4lld n_exp = %lld ids = %lld tok = 8",
                         ggml_type_name(type), (long long) k, (long long) n, (long long) n_e, (long long) n_ids);
                if (!ok_ref || !ok_out) {
                    ++n_failed;
                    printf("FAIL %s : graph compute failed (plain %d, repack %d)\n", desc, (int) ok_ref, (int) ok_out);
                } else {
                    compare_and_report(desc, ref, out);
                }
            }
        }

        // routing corners: one expert takes every row, other experts take none, and a -1 id leaves
        // the output row zeroed
        {
            const int64_t k = 1024, n = 512, n_e = 4;
            const std::vector<uint8_t> w_q = quantize_weights(type, n * n_e, k);

            std::vector<int32_t> ids_one((size_t) (2 * 8), (int32_t) (n_e - 1));
            test_mul_mat_id(backend, buft_repack, type, k, n, n_e, 2, 8, w_q, ids_one, "one-exp");

            // one expert takes 256 rows: several 64-row weight passes
            std::vector<int32_t> ids_one_big((size_t) (2 * 128), (int32_t) (n_e - 1));
            test_mul_mat_id(backend, buft_repack, type, k, n, n_e, 2, 128, w_q, ids_one_big, "one-exp-big");

            std::vector<int32_t> ids_skip((size_t) (2 * 3));
            for (size_t i = 0; i < ids_skip.size(); ++i) {
                ids_skip[i] = (i % 2 == 0) ? -1 : (int32_t) (i % n_e);
            }
            test_mul_mat_id(backend, buft_repack, type, k, n, n_e, 2, 3, w_q, ids_skip, "-1   ");

            std::vector<int32_t> ids_none((size_t) (1 * 4), -1);
            test_mul_mat_id(backend, buft_repack, type, k, n, n_e, 1, 4, w_q, ids_none, "-1all");
        }
    }

    ggml_backend_free(backend);

    if (n_failed > 0) {
        printf("%d case(s) failed\n", n_failed);
        return 1;
    }
    printf("all cases passed\n");
    return 0;
}
