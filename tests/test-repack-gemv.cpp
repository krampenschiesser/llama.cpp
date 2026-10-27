// Compares the x86 gemv and gemm of the 8-column interleaved q4_K and 4-bit layouts against the
// scalar references. Both kernels read the same packed bytes, so the repacked weights are
// synthesized here instead of going through the repack path.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

#define QK_K 256
#define QK8_0 32

// block_q4_Kx8 from ggml/src/ggml-cpu/repack.h, block_q8_K from ggml/src/ggml-common.h
struct block_q4_Kx8 {
    ggml_fp16_t d[8];
    ggml_fp16_t dmin[8];
    uint8_t     scales[96];
    uint8_t     qs[1024];
};

// block_q5_Kx8 from ggml/src/ggml-cpu/repack.h, the 5th quant bit lives in qh
struct block_q5_Kx8 {
    ggml_fp16_t d[8];
    ggml_fp16_t dmin[8];
    uint8_t     scales[96];
    uint8_t     qh[256];
    uint8_t     qs[1024];
};

// block_q6_Kx8 from ggml/src/ggml-cpu/repack.h, signed 8 bit scales and 6 bit quants
struct block_q6_Kx8 {
    ggml_fp16_t d[8];
    int8_t      scales[QK_K / 16 * 8];
    uint8_t     ql[QK_K / 2 * 8];
    uint8_t     qh[QK_K / 4 * 8];
};

struct block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};

// block_q8_Kx4 from ggml/src/ggml-cpu/repack.h, four q8_K rows interleaved for the gemm
struct block_q8_Kx4 {
    float   d[4];
    int8_t  qs[QK_K * 4];
    int16_t bsums[QK_K / 4];
};

// block_q4_0x8 / block_iq4_nlx8 / block_mxfp4x8 from ggml/src/ggml-cpu/repack.h, 8 x 32 quants
struct block_q4_0x8 {
    ggml_fp16_t d[8];
    uint8_t     qs[QK8_0 * 4];
};

struct block_iq4_nlx8 {
    ggml_fp16_t d[8];
    uint8_t     qs[QK8_0 * 4];
};

struct block_mxfp4x8 {
    uint8_t e[8];
    uint8_t qs[QK8_0 * 4];
};

// block_iq4_xsx8 from ggml/src/ggml-cpu/repack.h, 8 x 256 IQ4_XS quants, 6 bit scales
struct block_iq4_xsx8 {
    ggml_fp16_t d[8];
    uint8_t     scales[48];
    uint8_t     qs[QK_K * 4];
};

struct block_q8_0 {
    ggml_fp16_t d;
    int8_t      qs[QK8_0];
};

// block_q8_0x4 from ggml/src/ggml-cpu/repack.h, four q8_0 rows interleaved for the gemm
struct block_q8_0x4 {
    ggml_fp16_t d[4];
    int8_t      qs[QK8_0 * 4];
};

static_assert(sizeof(block_q4_Kx8) == 16 * sizeof(ggml_fp16_t) + 96 + 1024, "wrong block_q4_Kx8 size");
static_assert(sizeof(block_q5_Kx8) == 16 * sizeof(ggml_fp16_t) + 96 + 256 + 1024, "wrong block_q5_Kx8 size");
static_assert(sizeof(block_q6_Kx8) == 8 * sizeof(ggml_fp16_t) + QK_K / 16 * 8 + QK_K / 2 * 8 + QK_K / 4 * 8, "wrong block_q6_Kx8 size");
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + (QK_K / 16) * sizeof(int16_t), "wrong block_q8_K size");
static_assert(sizeof(block_q8_Kx4) == 4 * sizeof(float) + QK_K * 4 + (QK_K / 4) * sizeof(int16_t), "wrong block_q8_Kx4 size");
static_assert(sizeof(block_q4_0x8) == 8 * sizeof(ggml_fp16_t) + QK8_0 * 4, "wrong block_q4_0x8 size");
static_assert(sizeof(block_iq4_nlx8) == 8 * sizeof(ggml_fp16_t) + QK8_0 * 4, "wrong block_iq4_nlx8 size");
static_assert(sizeof(block_mxfp4x8) == 8 + QK8_0 * 4, "wrong block_mxfp4x8 size");
static_assert(sizeof(block_iq4_xsx8) == 8 * sizeof(ggml_fp16_t) + 48 + QK_K * 4, "wrong block_iq4_xsx8 size");
static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0, "wrong block_q8_0 size");
static_assert(sizeof(block_q8_0x4) == 4 * sizeof(ggml_fp16_t) + QK8_0 * 4, "wrong block_q8_0x4 size");

// gemv and gemm signatures shared by the 8x8 kernels and their generic references
typedef void (*gemv_fn)(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
typedef void (*gemm_fn)(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);

// repack.h is not on the test include path, so the exported symbols are declared here
extern "C" {
void ggml_gemv_q4_K_8x8_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemv_q4_K_8x8_q8_K_generic(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_q4_K_8x8_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_q4_K_8x8_q8_K_generic(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemv_q5_K_8x8_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemv_q5_K_8x8_q8_K_generic(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_q5_K_8x8_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_q5_K_8x8_q8_K_generic(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemv_q6_K_8x8_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemv_q6_K_8x8_q8_K_generic(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_q6_K_8x8_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_q6_K_8x8_q8_K_generic(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemv_q4_0_8x8_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemv_q4_0_8x8_q8_0_generic(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemv_iq4_nl_8x8_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemv_iq4_nl_8x8_q8_0_generic(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemv_iq4_xs_8x8_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemv_iq4_xs_8x8_q8_K_generic(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemv_mxfp4_8x8_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemv_mxfp4_8x8_q8_0_generic(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_q4_0_8x8_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_q4_0_8x8_q8_0_generic(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_iq4_nl_8x8_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_iq4_nl_8x8_q8_0_generic(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_iq4_xs_8x8_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_iq4_xs_8x8_q8_K_generic(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_mxfp4_8x8_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_mxfp4_8x8_q8_0_generic(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void quantize_row_q8_K_ref(const float * x, block_q8_K * y, int64_t k);
void quantize_row_q8_0_generic(const float * x, void * vy, int64_t k);
void ggml_quantize_mat_q8_K_4x8_generic(const float * x, void * vy, int64_t k);
void ggml_quantize_mat_q8_0_4x8_generic(const float * x, void * vy, int64_t k);
}

// failed shapes; main exits non-zero on any failures
static int n_failed = 0;

static float rand_f32(void) {
    return (float) rand() / (float) RAND_MAX;
}

// updates pointers to max_err and rms_err
static void compare_f32(const float * ref, const float * out, int64_t n, float * max_err, float * rms_err) {
    *max_err = 0.0f;
    double sum_sq_err = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const float err = fabsf(ref[i] - out[i]);
        if (err > *max_err) {
            *max_err = err;
        }
        sum_sq_err += (double) err * err;
    }
    *rms_err = (float) sqrt(sum_sq_err / n);
}

// the kernels under test and the reference read the same packed bytes, so any values are valid
template <typename block_t>
static void fill_packed(std::vector<block_t> & packed) {
    for (size_t i = 0; i < packed.size(); ++i) {
        for (int j = 0; j < 8; ++j) {
            packed[i].d[j] = ggml_fp32_to_fp16(0.25f + rand_f32());
        }
        if constexpr (std::is_same_v<block_t, block_iq4_xsx8>) {
            for (int j = 0; j < 48; ++j) {
                packed[i].scales[j] = (uint8_t) (rand() & 0xFF);
            }
            for (int j = 0; j < QK_K * 4; ++j) {
                packed[i].qs[j] = (uint8_t) (rand() & 0xFF);
            }
        } else if constexpr (std::is_same_v<block_t, block_q6_Kx8>) {
            for (int j = 0; j < QK_K / 16 * 8; ++j) {
                packed[i].scales[j] = (int8_t) ((rand() % 9) - 4);
            }
            for (int j = 0; j < QK_K / 2 * 8; ++j) {
                packed[i].ql[j] = (uint8_t) (rand() & 0xFF);
            }
            for (int j = 0; j < QK_K / 4 * 8; ++j) {
                packed[i].qh[j] = (uint8_t) (rand() & 0xFF);
            }
        } else {
            for (int j = 0; j < 8; ++j) {
                packed[i].dmin[j] = ggml_fp32_to_fp16(0.25f + rand_f32());
            }
            for (int j = 0; j < 96; ++j) {
                packed[i].scales[j] = (uint8_t) (rand() & 0xFF);
            }
            if constexpr (std::is_same_v<block_t, block_q5_Kx8>) {
                for (int j = 0; j < 256; ++j) {
                    packed[i].qh[j] = (uint8_t) (rand() & 0xFF);
                }
            }
            for (int j = 0; j < 1024; ++j) {
                packed[i].qs[j] = (uint8_t) (rand() & 0xFF);
            }
        }
    }
}

// some x86 builds fall back to the scalar single row reference for a gemv (e.g. the q5_K 8x8
// gemv without AVX-512 VNNI); that reference ignores nr, so only compare nr > 1 when the kernel
// actually writes the later rows
static bool gemv_writes_rows(gemv_fn kernel, int n, int nc, int nr, const void * vx, const void * vy) {
    const int64_t out_n = (int64_t) (nr - 1) * nr + nc;
    const float sentinel = -12345.0f;
    std::vector<float> out(out_n, sentinel);
    kernel(n, out.data(), nc, vx, vy, nr, nc);
    // a single row fallback writes only nc values, the last output value stays untouched
    return out[out_n - 1] != sentinel;
}

template <typename block_t>
static void test_shape(gemv_fn kernel, gemv_fn ref_fn, int n, int nc, int nr) {
    const int nb = n / QK_K;
    GGML_ASSERT(n % QK_K == 0);
    GGML_ASSERT(nc % 8 == 0);

    std::vector<block_t> packed((size_t) nb * (nc / 8));
    fill_packed(packed);

    std::vector<block_q8_K> act((size_t) nr * nb);
    std::vector<float> row(n);
    for (int y = 0; y < nr; ++y) {
        for (int i = 0; i < n; ++i) {
            row[i] = 5.0f * (rand_f32() - 0.5f);
        }
        quantize_row_q8_K_ref(row.data(), act.data() + (size_t) y * nb, n);
    }

    if (nr > 1 && !gemv_writes_rows(kernel, n, nc, nr, packed.data(), act.data())) {
        printf("SKIP n = %4d nc = %3d nr = %d : gemv only writes one row, the build uses the single row fallback\n", n, nc, nr);
        return;
    }

    // the kernel stores row y at s + y * nr, so the written range is [0, (nr - 1) * nr + nc)
    const int64_t out_n = (int64_t) (nr - 1) * nr + nc;
    const float sentinel = -12345.0f;

    // the reference ignores nr, so write its rows with the same layout as the arch kernel
    std::vector<float> ref(out_n, sentinel);
    std::vector<float> tmp(nc);
    for (int y = 0; y < nr; ++y) {
        ref_fn(n, tmp.data(), nc, packed.data(), act.data() + (size_t) y * nb, 1, nc);
        for (int i = 0; i < nc; ++i) {
            ref[(int64_t) y * nr + i] = tmp[i];
        }
    }

    std::vector<float> out(out_n, sentinel);
    kernel(n, out.data(), nc, packed.data(), act.data(), nr, nc);

    float max_err, rms_err, max_ref = 0.0f;
    compare_f32(ref.data(), out.data(), out_n, &max_err, &rms_err);
    for (int64_t i = 0; i < out_n; ++i) {
        if (fabsf(ref[i]) > max_ref) {
            max_ref = fabsf(ref[i]);
        }
    }

    // the reference converts fp16 with the ggml table, which is only filled after backend init
    if (max_ref == 0.0f) {
        ++n_failed;
        printf("FAIL n = %4d nc = %3d nr = %d : reference is all zero, CPU backend not initialized?\n", n, nc, nr);
        return;
    }

    const float tol  = 1e-4f * fmaxf(1.0f, max_ref);
    const bool  pass = max_err <= tol;
    if (!pass) {
        ++n_failed;
    }
    printf("%s n = %4d nc = %3d nr = %d : max_err = %.6g rms_err = %.6g max |ref| = %.6g\n",
           pass ? "OK  " : "FAIL", n, nc, nr, max_err, rms_err, max_ref);
}

// the kernels and the reference read the same packed bytes, so any values are valid; mxfp4 needs
// valid E8M0 exponents
template <typename block_t>
static void fill_b32_packed(std::vector<block_t> & packed) {
    for (size_t i = 0; i < packed.size(); ++i) {
        if constexpr (std::is_same_v<block_t, block_mxfp4x8>) {
            // E8M0 exponents 0 and 1 are the subnormal scales, 2 .. 132 the normal ones
            static const uint8_t e8m0[] = { 0, 1, 2, 3, 120, 124, 128, 132 };
            for (int j = 0; j < 8; ++j) {
                packed[i].e[j] = e8m0[(i + j) % 8];
            }
        } else {
            for (int j = 0; j < 8; ++j) {
                packed[i].d[j] = ggml_fp32_to_fp16(0.25f + rand_f32());
            }
        }
        for (int j = 0; j < QK8_0 * 4; ++j) {
            packed[i].qs[j] = (uint8_t) (rand() & 0xFF);
        }
    }
}

// gemv of the 8-column 4-bit layouts: nr activation rows of n quants against nc columns
template <typename block_t>
static void test_gemv_b32_shape(const char * fmt, gemv_fn kernel, gemv_fn ref_fn, int n, int nc, int nr) {
    const int nb = n / QK8_0;
    GGML_ASSERT(n % QK8_0 == 0);
    GGML_ASSERT(nc % 8 == 0);

    std::vector<block_t> packed((size_t) nb * (nc / 8));
    fill_b32_packed(packed);

    std::vector<block_q8_0> act((size_t) nr * nb);
    std::vector<float> row(n);
    for (int y = 0; y < nr; ++y) {
        for (int i = 0; i < n; ++i) {
            row[i] = 5.0f * (rand_f32() - 0.5f);
        }
        quantize_row_q8_0_generic(row.data(), act.data() + (size_t) y * nb, n);
    }

    // the kernel stores row y at s + y * nr, the reference ignores nr, so compare row by row
    const int64_t out_n = (int64_t) (nr - 1) * nr + nc;
    const float sentinel = -12345.0f;
    std::vector<float> ref(out_n, sentinel);
    std::vector<float> tmp(nc);
    for (int y = 0; y < nr; ++y) {
        ref_fn(n, tmp.data(), nc, packed.data(), act.data() + (size_t) y * nb, 1, nc);
        for (int i = 0; i < nc; ++i) {
            ref[(int64_t) y * nr + i] = tmp[i];
        }
    }

    std::vector<float> out(out_n, sentinel);
    kernel(n, out.data(), nc, packed.data(), act.data(), nr, nc);

    float max_err, rms_err, max_ref = 0.0f;
    compare_f32(ref.data(), out.data(), out_n, &max_err, &rms_err);
    for (int64_t i = 0; i < out_n; ++i) {
        if (fabsf(ref[i]) > max_ref) {
            max_ref = fabsf(ref[i]);
        }
    }

    // the reference reads the scales through the fp16 and E8M0 tables, filled by ggml_cpu_init
    if (max_ref == 0.0f) {
        ++n_failed;
        printf("FAIL %-6s n = %4d nc = %3d nr = %d : reference is all zero, CPU backend not initialized?\n",
               fmt, n, nc, nr);
        return;
    }

    const float tol  = 1e-4f * fmaxf(1.0f, max_ref);
    const bool  pass = max_err <= tol;
    if (!pass) {
        ++n_failed;
    }
    printf("%s %-6s n = %4d nc = %3d nr = %d : max_err = %.6g rms_err = %.6g max |ref| = %.6g\n",
           pass ? "OK  " : "FAIL", fmt, n, nc, nr, max_err, rms_err, max_ref);
}

// gemm over nr activation rows and nc columns, the layout requires nr % 4 == 0 and nc % 8 == 0
template <typename block_t>
static void test_gemm_shape(gemm_fn kernel, gemm_fn ref_fn, int n, int nc, int nr) {
    const int nb = n / QK_K;
    GGML_ASSERT(n % QK_K == 0);
    GGML_ASSERT(nc % 8 == 0);
    GGML_ASSERT(nr % 4 == 0);

    std::vector<block_t> packed((size_t) nb * (nc / 8));
    fill_packed(packed);

    // one block_q8_Kx4 per four activation rows and super block, quantized with the generic path
    std::vector<block_q8_Kx4> act((size_t) (nr / 4) * nb);
    std::vector<float> rows((size_t) 4 * n);
    for (int y = 0; y < nr / 4; ++y) {
        for (int i = 0; i < 4 * n; ++i) {
            rows[i] = 5.0f * (rand_f32() - 0.5f);
        }
        ggml_quantize_mat_q8_K_4x8_generic(rows.data(), act.data() + (size_t) y * nb, n);
    }

    const int64_t out_n = (int64_t) nr * nc;
    const float sentinel = -12345.0f;
    std::vector<float> ref(out_n, sentinel);
    std::vector<float> out(out_n, sentinel);

    ref_fn(n, ref.data(), nc, packed.data(), act.data(), nr, nc);
    kernel(n, out.data(), nc, packed.data(), act.data(), nr, nc);

    int64_t unwritten = 0;
    for (int64_t i = 0; i < out_n; ++i) {
        if (out[i] == sentinel) {
            ++unwritten;
        }
    }

    float max_err, rms_err, max_ref = 0.0f;
    compare_f32(ref.data(), out.data(), out_n, &max_err, &rms_err);
    for (int64_t i = 0; i < out_n; ++i) {
        if (fabsf(ref[i]) > max_ref) {
            max_ref = fabsf(ref[i]);
        }
    }

    // the reference converts fp16 with the ggml table, which is only filled after backend init
    if (max_ref == 0.0f) {
        ++n_failed;
        printf("FAIL gemm n = %4d nc = %3d nr = %2d : reference is all zero, CPU backend not initialized?\n", n, nc, nr);
        return;
    }
    if (unwritten != 0) {
        ++n_failed;
        printf("FAIL gemm n = %4d nc = %3d nr = %2d : %lld output values were not written\n",
               n, nc, nr, (long long) unwritten);
        return;
    }

    const float tol  = 1e-4f * fmaxf(1.0f, max_ref);
    const bool  pass = max_err <= tol;
    if (!pass) {
        ++n_failed;
    }
    printf("%s gemm n = %4d nc = %3d nr = %2d : max_err = %.6g rms_err = %.6g max |ref| = %.6g\n",
           pass ? "OK  " : "FAIL", n, nc, nr, max_err, rms_err, max_ref);
}

// gemm of the 8-column 4-bit layouts over nr activation rows and nc columns, nr % 4 == 0 and
// nc % 8 == 0; bs is the destination row stride in floats, so nc gives a dense [nr][nc] output
template <typename block_t>
static void test_gemm_b32_shape(const char * fmt, gemm_fn kernel, gemm_fn ref_fn, int n, int nc, int nr) {
    const int nb = n / QK8_0;
    GGML_ASSERT(n % QK8_0 == 0);
    GGML_ASSERT(nc % 8 == 0);
    GGML_ASSERT(nr % 4 == 0);

    std::vector<block_t> packed((size_t) nb * (nc / 8));
    fill_b32_packed(packed);

    // one block_q8_0x4 per four activation rows and 32-quant block
    std::vector<block_q8_0x4> act((size_t) (nr / 4) * nb);
    std::vector<float> rows((size_t) 4 * n);
    for (int y = 0; y < nr / 4; ++y) {
        for (int i = 0; i < 4 * n; ++i) {
            rows[i] = 5.0f * (rand_f32() - 0.5f);
        }
        ggml_quantize_mat_q8_0_4x8_generic(rows.data(), act.data() + (size_t) y * nb, n);
    }

    const int64_t out_n = (int64_t) nr * nc;
    const float sentinel = -12345.0f;
    std::vector<float> ref(out_n, sentinel);
    std::vector<float> out(out_n, sentinel);

    ref_fn(n, ref.data(), nc, packed.data(), act.data(), nr, nc);
    kernel(n, out.data(), nc, packed.data(), act.data(), nr, nc);

    int64_t unwritten = 0;
    for (int64_t i = 0; i < out_n; ++i) {
        if (out[i] == sentinel) {
            ++unwritten;
        }
    }

    float max_err, rms_err, max_ref = 0.0f;
    compare_f32(ref.data(), out.data(), out_n, &max_err, &rms_err);
    for (int64_t i = 0; i < out_n; ++i) {
        if (fabsf(ref[i]) > max_ref) {
            max_ref = fabsf(ref[i]);
        }
    }

    // the reference reads the scales through the fp16 and E8M0 tables, filled by ggml_cpu_init
    if (max_ref == 0.0f) {
        ++n_failed;
        printf("FAIL gemm %-6s n = %4d nc = %3d nr = %2d : reference is all zero, CPU backend not initialized?\n",
               fmt, n, nc, nr);
        return;
    }
    if (unwritten != 0) {
        ++n_failed;
        printf("FAIL gemm %-6s n = %4d nc = %3d nr = %2d : %lld output values were not written\n",
               fmt, n, nc, nr, (long long) unwritten);
        return;
    }

    const float tol  = 1e-4f * fmaxf(1.0f, max_ref);
    const bool  pass = max_err <= tol;
    if (!pass) {
        ++n_failed;
    }
    printf("%s gemm %-6s n = %4d nc = %3d nr = %2d : max_err = %.6g rms_err = %.6g max |ref| = %.6g\n",
           pass ? "OK  " : "FAIL", fmt, n, nc, nr, max_err, rms_err, max_ref);
}

int main(void) {
    srand(42);

    // the scalar reference reads d/dmin through the fp16 lookup table, which ggml_cpu_init fills
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        printf("failed to initialize the CPU backend\n");
        return 1;
    }

    const int ns[]  = { 256, 768, 1024 };  // nb = 1 (tail only), 3 (pair + tail), 4 (pairs only)
    const int ncs[] = { 8, 64 };

    for (size_t i = 0; i < sizeof(ns)/sizeof(ns[0]); ++i) {
        for (size_t j = 0; j < sizeof(ncs)/sizeof(ncs[0]); ++j) {
            for (int nr = 1; nr <= 3; ++nr) {
                test_shape<block_q4_Kx8>(ggml_gemv_q4_K_8x8_q8_K, ggml_gemv_q4_K_8x8_q8_K_generic, ns[i], ncs[j], nr);
                test_shape<block_q5_Kx8>(ggml_gemv_q5_K_8x8_q8_K, ggml_gemv_q5_K_8x8_q8_K_generic, ns[i], ncs[j], nr);
                test_shape<block_q6_Kx8>(ggml_gemv_q6_K_8x8_q8_K, ggml_gemv_q6_K_8x8_q8_K_generic, ns[i], ncs[j], nr);
                // the IQ4_XS kernels only exist in the x86 backend
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
                test_shape<block_iq4_xsx8>(ggml_gemv_iq4_xs_8x8_q8_K, ggml_gemv_iq4_xs_8x8_q8_K_generic, ns[i], ncs[j], nr);
#endif
            }
        }
    }

    // gemv of the 8-column 4-bit layouts, nc = 8 is one column group and nc = 24 is not a multiple of 16
    const int b32_ncs[] = { 8, 24, 64 };

    for (size_t i = 0; i < sizeof(ns)/sizeof(ns[0]); ++i) {
        for (size_t j = 0; j < sizeof(b32_ncs)/sizeof(b32_ncs[0]); ++j) {
            for (int nr = 1; nr <= 3; ++nr) {
                test_gemv_b32_shape<block_q4_0x8>  ("q4_0",   ggml_gemv_q4_0_8x8_q8_0,   ggml_gemv_q4_0_8x8_q8_0_generic,   ns[i], b32_ncs[j], nr);
                test_gemv_b32_shape<block_iq4_nlx8>("iq4_nl", ggml_gemv_iq4_nl_8x8_q8_0, ggml_gemv_iq4_nl_8x8_q8_0_generic, ns[i], b32_ncs[j], nr);
                test_gemv_b32_shape<block_mxfp4x8> ("mxfp4",  ggml_gemv_mxfp4_8x8_q8_0,  ggml_gemv_mxfp4_8x8_q8_0_generic,  ns[i], b32_ncs[j], nr);
            }
        }
    }

    // gemm: nr covers full 16 row groups (16, 20, 24, 28) and the 4, 8, 12 row tails, nc = 24 keeps
    // the last 8 columns on the 256 bit path
    const int gemm_ncs[] = { 16, 24, 32, 64 };
    const int gemm_nrs[] = { 4, 8, 12, 16, 20, 24, 28 };

    for (size_t i = 0; i < sizeof(ns)/sizeof(ns[0]); ++i) {
        for (size_t j = 0; j < sizeof(gemm_ncs)/sizeof(gemm_ncs[0]); ++j) {
            for (size_t k = 0; k < sizeof(gemm_nrs)/sizeof(gemm_nrs[0]); ++k) {
                test_gemm_shape<block_q4_Kx8>(ggml_gemm_q4_K_8x8_q8_K, ggml_gemm_q4_K_8x8_q8_K_generic, ns[i], gemm_ncs[j], gemm_nrs[k]);
                test_gemm_shape<block_q5_Kx8>(ggml_gemm_q5_K_8x8_q8_K, ggml_gemm_q5_K_8x8_q8_K_generic, ns[i], gemm_ncs[j], gemm_nrs[k]);
                test_gemm_shape<block_q6_Kx8>(ggml_gemm_q6_K_8x8_q8_K, ggml_gemm_q6_K_8x8_q8_K_generic, ns[i], gemm_ncs[j], gemm_nrs[k]);
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
                test_gemm_shape<block_iq4_xsx8>(ggml_gemm_iq4_xs_8x8_q8_K, ggml_gemm_iq4_xs_8x8_q8_K_generic, ns[i], gemm_ncs[j], gemm_nrs[k]);
#endif
            }
        }
    }

    // gemm of the 8-column 4-bit layouts against the scalar generic gemm, same nr/nc grid
    for (size_t i = 0; i < sizeof(ns)/sizeof(ns[0]); ++i) {
        for (size_t j = 0; j < sizeof(gemm_ncs)/sizeof(gemm_ncs[0]); ++j) {
            for (size_t k = 0; k < sizeof(gemm_nrs)/sizeof(gemm_nrs[0]); ++k) {
                test_gemm_b32_shape<block_q4_0x8>  ("q4_0",   ggml_gemm_q4_0_8x8_q8_0,   ggml_gemm_q4_0_8x8_q8_0_generic,   ns[i], gemm_ncs[j], gemm_nrs[k]);
                test_gemm_b32_shape<block_iq4_nlx8>("iq4_nl", ggml_gemm_iq4_nl_8x8_q8_0, ggml_gemm_iq4_nl_8x8_q8_0_generic, ns[i], gemm_ncs[j], gemm_nrs[k]);
                test_gemm_b32_shape<block_mxfp4x8> ("mxfp4",  ggml_gemm_mxfp4_8x8_q8_0,  ggml_gemm_mxfp4_8x8_q8_0_generic,  ns[i], gemm_ncs[j], gemm_nrs[k]);
            }
        }
    }

    ggml_backend_free(backend);

    if (n_failed > 0) {
        printf("%d shape(s) failed\n", n_failed);
        return 1;
    }
    printf("all shapes passed\n");
    return 0;
}
