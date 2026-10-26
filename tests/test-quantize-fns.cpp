// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <assert.h>
#include <algorithm>
#include <cmath>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_BINARY = 0.025f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_FP4 = 0.0030f;
constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_IQ2_XXS = 0.05f;
constexpr float MAX_DOT_PRODUCT_ERROR_FP4 = 0.03f;
constexpr float MAX_DOT_PRODUCT_ERROR_BINARY = 0.40f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

static const char* RESULT_STR[] = {"ok", "FAILED"};


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst, float amplitude = 2.0f) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + amplitude*cosf(i + offset);
    }
}

// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

// Total quantization error on test data
static float total_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);

    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);
    return array_rmse(test_data, tmp_out.data(), test_size);
}

// Total quantization error on test data
static float reference_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    // FIXME: why is done twice?
    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);

    qfns->from_float_ref(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

static float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

// Total dot product error
static float dot_product_error(const ggml_type_traits_cpu * qfns_cpu, ggml_type src0_type, size_t test_size,
                               const float * test_data1, const float * test_data2,
                               const float * test_data3, const float * test_data4,
                               const int nrc) {
    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);
    const size_t pad  = 64;
    const size_t bx   = ggml_row_size(src0_type, test_size) + pad;
    const size_t by   = ggml_row_size(qfns_cpu->vec_dot_type, test_size) + pad;

    std::vector<uint8_t> tmp_q1(bx * nrc);
    std::vector<uint8_t> tmp_q2(by * nrc);
    const std::vector<float> imatrix(test_size, 1.0f);

    // IQ types have no from_float, quantize them through ggml_quantize_chunk instead
    const auto quantize_src0 = [&](const float * src, uint8_t * dst) {
        if (qfns_cpu->from_float) {
            qfns_cpu->from_float(src, dst, test_size);
        } else {
            ggml_quantize_chunk(src0_type, src, dst, 0, 1, test_size, imatrix.data());
        }
    };

    quantize_src0(test_data1, tmp_q1.data());
    vdot->from_float(test_data2, tmp_q2.data(), test_size);

    if (nrc == 1) {
        float result = INFINITY;
        qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

        const float dot_ref = dot_product(test_data1, test_data2, test_size);
        return fabsf(result - dot_ref) / test_size;
    }

    // nrc == 2: kernel computes a 2x2 dot product matrix
    // Output layout: s[0]=dot(vx0,vy0), s[1]=dot(vx1,vy0), s[bs]=dot(vx0,vy1), s[bs+1]=dot(vx1,vy1)
    // row and output strides are padded, same as in the mul_mat path
    quantize_src0(test_data3, tmp_q1.data() + bx);
    vdot->from_float(test_data4, tmp_q2.data() + by, test_size);

    const size_t bs = 16;
    std::vector<float> result(bs + 2, INFINITY);
    qfns_cpu->vec_dot(test_size, result.data(), bs, tmp_q1.data(), bx, tmp_q2.data(), by, 2);

    const float ref00 = dot_product(test_data1, test_data2, test_size);
    const float ref10 = dot_product(test_data3, test_data2, test_size);
    const float ref01 = dot_product(test_data1, test_data4, test_size);
    const float ref11 = dot_product(test_data3, test_data4, test_size);

    const auto err = [test_size](float val, float ref) {
        const float e = fabsf(val - ref) / test_size;
        return std::isfinite(e) ? e : INFINITY;
    };

    return std::max({err(result[0], ref00), err(result[1], ref10), err(result[bs], ref01), err(result[bs + 1], ref11)});
}

static int test_vec_dot_f32(bool verbose) {
    const auto * f32 = ggml_get_type_traits_cpu(GGML_TYPE_F32);
    int num_failed = 0;
    for (int n : {1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 33, 63, 67, 127, 129, 193, 255, 1023}) {
        std::vector<float> a(n);
        std::vector<float> b(n);
        generate_data(0.0, n, a.data());
        generate_data(1.0, n, b.data());

        float result = 0.0f;
        f32->vec_dot(n, &result, 0, a.data(), 0, b.data(), 0, 1);
        const float ref = dot_product(a.data(), b.data(), n);
        const float error = fabsf(result - ref) / n;

        const bool failed = !(error < MAX_QUANTIZATION_REFERENCE_ERROR);
        num_failed += failed;
        if (failed || verbose) {
            printf(" f32 vec_dot n=%4d:                 %s (ref=%f got=%f err=%f)\n",
                   n, RESULT_STR[failed], ref, result, error);
        }
    }
    return num_failed;
}

static int test_vec_dot_q(bool verbose) {
    int num_failed = 0;

    const size_t test_size = 32 * 128;

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);
    std::vector<float> test_data3(test_size);
    std::vector<float> test_data4(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());
    generate_data(3.0, test_data3.size(), test_data3.data(), 1.0f);
    generate_data(4.0, test_data4.size(), test_data4.data(), 1.5f);

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        // deprecated - skip
        if (qfns->blck_size == 0) {
            continue;
        }

        const ggml_type ei = (ggml_type)i;

        printf("Testing %s\n", ggml_type_name((ggml_type) i));
        ggml_quantize_init(ei);

        if (qfns_cpu->from_float && qfns->to_float) {
            const float total_error = total_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            const float max_quantization_error =
                type == GGML_TYPE_Q1_0    ? MAX_QUANTIZATION_TOTAL_ERROR_BINARY :
                type == GGML_TYPE_TQ1_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_TQ2_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_0    ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_K    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_IQ2_S   ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_Q3_K    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_S   ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_XXS ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS :
                type == GGML_TYPE_NVFP4   ? MAX_QUANTIZATION_TOTAL_ERROR_FP4 : MAX_QUANTIZATION_TOTAL_ERROR;
            bool failed = !(total_error < max_quantization_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
            }
        }

        if (qfns_cpu->vec_dot) {
            const float vec_dot_error = dot_product_error(qfns_cpu, type, test_size, test_data.data(), test_data2.data(), nullptr, nullptr, 1);
            const float max_allowed_error = type == GGML_TYPE_IQ2_XXS
                ? MAX_DOT_PRODUCT_ERROR_IQ2_XXS
                : type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ||
                type == GGML_TYPE_IQ2_XS ||
                type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S
                ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                : type == GGML_TYPE_Q1_0
                ? MAX_DOT_PRODUCT_ERROR_BINARY
                : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0 || type == GGML_TYPE_Q2_0
                ? MAX_DOT_PRODUCT_ERROR_TERNARY
                : type == GGML_TYPE_NVFP4
                ? MAX_DOT_PRODUCT_ERROR_FP4
                : MAX_DOT_PRODUCT_ERROR;
            bool failed = !(vec_dot_error < max_allowed_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }

            // Test nrc=2 path for types that support it
            if (qfns_cpu->nrows == 2) {
                const float vec_dot_error_nrc2 = dot_product_error(qfns_cpu, type, test_size, test_data.data(), test_data2.data(), test_data3.data(), test_data4.data(), 2);
                failed = !(vec_dot_error_nrc2 < max_allowed_error);
                num_failed += failed;
                if (failed || verbose) {
                    printf("%5s dot product error (nrc=2):    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error_nrc2);
                }
            }
        }
    }

    return num_failed;
}

// In every group, all values with importance are equal, and the max (0) has none.
// The scale search in make_qkx3_quants then fits min == max and passes inf/nan to nearest_int (#29804).
static int test_quantize_imatrix_degenerate(bool verbose) {
    const int64_t n = 256;
    std::vector<float> x(n);
    std::vector<float> imatrix(n);
    std::vector<float> out(n);
    int num_failed = 0;

    printf("Testing degenerate imatrix:\n");
    for (ggml_type type : {GGML_TYPE_Q2_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1}) {
        const int64_t group = type == GGML_TYPE_Q2_K ? 16 : 32;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t g = i / group;
            const int64_t p = i % group;
            const bool important = p % 3 == 1;
            x[i] = important ? -0.02f*(g + 1) : (p % 2 ? -1.0f : 0.0f);
            imatrix[i] = important ? 1.0f : 0.0f;
        }
        printf("  - %s\n", ggml_type_name(type));

        std::vector<uint8_t> q(ggml_row_size(type, n));
        ggml_quantize_init(type);
        ggml_quantize_chunk(type, x.data(), q.data(), 0, 1, n, imatrix.data());
        ggml_get_type_traits(type)->to_float(q.data(), out.data(), n);

        const bool failed = !std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); });
        num_failed += failed;
        if (failed || verbose) {
            printf("%5s imatrix degenerate groups:      %s\n", ggml_type_name(type), RESULT_STR[failed]);
        }
    }

    return num_failed;
}


// Compare the CPU (ISA-specialized) dequantizer against the generic reference
static int test_dequantize_q(bool verbose) {
    const int64_t test_size = 4096;

    std::vector<float> test_data(test_size);
    generate_data(0, test_size, test_data.data());

    std::vector<float> imatrix(test_size, 1.0f);
    std::vector<float> out_ref(test_size + 8);
    std::vector<float> out_cpu(test_size + 8);

    printf("%s: dequantize_row_q\n", __func__);

    int num_failed = 0;

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const ggml_type type = (ggml_type) i;

        if (ggml_type_name(type) == NULL) {
            continue;
        }

        const struct ggml_type_traits     * qfns     = ggml_get_type_traits(type);
        const struct ggml_type_traits_cpu * qfns_cpu = ggml_get_type_traits_cpu(type);

        if (qfns->to_float == NULL ||
                type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || type == GGML_TYPE_F32) {
            continue;
        }

        const size_t row_size = ggml_row_size(type, test_size);
        std::vector<uint8_t> tmp_q(row_size, 0);

        ggml_quantize_init(type);
        ggml_quantize_chunk(type, test_data.data(), tmp_q.data(), 0, 1, test_size, imatrix.data());

        qfns->to_float(tmp_q.data(), out_ref.data(), test_size);

        // NULL means no ISA kernel for this type in this build: fall back to the reference
        const ggml_to_float_t to_float = qfns_cpu->to_float ? qfns_cpu->to_float : qfns->to_float;

        bool  failed   = false;
        float max_diff = 0.0f;

        // unaligned destinations must work too (GET_ROWS rows are not 64B aligned)
        for (int off : {0, 1, 3, 7}) {
            std::fill(out_cpu.begin(), out_cpu.end(), 0.0f);
            to_float(tmp_q.data(), out_cpu.data() + off, test_size);

            for (int64_t j = 0; j < test_size; j++) {
                const float a    = out_cpu[off + j];
                const float b    = out_ref[j];
                const float diff = fabsf(a - b);
                if (std::isnan(diff)) {
                    failed = true;
                    continue;
                }
                max_diff = std::max(max_diff, diff);
                if (diff > 1e-6f*fmaxf(1.0f, fabsf(b))) {
                    failed = true;
                }
            }
        }

        num_failed += failed;
        if (failed || verbose) {
            printf("%5s dequantize: %s (max diff %g)\n", ggml_type_name(type), RESULT_STR[failed], max_diff);
        }
    }

    return num_failed;
}

// The block layouts of ggml-common.h and the reference quantizers of ggml-quants.h
// are internal headers, so the test declares the small part it needs here.
struct block_q8_0 {
    ggml_fp16_t d;
    int8_t      qs[32];
};
static_assert(sizeof(block_q8_0) == 2 + 32, "wrong q8_0 block size/padding");

struct block_q8_1 {
    ggml_fp16_t d;
    ggml_fp16_t s;
    int8_t      qs[32];
};
static_assert(sizeof(block_q8_1) == 2 + 2 + 32, "wrong q8_1 block size/padding");

struct block_q8_K {
    float   d;
    int8_t  qs[256];
    int16_t bsums[16];
};
static_assert(sizeof(block_q8_K) == 4 + 256 + 32, "wrong q8_K block size/padding");

// 4-row blocks of the repack path: the quants of the 4 rows are interleaved in groups of 8
struct block_q8_0x4 {
    ggml_fp16_t d[4];
    int8_t      qs[4*32];
};
static_assert(sizeof(block_q8_0x4) == 4*2 + 4*32, "wrong q8_0x4 block size/padding");

struct block_q8_Kx4 {
    float   d[4];
    int8_t  qs[4*256];
    int16_t bsums[64];
};
static_assert(sizeof(block_q8_Kx4) == 4*4 + 4*256 + 64*2, "wrong q8_Kx4 block size/padding");

extern "C" {
void quantize_row_q8_0_ref(const float * x, block_q8_0 * y, int64_t k);
void quantize_row_q8_1_ref(const float * x, block_q8_1 * y, int64_t k);
void quantize_row_q8_K_ref(const float * x, block_q8_K * y, int64_t k);
void ggml_quantize_mat_q8_0_4x8(const float * x, void * vy, int64_t k);
void ggml_quantize_mat_q8_K_4x8(const float * x, void * vy, int64_t k);
void ggml_vec_dot_q8_0_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
}

// A code may differ by 1 where the value lands exactly on a rounding tie: the ISA
// kernels round half to even, the reference uses roundf (half away from zero).
static int count_quant_mismatches(const int8_t * a, const int8_t * b, size_t n) {
    int n_bad = 0;
    for (size_t i = 0; i < n; i++) {
        const int diff = (int) a[i] - (int) b[i];
        n_bad += diff > 1 || diff < -1;
    }
    return n_bad;
}

static float scale_rel_diff(float a, float b) {
    return fabsf(a - b)/fmaxf(1.0f, fabsf(b));
}

// Compare the CPU (ISA-specialized) quantizer against the reference
static int test_quantize_q(bool verbose) {
    const int64_t test_size = 4096;

    std::vector<float> test_data(test_size);
    generate_data(0, test_size, test_data.data());

    printf("%s: quantize_row_q\n", __func__);

    int num_failed = 0;

    {
        const int64_t nb = test_size / 32;

        std::vector<block_q8_0> q_cpu(nb);
        std::vector<block_q8_0> q_ref(nb);

        ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(test_data.data(), q_cpu.data(), test_size);
        quantize_row_q8_0_ref(test_data.data(), q_ref.data(), test_size);

        int   n_bad  = 0;
        float d_diff = 0.0f;

        for (int64_t i = 0; i < nb; i++) {
            n_bad  += count_quant_mismatches(q_cpu[i].qs, q_ref[i].qs, 32);
            d_diff  = std::max(d_diff, scale_rel_diff(ggml_fp16_to_fp32(q_cpu[i].d), ggml_fp16_to_fp32(q_ref[i].d)));
        }

        const bool failed = n_bad != 0 || d_diff > 1e-6f;
        num_failed += failed;
        if (failed || verbose) {
            printf("%5s quantize: %s (%d codes off by more than 1, max d diff %g)\n",
                   ggml_type_name(GGML_TYPE_Q8_0), RESULT_STR[failed], n_bad, d_diff);
        }
    }

    {
        const int64_t nb = test_size / 32;

        std::vector<block_q8_1> q_cpu(nb);
        std::vector<block_q8_1> q_ref(nb);

        ggml_get_type_traits_cpu(GGML_TYPE_Q8_1)->from_float(test_data.data(), q_cpu.data(), test_size);
        quantize_row_q8_1_ref(test_data.data(), q_ref.data(), test_size);

        int   n_bad    = 0;
        int   n_s_bad  = 0;
        float d_diff   = 0.0f;

        for (int64_t i = 0; i < nb; i++) {
            int n_ties = 0;
            for (int j = 0; j < 32; j++) {
                const int diff = (int) q_cpu[i].qs[j] - (int) q_ref[i].qs[j];
                n_bad  += diff > 1 || diff < -1;
                n_ties += diff == 1 || diff == -1;
            }
            const float d_cpu = ggml_fp16_to_fp32(q_cpu[i].d);
            const float d_ref = ggml_fp16_to_fp32(q_ref[i].d);
            d_diff = std::max(d_diff, scale_rel_diff(d_cpu, d_ref));

            // s = d * sum(qs), so every rounding tie may shift it by |d|
            const float s_cpu = ggml_fp16_to_fp32(q_cpu[i].s);
            const float s_ref = ggml_fp16_to_fp32(q_ref[i].s);
            n_s_bad += fabsf(s_cpu - s_ref) > 1e-6f*fmaxf(1.0f, fabsf(s_ref)) + fabsf(d_ref)*n_ties;
        }

        const bool failed = n_bad != 0 || n_s_bad != 0 || d_diff > 1e-6f;
        num_failed += failed;
        if (failed || verbose) {
            printf("%5s quantize: %s (%d codes off by more than 1, %d s mismatches, max d diff %g)\n",
                   ggml_type_name(GGML_TYPE_Q8_1), RESULT_STR[failed], n_bad, n_s_bad, d_diff);
        }
    }

    {
        const int64_t nb = test_size / 256;

        std::vector<block_q8_K> q_cpu(nb);
        std::vector<block_q8_K> q_ref(nb);

        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(test_data.data(), q_cpu.data(), test_size);
        quantize_row_q8_K_ref(test_data.data(), q_ref.data(), test_size);

        int   n_bad       = 0;
        int   n_bsums_bad = 0;
        float d_diff      = 0.0f;

        for (int64_t i = 0; i < nb; i++) {
            n_bad += count_quant_mismatches(q_cpu[i].qs, q_ref[i].qs, 256);
            d_diff = std::max(d_diff, scale_rel_diff(q_cpu[i].d, q_ref[i].d));

            for (int j = 0; j < 16; j++) {
                int sum = 0;
                for (int k = 0; k < 16; k++) {
                    sum += q_cpu[i].qs[16*j + k];
                }
                n_bsums_bad += sum != q_cpu[i].bsums[j];
            }
        }

        const bool failed = n_bad != 0 || n_bsums_bad != 0 || d_diff > 1e-6f;
        num_failed += failed;
        if (failed || verbose) {
            printf("%5s quantize: %s (%d codes off by more than 1, %d bsums mismatches, max d diff %g)\n",
                   ggml_type_name(GGML_TYPE_Q8_K), RESULT_STR[failed], n_bad, n_bsums_bad, d_diff);
        }
    }

    // The repack path quantizes 4 rows at once and interleaves 8 columns of each row
    {
        const int64_t ns[] = { 256, 1024, 4096 };

        for (size_t it = 0; it < sizeof(ns)/sizeof(ns[0]); it++) {
            const int64_t n = ns[it];
            const int64_t nb = n / 32;

            std::vector<float> data_in(4*n);
            for (int r = 0; r < 4; r++) {
                generate_data(0.5f*r, n, data_in.data() + r*n);
            }

            std::vector<block_q8_0x4> q_cpu(nb);
            std::vector<block_q8_0>   q_ref(4*nb);

            ggml_quantize_mat_q8_0_4x8(data_in.data(), q_cpu.data(), n);
            for (int r = 0; r < 4; r++) {
                quantize_row_q8_0_ref(data_in.data() + r*n, q_ref.data() + r*nb, n);
            }

            int n_bad = 0;
            int n_d_bad = 0;

            for (int64_t i = 0; i < nb; i++) {
                for (int r = 0; r < 4; r++) {
                    const block_q8_0 & ref = q_ref[r*nb + i];
                    n_d_bad += ggml_fp16_to_fp32(q_cpu[i].d[r]) != ggml_fp16_to_fp32(ref.d);

                    for (int m = 0; m < 32; m++) {
                        // chunk of 8 columns of row r
                        const int8_t q = q_cpu[i].qs[32*(m/8) + 8*r + (m%8)];
                        const int diff = (int) q - (int) ref.qs[m];
                        n_bad += diff > 1 || diff < -1;
                    }
                }
            }

            const bool failed = n_bad != 0 || n_d_bad != 0;
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s quantize 4x8 n=%4d: %s (%d codes off by more than 1, %d d mismatches)\n",
                       ggml_type_name(GGML_TYPE_Q8_0), (int) n, RESULT_STR[failed], n_bad, n_d_bad);
            }
        }

        for (size_t it = 0; it < sizeof(ns)/sizeof(ns[0]); it++) {
            const int64_t n = ns[it];
            const int64_t nb = n / 256;

            std::vector<float> data_in(4*n);
            for (int r = 0; r < 4; r++) {
                generate_data(0.5f*r, n, data_in.data() + r*n);
            }

            std::vector<block_q8_Kx4> q_cpu(nb);
            std::vector<block_q8_K>   q_ref(4*nb);

            ggml_quantize_mat_q8_K_4x8(data_in.data(), q_cpu.data(), n);
            for (int r = 0; r < 4; r++) {
                quantize_row_q8_K_ref(data_in.data() + r*n, q_ref.data() + r*nb, n);
            }

            int n_bad = 0;
            int n_d_bad = 0;
            int n_bsums_bad = 0;

            for (int64_t i = 0; i < nb; i++) {
                for (int r = 0; r < 4; r++) {
                    const block_q8_K & ref = q_ref[r*nb + i];
                    n_d_bad += q_cpu[i].d[r] != ref.d;

                    int n_row_diff = 0;
                    for (int m = 0; m < 256; m++) {
                        const int8_t q = q_cpu[i].qs[32*(m/8) + 8*r + (m%8)];
                        const int diff = (int) q - (int) ref.qs[m];
                        n_bad += diff > 1 || diff < -1;
                        n_row_diff += diff != 0;
                    }

                    for (int g = 0; g < 16; g++) {
                        int sum = 0;
                        for (int j = 0; j < 16; j++) {
                            sum += q_cpu[i].qs[32*(2*g + j/8) + 8*r + (j%8)];
                        }
                        const int16_t bsums = q_cpu[i].bsums[16*(g/4) + 4*r + (g%4)];
                        n_bsums_bad += sum != bsums;
                        // the reference bsums only match when no code differs
                        n_bsums_bad += n_row_diff == 0 && bsums != ref.bsums[g];
                    }
                }
            }

            const bool failed = n_bad != 0 || n_d_bad != 0 || n_bsums_bad != 0;
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s quantize 4x8 n=%4d: %s (%d codes off by more than 1, %d d mismatches, %d bsums mismatches)\n",
                       ggml_type_name(GGML_TYPE_Q8_K), (int) n, RESULT_STR[failed], n_bad, n_d_bad, n_bsums_bad);
            }
        }
    }

    return num_failed;
}

// q8_0 x q8_0 vec_dot for even and odd numbers of blocks: an odd nb must not drop the
// last block, which the AVX512 branch leaves to the shared scalar tail
static int test_vec_dot_q8_0_q8_0(bool verbose) {
    printf("%s: vec_dot q8_0 x q8_0\n", __func__);

    int num_failed = 0;

    const int ns[] = { 32, 64, 96, 128, 160, 224, 256 };

    for (size_t c = 0; c < sizeof(ns)/sizeof(ns[0]); c++) {
        const int n  = ns[c];
        const int nb = n / 32;

        std::vector<float> a(n);
        std::vector<float> b(n);
        generate_data(0.0, n, a.data());
        generate_data(1.0, n, b.data());

        std::vector<block_q8_0> qa(nb);
        std::vector<block_q8_0> qb(nb);
        quantize_row_q8_0_ref(a.data(), qa.data(), n);
        quantize_row_q8_0_ref(b.data(), qb.data(), n);

        float ref = 0.0f;
        for (int i = 0; i < nb; i++) {
            int sumi = 0;
            for (int j = 0; j < 32; j++) {
                sumi += (int) qa[i].qs[j] * (int) qb[i].qs[j];
            }
            ref += (float) sumi * (ggml_fp16_to_fp32(qa[i].d) * ggml_fp16_to_fp32(qb[i].d));
        }

        float result = 0.0f;
        ggml_vec_dot_q8_0_q8_0(n, &result, 0, qa.data(), 0, qb.data(), 0, 1);

        const float err = fabsf(result - ref) / std::max(1.0f, fabsf(ref));
        const bool failed = !(err < 1e-5f);
        num_failed += failed;
        if (failed || verbose) {
            printf("%5s vec_dot n=%3d (%3s nb): %s (ref=%f got=%f err=%g)\n",
                   ggml_type_name(GGML_TYPE_Q8_0), n, nb % 2 ? "odd" : "even", RESULT_STR[failed], ref, result, err);
        }
    }

    return num_failed;
}

int main(int argc, char * argv[]) {
    bool verbose = false;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    ggml_cpu_init();

    int num_failed = 0;

    num_failed += test_vec_dot_f32(verbose);
    num_failed += test_vec_dot_q(verbose);
    num_failed += test_vec_dot_q8_0_q8_0(verbose);
    num_failed += test_dequantize_q(verbose);
    num_failed += test_quantize_q(verbose);
    num_failed += test_quantize_imatrix_degenerate(verbose);

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}
