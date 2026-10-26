#include "unary-ops.h"
#include "vec.h"

#include <type_traits>

static inline float op_abs(float x) {
    return fabsf(x);
}

static inline float op_sgn(float x) {
    return (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

static inline float op_neg(float x) {
    return -x;
}

static inline float op_step(float x) {
    return (x > 0.f) ? 1.f : 0.f;
}

static inline float op_tanh(float x) {
    return tanhf(x);
}

static inline float op_elu(float x) {
    return (x > 0.f) ? x : expm1f(x);
}

static inline float op_relu(float x) {
    return (x > 0.f) ? x : 0.f;
}

static inline float op_sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

static inline float op_hardsigmoid(float x) {
    return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static inline float op_exp(float x) {
    return expf(x);
}

static inline float op_hardswish(float x) {
    return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static inline float op_sqr(float x) {
    return x * x;
}

static inline float op_sqrt(float x) {
    return sqrtf(x);
}

static inline float op_xielu(float x, float alpha_n, float alpha_p, float beta, float eps) {
    if (x > 0.0f) {
        return alpha_p * x * x + beta * x;
    } else {
        const float min_x_eps = fminf(x, eps);
        return (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

static inline float op_sin(float x) {
    return sinf(x);
}

static inline float op_cos(float x) {
    return cosf(x);
}

static inline float op_log(float x) {
    return logf(x);
}

static inline float op_expm1(float x) {
    return expf(x) - 1.0f;
}

static inline float op_softplus(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

static inline float op_floor(float x) {
    return floorf(x);
}

static inline float op_ceil(float x) {
    return ceilf(x);
}

static inline float op_round(float x) {
    return roundf(x);
}

static inline float op_trunc(float x) {
    return truncf(x);
}

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__FMA__)
// log(x) for finite x > 0: frexp by bits, then the atanh series on the mantissa
static inline __m512 ggml_v_logf(__m512 x) {
    const __m512i bits = _mm512_castps_si512(x);
    __m512i e = _mm512_sub_epi32(_mm512_srli_epi32(bits, 23), _mm512_set1_epi32(127));
    __m512 m = _mm512_castsi512_ps(_mm512_or_si512(
            _mm512_and_si512(bits, _mm512_set1_epi32(0x007fffff)), _mm512_set1_epi32(0x3f800000)));
    // keep the mantissa in [sqrt(0.5), sqrt(2))
    const __mmask16 hi = _mm512_cmp_ps_mask(m, _mm512_set1_ps(1.4142135f), _CMP_GT_OQ);
    m = _mm512_mask_mul_ps(m, hi, m, _mm512_set1_ps(0.5f));
    e = _mm512_mask_add_epi32(e, hi, e, _mm512_set1_epi32(1));

    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 s = _mm512_div_ps(_mm512_sub_ps(m, one), _mm512_add_ps(m, one));
    const __m512 s2 = _mm512_mul_ps(s, s);
    __m512 p = _mm512_set1_ps(1.0f/9.0f);
    p = _mm512_fmadd_ps(p, s2, _mm512_set1_ps(1.0f/7.0f));
    p = _mm512_fmadd_ps(p, s2, _mm512_set1_ps(1.0f/5.0f));
    p = _mm512_fmadd_ps(p, s2, _mm512_set1_ps(1.0f/3.0f));
    p = _mm512_fmadd_ps(p, s2, one);

    const __m512 lm = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(2.0f), s), p);
    return _mm512_fmadd_ps(_mm512_cvtepi32_ps(e), _mm512_set1_ps(0.6931471805599453f), lm);
}

// NaN has no defined path through the bit tricks in ggml_v_expf/ggml_v_logf, so keep it explicit
static inline __m512 ggml_v_nan_passthrough(__m512 r, __m512 vx) {
    return _mm512_mask_mov_ps(r, _mm512_cmp_ps_mask(vx, vx, _CMP_UNORD_Q), vx);
}

// expm1(x) for x <= 0: series near zero, exp(x) - 1 where that cannot cancel
static inline __m512 ggml_v_expm1_neg(__m512 vx, __m512 vexp) {
    const __m512 one = _mm512_set1_ps(1.0f);
    // x + x^2*(1/2 + x/6 + x^2/24 + x^3/120 + x^4/720), error ~1.2e-8 at |x| = 0.25
    __m512 p = _mm512_set1_ps(1.0f/720.0f);
    p = _mm512_fmadd_ps(p, vx, _mm512_set1_ps(1.0f/120.0f));
    p = _mm512_fmadd_ps(p, vx, _mm512_set1_ps(1.0f/24.0f));
    p = _mm512_fmadd_ps(p, vx, _mm512_set1_ps(1.0f/6.0f));
    p = _mm512_fmadd_ps(p, vx, _mm512_set1_ps(0.5f));
    const __m512 ser = _mm512_fmadd_ps(_mm512_mul_ps(vx, vx), p, vx);
    const __m512 expm1 = _mm512_sub_ps(vexp, one);
    return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(vx, _mm512_set1_ps(-0.25f), _CMP_GT_OQ), expm1, ser);
}

// the f32 kernels for the ops that only need exp/log, reused for the 16-bit types below
template <float (*op)(float)>
static inline __m512 ggml_v_unary_f32(__m512 vx) {
    const __m512 one  = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_setzero_ps();
    __m512 r;
    if constexpr (op == op_tanh) {
        r = ggml_v_tanh(vx);
    } else if constexpr (op == op_sigmoid) {
        r = _mm512_div_ps(one, _mm512_add_ps(one, ggml_v_expf(_mm512_sub_ps(zero, vx))));
    } else if constexpr (op == op_exp) {
        r = ggml_v_expf(vx);
    } else if constexpr (op == op_expm1) {
        r = _mm512_sub_ps(ggml_v_expf(vx), one);
    } else if constexpr (op == op_elu) {
        const __m512 em1 = ggml_v_expm1_neg(vx, ggml_v_expf(vx));
        r = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(vx, zero, _CMP_GT_OQ), em1, vx);
    } else { // op_softplus: x > 20 keeps x, else log1p(exp(x)); series when exp(x) is tiny
        const __m512 t = ggml_v_expf(vx);
        const __m512 small = _mm512_mul_ps(t, _mm512_sub_ps(one, _mm512_mul_ps(_mm512_set1_ps(0.5f), t)));
        const __m512 l = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(vx, _mm512_set1_ps(-8.0f), _CMP_LE_OQ),
                                              ggml_v_logf(_mm512_add_ps(one, t)), small);
        r = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(vx, _mm512_set1_ps(20.0f), _CMP_GT_OQ), l, vx);
    }
    return ggml_v_nan_passthrough(r, vx);
}
// xielu: alpha_p*x^2 + beta*x on the right, (expm1(min(x, eps)) - x)*alpha_n + beta*x on the left
static inline void vec_unary_op_xielu_f32(int64_t n, float * y, const float * x,
        float alpha_n, float alpha_p, float beta, float eps, int64_t & i) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 va  = _mm512_set1_ps(alpha_n);
    const __m512 vp  = _mm512_set1_ps(alpha_p);
    const __m512 vb  = _mm512_set1_ps(beta);
    const __m512 ve  = _mm512_set1_ps(eps);
    for (; i + 15 < n; i += 16) {
        const __m512 vx = _mm512_loadu_ps(x + i);
        const __mmask16 pos = _mm512_cmp_ps_mask(vx, _mm512_setzero_ps(), _CMP_GT_OQ);
        const __m512 yp = _mm512_fmadd_ps(vp, _mm512_mul_ps(vx, vx), _mm512_mul_ps(vb, vx));
        const __m512 mn = _mm512_min_ps(vx, ve);
        const __m512 em1 = ggml_v_expm1_neg(mn, ggml_v_expf(mn));
        const __m512 yn = _mm512_fmadd_ps(_mm512_sub_ps(em1, vx), va, _mm512_mul_ps(vb, vx));
        _mm512_storeu_ps(y + i, ggml_v_nan_passthrough(_mm512_mask_blend_ps(pos, yn, yp), vx));
    }
}

#endif

template <float (*op)(float), typename src0_t, typename dst_t>
static inline void vec_unary_op(int64_t n, dst_t * y, const src0_t * x) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__FMA__)
    // ops with a vector exp/log kernel; same-type 16-bit is handled by converting in and out
    constexpr bool vectorized =
        (op == op_tanh || op == op_sigmoid || op == op_exp || op == op_elu ||
         op == op_expm1 || op == op_softplus) &&
        ((std::is_same_v<src0_t, float>       && std::is_same_v<dst_t, float>) ||
         (std::is_same_v<src0_t, ggml_fp16_t> && std::is_same_v<dst_t, ggml_fp16_t>) ||
         (std::is_same_v<src0_t, ggml_bf16_t> && std::is_same_v<dst_t, ggml_bf16_t>) ||
         (std::is_same_v<src0_t, ggml_fp16_t> && std::is_same_v<dst_t, float>) ||
         (std::is_same_v<src0_t, ggml_bf16_t> && std::is_same_v<dst_t, float>));
    if constexpr (vectorized) {
        int i = 0;
        for (; i + 15 < n; i += 16) {
            __m512 vx;
            if constexpr (std::is_same_v<src0_t, float>) {
                vx = _mm512_loadu_ps(x + i);
            } else if constexpr (std::is_same_v<src0_t, ggml_fp16_t>) {
                vx = ggml_v_f16_to_f32(x + i);
            } else {
                vx = ggml_v_bf16_to_f32(x + i);
            }

            const __m512 vy = ggml_v_unary_f32<op>(vx);

            if constexpr (std::is_same_v<dst_t, float>) {
                _mm512_storeu_ps(y + i, vy);
            } else if constexpr (std::is_same_v<dst_t, ggml_fp16_t>) {
                ggml_v_f32_to_f16(y + i, vy);
            } else {
                ggml_v_f32_to_bf16(y + i, vy);
            }
        }
        for (; i < n; ++i) {
            y[i] = f32_to_dst(op(src0_to_f32(x[i])));
        }
        return;
    }
#endif

    for (int i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

template <float (*op)(float), typename src0_t, typename dst_t>
static void apply_unary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((char *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        const src0_t * src0_ptr = (const src0_t *) ((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <float (*op)(float)>
static void unary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    /*  */ if (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, float>(params, dst);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

template <float (*op)(float, ggml_tensor *)>
static void unary_op_params(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    /*  */ if (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, float>(params, dst);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

template <typename src0_t, typename dst_t>
static inline void vec_unary_op_xielu(int64_t n, dst_t * y, const src0_t * x,
        float alpha_n, float alpha_p, float beta, float eps) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    int64_t i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__FMA__)
    if constexpr (std::is_same_v<src0_t, float> && std::is_same_v<dst_t, float>) {
        vec_unary_op_xielu_f32(n, y, x, alpha_n, alpha_p, beta, eps, i);
    }
#endif
    for (; i < n; i++) {
        y[i] = f32_to_dst(op_xielu(src0_to_f32(x[i]), alpha_n, alpha_p, beta, eps));
    }
}

template <typename src0_t, typename dst_t>
static void apply_unary_op_xielu(const ggml_compute_params * params, ggml_tensor * dst,
        float alpha_n, float alpha_p, float beta, float eps) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((char *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        const src0_t * src0_ptr = (const src0_t *) ((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_xielu(ne0, dst_ptr, src0_ptr, alpha_n, alpha_p, beta, eps);
    }
}

static void unary_op_xielu(const ggml_compute_params * params, ggml_tensor * dst,
        float alpha_n, float alpha_p, float beta, float eps) {
    const ggml_tensor * src0 = dst->src[0];

    /*  */ if (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_xielu<float, float>(params, dst, alpha_n, alpha_p, beta, eps);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) {
        apply_unary_op_xielu<ggml_fp16_t, ggml_fp16_t>(params, dst, alpha_n, alpha_p, beta, eps);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        apply_unary_op_xielu<ggml_bf16_t, ggml_bf16_t>(params, dst, alpha_n, alpha_p, beta, eps);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_xielu<ggml_bf16_t, float>(params, dst, alpha_n, alpha_p, beta, eps);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_xielu<ggml_fp16_t, float>(params, dst, alpha_n, alpha_p, beta, eps);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

void ggml_compute_forward_abs(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

void ggml_compute_forward_sgn(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

void ggml_compute_forward_neg(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

void ggml_compute_forward_step(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_step>(params, dst);
}

void ggml_compute_forward_tanh(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

void ggml_compute_forward_elu(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

void ggml_compute_forward_relu(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

void ggml_compute_forward_sigmoid(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

void ggml_compute_forward_hardsigmoid(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

void ggml_compute_forward_exp(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

void ggml_compute_forward_hardswish(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

void ggml_compute_forward_sqr(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

void ggml_compute_forward_sqrt(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

void ggml_compute_forward_sin(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

void ggml_compute_forward_cos(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

void ggml_compute_forward_log(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

void ggml_compute_forward_expm1(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

void ggml_compute_forward_softplus(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

void ggml_compute_forward_floor(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

void ggml_compute_forward_ceil(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

void ggml_compute_forward_round(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_round>(params, dst);
}

void ggml_compute_forward_trunc(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

void ggml_compute_forward_xielu(const ggml_compute_params * params, ggml_tensor * dst) {
    const float alpha_n = ggml_get_op_params_f32(dst, 1);
    const float alpha_p = ggml_get_op_params_f32(dst, 2);
    const float beta = ggml_get_op_params_f32(dst, 3);
    const float eps = ggml_get_op_params_f32(dst, 4);

    unary_op_xielu(params, dst, alpha_n, alpha_p, beta, eps);
}

