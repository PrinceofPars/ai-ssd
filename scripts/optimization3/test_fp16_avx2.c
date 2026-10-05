#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>
#include <math.h>

// Convert float to fp16 uint16 using hardware instruction
static inline uint16_t float_to_fp16(float f) {
    __m128 v = _mm_set_ss(f);
    __m128i h = _mm_cvtps_ph(v, 0);
    return (uint16_t)_mm_cvtsi128_si32(h);
}

// Convert fp16 uint16 to float using hardware instruction
static inline float fp16_to_float(uint16_t h) {
    __m128i v = _mm_cvtsi32_si128((int)h);
    __m128 f = _mm_cvtph_ps(v);
    return _mm_cvtss_f32(f);
}

// 128-dim dot product with float32 Query and float16 Key (native F16C + AVX2 + FMA)
static inline float dot_product_128_fp16_avx2(const float* a, const uint16_t* b) {
    __m256 b0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)b));
    __m256 b1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + 8)));
    __m256 b2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + 16)));
    __m256 b3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + 24)));

    __m256 acc0 = _mm256_mul_ps(_mm256_loadu_ps(a), b0);
    __m256 acc1 = _mm256_mul_ps(_mm256_loadu_ps(a + 8), b1);
    __m256 acc2 = _mm256_mul_ps(_mm256_loadu_ps(a + 16), b2);
    __m256 acc3 = _mm256_mul_ps(_mm256_loadu_ps(a + 24), b3);

    __m256 b4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + 32)));
    __m256 b5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + 40)));
    __m256 b6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + 48)));
    __m256 b7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + 56)));

    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 32), b4, acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 40), b5, acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 48), b6, acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 56), b7, acc3);

    __m256 b8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + 64)));
    __m256 b9 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + 72)));
    __m256 b10 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + 80)));
    __m256 b11 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + 88)));

    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 64), b8, acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 72), b9, acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 80), b10, acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 88), b11, acc3);

    __m256 b12 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + 96)));
    __m256 b13 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + 104)));
    __m256 b14 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + 112)));
    __m256 b15 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + 120)));

    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 96), b12, acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 104), b13, acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 112), b14, acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 120), b15, acc3);

    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

int main() {
    float a[128];
    uint16_t b[128];
    float b_f32[128];

    for (int i = 0; i < 128; i++) {
        a[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float val = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        b[i] = float_to_fp16(val);
        b_f32[i] = fp16_to_float(b[i]);
    }

    // Scalar reference
    float expected = 0.0f;
    for (int i = 0; i < 128; i++) {
        expected += a[i] * b_f32[i];
    }

    float measured = dot_product_128_fp16_avx2(a, b);
    printf("Expected: %f | Measured: %f | Diff: %e\n", expected, measured, fabsf(expected - measured));
    if (fabsf(expected - measured) < 1e-4) {
        printf("[SUCCESS] 128-dim FP16 AVX2 kernel verified!\n");
        return 0;
    } else {
        printf("[FAIL] Mismatch!\n");
        return 1;
    }
}
