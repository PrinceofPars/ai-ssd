#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>
#include <math.h>

#define NUM_BLOCKS 255
#define ACTUAL_TOKENS 16
#define Q_HEADS 14
#define KV_HEADS 2
#define HEAD_DIM 64
#define SCALE (1.0f / 8.0f)
#define NUM_LAYERS 36
#define DECODE_STEPS 15

// Current implementation in nvme_guest_daemon.c
static inline float dot_product_64_avx2_orig(const float* a, const float* b) {
    __m256 acc0 = _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 48), _mm256_loadu_ps(b + 48), acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 56), _mm256_loadu_ps(b + 56), acc0);

    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

static inline float score_block_orig(const float* query, const float* k_block, int actual_tokens, int q_heads, int kv_heads, int head_dim, float scale) {
    float max_score = -1e30f;
    int gqa_ratio = (kv_heads > 0) ? (q_heads / kv_heads) : 1;
    int kv_stride = kv_heads * head_dim;

    for (int t = 0; t < actual_tokens; t++) {
        const float* k_token = k_block + (t * kv_stride);
        for (int qh = 0; qh < q_heads; qh++) {
            int kh = qh / gqa_ratio;
            const float* q_vec = query + (qh * head_dim);
            const float* k_vec = k_token + (kh * head_dim);
            float dot = dot_product_64_avx2_orig(q_vec, k_vec);
            float scaled_dot = dot * scale;
            if (scaled_dot > max_score) {
                max_score = scaled_dot;
            }
        }
    }
    return max_score;
}

// Optimized 4-way independent accumulator implementation
static inline float dot_product_64_avx2_opt(const float* a, const float* b) {
    __m256 acc0 = _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
    __m256 acc1 = _mm256_mul_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
    __m256 acc2 = _mm256_mul_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
    __m256 acc3 = _mm256_mul_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));

    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 48), _mm256_loadu_ps(b + 48), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 56), _mm256_loadu_ps(b + 56), acc3);

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

static inline float score_block_opt(const float* query, const float* k_block, int actual_tokens, int q_heads, int kv_heads, int head_dim, float scale) {
    float max_dot = -1e30f;
    int gqa_ratio = (kv_heads > 0) ? (q_heads / kv_heads) : 1;
    int kv_stride = kv_heads * head_dim;

    for (int t = 0; t < actual_tokens; t++) {
        const float* k_token = k_block + (t * kv_stride);
        _mm_prefetch((const char*)(k_token + kv_stride), _MM_HINT_T0);
        
        for (int kh = 0; kh < kv_heads; kh++) {
            const float* k_vec = k_token + (kh * head_dim);
            int q_start = kh * gqa_ratio;
            int q_end = q_start + gqa_ratio;
            for (int qh = q_start; qh < q_end; qh++) {
                const float* q_vec = query + (qh * head_dim);
                float dot = dot_product_64_avx2_opt(q_vec, k_vec);
                if (dot > max_dot) {
                    max_dot = dot;
                }
            }
        }
    }
    return max_dot * scale;
}

int main() {
    float query[Q_HEADS * HEAD_DIM];
    float k_blocks[NUM_BLOCKS][ACTUAL_TOKENS * KV_HEADS * HEAD_DIM];

    for (int i = 0; i < Q_HEADS * HEAD_DIM; i++) query[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int b = 0; b < NUM_BLOCKS; b++) {
        for (int i = 0; i < ACTUAL_TOKENS * KV_HEADS * HEAD_DIM; i++) {
            k_blocks[b][i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }

    // Warmup
    float s1 = 0, s2 = 0;
    for (int b = 0; b < NUM_BLOCKS; b++) {
        s1 += score_block_orig(query, k_blocks[b], ACTUAL_TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM, SCALE);
        s2 += score_block_opt(query, k_blocks[b], ACTUAL_TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM, SCALE);
    }
    printf("Verification check: diff = %e (s1=%f, s2=%f)\n", fabsf(s1 - s2), s1, s2);

    int total_layers = NUM_LAYERS * DECODE_STEPS; // 540 scoring calls of 255 blocks each
    
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    float dummy1 = 0;
    for (int call = 0; call < total_layers; call++) {
        for (int b = 0; b < NUM_BLOCKS; b++) {
            dummy1 += score_block_orig(query, k_blocks[b], ACTUAL_TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM, SCALE);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double t_orig = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

    clock_gettime(CLOCK_MONOTONIC, &ts0);
    float dummy2 = 0;
    for (int call = 0; call < total_layers; call++) {
        for (int b = 0; b < NUM_BLOCKS; b++) {
            dummy2 += score_block_opt(query, k_blocks[b], ACTUAL_TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM, SCALE);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double t_opt = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

    printf("Original 540 scoring calls (137,700 blocks): %6.3f s\n", t_orig);
    printf("Optimized 540 scoring calls (137,700 blocks): %6.3f s\n", t_opt);
    printf("Speedup: %.2fx (Savings: %.3f s, dummy diff=%e)\n", t_orig / t_opt, t_orig - t_opt, fabsf(dummy1 - dummy2));

    return 0;
}
