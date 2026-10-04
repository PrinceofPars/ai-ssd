#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path("/home/ubuntu/ai-ssd")
daemon_path = ROOT / "scripts" / "nvme_guest_daemon.c"
content = daemon_path.read_text()

# 1. Add dot_product_128_avx2 function
avx64_func = """static inline float dot_product_64_avx2(const float* a, const float* b) {
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
}"""

avx128_func = """static inline float dot_product_64_avx2(const float* a, const float* b) {
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

static inline float dot_product_128_avx2(const float* a, const float* b) {
    __m256 acc0 = _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
    __m256 acc1 = _mm256_mul_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
    __m256 acc2 = _mm256_mul_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
    __m256 acc3 = _mm256_mul_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));

    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 48), _mm256_loadu_ps(b + 48), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 56), _mm256_loadu_ps(b + 56), acc3);

    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 64), _mm256_loadu_ps(b + 64), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 72), _mm256_loadu_ps(b + 72), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 80), _mm256_loadu_ps(b + 80), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 88), _mm256_loadu_ps(b + 88), acc3);

    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 96), _mm256_loadu_ps(b + 96), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 104), _mm256_loadu_ps(b + 104), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 112), _mm256_loadu_ps(b + 112), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 120), _mm256_loadu_ps(b + 120), acc3);

    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}"""

assert avx64_func in content, "avx64_func not found"
content = content.replace(avx64_func, avx128_func, 1)

# 2. Update compute_block_score_gqa to branch for head_dim == 128
old_branch = """            if (head_dim == 64) {
                dot = dot_product_64_avx2(q_vec, k_vec);
            } else {
                dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += q_vec[d] * k_vec[d];
                }
            }"""

new_branch = """            if (head_dim == 128) {
                dot = dot_product_128_avx2(q_vec, k_vec);
            } else if (head_dim == 64) {
                dot = dot_product_64_avx2(q_vec, k_vec);
            } else {
                dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += q_vec[d] * k_vec[d];
                }
            }"""

assert old_branch in content, "old_branch not found"
content = content.replace(old_branch, new_branch, 1)

daemon_path.write_text(content)
print("Updated scripts/nvme_guest_daemon.c with dot_product_128_avx2.")
