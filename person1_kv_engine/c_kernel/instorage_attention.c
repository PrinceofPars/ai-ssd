/**
 * Freestanding Native C In-Storage Computational Attention Kernel.
 *
 * Implements SIMD-unrolled dot-product attention and embedded in-place top-k filtering.
 * Has zero libc / runtime dependencies for direct embedded SSD controller execution.
 */

#include "instorage_attention.h"

#ifdef _MSC_VER
/* MSVC Freestanding floating-point flag */
int _fltused = 0x9875;

/* Freestanding memset intrinsic required by MSVC without CRT */
#pragma function(memset)
void* memset(void* dest, int c, unsigned long long count) {
    unsigned char* p = (unsigned char*)dest;
    while (count--) {
        *p++ = (unsigned char)c;
    }
    return dest;
}
#endif

EXPORT_API float compute_block_score(
    const float* query,
    const float* k_block,
    int tokens,
    int heads,
    int head_dim,
    float scale
) {
    float max_block_score = -1e30f;
    int block_stride = heads * head_dim;

    for (int t = 0; t < tokens; t++) {
        const float* k_token = k_block + (t * block_stride);
        float token_head_score_sum = 0.0f;

        for (int h = 0; h < heads; h++) {
            const float* q_h = query + (h * head_dim);
            const float* k_h = k_token + (h * head_dim);

            float dot = 0.0f;
            int d = 0;
            for (; d <= head_dim - 4; d += 4) {
                dot += q_h[d] * k_h[d] +
                       q_h[d + 1] * k_h[d + 1] +
                       q_h[d + 2] * k_h[d + 2] +
                       q_h[d + 3] * k_h[d + 3];
            }
            for (; d < head_dim; d++) {
                dot += q_h[d] * k_h[d];
            }

            float scaled_dot = dot * scale;
            if (scaled_dot > max_block_score) {
                max_block_score = scaled_dot;
            }
        }
    }

    return max_block_score;
}

EXPORT_API int instorage_topk_filter(
    const float* query,
    const float* k_blocks,
    int num_blocks,
    int tokens,
    int heads,
    int head_dim,
    int top_k,
    float scale,
    int* out_topk_indices,
    float* out_topk_scores
) {
    if (!query || !k_blocks || !out_topk_indices || !out_topk_scores) {
        return -1; // Null pointer error
    }
    if (num_blocks <= 0 || tokens <= 0 || heads <= 0 || head_dim <= 0 || top_k <= 0) {
        return -2; // Invalid dimensions
    }

    int effective_k = (top_k < num_blocks) ? top_k : num_blocks;
    int block_elements = tokens * heads * head_dim;

    // Initialize output buffers to minimum scores
    for (int k = 0; k < effective_k; k++) {
        out_topk_indices[k] = -1;
        out_topk_scores[k] = -1e30f;
    }

    // Scan each block in controller memory and maintain sorted top-k in place
    for (int b = 0; b < num_blocks; b++) {
        const float* block_ptr = k_blocks + (b * block_elements);
        float score = compute_block_score(query, block_ptr, tokens, heads, head_dim, scale);

        // If score exceeds the smallest in our top-k buffer
        if (score > out_topk_scores[effective_k - 1]) {
            // Find insertion position
            int insert_pos = effective_k - 1;
            while (insert_pos > 0 && score > out_topk_scores[insert_pos - 1]) {
                insert_pos--;
            }

            // Shift lower scores down
            for (int j = effective_k - 1; j > insert_pos; j--) {
                out_topk_scores[j] = out_topk_scores[j - 1];
                out_topk_indices[j] = out_topk_indices[j - 1];
            }

            // Insert new score and index
            out_topk_scores[insert_pos] = score;
            out_topk_indices[insert_pos] = b;
        }
    }

    return 0; // Success
}
