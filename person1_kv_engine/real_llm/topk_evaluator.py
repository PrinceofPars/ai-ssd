"""Top-K Sparse Attention Evaluator against Dense Reference for AI-SSD V2.

Compares Dense Reference Attention vs In-Storage Top-k Sparse Attention
across multiple sparsity budgets: 1%, 5%, 10%, 20%, 50%.

Measures actual empirical quality and bandwidth metrics without fabrication:
- selected block count
- bytes transferred over PCIe
- attention output cosine similarity
- relative Frobenius error
- output MSE divergence
- attention mass recall
- latency and PCIe traffic reduction
"""

from typing import List, Dict, Tuple, Any, Optional
import time
import numpy as np

from person1_kv_engine.c_kernel.kernel_binding import get_native_c_kernel


class TopKEvaluator:
    """Evaluates accuracy and storage reduction of in-storage Top-k attention."""

    def __init__(
        self,
        head_dim: int = 64,
        tokens_per_block: int = 16,
        attention_sink_tokens: int = 4,
        recent_window_tokens: int = 16,
        bytes_per_elem: int = 4,
    ):
        self.head_dim = head_dim
        self.tokens_per_block = tokens_per_block
        self.attention_sink_tokens = attention_sink_tokens
        self.recent_window_tokens = recent_window_tokens
        self.bytes_per_elem = bytes_per_elem
        self.kernel = get_native_c_kernel()

    def evaluate_query(
        self,
        query: np.ndarray,      # [q_heads, head_dim]
        k_seq: np.ndarray,      # [kv_heads, seq_len, head_dim]
        v_seq: np.ndarray,      # [kv_heads, seq_len, head_dim]
        sparsity_levels: List[float] = [1.0, 5.0, 10.0, 20.0, 50.0],
    ) -> Dict[str, Any]:
        """Runs dense attention reference and evaluates multiple Top-k sparsity levels.
        
        Returns:
            Dictionary mapping sparsity level string to measured metrics dictionary.
        """
        q_heads, head_dim = query.shape
        kv_heads, seq_len, _ = k_seq.shape
        scale = 1.0 / np.sqrt(head_dim)
        gqa_ratio = max(1, q_heads // kv_heads)

        # -----------------------------------------------------------------
        # 1. DENSE REFERENCE ATTENTION
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        dense_outputs = np.zeros((q_heads, head_dim), dtype=np.float32)
        dense_attn_weights = []

        for h in range(q_heads):
            kv_h = h // gqa_ratio
            q_h = query[h]                       # [head_dim]
            k_h = k_seq[kv_h]                    # [seq_len, head_dim]
            v_h = v_seq[kv_h]                    # [seq_len, head_dim]

            # Dot-product logits
            logits = np.dot(k_h, q_h) * scale    # [seq_len]
            # Numerically stable softmax
            logits_max = np.max(logits)
            exp_logits = np.exp(logits - logits_max)
            weights = exp_logits / np.sum(exp_logits)
            dense_attn_weights.append(weights)

            # Output vector
            dense_outputs[h] = np.dot(weights, v_h)

        dense_time_us = (time.perf_counter() - t0) * 1e6
        dense_total_bytes = 2 * seq_len * kv_heads * head_dim * self.bytes_per_elem

        # -----------------------------------------------------------------
        # 2. BLOCK PARTITIONING
        # -----------------------------------------------------------------
        num_blocks = int(np.ceil(seq_len / self.tokens_per_block))
        sink_blocks = set(range(int(np.ceil(self.attention_sink_tokens / self.tokens_per_block))))
        recent_cutoff = max(0, seq_len - self.recent_window_tokens)
        recent_blocks = set(range(recent_cutoff // self.tokens_per_block, num_blocks))
        host_dram_blocks = sink_blocks.union(recent_blocks)
        ssd_candidate_blocks = [b for b in range(num_blocks) if b not in host_dram_blocks]

        # -----------------------------------------------------------------
        # 3. EVALUATE EACH SPARSITY LEVEL
        # -----------------------------------------------------------------
        results = {
            "dense_reference": {
                "latency_us": dense_time_us,
                "bytes_transferred": dense_total_bytes,
                "total_blocks": num_blocks,
                "seq_len": seq_len,
            },
            "sparsity_evaluations": {},
        }

        for p in sparsity_levels:
            t_sparse_start = time.perf_counter()

            # Determine Top-k count
            k_candidates = len(ssd_candidate_blocks)
            k_val = max(1, int(np.ceil(k_candidates * (p / 100.0)))) if k_candidates > 0 else 0

            # Score SSD blocks using in-storage Key scan
            selected_ssd_blocks = set()
            if k_candidates > 0 and k_val > 0:
                block_scores = []
                for b in ssd_candidate_blocks:
                    t_start = b * self.tokens_per_block
                    t_end = min(t_start + self.tokens_per_block, seq_len)

                    # In-storage max dot-product score
                    b_max = -1e30
                    for h in range(q_heads):
                        kv_h = h // gqa_ratio
                        q_h = query[h]
                        k_b = k_seq[kv_h, t_start:t_end, :]
                        dots = np.dot(k_b, q_h) * scale
                        m = float(np.max(dots))
                        if m > b_max:
                            b_max = m
                    block_scores.append((b_max, b))

                block_scores.sort(key=lambda x: x[0], reverse=True)
                selected_ssd_blocks = set(b for _, b in block_scores[:k_val])

            # Active sparse blocks: host DRAM (sinks + recent) + selected SSD blocks
            active_blocks = sorted(list(host_dram_blocks.union(selected_ssd_blocks)))

            # Gather active token indices
            active_token_indices = []
            for b in active_blocks:
                t_start = b * self.tokens_per_block
                t_end = min(t_start + self.tokens_per_block, seq_len)
                active_token_indices.extend(range(t_start, t_end))
            active_token_indices = np.array(active_token_indices, dtype=np.int64)

            # Compute sparse attention
            sparse_outputs = np.zeros((q_heads, head_dim), dtype=np.float32)
            cos_sims = []
            rel_errors = []
            mses = []
            attention_mass_recalls = []

            for h in range(q_heads):
                kv_h = h // gqa_ratio
                q_h = query[h]
                k_active = k_seq[kv_h, active_token_indices, :]
                v_active = v_seq[kv_h, active_token_indices, :]

                sparse_logits = np.dot(k_active, q_h) * scale
                s_max = np.max(sparse_logits)
                exp_s = np.exp(sparse_logits - s_max)
                sparse_weights = exp_s / np.sum(exp_s)

                sparse_out = np.dot(sparse_weights, v_active)
                sparse_outputs[h] = sparse_out

                # Quality metrics vs dense reference
                dense_out = dense_outputs[h]
                norm_d = np.linalg.norm(dense_out)
                norm_s = np.linalg.norm(sparse_out)
                cos = float(np.dot(dense_out, sparse_out) / (max(1e-12, norm_d * norm_s)))
                cos_sims.append(cos)

                rel_err = float(np.linalg.norm(dense_out - sparse_out) / max(1e-12, norm_d))
                rel_errors.append(rel_err)

                mse = float(np.mean((dense_out - sparse_out) ** 2))
                mses.append(mse)

                # Attention mass captured by active tokens
                dense_weights_h = dense_attn_weights[h]
                captured_mass = float(np.sum(dense_weights_h[active_token_indices]))
                attention_mass_recalls.append(captured_mass)

            sparse_time_us = (time.perf_counter() - t_sparse_start) * 1e6

            # Byte calculation:
            # SSD Key scan (in-storage controller internal): k_candidates * tokens_per_block * head_dim * bytes
            # PCIe transfer: only active SSD blocks (Key + Value) + DRAM resident
            pcie_transferred_bytes = (
                len(host_dram_blocks) * self.tokens_per_block * kv_heads * head_dim * self.bytes_per_elem * 2 +
                len(selected_ssd_blocks) * self.tokens_per_block * kv_heads * head_dim * self.bytes_per_elem * 2
            )
            traffic_reduction = max(0.0, 1.0 - (pcie_transferred_bytes / max(1.0, float(dense_total_bytes))))

            results["sparsity_evaluations"][f"{p:.1f}%"] = {
                "sparsity_budget_percent": p,
                "selected_ssd_blocks": len(selected_ssd_blocks),
                "total_active_blocks": len(active_blocks),
                "total_candidate_blocks": k_candidates,
                "active_tokens": len(active_token_indices),
                "total_tokens": seq_len,
                "token_selection_ratio": float(len(active_token_indices)) / float(seq_len),
                "pcie_bytes_transferred": pcie_transferred_bytes,
                "pcie_traffic_reduction_percent": traffic_reduction * 100.0,
                "mean_cosine_similarity": float(np.mean(cos_sims)),
                "min_cosine_similarity": float(np.min(cos_sims)),
                "mean_relative_error": float(np.mean(rel_errors)),
                "mean_mse": float(np.mean(mses)),
                "mean_attention_mass_recall": float(np.mean(attention_mass_recalls)),
                "latency_us": sparse_time_us,
            }

        return results
