"""In-Storage Computational Attention Pruner and Host Integration Engine.

Combines local Host RAM attention on HOT tokens with in-storage hardware-accelerated
top-k attention on COLD tokens, achieving massive PCIe bus traffic reductions.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

from person1_kv_engine.baseline.transformer_attention import AttentionConfig
from person1_kv_engine.storage_backend.kv_storage_api import KVStorageInterface
from person1_kv_engine.tiering.tiered_kv_manager import TieredKVManager
from person1_kv_engine.computational_storage.streaming_softmax import OnlineSoftmaxAccumulator


class InStorageAttentionPruner:
    """Dispatches query to SSD controller, performs in-storage pruning, and merges results."""

    def __init__(
        self,
        config: AttentionConfig,
        storage: KVStorageInterface,
        manager: TieredKVManager,
        default_top_k: int = 4,
    ):
        self.config = config
        self.storage = storage
        self.manager = manager
        self.default_top_k = default_top_k
        self.scale = 1.0 / np.sqrt(self.config.head_dim, dtype=np.float32)

    def compute_decode_attention(
        self,
        query: np.ndarray,  # [num_heads, head_dim]
        layer_id: int,
        top_k: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Computes hybrid decode attention combining Host RAM and In-Storage Top-k.
        
        Args:
            query: Query token vector [num_heads, head_dim]
            layer_id: Transformer layer index
            top_k: Number of cold blocks to retrieve (defaults to default_top_k)
            
        Returns:
            Tuple of:
            - context_output: [num_heads, head_dim]
            - stats: Dictionary with performance breakdown and byte counters
        """
        k_val = top_k if top_k is not None else self.default_top_k
        accumulator = OnlineSoftmaxAccumulator(
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            dtype=self.config.dtype,
        )

        # 1. Host RAM: Process HOT tokens (Attention sinks + sliding window)
        hot_k, hot_v, hot_indices = self.manager.get_hot_kv(layer_id)
        num_hot_tokens = len(hot_indices)

        if num_hot_tokens > 0:
            # Transpose hot_k: [num_hot_tokens, heads, head_dim] -> [heads, head_dim, num_hot_tokens]
            k_t = np.transpose(hot_k, (1, 2, 0))
            # Q: [heads, 1, head_dim] * k_t -> [heads, 1, num_hot_tokens]
            q_expanded = query[:, np.newaxis, :]
            logits_hot = np.matmul(q_expanded, k_t).squeeze(axis=1) * self.scale
            accumulator.update_with_partition(logits=logits_hot, values=hot_v)

        # 2. SSD Domain: Process COLD tokens via In-Storage Top-k Pruning
        cold_blocks = self.manager.get_cold_blocks(layer_id)
        num_cold_blocks = len(cold_blocks)
        num_cold_tokens = sum(b.token_count for b in cold_blocks)
        retrieved_tokens = 0

        if num_cold_blocks > 0:
            # Send Q to SSD controller, compute dot products in controller DRAM, stream ONLY top-k
            topk_ids, topk_vals, topk_logits, topk_scores = self.storage.in_storage_topk_attention(
                query=query,
                layer_id=layer_id,
                top_k=k_val,
            )

            if len(topk_ids) > 0:
                # Reshape topk_vals: [top_k, tokens_per_block, heads, head_dim] -> [top_k * tokens, heads, head_dim]
                effective_k, tok_per_block, heads, h_dim = topk_vals.shape
                topk_vals_flat = topk_vals.reshape(effective_k * tok_per_block, heads, h_dim)
                retrieved_tokens = effective_k * tok_per_block

                # Merge cold partition into running accumulator
                accumulator.update_with_partition(logits=topk_logits, values=topk_vals_flat)

        # 3. Finalize normalized attention context vector
        context_vector = accumulator.finalize()

        stats = {
            "layer_id": layer_id,
            "hot_tokens": num_hot_tokens,
            "total_cold_tokens": num_cold_tokens,
            "cold_blocks_total": num_cold_blocks,
            "cold_blocks_retrieved": min(k_val, num_cold_blocks),
            "retrieved_tokens": retrieved_tokens,
            "total_effective_tokens": num_hot_tokens + retrieved_tokens,
            "pruning_ratio": 1.0 - (retrieved_tokens / max(1, num_cold_tokens)) if num_cold_tokens > 0 else 0.0,
        }

        return context_vector, stats
