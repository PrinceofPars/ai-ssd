"""
Hot/Cold Classification: Identifies attention sinks and sliding window tokens as HOT,
and older intermediate tokens as COLD candidates for SSD offloading.
"""

from typing import List, Tuple
from common.schemas.kv_block import KVBlock
from common.constants import DEFAULT_ATTENTION_SINK_TOKENS, DEFAULT_RECENT_WINDOW_TOKENS


class HotColdClassifier:
    def __init__(
        self,
        sink_tokens: int = DEFAULT_ATTENTION_SINK_TOKENS,
        recent_tokens: int = DEFAULT_RECENT_WINDOW_TOKENS,
    ):
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens

    def classify_block(self, block: KVBlock, total_context_tokens: int) -> bool:
        """
        Returns True if HOT (should stay in GPU/DRAM), False if COLD (eligible for SSD).
        - Attention Sinks: Initial tokens [0, sink_tokens)
        - Recent Window: [total_context_tokens - recent_tokens, total_context_tokens)
        """
        token_end = block.token_start + block.token_count
        # Check if overlaps with attention sinks
        if block.token_start < self.sink_tokens:
            return True
        # Check if overlaps with recent sliding window
        window_start = max(0, total_context_tokens - self.recent_tokens)
        if token_end > window_start:
            return True
        return False

    def partition_blocks(
        self, blocks: List[KVBlock], total_context_tokens: int
    ) -> Tuple[List[KVBlock], List[KVBlock]]:
        hot, cold = [], []
        for b in blocks:
            if self.classify_block(b, total_context_tokens):
                b.hotness = 1.0
                hot.append(b)
            else:
                b.hotness = max(0.1, 1.0 - (total_context_tokens - b.token_start) / float(total_context_tokens))
                cold.append(b)
        return hot, cold
