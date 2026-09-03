"""Hot/Cold KV Block Classifier and Tiering Policy."""

from dataclasses import dataclass
from typing import List, Tuple
from person1_kv_engine.tiering.kv_block import KVBlock


@dataclass
class TieringPolicy:
    sink_tokens: int = 16
    sliding_window_tokens: int = 32
    max_hot_blocks_per_layer: int = 4
    tokens_per_block: int = 16


class HotColdClassifier:
    def __init__(self, policy: TieringPolicy):
        self.policy = policy

    def is_hot(self, block: KVBlock, total_tokens: int) -> bool:
        """Determines if a block should be resident in Host RAM or offloaded to SSD."""
        # Sink tokens: initial tokens [0, sink_tokens)
        if block.token_start < self.policy.sink_tokens:
            return True

        # Recent sliding window tokens: [total_tokens - sliding_window_tokens, total_tokens)
        window_start = max(0, total_tokens - self.policy.sliding_window_tokens)
        token_end = block.token_start + block.token_count
        if token_end > window_start:
            return True

        return False
