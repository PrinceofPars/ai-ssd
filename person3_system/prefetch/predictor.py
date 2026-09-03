"""
Predictor for speculative next-layer KV block prefetching.
"""

from typing import List


class NextLayerPredictor:
    def __init__(self, total_layers: int = 32):
        self.total_layers = total_layers

    def predict_next_layer_blocks(self, current_layer_id: int, current_block_ids: List[int]) -> List[int]:
        """
        In standard transformer inference, attention moves from layer L to L+1.
        Predicts blocks needed at L+1 based on tokens referenced at L.
        """
        next_layer = (current_layer_id + 1) % self.total_layers
        # Predict corresponding blocks in the next layer
        return [bid + 1 for bid in current_block_ids]
