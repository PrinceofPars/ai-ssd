"""
Workload Generator: Generates synthetic prompt tokens and autoregressive decode traces.
"""

from typing import List, Dict, Any


class WorkloadGenerator:
    def __init__(self, context_length: int = 32768, layers: int = 32, heads: int = 32):
        self.context_length = context_length
        self.layers = layers
        self.heads = heads

    def generate_decode_step(self, step: int, current_length: int) -> Dict[str, Any]:
        """
        Generates simulated attention access request for one decoding step.
        """
        return {
            "step": step,
            "context_length": current_length,
            "query_token_id": current_length,
            "layers": self.layers,
            "heads": self.heads,
        }
