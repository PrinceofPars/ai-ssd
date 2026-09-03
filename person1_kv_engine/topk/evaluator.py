"""Top-K Evaluator: Computes recall and retrieval accuracy compared to dense baseline."""

from typing import Set, List
import numpy as np


class TopKEvaluator:
    @staticmethod
    def calculate_recall(ground_truth_ids: List[int], retrieved_ids: List[int]) -> float:
        if not ground_truth_ids:
            return 1.0
        gt_set: Set[int] = set(ground_truth_ids)
        ret_set: Set[int] = set(retrieved_ids)
        intersection = gt_set.intersection(ret_set)
        return len(intersection) / float(len(gt_set))

    @staticmethod
    def calculate_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Computes average cosine similarity across heads between baseline and pruned outputs."""
        dot = np.sum(a * b, axis=-1)
        norm_a = np.linalg.norm(a, axis=-1)
        norm_b = np.linalg.norm(b, axis=-1)
        cos = dot / (norm_a * norm_b + 1e-12)
        return float(np.mean(cos))

    @staticmethod
    def calculate_mse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean((a - b) ** 2))
