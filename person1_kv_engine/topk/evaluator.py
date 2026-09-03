"""
Top-K Evaluator: Computes recall and retrieval accuracy compared to dense baseline.
"""

from typing import Set, List


class TopKEvaluator:
    @staticmethod
    def calculate_recall(ground_truth_ids: List[int], retrieved_ids: List[int]) -> float:
        if not ground_truth_ids:
            return 1.0
        gt_set: Set[int] = set(ground_truth_ids)
        ret_set: Set[int] = set(retrieved_ids)
        intersection = gt_set.intersection(ret_set)
        return len(intersection) / float(len(gt_set))
