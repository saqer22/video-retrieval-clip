from typing import Dict, List
import numpy as np


def recall_at_k(ranks: List[int], k: int) -> float:
    if not ranks:
        return 0.0
    return float(np.mean([r < k for r in ranks]))


def median_rank(ranks: List[int]) -> float:
    if not ranks:
        return 0.0
    return float(np.median([r + 1 for r in ranks]))


def mean_rank(ranks: List[int]) -> float:
    if not ranks:
        return 0.0
    return float(np.mean([r + 1 for r in ranks]))


def mean_average_precision(ranks: List[int]) -> float:
    if not ranks:
        return 0.0
    ap = [1.0 / (r + 1) for r in ranks]
    return float(np.mean(ap))


def compute_metrics(ranks: List[int], topk: List[int]) -> Dict[str, float]:
    metrics = {f"R@{k}": recall_at_k(ranks, k) for k in topk}
    metrics["MedR"] = median_rank(ranks)
    metrics["MeanR"] = mean_rank(ranks)
    metrics["mAP"] = mean_average_precision(ranks)
    return metrics
