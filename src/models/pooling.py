import numpy as np


def mean_pool(frames: np.ndarray) -> np.ndarray:
    """Mean pool across frames (first dimension)."""
    if frames.ndim == 1:
        return frames
    pooled = frames.mean(axis=0)
    # Normalize
    norm = np.linalg.norm(pooled)
    if norm > 1e-6:
        pooled = pooled / norm
    return pooled
