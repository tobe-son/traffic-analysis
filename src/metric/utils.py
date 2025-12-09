"""Utility helpers for metric-learning workflows."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def compute_stats_by_label(values: np.ndarray, labels: Sequence[int]) -> dict[int, float]:
    """Return mean values per label."""
    output: dict[int, float] = {}
    labels_arr = np.asarray(labels)
    for label in np.unique(labels_arr):
        mask = labels_arr == label
        if np.any(mask):
            output[int(label)] = float(np.mean(values[mask]))
    return output


def global_average_pool(features: torch.Tensor) -> torch.Tensor:
    """Apply global average pooling regardless of feature dimensionality."""
    if features.dim() == 2:
        return features
    if features.dim() == 3:
        return features.mean(dim=-1)
    if features.dim() == 4:
        return features.mean(dim=(2, 3))
    raise ValueError(f"Unsupported feature dimension: {features.dim()}")


def normalise_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """L2-normalise embeddings along the feature axis."""
    return F.normalize(embedding, p=2, dim=1)


def extract_embedding(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Forward pass followed by global average pooling and normalisation."""
    features = model(inputs)
    pooled = global_average_pool(features)
    pooled = pooled.view(pooled.size(0), -1)
    return normalise_embedding(pooled)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def describe_continuous(values: Sequence[float]) -> dict[str, float]:
    """Return simple descriptive statistics for a 1-D numeric sequence."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
    }
