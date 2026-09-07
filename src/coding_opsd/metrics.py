"""Small, deterministic metrics shared by Phase-0 direction runners."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .runtime import named_rng


def _probabilities_with_floor(probabilities: np.ndarray | Iterable[float], floor: float) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("probabilities must be a non-empty two-dimensional array")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if floor <= 0:
        raise ValueError("floor must be positive")
    floored = np.maximum(values, floor)
    totals = floored.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("each probability row must have positive mass")
    return floored / totals


def categorical_nll(probabilities: np.ndarray | Iterable[float], labels: np.ndarray | Iterable[int], floor: float = 1e-8) -> float:
    """Mean categorical negative log likelihood after flooring and renormalizing."""

    normalized = _probabilities_with_floor(probabilities, floor)
    targets = np.asarray(labels, dtype=int)
    if targets.ndim != 1 or targets.shape[0] != normalized.shape[0]:
        raise ValueError("labels must be one-dimensional and match probability rows")
    if (targets < 0).any() or (targets >= normalized.shape[1]).any():
        raise ValueError("labels are outside the categorical support")
    return float(-np.log(normalized[np.arange(targets.size), targets]).mean())


def brier_score(probabilities: np.ndarray | Iterable[float], labels: np.ndarray | Iterable[int]) -> float:
    """Mean multiclass Brier score."""

    values = np.asarray(probabilities, dtype=float)
    targets = np.asarray(labels, dtype=int)
    if values.ndim != 2 or targets.ndim != 1 or len(targets) != len(values):
        raise ValueError("probabilities and labels have incompatible shapes")
    if (targets < 0).any() or (targets >= values.shape[1]).any():
        raise ValueError("labels are outside the categorical support")
    one_hot = np.zeros_like(values)
    one_hot[np.arange(targets.size), targets] = 1.0
    return float(np.mean(np.sum((values - one_hot) ** 2, axis=1)))


def cosine_similarity(left: np.ndarray | Iterable[float], right: np.ndarray | Iterable[float]) -> float:
    """Cosine similarity, returning zero when either vector has zero norm."""

    first = np.asarray(left, dtype=float).ravel()
    second = np.asarray(right, dtype=float).ravel()
    if first.shape != second.shape:
        raise ValueError("vectors must have the same shape")
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return 0.0
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def effective_sample_size(weights: np.ndarray | Iterable[float]) -> float:
    """Return the standard importance-weight effective sample size."""

    values = np.asarray(weights, dtype=float).ravel()
    if values.size == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("weights must be a non-empty finite non-negative vector")
    total = values.sum()
    return 0.0 if total == 0 else float(total * total / np.dot(values, values))


def paired_bootstrap_ci(
    left: np.ndarray | Iterable[float],
    right: np.ndarray | Iterable[float],
    *,
    seed: int,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Return a deterministic percentile CI for the mean paired difference."""

    left_values = np.asarray(left, dtype=float).ravel()
    right_values = np.asarray(right, dtype=float).ravel()
    if left_values.shape != right_values.shape:
        raise ValueError("paired samples must have equal lengths")
    differences = left_values - right_values
    if differences.size == 0 or not np.isfinite(differences).all():
        raise ValueError("paired samples must be non-empty and finite")
    if resamples < 1 or not 0 < confidence_level < 1:
        raise ValueError("invalid bootstrap resamples or confidence level")
    rng = named_rng(seed, "paired_bootstrap_ci")
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    estimates = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return (float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha)))
