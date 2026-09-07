"""Fixed-target LOO, AIPW, and exact design-risk operators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable

import numpy as np

from .subsets import SubsetDesign, enumerate_subsets


def validate_psd(matrix: np.ndarray | Iterable[Iterable[float]], name: str) -> np.ndarray:
    """Return a finite symmetric PSD matrix or fail closed."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite square matrix")
    if not np.allclose(values, values.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    symmetric = (values + values.T) / 2.0
    scale = max(1.0, float(np.linalg.norm(symmetric, ord=2)))
    if float(np.linalg.eigvalsh(symmetric).min()) < -1e-12 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return symmetric


def _validate_inclusion_moments(pi: np.ndarray, pi2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pi.ndim != 1 or pi.size < 1 or pi2.shape != (pi.size, pi.size):
        raise ValueError("pi and pi2 have incompatible shapes")
    if not np.isfinite(pi).all() or (pi <= 0.0).any() or (pi > 1.0).any() or not np.isfinite(pi2).all():
        raise ValueError("pi and pi2 must be finite valid inclusion probabilities")
    if not np.allclose(pi2, pi2.T, atol=1e-12, rtol=0.0):
        raise ValueError("pi2 must be symmetric")
    if not np.allclose(np.diag(pi2), pi, atol=1e-12, rtol=0.0):
        raise ValueError("the pi2 diagonal must equal pi")
    lower = np.maximum(0.0, pi[:, None] + pi[None, :] - 1.0)
    upper = np.minimum(pi[:, None], pi[None, :])
    if (pi2 < lower - 1e-12).any() or (pi2 > upper + 1e-12).any():
        raise ValueError("pi2 violates Frechet bounds")
    return pi, pi2


def loo_influence(scores: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    """Return the ``P x K`` linear influence matrix for the LOO gradient."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite K by P matrix with K at least two")
    candidates = values.shape[0]
    other_sum = values.sum(axis=0, keepdims=True) - values
    columns = (values - other_sum / (candidates - 1)) / candidates
    return columns.T


def direct_loo_gradient(
    scores: np.ndarray | Iterable[Iterable[float]], labels: np.ndarray | Iterable[float]
) -> np.ndarray:
    """Compute the LOO gradient directly from candidate advantages."""

    values = np.asarray(scores, dtype=np.float64)
    outcomes = np.asarray(labels, dtype=np.float64)
    if values.ndim != 2 or outcomes.ndim != 1 or len(outcomes) != len(values) or len(values) < 2:
        raise ValueError("scores and labels have incompatible shapes")
    if not np.isfinite(values).all() or not np.isfinite(outcomes).all():
        raise ValueError("scores and labels must be finite")
    baselines = (outcomes.sum() - outcomes) / (len(outcomes) - 1)
    return np.mean(values * (outcomes - baselines)[:, None], axis=0)


def _observed_vector(selected_values: Mapping[int, float] | np.ndarray | Iterable[float], candidates: int) -> tuple[np.ndarray, np.ndarray]:
    observed = np.full(candidates, np.nan, dtype=np.float64)
    if isinstance(selected_values, Mapping):
        for raw_index, raw_value in selected_values.items():
            index = int(raw_index)
            if index != raw_index or not 0 <= index < candidates:
                raise ValueError("selected label index is outside the candidate range")
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError("mapping values identify selected labels and must be finite")
            observed[index] = value
    else:
        observed = np.asarray(selected_values, dtype=np.float64)
        if observed.ndim != 1 or len(observed) != candidates:
            raise ValueError("indexed selected values must match the candidate count")
        observed = observed.copy()
    selected = ~np.isnan(observed)
    if not np.isfinite(observed[selected]).all():
        raise ValueError("selected values must be finite; only unselected entries may be NaN")
    return observed, selected


def aipw_pseudolabel(
    mu: np.ndarray | Iterable[float],
    selected_values: Mapping[int, float] | np.ndarray | Iterable[float],
    pi: np.ndarray | Iterable[float],
) -> np.ndarray:
    """Return un-clipped, unnormalized augmented inverse-inclusion labels."""

    means = np.asarray(mu, dtype=np.float64)
    inclusion = np.asarray(pi, dtype=np.float64)
    if means.ndim != 1 or inclusion.ndim != 1 or means.shape != inclusion.shape:
        raise ValueError("mu and pi must be equal-length vectors")
    if not np.isfinite(means).all():
        raise ValueError("mu must be finite")
    if not np.isfinite(inclusion).all() or (inclusion <= 0.0).any() or (inclusion > 1.0).any():
        raise ValueError("pi must be finite and lie in (0, 1]")
    observed, selected = _observed_vector(selected_values, len(means))
    result = means.copy()
    result[selected] += (observed[selected] - means[selected]) / inclusion[selected]
    return result


def aipw_gradient(
    influence: np.ndarray | Iterable[Iterable[float]],
    mu: np.ndarray | Iterable[float],
    selected_values: Mapping[int, float] | np.ndarray | Iterable[float],
    pi: np.ndarray | Iterable[float],
) -> np.ndarray:
    matrix = np.asarray(influence, dtype=np.float64)
    means = np.asarray(mu, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(means):
        raise ValueError("influence must have one column per candidate")
    return matrix @ aipw_pseudolabel(means, selected_values, pi)


def _metric_gram(influence: np.ndarray, metric: np.ndarray | None) -> np.ndarray:
    if influence.ndim != 2 or not np.isfinite(influence).all():
        raise ValueError("influence must be a finite P by K matrix")
    if metric is None:
        return influence.T @ influence
    weight = validate_psd(metric, "metric")
    if weight.shape != (influence.shape[0], influence.shape[0]):
        raise ValueError("metric must be a P by P matrix")
    return influence.T @ weight @ influence


def design_risk(
    covariance: np.ndarray | Iterable[Iterable[float]],
    influence: np.ndarray | Iterable[Iterable[float]],
    design_or_pi: SubsetDesign | np.ndarray | Iterable[float],
    pi2: np.ndarray | Iterable[Iterable[float]] | None = None,
    metric: np.ndarray | Iterable[Iterable[float]] | None = None,
) -> float:
    """Evaluate the exact posterior AIPW design-MSE formula."""

    residual_second_moment = np.asarray(covariance, dtype=np.float64)
    matrix = np.asarray(influence, dtype=np.float64)
    if isinstance(design_or_pi, SubsetDesign):
        inclusion = design_or_pi.pi
        joint = design_or_pi.pi2
    else:
        inclusion = np.asarray(design_or_pi, dtype=np.float64)
        if pi2 is None:
            raise ValueError("pi2 is required when a SubsetDesign is not supplied")
        joint = np.asarray(pi2, dtype=np.float64)
    candidates = matrix.shape[1] if matrix.ndim == 2 else -1
    if residual_second_moment.shape != (candidates, candidates) or inclusion.shape != (candidates,) or joint.shape != (candidates, candidates):
        raise ValueError("covariance, influence, pi, and pi2 have incompatible shapes")
    residual_second_moment = validate_psd(residual_second_moment, "covariance")
    inclusion, joint = _validate_inclusion_moments(inclusion, joint)
    gram = _metric_gram(matrix, None if metric is None else np.asarray(metric, dtype=np.float64))
    factors = joint / np.outer(inclusion, inclusion) - 1.0
    return float(np.sum(factors * residual_second_moment * gram))


def exact_realized_design_mse(
    residual: np.ndarray | Iterable[float],
    influence: np.ndarray | Iterable[Iterable[float]],
    design: SubsetDesign,
    metric: np.ndarray | Iterable[Iterable[float]] | None = None,
) -> float:
    """Explicitly enumerate the realized fixed-residual gradient MSE."""

    errors = np.asarray(residual, dtype=np.float64)
    matrix = np.asarray(influence, dtype=np.float64)
    if errors.ndim != 1 or matrix.ndim != 2 or matrix.shape[1] != len(errors) or len(design.pi) != len(errors):
        raise ValueError("residual, influence, and design have incompatible shapes")
    if not np.isfinite(errors).all() or not np.isfinite(matrix).all():
        raise ValueError("residual and influence must be finite")
    subsets = enumerate_subsets(len(errors)).astype(np.float64)
    candidate_errors = (subsets / design.pi[None, :] - 1.0) * errors[None, :]
    gradient_errors = candidate_errors @ matrix.T
    if metric is None:
        squared = np.einsum("sp,sp->s", gradient_errors, gradient_errors)
    else:
        weight = validate_psd(metric, "metric")
        if weight.shape != (matrix.shape[0], matrix.shape[0]):
            raise ValueError("metric must be P by P")
        squared = np.einsum("sp,pq,sq->s", gradient_errors, weight, gradient_errors)
    return float(np.dot(design.probabilities, squared))
