"""Synthetic correlated test noise and its exact finite joint posterior."""

from __future__ import annotations

from math import lgamma
from typing import Iterable

import numpy as np

from coding_opsd.runtime import named_rng

from .subsets import enumerate_subsets


def _validate_error_parameters(false_negative: float, false_positive: float, rho: float) -> None:
    if not 0.0 <= false_negative <= 1.0 or not 0.0 <= false_positive <= 1.0:
        raise ValueError("false-negative and false-positive means must lie in [0, 1]")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")


def shared_error_moments(mean: float, rho: float) -> tuple[float, float, float, float]:
    """Return exact marginal variance and distinct-draw covariance/ICC for shared beta noise."""

    if not 0.0 <= mean <= 1.0 or not 0.0 <= rho < 1.0:
        raise ValueError("mean and rho must define a valid shared-error model")
    variance = mean * (1.0 - mean)
    covariance = rho * variance
    correlation = 0.0 if variance == 0.0 or rho == 0.0 else covariance / variance
    return float(mean), float(variance), float(covariance), float(correlation)


def simulate_synthetic_oracle(
    groups: int,
    candidates: int,
    cluster_ids: np.ndarray | Iterable[int],
    false_negative: float,
    false_positive: float,
    rho: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw task latents, labels, shared cluster error rates, and cheap outcomes."""

    if not isinstance(groups, int) or groups < 1 or not isinstance(candidates, int) or candidates < 1:
        raise ValueError("groups and candidates must be positive integers")
    clusters = np.asarray(cluster_ids)
    if clusters.ndim != 1 or clusters.size < 1:
        raise ValueError("cluster_ids must be a non-empty vector")
    _validate_error_parameters(false_negative, false_positive, rho)
    rng = named_rng(seed, "goav_synthetic_oracle")
    success = rng.beta(2.0, 6.0, size=groups)
    labels = rng.random((groups, candidates)) < success[:, None]
    evidence = np.empty((groups, candidates, len(clusters)), dtype=np.int8)
    for cluster in np.unique(clusters):
        tests = np.flatnonzero(clusters == cluster)
        if rho == 0.0:
            fn_rates = np.full(groups, false_negative)
            fp_rates = np.full(groups, false_positive)
        else:
            concentration = 1.0 / rho - 1.0
            fn_rates = _draw_beta_or_boundary(rng, groups, false_negative, concentration)
            fp_rates = _draw_beta_or_boundary(rng, groups, false_positive, concentration)
        for test in tests:
            error_rate = np.where(labels, fn_rates[:, None], fp_rates[:, None])
            errors = rng.random((groups, candidates)) < error_rate
            evidence[:, :, test] = np.where(labels, ~errors, errors)
    return labels, evidence


def _draw_beta_or_boundary(rng: np.random.Generator, size: int, mean: float, concentration: float) -> np.ndarray:
    if mean == 0.0 or mean == 1.0:
        return np.full(size, mean)
    return rng.beta(mean * concentration, (1.0 - mean) * concentration, size=size)


def _log_beta(a: float, b: float) -> float:
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def _log_error_marginal(errors: int, trials: int, mean: float, rho: float) -> float:
    if trials == 0:
        return 0.0
    if mean == 0.0:
        return 0.0 if errors == 0 else -np.inf
    if mean == 1.0:
        return 0.0 if errors == trials else -np.inf
    if rho == 0.0:
        return errors * np.log(mean) + (trials - errors) * np.log1p(-mean)
    concentration = 1.0 / rho - 1.0
    alpha = mean * concentration
    beta = (1.0 - mean) * concentration
    return _log_beta(alpha + errors, beta + trials - errors) - _log_beta(alpha, beta)


def oracle_joint_posterior(
    evidence: np.ndarray | Iterable[Iterable[int]],
    false_negative: float,
    false_positive: float,
    rho: float,
    cluster_ids: np.ndarray | Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate the exact label posterior with beta/beta-binomial integration."""

    observations = np.asarray(evidence)
    clusters = np.asarray(cluster_ids)
    if observations.ndim != 2 or observations.shape[0] < 1 or observations.shape[1] < 1:
        raise ValueError("evidence must be a non-empty K by T matrix")
    if not np.isin(observations, [0, 1]).all() or clusters.ndim != 1 or len(clusters) != observations.shape[1]:
        raise ValueError("evidence must be binary and cluster_ids must identify each test")
    _validate_error_parameters(false_negative, false_positive, rho)
    states = enumerate_subsets(observations.shape[0]).astype(np.int8)
    log_weights = np.empty(len(states), dtype=np.float64)
    log_prior_normalizer = _log_beta(2.0, 6.0)
    for state_index, labels in enumerate(states):
        positives = int(labels.sum())
        log_probability = _log_beta(2.0 + positives, 6.0 + len(labels) - positives) - log_prior_normalizer
        for cluster in np.unique(clusters):
            cluster_observations = observations[:, clusters == cluster]
            positive_observations = cluster_observations[labels.astype(bool), :]
            negative_observations = cluster_observations[~labels.astype(bool), :]
            fn_errors = int(np.count_nonzero(positive_observations == 0))
            fp_errors = int(np.count_nonzero(negative_observations == 1))
            log_probability += _log_error_marginal(fn_errors, positive_observations.size, false_negative, rho)
            log_probability += _log_error_marginal(fp_errors, negative_observations.size, false_positive, rho)
        log_weights[state_index] = log_probability
    maximum = float(np.max(log_weights))
    probabilities = np.exp(log_weights - maximum)
    probabilities /= probabilities.sum()
    log_q = np.log(probabilities)
    mu = states.T @ probabilities
    centered = states - mu[None, :]
    covariance = centered.T @ (probabilities[:, None] * centered)
    covariance = (covariance + covariance.T) / 2.0
    return log_q, np.asarray(mu, dtype=np.float64), np.asarray(covariance, dtype=np.float64)
