"""Independent baselines and the deterministic exact GOAV subset solver."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from coding_opsd.runtime import named_rng

from .estimator import design_risk, validate_psd
from .subsets import SubsetDesign, enumerate_subsets


def _positive_costs(costs: np.ndarray | Iterable[float], candidates: int | None = None) -> np.ndarray:
    values = np.asarray(costs, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or (candidates is not None and values.size != candidates):
        raise ValueError("costs must be a non-empty candidate vector")
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("costs must be finite and strictly positive")
    return values


def _validate_budget(costs: np.ndarray, budget: float, floor: float) -> None:
    if not np.isfinite(budget) or not np.isfinite(floor) or not 0.0 < floor < 1.0:
        raise ValueError("budget must be finite and floor must lie strictly between zero and one")
    minimum = float(floor * costs.sum())
    maximum = float(costs.sum())
    if budget < minimum - 1e-12 or budget > maximum + 1e-12:
        raise ValueError("budget is infeasible for the requested inclusion floor")


def _independent_probabilities(pi: np.ndarray) -> np.ndarray:
    subsets = enumerate_subsets(len(pi)).astype(np.float64)
    probabilities = np.prod(np.where(subsets == 1.0, pi[None, :], 1.0 - pi[None, :]), axis=1)
    probabilities /= probabilities.sum()
    return probabilities


def bernoulli_design(
    pi: np.ndarray | Iterable[float], costs: np.ndarray | Iterable[float] | None = None
) -> SubsetDesign:
    """Construct the exact full-support independent Bernoulli subset design."""

    inclusion = np.asarray(pi, dtype=np.float64)
    if inclusion.ndim != 1 or inclusion.size < 1 or not np.isfinite(inclusion).all():
        raise ValueError("pi must be a non-empty finite vector")
    if (inclusion <= 0.0).any() or (inclusion >= 1.0).any():
        raise ValueError("an exact Bernoulli full-support design requires 0 < pi < 1")
    c = np.ones(len(inclusion)) if costs is None else _positive_costs(costs, len(inclusion))
    return SubsetDesign.from_probabilities(_independent_probabilities(inclusion), c)


def full_audit_design(candidates: int, costs: np.ndarray | Iterable[float] | None = None) -> SubsetDesign:
    """Return the deterministic all-candidate ceiling, exempt from full support."""

    if not isinstance(candidates, (int, np.integer)) or candidates < 1:
        raise ValueError("candidates must be a positive integer")
    c = np.ones(int(candidates)) if costs is None else _positive_costs(costs, int(candidates))
    probabilities = np.zeros(1 << int(candidates), dtype=np.float64)
    probabilities[-1] = 1.0
    return SubsetDesign.from_probabilities(probabilities, c, full_audit=True)


def _scaled_marginals(weights: np.ndarray, costs: np.ndarray, budget: float, floor: float) -> np.ndarray:
    _validate_budget(costs, budget, floor)
    if np.isclose(budget, costs.sum(), atol=1e-12, rtol=0.0):
        # A deterministic ceiling is a separate design; randomized baselines require full support.
        raise ValueError("use full_audit_design for the all-candidate budget")
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != costs.shape or not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("allocation weights must be finite and non-negative")
    if not np.any(values > 0.0):
        values = 1.0 / costs

    minimum_budget = float(floor * costs.sum())
    marginal_floor = floor
    if budget > minimum_budget + 1e-12:
        # Subset-product reconstruction can lose a few ulps even when the
        # requested marginal equals the floor exactly.  Keep a tiny numerical
        # interior margin while bisection preserves the registered budget.
        marginal_floor = min(
            floor + 32.0 * float(np.spacing(floor)),
            float(np.nextafter(1.0, 0.0)),
        )

    positive = values > 0.0
    positive_capacity = np.where(positive, 1.0 - 1e-14, marginal_floor)
    if float(np.dot(positive_capacity, costs)) < budget - 1e-12:
        result = positive_capacity.copy()
        zero = ~positive

        def filled(scale: float) -> np.ndarray:
            candidate = result.copy()
            candidate[zero] = np.clip(marginal_floor + scale / costs[zero], marginal_floor, 1.0 - 1e-14)
            return candidate

        low, high = 0.0, 1.0
        while float(np.dot(filled(high), costs)) < budget:
            high *= 2.0
        for _ in range(120):
            middle = (low + high) / 2.0
            if float(np.dot(filled(middle), costs)) < budget:
                low = middle
            else:
                high = middle
        result = filled((low + high) / 2.0)
        if abs(float(np.dot(result, costs)) - budget) > 1e-9:
            raise ValueError("could not allocate degenerate zero-risk budget")
        return result

    def marginals(scale: float) -> np.ndarray:
        return np.clip(scale * values, marginal_floor, 1.0 - 1e-14)

    low, high = 0.0, 1.0
    while float(np.dot(marginals(high), costs)) < budget:
        high *= 2.0
        if high > 1e300:
            raise ValueError("positive-weight candidates cannot absorb the requested budget")
    for _ in range(120):
        middle = (low + high) / 2.0
        if float(np.dot(marginals(middle), costs)) < budget:
            low = middle
        else:
            high = middle
    result = marginals((low + high) / 2.0)
    if abs(float(np.dot(result, costs)) - budget) > 1e-9:
        raise ValueError("could not match the requested budget")
    return result


def poisson_neyman_design(
    covariance: np.ndarray | Iterable[Iterable[float]],
    influence: np.ndarray | Iterable[Iterable[float]],
    costs: np.ndarray | Iterable[float],
    budget: float,
    floor: float,
) -> SubsetDesign:
    """Solve the independent-Poisson Neyman relaxation by common-lambda bisection."""

    residual_second_moment = np.asarray(covariance, dtype=np.float64)
    matrix = np.asarray(influence, dtype=np.float64)
    c = _positive_costs(costs)
    if residual_second_moment.shape != (len(c), len(c)) or matrix.ndim != 2 or matrix.shape[1] != len(c):
        raise ValueError("covariance, influence, and costs have incompatible shapes")
    residual_second_moment = validate_psd(residual_second_moment, "covariance")
    if not np.isfinite(matrix).all():
        raise ValueError("influence must be finite")
    importance = np.maximum(np.diag(residual_second_moment), 0.0) * np.einsum("pk,pk->k", matrix, matrix)
    base = np.sqrt(importance / c)
    pi = _scaled_marginals(base, c, float(budget), float(floor))
    return bernoulli_design(pi, c)


def bayes_voi_scores(
    covariance: np.ndarray | Iterable[Iterable[float]],
    influence: np.ndarray | Iterable[Iterable[float]],
) -> np.ndarray:
    """Return single-label posterior-mean risk reductions ``||L C_:j||^2 / C_jj``."""

    residual_second_moment = validate_psd(covariance, "covariance")
    matrix = np.asarray(influence, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(residual_second_moment) or not np.isfinite(matrix).all():
        raise ValueError("influence must be finite with one column per candidate")
    projected = matrix @ residual_second_moment
    diagonal = np.diag(residual_second_moment)
    return np.divide(
        np.einsum("pk,pk->k", projected, projected),
        diagonal,
        out=np.zeros(len(diagonal), dtype=np.float64),
        where=diagonal > 0.0,
    )


def score_design(
    scores: np.ndarray | Iterable[float],
    budget: float,
    floor: float,
    costs: np.ndarray | Iterable[float] | None = None,
) -> SubsetDesign:
    """Exponentially tilt nonnegative acquisition scores into exact Bernoulli marginals."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("scores must be a non-empty finite non-negative vector")
    c = np.ones(len(values)) if costs is None else _positive_costs(costs, len(values))
    centered = values - values.max()
    scale = max(float(np.std(values)), 1.0)
    weights = np.exp(centered / scale) / c
    pi = _scaled_marginals(weights, c, float(budget), float(floor))
    return bernoulli_design(pi, c)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def _budget_project(logits: np.ndarray, subset_costs: np.ndarray, budget: float) -> tuple[np.ndarray, np.ndarray]:
    def projected(dual: float) -> np.ndarray:
        return _softmax(logits - dual * subset_costs)

    low, high = -1.0, 1.0
    while float(np.dot(projected(low), subset_costs)) < budget:
        low *= 2.0
    while float(np.dot(projected(high), subset_costs)) > budget:
        high *= 2.0
    for _ in range(100):
        middle = (low + high) / 2.0
        probabilities = projected(middle)
        if float(np.dot(probabilities, subset_costs)) > budget:
            low = middle
        else:
            high = middle
    dual = (low + high) / 2.0
    return projected(dual), logits - dual * subset_costs


def _risk_probability_gradient(covariance: np.ndarray, influence: np.ndarray, design: SubsetDesign) -> np.ndarray:
    subsets = enumerate_subsets(len(design.pi)).astype(np.float64)
    gram_weight = covariance * (influence.T @ influence)
    scaled = gram_weight / np.outer(design.pi, design.pi)
    numerator_rows = np.sum(scaled * design.pi2, axis=1)
    first = np.einsum("si,ij,sj->s", subsets, scaled, subsets)
    second = 2.0 * (subsets @ (numerator_rows / design.pi))
    return first - second


def solve_goav(
    covariance: np.ndarray | Iterable[Iterable[float]],
    influence: np.ndarray | Iterable[Iterable[float]],
    costs: np.ndarray | Iterable[float],
    budget_fraction: float,
    floor: float,
    steps: int = 200,
    lr: float = 0.05,
    restarts: int = 8,
) -> SubsetDesign:
    """Optimize the exact subset risk with deterministic mirror steps and cost projection."""

    residual_second_moment = np.asarray(covariance, dtype=np.float64)
    matrix = np.asarray(influence, dtype=np.float64)
    c = _positive_costs(costs)
    if residual_second_moment.shape != (len(c), len(c)) or matrix.ndim != 2 or matrix.shape[1] != len(c):
        raise ValueError("covariance, influence, and costs have incompatible shapes")
    residual_second_moment = validate_psd(residual_second_moment, "covariance")
    if not np.isfinite(matrix).all():
        raise ValueError("influence must be finite")
    if not 0.0 < budget_fraction < 1.0 or not 0.0 < floor <= budget_fraction:
        raise ValueError("require 0 < floor <= budget_fraction < 1")
    if not isinstance(steps, int) or steps < 0 or not isinstance(restarts, int) or restarts < 1 or not np.isfinite(lr) or lr <= 0.0:
        raise ValueError("steps, restarts, and learning rate must be positive")

    budget = float(budget_fraction * c.sum())
    _validate_budget(c, budget, floor)
    subsets = enumerate_subsets(len(c)).astype(np.float64)
    subset_costs = subsets @ c
    exploration = bernoulli_design(np.full(len(c), budget_fraction), c)
    mixture_weight = float(floor / budget_fraction)

    uniform_logits = np.zeros(len(subsets), dtype=np.float64)
    neyman = poisson_neyman_design(residual_second_moment, matrix, c, budget, max(min(floor, budget_fraction * 0.5), 1e-8))
    voi_scores = bayes_voi_scores(residual_second_moment, matrix)
    voi = score_design(voi_scores, budget, max(min(floor, budget_fraction * 0.5), 1e-8), c)
    initial_logits = [uniform_logits, np.log(neyman.probabilities), np.log(voi.probabilities)]
    rng = named_rng(0, "solve_goav_restarts")
    while len(initial_logits) < restarts:
        initial_logits.append(rng.normal(0.0, 0.25, size=len(subsets)))

    best_design: SubsetDesign | None = None
    best_risk = np.inf
    for restart, raw_logits in enumerate(initial_logits[:restarts]):
        logits = raw_logits.copy()
        for step in range(steps + 1):
            base, projected_logits = _budget_project(logits, subset_costs, budget)
            probabilities = (1.0 - mixture_weight) * base + mixture_weight * exploration.probabilities
            design = SubsetDesign.from_probabilities(probabilities, c, logits=projected_logits, restart=restart)
            risk = design_risk(residual_second_moment, matrix, design)
            feasible = abs(design.expected_cost - budget) <= 1e-4 and float(design.pi.min()) >= floor - 1e-12
            if feasible and risk < best_risk:
                best_risk = risk
                best_design = design
            if step == steps:
                break
            probability_gradient = (1.0 - mixture_weight) * _risk_probability_gradient(residual_second_moment, matrix, design)
            probability_gradient -= float(np.dot(base, probability_gradient))
            infinity_norm = float(np.max(np.abs(probability_gradient)))
            if infinity_norm > 100.0:
                probability_gradient *= 100.0 / infinity_norm
            logits = projected_logits - lr * probability_gradient
    if best_design is None:
        raise RuntimeError("GOAV solver found no feasible iterate")
    return best_design
