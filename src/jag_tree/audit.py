"""Outcome-blind deterministic allocation replay over an immutable tree bank.

Allocation consumes a bank that was generated once and a moment predictor fitted
on a disjoint calibration split.  Audit outcomes are used only after selection,
to score the replay against the full-bank registered estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .bank import TreeBank, verify_bank
from .estimator import TreeNode as EstimateNode, recursive_estimate
from .ledger import RunLedger
from .registry import ALLOCATOR_ARMS


@dataclass(frozen=True)
class MomentPrediction:
    """Frozen, outcome-free moment features for one available edge."""

    value_variance: float
    gradient_variance_trace: float
    value_gradient_covariance: np.ndarray
    entropy: float
    trace_score: float

    def __post_init__(self) -> None:
        source = np.asarray(self.value_gradient_covariance, dtype=np.float64).reshape(-1)
        covariance = np.frombuffer(source.tobytes(), dtype=source.dtype)
        object.__setattr__(self, "value_gradient_covariance", covariance)
        values = (self.value_variance, self.gradient_variance_trace, self.entropy, self.trace_score)
        if not np.isfinite(values).all() or min(values) < 0 or not np.isfinite(covariance).all():
            raise ValueError("moment predictions must be finite with nonnegative scalar features")


@dataclass(frozen=True)
class FrozenMomentPredictor:
    """Immutable predictions carrying the calibration artifact identity."""

    calibration_identity: str
    predictions: Mapping[str, MomentPrediction]

    def __post_init__(self) -> None:
        if not self.calibration_identity:
            raise ValueError("calibration identity must be non-empty")
        prepared = {str(key): value for key, value in self.predictions.items()}
        if not prepared or not all(isinstance(value, MomentPrediction) for value in prepared.values()):
            raise ValueError("frozen predictor requires typed moment predictions")
        object.__setattr__(self, "predictions", MappingProxyType(prepared))

    def predict(self, node_id: str) -> MomentPrediction:
        try:
            return self.predictions[node_id]
        except KeyError as exc:
            raise ValueError(f"calibration predictor has no prediction for edge {node_id}") from exc


@dataclass(frozen=True)
class AuditReplay:
    arm: str
    seed: int
    allocation: dict[str, int]
    rows: tuple[dict[str, object], ...]
    ledger: RunLedger
    selected_leaf_ids: tuple[str, ...]
    selected_edge_ids: tuple[str, ...]
    metrics: Mapping[str, float]
    calibration_identity: str


def _score_dimension(bank: TreeBank) -> int:
    dimensions = {np.asarray(value).size for value in bank.score_arrays.values()}
    if not dimensions:
        raise ValueError("audit bank has no policy-score gradients")
    if len(dimensions) != 1 or next(iter(dimensions)) < 1:
        raise ValueError("policy-score gradients must have one positive common dimension")
    return next(iter(dimensions))


def _topology(bank: TreeBank) -> tuple[dict[str, object], dict[str, list[object]], list[object]]:
    by_id = {node.node_id: node for node in bank.nodes}
    children: dict[str, list[object]] = {}
    for node in bank.nodes:
        if node.parent_id is not None:
            children.setdefault(node.parent_id, []).append(node)
    for rows in children.values():
        rows.sort(key=lambda item: item.node_id)
    leaves = sorted((node for node in bank.nodes if node.node_id not in children), key=lambda item: item.node_id)
    return by_id, children, leaves


def _path(node: object, by_id: Mapping[str, object]) -> tuple[object, ...]:
    result: list[object] = []
    current = node
    seen: set[str] = set()
    while current.parent_id is not None:
        if current.node_id in seen:
            raise ValueError("bank genealogy contains a cycle")
        seen.add(current.node_id)
        result.append(current)
        current = by_id[current.parent_id]
    result.reverse()
    return tuple(result)


def _risk(arm: str, prediction: MomentPrediction, path_score: np.ndarray) -> float:
    covariance = prediction.value_gradient_covariance
    if covariance.shape != path_score.shape:
        raise ValueError("predicted covariance dimension does not match policy-score gradients")
    if arm == "uniform":
        return 1.0
    if arm == "entropy":
        return prediction.entropy
    if arm == "value_variance":
        return prediction.value_variance
    if arm == "gradient_only":
        return prediction.gradient_variance_trace
    if arm == "trace_score":
        return prediction.trace_score
    no_cross = prediction.gradient_variance_trace + prediction.value_variance * float(path_score @ path_score)
    if arm == "jag_no_cross":
        return no_cross
    return no_cross + 2.0 * float(path_score @ covariance)


def _estimate_tree(bank: TreeBank, selected_edges: set[str] | None = None) -> tuple[float, np.ndarray]:
    dimension = _score_dimension(bank)
    by_id, children, _ = _topology(bank)

    def build(node: object) -> EstimateNode | None:
        score = np.zeros(dimension, dtype=np.float64) if node.parent_id is None else np.asarray(bank.score_arrays[node.node_id], dtype=np.float64).reshape(-1)
        if node.node_id not in children:
            if selected_edges is not None and node.node_id not in selected_edges:
                return None
            if node.reward is None:
                raise ValueError(f"audit leaf {node.node_id} has no sealed outcome")
            return EstimateNode(score, reward=float(node.reward))
        descendants = [built for child in children[node.node_id] if (built := build(child)) is not None]
        if not descendants:
            return None
        return EstimateNode(score, children=descendants)

    roots = sorted((node for node in bank.nodes if node.parent_id is None), key=lambda item: item.node_id)
    estimates = [built for root in roots if (built := build(root)) is not None]
    if not estimates:
        return 0.0, np.zeros(dimension, dtype=np.float64)
    root = EstimateNode(np.zeros(dimension, dtype=np.float64), children=estimates)
    return recursive_estimate(root)


def _gradient_metrics(estimate: np.ndarray, target: np.ndarray) -> Mapping[str, float]:
    difference = np.asarray(estimate) - np.asarray(target)
    denominator = float(np.linalg.norm(estimate) * np.linalg.norm(target))
    cosine = float(estimate @ target / denominator) if denominator else float(1.0 if np.allclose(estimate, target) else 0.0)
    return MappingProxyType({
        "gradient_mse": float(np.mean(difference * difference)),
        "gradient_bias_norm": float(np.linalg.norm(difference)),
        "gradient_cosine": cosine,
    })


def audit_replay(
    bank: TreeBank | str | Path,
    arm: str,
    seed: int,
    *,
    budget: int = 0,
    predictor: FrozenMomentPredictor | None = None,
) -> AuditReplay:
    loaded = bank if isinstance(bank, TreeBank) else verify_bank(bank)
    if arm not in ALLOCATOR_ARMS:
        raise ValueError(f"unknown allocator arm: {arm!r}")
    if budget < 0:
        raise ValueError("audit budget must be nonnegative")
    if arm != "oracle_moments" and predictor is None:
        raise ValueError("non-oracle replay requires a frozen calibration predictor")
    dimension = _score_dimension(loaded)
    by_id, children, leaves = _topology(loaded)
    paths = {leaf.node_id: _path(leaf, by_id) for leaf in leaves}

    # The oracle is deliberately labelled and isolated.  It may inspect sealed
    # outcomes; every advertised non-oracle arm is prohibited from doing so.
    if arm == "oracle_moments":
        predictions = {
            leaf.node_id: MomentPrediction(
                value_variance=float((float(leaf.reward or 0.0) - 0.5) ** 2),
                gradient_variance_trace=float(np.asarray(loaded.score_arrays[leaf.node_id]) @ np.asarray(loaded.score_arrays[leaf.node_id])),
                value_gradient_covariance=(float(leaf.reward or 0.0) - 0.5) * np.asarray(loaded.score_arrays[leaf.node_id]),
                entropy=float(leaf.edge_entropy),
                trace_score=float(np.asarray(loaded.score_arrays[leaf.node_id]) @ np.asarray(loaded.score_arrays[leaf.node_id])),
            )
            for leaf in leaves
        }
        active_predictor = FrozenMomentPredictor("oracle:audit-outcomes", predictions)
    else:
        assert predictor is not None
        active_predictor = predictor

    roots = [node for node in loaded.nodes if node.parent_id is None]
    candidates: list[tuple[object, tuple[object, ...], int, float, float, np.ndarray]] = []
    for leaf in leaves:
        path = paths[leaf.node_id]
        cost = sum(int(edge.unique_tokens) for edge in path)
        if cost <= 0:
            raise ValueError("every replay trajectory must have positive measured token cost")
        path_score = np.sum(np.stack([np.asarray(loaded.score_arrays[edge.node_id], dtype=np.float64).reshape(-1) for edge in path]), axis=0)
        target_probability = 1.0 / len(roots)
        for edge in path:
            target_probability /= len(children[edge.parent_id])
        if arm == "oracle_moments":
            # Full-information variance-optimal trajectory magnitude. Outcome
            # access is confined to this explicitly labelled oracle arm.
            risk = target_probability * abs(float(leaf.reward or 0.0)) * float(np.linalg.norm(path_score))
        else:
            risk = max(0.0, _risk(arm, active_predictor.predict(leaf.node_id), path_score))
        candidates.append((leaf, path, cost, risk, target_probability, path_score))
    max_cost = max((row[2] for row in candidates), default=budget + 1)
    draw_count = int(budget // max_cost) if candidates else 0
    priorities = np.asarray([row[3] if arm == "oracle_moments" else row[3] / row[2] for row in candidates], dtype=np.float64)
    support_floor = max(float(priorities.max(initial=0.0)) * 1e-6, 1e-12)
    probabilities = (priorities + support_floor) / float(np.sum(priorities + support_floor)) if candidates else np.zeros(0)
    cumulative = np.cumsum(probabilities)
    rng = np.random.default_rng(int(seed))
    selected_edges: set[str] = set()
    selected_leaves: set[str] = set()
    events: list[dict[str, object]] = []
    draw_gradients: list[np.ndarray] = []
    draw_values: list[float] = []
    spent = 0
    for draw in range(draw_count):
        uniform = float(rng.random())
        index = min(int(np.searchsorted(cumulative, uniform, side="right")), len(candidates) - 1)
        leaf, path, cost, risk, target_probability, path_score = candidates[index]
        proposal = float(probabilities[index])
        importance = float(target_probability / proposal)
        reward = float(leaf.reward if leaf.reward is not None else 0.0)
        gradient = importance * reward * path_score
        draw_gradients.append(gradient)
        draw_values.append(importance * reward)
        spent += cost
        selected_leaves.add(leaf.node_id)
        selected_edges.update(edge.node_id for edge in path)
        events.append({
            "draw": draw,
            "leaf_id": leaf.node_id,
            "trajectory_tokens": cost,
            "risk": risk,
            "proposal_probability": proposal,
            "target_probability": target_probability,
            "importance_weight": importance,
            "rng_uniform": uniform,
            "path_edge_ids": tuple(edge.node_id for edge in path),
        })

    allocation: dict[str, int] = {}
    for edge_id in selected_edges:
        parent_id = by_id[edge_id].parent_id
        if parent_id is not None:
            allocation[parent_id] = allocation.get(parent_id, 0) + 1

    target_value, target_gradient = _estimate_tree(loaded)
    if draw_gradients:
        estimates = np.stack(draw_gradients)
        estimated_gradient = np.mean(estimates, axis=0)
        estimated_value = float(np.mean(draw_values))
        per_draw_error = estimates - target_gradient
        per_draw_gradient_mse = float(np.mean(per_draw_error * per_draw_error))
        mcse = float(np.sqrt(np.mean(np.var(estimates, axis=0, ddof=1)) / len(estimates))) if len(estimates) > 1 else float("inf")
    else:
        estimated_gradient = np.zeros(dimension)
        estimated_value = 0.0
        per_draw_gradient_mse = float("inf")
        mcse = float("inf")
    metrics = dict(_gradient_metrics(estimated_gradient, target_gradient))
    estimator_mean_mse = per_draw_gradient_mse / draw_count if draw_count else float("inf")
    metrics["per_draw_gradient_mse"] = per_draw_gradient_mse
    metrics["estimator_mean_gradient_mse"] = estimator_mean_mse
    metrics["gradient_mse"] = estimator_mean_mse
    metrics.update({
        "value_bias": float(estimated_value - target_value),
        "selected_leaves": float(len(selected_leaves)),
        "draw_count": float(draw_count),
        "gradient_mcse": mcse,
        "gradient_bias_z": float(metrics["gradient_bias_norm"] / mcse) if np.isfinite(mcse) and mcse > 0 else float("inf"),
    })
    rows: list[dict[str, object]] = []
    for event in events:
        rows.append({**event, "arm": arm, "seed": int(seed), "calibration_identity": active_predictor.calibration_identity})
    ledger = RunLedger(replay_sampling_tokens=spent)
    return AuditReplay(
        arm,
        int(seed),
        allocation,
        tuple(rows),
        ledger,
        tuple(sorted(selected_leaves)),
        tuple(sorted(selected_edges)),
        MappingProxyType(metrics),
        active_predictor.calibration_identity,
    )
