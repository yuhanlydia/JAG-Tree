"""Joint value-gradient covariance and deterministic integer allocation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class JointMoments:
    """Population moments for one child contribution ``(V, g)``."""

    value_mean: float
    gradient_mean: np.ndarray
    value_variance: float
    gradient_covariance: np.ndarray
    value_gradient_covariance: np.ndarray

    @classmethod
    def from_samples(cls, values: np.ndarray | Iterable[float], gradients: np.ndarray) -> "JointMoments":
        values_array = np.asarray(values, dtype=np.float64).reshape(-1)
        gradients_array = np.asarray(gradients, dtype=np.float64)
        if gradients_array.ndim != 2 or gradients_array.shape[0] != values_array.size or not values_array.size:
            raise ValueError("values and gradients must contain the same positive sample count")
        centered_values = values_array - np.mean(values_array)
        centered_gradients = gradients_array - np.mean(gradients_array, axis=0)
        count = values_array.size
        return cls(
            float(np.mean(values_array)),
            np.mean(gradients_array, axis=0),
            float(centered_values @ centered_values / count),
            centered_gradients.T @ centered_gradients / count,
            centered_values @ centered_gradients / count,
        )


@dataclass(frozen=True)
class JointTraceMoments:
    """Compact moments when only trace-gradient risk is required."""

    value_mean: float
    gradient_mean: np.ndarray
    value_variance: float
    gradient_variance_trace: float
    value_gradient_covariance: np.ndarray


def _gradient_trace(moments: JointMoments | JointTraceMoments) -> float:
    if isinstance(moments, JointTraceMoments):
        return float(moments.gradient_variance_trace)
    return float(np.trace(moments.gradient_covariance))


def joint_risk(moments: JointMoments | JointTraceMoments, accumulated_score: np.ndarray | Iterable[float]) -> float:
    """Population trace of ``Cov(g + V * S)`` including both cross terms."""

    score = np.asarray(accumulated_score, dtype=np.float64).reshape(-1)
    if score.shape != moments.value_gradient_covariance.shape:
        raise ValueError("accumulated score dimension does not match moments")
    return float(
        _gradient_trace(moments)
        + 2.0 * score @ moments.value_gradient_covariance
        + moments.value_variance * score @ score
    )


def joint_risk_no_cross(moments: JointMoments | JointTraceMoments, accumulated_score: np.ndarray | Iterable[float]) -> float:
    """Ablation of :func:`joint_risk` with value-gradient covariance removed."""

    score = np.asarray(accumulated_score, dtype=np.float64).reshape(-1)
    return float(_gradient_trace(moments) + moments.value_variance * score @ score)


def transport_weight(ancestor_branching: Iterable[int]) -> float:
    """Transport a local error through strict ancestors to the synthetic root."""

    result = 1.0
    for branching in ancestor_branching:
        if int(branching) < 1:
            raise ValueError("branching factors must be positive")
        result /= int(branching)
    return result


def marginal_score(risk: float, branching: int, cost: float, ancestor_branching: Iterable[int] = ()) -> float:
    """Greedy benefit of the next child under the registered ``b(b+1)`` rule."""

    if branching < 1 or cost <= 0:
        raise ValueError("branching and cost must be positive")
    weight = transport_weight(ancestor_branching)
    return float(weight * weight * risk / (cost * branching * (branching + 1)))


@dataclass(frozen=True)
class AllocationItem:
    """A node with one or more already allocated children."""

    node_id: str
    branching: int
    risk: float
    cost: float = 1.0
    ancestor_branching: tuple[int, ...] = ()
    parent_id: str | None = None
    max_branching: int | None = None

    def __post_init__(self) -> None:
        if self.branching < 1 or self.cost <= 0 or self.risk < 0:
            raise ValueError("allocation items need positive branching/cost and nonnegative risk")
        if self.parent_id is not None and self.ancestor_branching:
            raise ValueError("parent-linked items inherit fixed ancestry from the top linked ancestor")
        if self.max_branching is not None and self.max_branching < self.branching:
            raise ValueError("max_branching cannot be below current branching")
        transport_weight(self.ancestor_branching)


@dataclass(frozen=True)
class AllocationNode:
    """Observed, outcome-free features available when an allocation is made."""

    node_id: str
    branching: int
    cost: float
    ancestor_score: np.ndarray
    entropy: float = 0.0
    trace_score: float = 0.0
    parent_id: str | None = None
    max_branching: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ancestor_score", np.asarray(self.ancestor_score, dtype=np.float64).reshape(-1))
        if not self.node_id or self.branching < 1 or self.cost <= 0:
            raise ValueError("allocation nodes require an ID and positive branching/cost")
        if min(self.entropy, self.trace_score) < 0:
            raise ValueError("allocation features must be nonnegative")


def _coerce_node(value: AllocationNode | dict[str, Any]) -> AllocationNode:
    if isinstance(value, AllocationNode):
        return value
    forbidden = {"outcome", "reward", "passed", "test_result"}.intersection(value)
    if forbidden:
        raise ValueError(f"outcome-leaking allocation fields are forbidden: {sorted(forbidden)}")
    return AllocationNode(**value)


def allocate(
    nodes: Iterable[AllocationNode | dict[str, Any]],
    moments: dict[str, JointMoments | JointTraceMoments],
    budget: int | float,
    arm: str,
) -> dict[str, int]:
    """Allocate an additive budget through the common controlled-arm boundary."""

    from .baselines import arm_risk

    prepared = tuple(_coerce_node(node) for node in nodes)
    missing = sorted(node.node_id for node in prepared if node.node_id not in moments)
    if missing:
        raise ValueError(f"moments missing for nodes: {missing}")
    items = [AllocationItem(node.node_id, node.branching, max(0.0, arm_risk(node, moments[node.node_id], arm)), node.cost, parent_id=node.parent_id, max_branching=node.max_branching) for node in prepared]
    return greedy_allocate(items, budget)


@dataclass(frozen=True)
class FrontierItem:
    """One already-known unopened node in a frozen allocation frontier."""

    node_id: str
    parent_id: str | None
    branching: int
    risk: float
    cost: float
    ancestor_branching: tuple[int, ...] = ()
    max_branching: int | None = None

    def __post_init__(self) -> None:
        if self.branching < 1 or self.cost <= 0 or self.risk < 0:
            raise ValueError("frontier items need positive branching/cost and nonnegative risk")
        if self.parent_id is not None and self.ancestor_branching:
            raise ValueError("parent-linked frontier items inherit fixed ancestry from the top linked ancestor")
        if self.max_branching is not None and self.max_branching < self.branching:
            raise ValueError("max_branching cannot be below current branching")
        transport_weight(self.ancestor_branching)


@dataclass(frozen=True)
class FrontierPlan:
    branching: dict[str, int]
    events: tuple[dict[str, Any], ...]
    remaining_budget: float = 0.0
    objective_before: float | None = None
    objective_after: float | None = None
    enumerated_states: int | None = None


@dataclass(frozen=True)
class OracleScope:
    """Hard limits for the deliberately tiny exhaustive frontier oracle."""

    max_horizon: int = 3
    max_budget: int = 12
    max_frontier_nodes: int = 8
    max_states: int = 100_000

    def __post_init__(self) -> None:
        if min(self.max_horizon, self.max_budget, self.max_frontier_nodes, self.max_states) < 1:
            raise ValueError("oracle scope limits must be positive")


class OracleUnavailableError(RuntimeError):
    """Raised instead of silently replacing the exact oracle by a heuristic."""


def _frontier_ancestry(item: FrontierItem, by_id: dict[str, FrontierItem], branching: dict[str, int]) -> tuple[int, ...]:
    factors: list[int] = []
    current = item
    seen = {item.node_id}
    while current.parent_id is not None:
        parent = current.parent_id
        if parent in seen or parent not in by_id:
            raise ValueError("frontier parents must be known and acyclic")
        seen.add(parent)
        current = by_id[parent]
        factors.append(branching[parent])
    factors.extend(current.ancestor_branching)
    return tuple(factors)


def greedy_frontier_plan(items: Iterable[FrontierItem], extra_budget: int | float) -> FrontierPlan:
    """Allocate live frontier expansions by the registered marginal score."""

    ordered = tuple(sorted(items, key=lambda item: item.node_id))
    by_id = {item.node_id: item for item in ordered}
    if len(by_id) != len(ordered):
        raise ValueError("frontier node IDs must be unique")
    branching = {item.node_id: item.branching for item in ordered}
    for item in ordered:
        _frontier_ancestry(item, by_id, branching)
    remaining = float(extra_budget)
    if remaining < 0:
        raise ValueError("extra_budget must be nonnegative")
    events: list[dict[str, Any]] = []
    while True:
        eligible = [
            item
            for item in ordered
            if item.cost <= remaining + 1e-12
            and (item.max_branching is None or branching[item.node_id] < item.max_branching)
        ]
        if not eligible:
            break
        candidates = []
        for item in eligible:
            ancestors = _frontier_ancestry(item, by_id, branching)
            item_score = marginal_score(item.risk, branching[item.node_id], item.cost, ancestors)
            candidates.append(
                {
                    "node_id": item.node_id,
                    "risk": float(item.risk),
                    "cost": float(item.cost),
                    "before": int(branching[item.node_id]),
                    "ancestor_branching": list(ancestors),
                    "score": float(item_score),
                }
            )
        selected_row = min(candidates, key=lambda row: (-float(row["score"]), str(row["node_id"])))
        selected = by_id[str(selected_row["node_id"])]
        before = branching[selected.node_id]
        recomputed = marginal_score(
            selected.risk,
            before,
            selected.cost,
            tuple(int(value) for value in selected_row["ancestor_branching"]),
        )
        events.append(
            {
                **selected_row,
                "after": before + 1,
                "score_identity_error": float(abs(float(selected_row["score"]) - recomputed)),
                "candidates": candidates,
            }
        )
        branching[selected.node_id] = before + 1
        remaining -= selected.cost
    return FrontierPlan(branching, tuple(events), remaining_budget=float(max(0.0, remaining)))


def exact_frontier_plan(items: Iterable[FrontierItem], extra_budget: int | float, scope: OracleScope) -> FrontierPlan:
    """Exhaustively optimize one live frontier or declare it unavailable.

    This is intentionally not advertised as a global tree dynamic program.  It
    is exact for the already observed, unopened frontier and its fixed/live
    ancestry.  The caller must freeze the returned branch counts before draws.
    """

    ordered = tuple(sorted(items, key=lambda item: item.node_id))
    if float(extra_budget) > float(scope.max_budget):
        raise OracleUnavailableError(
            f"frontier budget {extra_budget} exceeds max_budget={scope.max_budget}"
        )
    if len(ordered) > scope.max_frontier_nodes:
        raise OracleUnavailableError(
            f"frontier node count {len(ordered)} exceeds max_frontier_nodes={scope.max_frontier_nodes}"
        )
    budget = float(extra_budget)
    if budget < 0:
        raise ValueError("extra_budget must be nonnegative")
    state_count = 1
    for item in ordered:
        possible_extras = int(np.floor(budget / item.cost))
        if item.max_branching is not None:
            possible_extras = min(possible_extras, item.max_branching - item.branching)
        state_count *= possible_extras + 1
        if state_count > scope.max_states:
            raise OracleUnavailableError(
                f"frontier enumeration needs more than max_states={scope.max_states}"
            )
    allocation_items = [
        AllocationItem(
            node_id=item.node_id,
            branching=item.branching,
            risk=item.risk,
            cost=item.cost,
            ancestor_branching=item.ancestor_branching,
            parent_id=item.parent_id,
            max_branching=item.max_branching,
        )
        for item in ordered
    ]
    before = {item.node_id: item.branching for item in ordered}
    after = integer_oracle(allocation_items, budget)
    used = float(sum((after[item.node_id] - item.branching) * item.cost for item in ordered))
    before_objective = _objective(tuple(allocation_items), before, {item.node_id: item for item in allocation_items})
    after_objective = _objective(tuple(allocation_items), after, {item.node_id: item for item in allocation_items})
    event = {
        "kind": "exact_frontier_solution",
        "enumerated_states": int(state_count),
        "objective_before": float(before_objective),
        "objective_after": float(after_objective),
        "optimality_gap": 0.0,
        "budget_used": used,
    }
    return FrontierPlan(
        after,
        (event,),
        remaining_budget=float(max(0.0, budget - used)),
        objective_before=float(before_objective),
        objective_after=float(after_objective),
        enumerated_states=int(state_count),
    )


def _current_ancestry(item: AllocationItem, by_id: dict[str, AllocationItem], allocations: dict[str, int]) -> tuple[int, ...]:
    """Resolve strict-ancestor branchings from the current allocation state."""

    if item.parent_id is None:
        return item.ancestor_branching
    factors: list[int] = []
    parent_id = item.parent_id
    seen = {item.node_id}
    while parent_id is not None:
        if parent_id in seen or parent_id not in by_id:
            raise ValueError("allocation parent links must form a known acyclic tree")
        seen.add(parent_id)
        parent_item = by_id[parent_id]
        factors.append(allocations[parent_id])
        parent_id = parent_item.parent_id
        if parent_id is None:
            factors.extend(parent_item.ancestor_branching)
    return tuple(factors)


def _objective(items: tuple[AllocationItem, ...], allocations: dict[str, int], by_id: dict[str, AllocationItem]) -> float:
    return float(sum(transport_weight(_current_ancestry(item, by_id, allocations)) ** 2 * item.risk / allocations[item.node_id] for item in items))


def greedy_allocate(items: Iterable[AllocationItem], total_extra: int | float) -> dict[str, int]:
    """Allocate extra additive budget greedily, with node-ID tie breaking."""

    ordered = tuple(sorted(items, key=lambda item: item.node_id))
    if len({item.node_id for item in ordered}) != len(ordered):
        raise ValueError("node IDs must be unique")
    by_id = {item.node_id: item for item in ordered}
    remaining = float(total_extra)
    if remaining < 0:
        raise ValueError("total_extra must be nonnegative")
    allocated = {item.node_id: item.branching for item in ordered}
    for item in ordered:
        _current_ancestry(item, by_id, allocated)
    while True:
        eligible = [
            item
            for item in ordered
            if item.cost <= remaining + 1e-12
            and (item.max_branching is None or allocated[item.node_id] < item.max_branching)
        ]
        if not eligible:
            break
        selected = min(
            eligible,
            key=lambda item: (-marginal_score(item.risk, allocated[item.node_id], item.cost, _current_ancestry(item, by_id, allocated)), item.node_id),
        )
        allocated[selected.node_id] += 1
        remaining -= selected.cost
    return allocated


def integer_oracle(items: Iterable[AllocationItem], total_extra: int | float) -> dict[str, int]:
    """Exhaustively minimize ``sum(w**2 * a / b)`` for tiny additive budgets."""

    ordered = tuple(sorted(items, key=lambda item: item.node_id))
    by_id = {item.node_id: item for item in ordered}
    if len(by_id) != len(ordered):
        raise ValueError("node IDs must be unique")
    budget = float(total_extra)
    if budget < 0:
        raise ValueError("total_extra must be nonnegative")
    ranges = [
        range(
            item.branching,
            min(
                item.branching + int(np.floor(budget / item.cost)),
                item.max_branching if item.max_branching is not None else item.branching + int(np.floor(budget / item.cost)),
            )
            + 1,
        )
        for item in ordered
    ]
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for allocation in product(*ranges):
        used = sum((branch - item.branching) * item.cost for item, branch in zip(ordered, allocation))
        if used <= budget + 1e-12:
            by_node = {item.node_id: branch for item, branch in zip(ordered, allocation)}
            candidates.append((_objective(ordered, by_node, by_id), allocation))
    if not candidates:
        return {item.node_id: item.branching for item in ordered}
    # At equal objective, assign the first available marginal unit to the
    # lexicographically lowest node ID.
    _, best = min(candidates, key=lambda candidate: (candidate[0], tuple(-branch for branch in candidate[1])))
    return {item.node_id: branch for item, branch in zip(ordered, best)}
