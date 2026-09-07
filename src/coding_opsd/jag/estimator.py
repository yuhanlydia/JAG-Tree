"""Bottom-up recursive JAG value/gradient estimator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class TreeNode:
    """A sampled node whose incoming edge carries a policy score."""

    edge_score: np.ndarray
    children: list["TreeNode"] = field(default_factory=list)
    reward: float | None = None
    prefix: tuple[int, ...] = ()
    planned_branching: int | None = None
    planning_events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.edge_score = np.asarray(self.edge_score, dtype=np.float64)


def recursive_estimate(root: TreeNode, baseline: Callable[[TreeNode], float] | float = 0.0) -> tuple[float, np.ndarray]:
    """Return the registered recursive value and score-function gradient.

    A node's baseline is evaluated once and shared by all of its children.
    Leaves deliberately contribute no gradient: their incoming edge is handled by
    their parent, which makes every edge appear exactly once.
    """

    baseline_for = baseline if callable(baseline) else lambda _: float(baseline)

    def visit(node: TreeNode) -> tuple[float, np.ndarray]:
        if not node.children:
            if node.reward is None:
                raise ValueError("leaf nodes require a reward")
            return float(node.reward), np.zeros_like(node.edge_score, dtype=np.float64)
        value_gradient = [visit(child) for child in node.children]
        values = [item[0] for item in value_gradient]
        group_baseline = float(baseline_for(node))
        gradients = [item[1] + (item[0] - group_baseline) * child.edge_score for child, item in zip(node.children, value_gradient)]
        return float(np.mean(values)), np.mean(np.stack(gradients), axis=0, dtype=np.float64)

    return visit(root)


def leaf_equal_estimate(root: TreeNode, baseline: Callable[[TreeNode], float] | float = 0.0) -> tuple[float, np.ndarray]:
    """Naive leaf-weighted estimator, retained as a labelled non-JAG arm."""

    leaves: list[tuple[float, np.ndarray]] = []
    baseline_for = baseline if callable(baseline) else lambda _: float(baseline)

    def visit(node: TreeNode, path_terms: list[tuple[np.ndarray, float]]) -> None:
        if not node.children:
            if node.reward is None:
                raise ValueError("leaf nodes require a reward")
            reward = float(node.reward)
            gradient = sum(
                ((reward - group_baseline) * score for score, group_baseline in path_terms),
                np.zeros_like(node.edge_score, dtype=np.float64),
            )
            leaves.append((reward, gradient))
            return
        group_baseline = float(baseline_for(node))
        for child in node.children:
            visit(child, path_terms + [(child.edge_score, group_baseline)])

    visit(root, [])
    if not leaves:
        raise ValueError("tree must contain a leaf")
    rewards = np.asarray([reward for reward, _ in leaves], dtype=np.float64)
    gradients = np.stack([gradient for _, gradient in leaves])
    return float(np.mean(rewards)), np.mean(gradients, axis=0)
