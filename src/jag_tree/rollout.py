"""Policy-generation contracts and deterministic offline backend."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Mapping, Protocol

import numpy as np

from .schema import TaskRecord, TreeNode


@dataclass(frozen=True)
class GenerationRequest:
    root_samples: int = 1
    branch_depths: tuple[int, ...] = ()
    children_per_branch: int = 1
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0


class PolicyBackend(Protocol):
    revision: str

    def generate_tree(self, task: TaskRecord, seed: int, request: GenerationRequest | None = None) -> tuple[TreeNode, ...]: ...

    def score_tree(self, task: TaskRecord, nodes: tuple[TreeNode, ...]) -> Mapping[str, np.ndarray]: ...

    def frozen_predictor(self, nodes: tuple[TreeNode, ...]) -> object: ...


def _edge_seed(seed: int, path: tuple[int, ...]) -> int:
    digest = sha256(f"{seed}:{','.join(map(str, path))}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def build_genealogy(
    task: TaskRecord,
    prompt: str,
    policy_revision: str,
    template_version: str,
    seed: int,
    request: GenerationRequest,
    sample_edge: Callable[[tuple[int, ...], int, int], tuple[int, ...]],
    decode: Callable[[tuple[int, ...]], str],
    is_terminal: Callable[[tuple[int, ...]], bool] | None = None,
) -> tuple[TreeNode, ...]:
    """Sample a full tree while storing each conditional edge exactly once."""

    depths = tuple(int(depth) for depth in request.branch_depths)
    terminal = is_terminal or (lambda _: False)
    if request.root_samples < 1 or request.children_per_branch < 1 or request.max_new_tokens < 1:
        raise ValueError("generation counts must be positive")
    if tuple(sorted(set(depths))) != depths or any(depth <= 0 or depth >= request.max_new_tokens for depth in depths):
        raise ValueError("branch depths must be unique, increasing, and below the response cap")
    root_id = f"{task.task_id}:root"
    nodes = [TreeNode(root_id, task.task_id, None, 0, prompt, (), 0, policy_revision, template_version)]
    first_target = depths[0] if depths else request.max_new_tokens
    frontier: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for root_index in range(request.root_samples):
        path = (root_index,)
        edge = sample_edge((), first_target, _edge_seed(seed, path))
        node_id = f"{task.task_id}:edge:{'.'.join(map(str, path))}"
        nodes.append(TreeNode(node_id, task.task_id, root_id, 1, decode(edge), edge, len(edge), policy_revision, template_version))
        if not terminal(edge):
            frontier.append((node_id, edge, path))
    for target in depths[1:] + ((request.max_new_tokens,) if depths else ()):
        expanded: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
        for parent_id, prefix, parent_path in frontier:
            for child_index in range(request.children_per_branch):
                path = parent_path + (child_index,)
                edge = sample_edge(prefix, target - len(prefix), _edge_seed(seed, path))
                full = prefix + edge
                node_id = f"{task.task_id}:edge:{'.'.join(map(str, path))}"
                parent_depth = next(node.depth for node in nodes if node.node_id == parent_id)
                nodes.append(TreeNode(node_id, task.task_id, parent_id, parent_depth + 1, decode(full), edge, len(edge), policy_revision, template_version))
                if not terminal(edge):
                    expanded.append((node_id, full, path))
        frontier = expanded
    return tuple(nodes)


class FakePolicyBackend:
    """Deterministic schema-complete backend for CPU integration tests."""

    def __init__(self, revision: str) -> None:
        self.revision = revision
        self.calls: list[tuple[str, int]] = []

    def generate_tree(self, task: TaskRecord, seed: int, request: GenerationRequest | None = None) -> tuple[TreeNode, ...]:
        _ = request
        self.calls.append((task.task_id, int(seed)))
        root_id = f"{task.task_id}:root"
        root = TreeNode(root_id, task.task_id, None, 0, task.statement, (), 0, self.revision, "fake-v1")
        token_ids = (int(seed) % 997, len(task.task_id))
        leaf = TreeNode(f"{task.task_id}:leaf:0", task.task_id, root_id, 1, f"candidate-{task.task_id}", token_ids, len(token_ids), self.revision, "fake-v1", edge_logprob=-1.0, edge_entropy=0.5)
        return root, leaf

    def score_tree(self, task: TaskRecord, nodes: tuple[TreeNode, ...]) -> Mapping[str, np.ndarray]:
        """Return deterministic fixture score gradients, never token IDs."""

        del task
        result: dict[str, np.ndarray] = {}
        for node in nodes:
            if node.parent_id is None:
                continue
            digest = sha256(f"fake-policy-score:{self.revision}:{node.node_id}:{node.text}".encode()).digest()
            result[node.node_id] = np.asarray([(digest[index] - 127.5) / 127.5 for index in range(4)], dtype=np.float64)
        return result

    def frozen_predictor(self, nodes: tuple[TreeNode, ...]) -> object:
        """Fixture-only outcome-free predictor used by offline CPU tests."""

        from .audit import FrozenMomentPredictor, MomentPrediction

        predictions = {}
        for node in nodes:
            if node.parent_id is None:
                continue
            digest = sha256(f"fake-moments:{self.revision}:{node.node_id}:{node.text}".encode()).digest()
            covariance = np.asarray([(digest[index + 4] - 127.5) / 255.0 for index in range(4)])
            predictions[node.node_id] = MomentPrediction(
                value_variance=0.25 + digest[0] / 255.0,
                gradient_variance_trace=0.25 + digest[1] / 64.0,
                value_gradient_covariance=covariance,
                entropy=max(0.0, float(node.edge_entropy)) + digest[2] / 255.0,
                trace_score=0.25 + digest[3] / 64.0,
            )
        identity = sha256(f"fake-calibration:{self.revision}".encode()).hexdigest()
        return FrozenMomentPredictor(f"fake-calibration:{identity}", predictions)
