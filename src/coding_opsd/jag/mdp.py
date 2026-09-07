"""Finite softmax MDP, exact enumeration, and seeded tree sampling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Iterable

import numpy as np

from ..runtime import named_rng
from .allocator import JointMoments, JointTraceMoments
from .estimator import TreeNode


_REWARD_FAMILIES = frozenset({"root_only", "suffix_only", "entropy_distractor", "covariance_reversal"})
_COVARIANCE_LEFT_PREFIX = (0,)
_COVARIANCE_RIGHT_PREFIX = (1,)
_COVARIANCE_STOCHASTIC_DEPTHS = 2


def _policy_cache_key(policy: Any) -> tuple[Any, ...]:
    """Return a stable in-process key for a frozen branching/baseline policy."""

    if isinstance(policy, Mapping):
        return (
            "mapping",
            tuple(sorted((tuple(prefix), int(value)) for prefix, value in policy.items())),
        )
    if callable(policy):
        # Learned predictors are frozen objects for the lifetime of one MDP.
        # Object identity deliberately avoids pretending two closures are the
        # same policy merely because they share a display name.
        return ("callable", id(policy))
    return ("constant", float(policy))


@dataclass(frozen=True)
class ExactTarget:
    probability_sum: float
    expected_reward: float
    gradient: np.ndarray


class FiniteMDP:
    """A bounded finite-horizon categorical policy with shared parameters."""

    def __init__(self, horizon: int, actions: int, theta: Iterable[float], feature_weights: np.ndarray, reward_family: str, reward_seed: int):
        if horizon < 1 or actions < 1:
            raise ValueError("horizon and actions must be positive")
        if reward_family not in _REWARD_FAMILIES:
            raise ValueError(f"unknown reward family {reward_family!r}")
        self.horizon, self.actions = int(horizon), int(actions)
        self.theta = np.asarray(theta, dtype=np.float64).reshape(-1).copy()
        self.feature_weights = np.asarray(feature_weights, dtype=np.float64).copy()
        self.feature_size = 2 * self.actions + 2
        if self.theta.shape != (self.actions,) or self.feature_weights.shape != (self.actions, self.feature_size):
            raise ValueError("theta or feature_weights has an incompatible shape")
        self.reward_family, self.reward_seed = reward_family, int(reward_seed)
        rng = named_rng(self.reward_seed, f"jag-reward:{self.reward_family}:{self.horizon}:{self.actions}")
        self._root_rewards = rng.uniform(-1.0, 1.0, self.actions)
        self._suffix_rewards = rng.uniform(-1.0, 1.0, self.actions)
        self._diagnostic_scale = float(rng.uniform(0.55, 0.9))
        self._entropy_signal = np.linspace(-1.0, 1.0, self.actions, dtype=np.float64) * self._diagnostic_scale
        self._covariance_signal = _covariance_signal(self.actions) * self._diagnostic_scale
        self._conditional_value_cache: dict[tuple[int, ...], float] = {}
        self._conditional_target_cache: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
        self._exact_target_cache: ExactTarget | None = None
        self._estimator_moment_caches: dict[tuple[Any, ...], tuple[dict[tuple[int, ...], Any], dict[tuple[int, ...], Any]]] = {}

    @property
    def parameter_size(self) -> int:
        return self.actions + self.actions * self.feature_size

    @property
    def parameters(self) -> np.ndarray:
        return np.concatenate((self.theta, self.feature_weights.reshape(-1))).copy()

    def with_parameters(self, parameters: Iterable[float]) -> "FiniteMDP":
        flattened = np.asarray(parameters, dtype=np.float64).reshape(-1)
        if flattened.size != self.parameter_size:
            raise ValueError("parameter vector has an incompatible size")
        return FiniteMDP(self.horizon, self.actions, flattened[: self.actions], flattened[self.actions :].reshape(self.actions, self.feature_size), self.reward_family, self.reward_seed)

    def paths(self) -> Iterable[tuple[int, ...]]:
        return product(range(self.actions), repeat=self.horizon)

    def features(self, prefix: Iterable[int]) -> np.ndarray:
        path = tuple(prefix)
        self._validate_prefix(path)
        features = np.zeros(self.feature_size, dtype=np.float64)
        if self.reward_family == "covariance_reversal":
            return features
        if path:
            features[0] = len(path) / self.horizon
            features[1 + path[-1]] = 1.0
            features[1 + self.actions + path[0]] = 1.0
            features[-1] = sum(path) / (self.horizon * max(1, self.actions - 1))
        return features

    def logits(self, prefix: Iterable[int]) -> np.ndarray:
        return self.theta + self.feature_weights @ self.features(prefix)

    def action_probabilities(self, prefix: Iterable[int]) -> np.ndarray:
        path = tuple(prefix)
        self._validate_prefix(path)
        if self.reward_family == "covariance_reversal":
            # The diagnostic isolates the cross term at the first child
            # frontier.  Two masked softmax decisions make both reversal
            # states on-policy; the remaining horizon is deterministic padding
            # so every sampled edge still counts against the tree budget.
            probabilities = np.zeros(self.actions, dtype=np.float64)
            if len(path) >= _COVARIANCE_STOCHASTIC_DEPTHS:
                probabilities[0] = 1.0
                return probabilities
            active_logits = self.logits(path)[:2]
            shifted = active_logits - np.max(active_logits)
            exponentials = np.exp(shifted)
            probabilities[:2] = exponentials / np.sum(exponentials)
            return probabilities
        logits = self.logits(prefix)
        shifted = logits - np.max(logits)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials)

    def action_probability(self, prefix: Iterable[int], action: int) -> float:
        self._validate_action(action)
        return float(self.action_probabilities(prefix)[action])

    def score(self, prefix: Iterable[int], action: int) -> np.ndarray:
        path = tuple(prefix)
        self._validate_prefix(path)
        self._validate_action(action)
        probabilities = self.action_probabilities(path)
        if self.reward_family == "covariance_reversal" and (
            len(path) >= _COVARIANCE_STOCHASTIC_DEPTHS or action >= 2
        ):
            # Masked and absorbing actions are fixed, so their log-probability
            # has no derivative with respect to the softmax parameters.
            return np.zeros(self.parameter_size, dtype=np.float64)
        logit_gradient = -probabilities
        logit_gradient[action] += 1.0
        return np.concatenate((logit_gradient, np.outer(logit_gradient, self.features(path)).reshape(-1)))

    def path_probability(self, path: Iterable[int]) -> float:
        actions = tuple(path)
        if len(actions) != self.horizon:
            raise ValueError("path must have exactly horizon actions")
        probability = 1.0
        for depth, action in enumerate(actions):
            probability *= self.action_probability(actions[:depth], action)
        return float(probability)

    def reward(self, path: Iterable[int]) -> float:
        actions = tuple(path)
        if len(actions) != self.horizon:
            raise ValueError("path must have exactly horizon actions")
        if self.reward_family == "root_only":
            return float(self._root_rewards[actions[0]])
        if self.reward_family == "suffix_only":
            return float(self._suffix_rewards[actions[-1]])
        if self.reward_family == "entropy_distractor":
            if self.horizon < 2 or self.actions < 2:
                return 0.0
            # Prefix (0,) is a true distractor: no downstream action changes
            # reward.  Prefix (1,) is informative through the second action.
            return 0.0 if actions[0] == 0 else float(self._entropy_signal[actions[1]])
        if self.horizon < _COVARIANCE_STOCHASTIC_DEPTHS or self.actions < 2:
            return float(self._covariance_signal[actions[0]])
        if actions[1] != 1:
            return 0.0
        if actions[0] == 0:
            return self._diagnostic_scale
        if actions[0] == 1:
            return -self._diagnostic_scale
        return 0.0

    def expected_reward(self) -> float:
        return exact_target(self).expected_reward

    def structural_audit(self) -> dict[str, Any]:
        """Return the measured structural diagnostics for this environment."""

        return structural_audit(self)

    def conditional_moments(
        self,
        prefix: Iterable[int],
        baseline: float | Callable[[tuple[int, ...]], float] = 0.0,
        continuation_branching: int | Mapping[tuple[int, ...], int] | Callable[[tuple[int, ...]], int] = 1,
        *,
        full_covariance: bool = True,
    ) -> JointMoments | JointTraceMoments:
        """Moments of one *sampled* child continuation contribution.

        Descendant nodes use ``continuation_branching`` and the recursive JAG
        estimator.  The current prefix's own branching is deliberately absent:
        this is the single-child covariance divided by ``b`` by the allocator.
        Unlike a covariance of exact child means, this recurrence retains all
        stochastic suffix variance and the baseline at every descendant.
        """

        path = tuple(prefix)
        self._validate_prefix(path)
        if len(path) == self.horizon:
            raise ValueError("terminal prefixes have no child distribution")
        baseline_key = _policy_cache_key(baseline)
        branching_key = _policy_cache_key(continuation_branching)
        cache_key = ("full" if full_covariance else "trace", baseline_key, branching_key)
        if cache_key not in self._estimator_moment_caches:
            self._estimator_moment_caches[cache_key] = ({}, {})
        node_cache, contribution_cache = self._estimator_moment_caches[cache_key]

        def baseline_at(current: tuple[int, ...]) -> float:
            value = baseline(current) if callable(baseline) else baseline
            result = float(value)
            if not np.isfinite(result):
                raise ValueError("baseline must be finite")
            return result

        def branching_at(current: tuple[int, ...]) -> int:
            if callable(continuation_branching):
                value = continuation_branching(current)
            elif isinstance(continuation_branching, Mapping):
                value = continuation_branching.get(current, 1)
            else:
                value = continuation_branching
            count = int(value)
            if count < 1 or count != value:
                raise ValueError("continuation branching factors must be positive integers")
            return count

        def terminal(current: tuple[int, ...]) -> JointMoments | JointTraceMoments:
            gradient = np.zeros(self.parameter_size, dtype=np.float64)
            covariance = np.zeros_like(gradient)
            if full_covariance:
                return JointMoments(
                    self.reward(current),
                    gradient,
                    0.0,
                    np.zeros((self.parameter_size, self.parameter_size), dtype=np.float64),
                    covariance,
                )
            return JointTraceMoments(self.reward(current), gradient, 0.0, 0.0, covariance)

        def node_moments(current: tuple[int, ...]) -> JointMoments | JointTraceMoments:
            if current in node_cache:
                return node_cache[current]
            if len(current) == self.horizon:
                result = terminal(current)
            else:
                one_child = contribution_moments(current)
                count = branching_at(current)
                if full_covariance:
                    assert isinstance(one_child, JointMoments)
                    result = JointMoments(
                        one_child.value_mean,
                        one_child.gradient_mean,
                        one_child.value_variance / count,
                        one_child.gradient_covariance / count,
                        one_child.value_gradient_covariance / count,
                    )
                else:
                    assert isinstance(one_child, JointTraceMoments)
                    result = JointTraceMoments(
                        one_child.value_mean,
                        one_child.gradient_mean,
                        one_child.value_variance / count,
                        one_child.gradient_variance_trace / count,
                        one_child.value_gradient_covariance / count,
                    )
            node_cache[current] = result
            return result

        def contribution_moments(current: tuple[int, ...]) -> JointMoments | JointTraceMoments:
            if current in contribution_cache:
                return contribution_cache[current]
            probabilities = self.action_probabilities(current)
            transformed = []
            current_baseline = baseline_at(current)
            for action in range(self.actions):
                child = node_moments(current + (action,))
                score = self.score(current, action)
                gradient_mean = child.gradient_mean + (child.value_mean - current_baseline) * score
                value_gradient_covariance = child.value_gradient_covariance + child.value_variance * score
                if full_covariance:
                    assert isinstance(child, JointMoments)
                    gradient_covariance = (
                        child.gradient_covariance
                        + np.outer(score, child.value_gradient_covariance)
                        + np.outer(child.value_gradient_covariance, score)
                        + child.value_variance * np.outer(score, score)
                    )
                    transformed.append(
                        (
                            child.value_mean,
                            gradient_mean,
                            child.value_variance,
                            gradient_covariance,
                            value_gradient_covariance,
                        )
                    )
                else:
                    assert isinstance(child, JointTraceMoments)
                    gradient_trace = (
                        child.gradient_variance_trace
                        + 2.0 * score @ child.value_gradient_covariance
                        + child.value_variance * score @ score
                    )
                    transformed.append(
                        (
                            child.value_mean,
                            gradient_mean,
                            child.value_variance,
                            float(gradient_trace),
                            value_gradient_covariance,
                        )
                    )
            value_mean = float(sum(probability * item[0] for probability, item in zip(probabilities, transformed)))
            gradient_mean = sum(
                (probability * item[1] for probability, item in zip(probabilities, transformed)),
                np.zeros(self.parameter_size, dtype=np.float64),
            )
            value_variance = float(
                sum(
                    probability * (item[2] + (item[0] - value_mean) ** 2)
                    for probability, item in zip(probabilities, transformed)
                )
            )
            value_gradient_covariance = sum(
                (
                    probability
                    * (item[4] + (item[0] - value_mean) * (item[1] - gradient_mean))
                    for probability, item in zip(probabilities, transformed)
                ),
                np.zeros(self.parameter_size, dtype=np.float64),
            )
            if full_covariance:
                gradient_covariance = sum(
                    (
                        probability
                        * (item[3] + np.outer(item[1] - gradient_mean, item[1] - gradient_mean))
                        for probability, item in zip(probabilities, transformed)
                    ),
                    np.zeros((self.parameter_size, self.parameter_size), dtype=np.float64),
                )
                result = JointMoments(
                    value_mean,
                    gradient_mean,
                    value_variance,
                    gradient_covariance,
                    value_gradient_covariance,
                )
            else:
                gradient_trace = float(
                    sum(
                        probability * (item[3] + np.dot(item[1] - gradient_mean, item[1] - gradient_mean))
                        for probability, item in zip(probabilities, transformed)
                    )
                )
                result = JointTraceMoments(
                    value_mean,
                    gradient_mean,
                    value_variance,
                    gradient_trace,
                    value_gradient_covariance,
                )
            contribution_cache[current] = result
            return result

        return contribution_moments(path)

    def _conditional_value(self, prefix: tuple[int, ...]) -> float:
        if prefix not in self._conditional_value_cache:
            if len(prefix) == self.horizon:
                self._conditional_value_cache[prefix] = self.reward(prefix)
            else:
                probabilities = self.action_probabilities(prefix)
                self._conditional_value_cache[prefix] = float(sum(probability * self._conditional_value(prefix + (action,)) for action, probability in enumerate(probabilities)))
        return self._conditional_value_cache[prefix]

    def _conditional_target(self, prefix: tuple[int, ...]) -> tuple[float, np.ndarray]:
        """Exact downstream value and gradient conditional on ``prefix``."""

        if prefix not in self._conditional_target_cache:
            if len(prefix) == self.horizon:
                result = (self.reward(prefix), np.zeros(self.parameter_size, dtype=np.float64))
            else:
                probabilities = self.action_probabilities(prefix)
                children = [self._conditional_target(prefix + (action,)) for action in range(self.actions)]
                value = float(sum(probability * child_value for probability, (child_value, _) in zip(probabilities, children)))
                gradient = np.zeros(self.parameter_size, dtype=np.float64)
                for action, (probability, (child_value, child_gradient)) in enumerate(zip(probabilities, children)):
                    gradient += probability * (child_gradient + child_value * self.score(prefix, action))
                result = (value, gradient)
            self._conditional_target_cache[prefix] = result
        return self._conditional_target_cache[prefix]

    def _validate_prefix(self, prefix: tuple[int, ...]) -> None:
        if len(prefix) > self.horizon:
            raise ValueError("prefix cannot exceed horizon")
        if any(action < 0 or action >= self.actions for action in prefix):
            raise ValueError("prefix contains an invalid action")

    def _validate_action(self, action: int) -> None:
        if action < 0 or action >= self.actions:
            raise ValueError("invalid action")


def _covariance_signal(actions: int) -> np.ndarray:
    """A bounded vector whose uniform-policy covariance has c0 == -c1."""

    if actions == 1:
        return np.zeros(1, dtype=np.float64)
    if actions == 2:
        return np.array([0.0, 1.0], dtype=np.float64)
    # For r=[0,1,x,...,x], solve x(x-mu)=(1-mu)/2.  Then the
    # value-gradient covariance has only two nonzero theta coordinates, equal
    # and opposite.  The positive root lies in [0,1].
    discriminant = float((actions - 4) ** 2 + 16 * (actions - 1))
    x = (-(actions - 4) + np.sqrt(discriminant)) / 8.0
    return np.array([0.0, 1.0] + [float(x)] * (actions - 2), dtype=np.float64)


def make_phase0_mdp(horizon: int, actions: int, reward_family: str, seed: int) -> FiniteMDP:
    """Create one seeded Phase-0 environment, including diagnostic policies.

    The two diagnostic families use seed-varying common-logit nuisance
    parameters.  These keep the shared-softmax parameterization while making
    their designated entropy/covariance relations exact rather than accidental.
    """

    if horizon < 1 or actions < 1:
        raise ValueError("horizon and actions must be positive")
    feature_size = 2 * actions + 2
    rng = named_rng(seed, f"jag-mdp:{reward_family}")
    if reward_family == "entropy_distractor":
        common_theta = float(rng.normal(0.0, 0.3))
        theta = np.full(actions, common_theta, dtype=np.float64)
        common_features = rng.normal(0.0, 0.05, feature_size)
        weights = np.tile(common_features, (actions, 1))
        if actions >= 2:
            contrast = np.linspace(1.4, -1.4, actions, dtype=np.float64)
            contrast *= float(rng.uniform(0.9, 1.1))
            weights[:, 1 + actions + 1] += contrast
    elif reward_family == "covariance_reversal":
        theta = np.full(actions, float(rng.normal(0.0, 0.3)), dtype=np.float64)
        weights = np.tile(rng.normal(0.0, 0.05, feature_size), (actions, 1))
    else:
        theta = rng.normal(0.0, 0.3, actions)
        weights = rng.normal(0.0, 0.2, (actions, feature_size))
    return FiniteMDP(horizon, actions, theta, weights, reward_family, reward_seed=seed)


def exact_target(mdp: FiniteMDP) -> ExactTarget:
    """Enumerate every terminal path and compute the exact REINFORCE gradient."""

    if mdp._exact_target_cache is None:
        # The recursion visits every prefix and terminal action path once.  Its
        # local derivative is exactly g_v + V_v*s_uv, so it is algebraically
        # equivalent to summing REINFORCE scores over all terminal paths.
        expected_reward, gradient = mdp._conditional_target(())
        masses: dict[tuple[int, ...], float] = {}
        def mass(prefix: tuple[int, ...]) -> float:
            if prefix not in masses:
                masses[prefix] = 1.0 if len(prefix) == mdp.horizon else float(sum(mdp.action_probability(prefix, action) * mass(prefix + (action,)) for action in range(mdp.actions)))
            return masses[prefix]
        mdp._exact_target_cache = ExactTarget(mass(()), expected_reward, gradient.copy())
    cached = mdp._exact_target_cache
    return ExactTarget(cached.probability_sum, cached.expected_reward, cached.gradient.copy())


def sample_chain(mdp: FiniteMDP, rng: np.random.Generator) -> TreeNode:
    """Draw a single trajectory and materialize it as a one-child tree."""

    root = TreeNode(np.zeros(mdp.parameter_size), prefix=())
    current = root
    prefix: tuple[int, ...] = ()
    for _ in range(mdp.horizon):
        action = int(rng.choice(mdp.actions, p=mdp.action_probabilities(prefix)))
        prefix = prefix + (action,)
        child = TreeNode(mdp.score(prefix[:-1], action), prefix=prefix)
        current.children = [child]
        current.planned_branching = 1
        current = child
    current.reward = mdp.reward(prefix)
    return root


def sample_balanced_tree(mdp: FiniteMDP, branching: int | dict[tuple[int, ...], int] | Callable[[tuple[int, ...]], int], rng: np.random.Generator) -> TreeNode:
    """Sample a replacement-child tree, fixing a node's allocation before draws."""

    def chosen_branching(prefix: tuple[int, ...]) -> int:
        value = branching(prefix) if callable(branching) else branching.get(prefix, 1) if isinstance(branching, dict) else branching
        if int(value) < 1:
            raise ValueError("branching must be positive")
        return int(value)

    def build(prefix: tuple[int, ...], score: np.ndarray) -> TreeNode:
        node = TreeNode(score, prefix=prefix)
        if len(prefix) == mdp.horizon:
            node.reward = mdp.reward(prefix)
            return node
        count = chosen_branching(prefix)  # Frozen before any child outcome is available.
        node.planned_branching = count
        node.children = []
        for _ in range(count):
            action = int(rng.choice(mdp.actions, p=mdp.action_probabilities(prefix)))
            node.children.append(build(prefix + (action,), mdp.score(prefix, action)))
        return node

    return build((), np.zeros(mdp.parameter_size))


def structural_audit(mdp: FiniteMDP, target: ExactTarget | None = None) -> dict[str, Any]:
    """Measure the advertised diagnostic relation at both designated nodes."""

    _ = target
    if mdp.reward_family == "root_only":
        rewards = mdp._root_rewards
    elif mdp.reward_family == "suffix_only":
        rewards = mdp._suffix_rewards
    elif mdp.reward_family == "entropy_distractor":
        rewards = np.concatenate((np.zeros(1), mdp._entropy_signal))
    else:
        rewards = mdp._covariance_signal
    result: dict[str, Any] = {
        "family": mdp.reward_family,
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "status": "NOT_APPLICABLE",
        "nodes": {},
        "checks": {},
    }
    if mdp.actions < 2:
        result["reason"] = "diagnostic node pair requires actions>=2"
        return result

    if mdp.reward_family == "covariance_reversal":
        if mdp.horizon < _COVARIANCE_STOCHASTIC_DEPTHS:
            result["reason"] = "registered covariance pair requires horizon>=2"
            return result
        left_prefix, right_prefix = _COVARIANCE_LEFT_PREFIX, _COVARIANCE_RIGHT_PREFIX
    else:
        if mdp.horizon < 2:
            result["reason"] = "diagnostic node pair requires horizon>=2"
            return result
        left_prefix, right_prefix = (0,), (1,)

    def node_measure(prefix: tuple[int, ...]) -> dict[str, Any]:
        probabilities = mdp.action_probabilities(prefix)
        positive_probabilities = probabilities[probabilities > 0.0]
        moments = mdp.conditional_moments(prefix, full_covariance=False)
        accumulated_score = _accumulated_prefix_score(mdp, prefix)
        cross_term = float(2.0 * accumulated_score @ moments.value_gradient_covariance)
        return {
            "prefix": list(prefix),
            "local_features": mdp.features(prefix).tolist(),
            "accumulated_score": accumulated_score.tolist(),
            "entropy": float(-np.sum(positive_probabilities * np.log(positive_probabilities))),
            "value_variance": float(moments.value_variance),
            "cross_term": cross_term,
            "joint_risk": float(max(0.0, joint_risk_for_audit(moments, accumulated_score))),
            "value_gradient_covariance": moments.value_gradient_covariance.tolist(),
            "value_gradient_covariance_norm": float(np.linalg.norm(moments.value_gradient_covariance)),
        }

    left, right = node_measure(left_prefix), node_measure(right_prefix)
    # Preserve the original scalar field for downstream readers while adding
    # the paired evidence needed to validate the construction.
    result["cross_term_prefix"] = "/".join(map(str, left_prefix))
    result["cross_term_comparator_prefix"] = "/".join(map(str, right_prefix))
    result["measured_cross_term"] = left["cross_term"]
    tolerance = 1e-10 * max(
        1.0,
        abs(float(left["value_variance"])),
        abs(float(right["value_variance"])),
        abs(float(left["cross_term"])),
        abs(float(right["cross_term"])),
    )
    result["tolerance"] = float(tolerance)
    if mdp.reward_family == "entropy_distractor":
        valid = bool(
            float(left["entropy"]) > float(right["entropy"]) + tolerance
            and float(left["joint_risk"]) + tolerance < float(right["joint_risk"])
        )
        result["nodes"] = {"distractor": left, "informative": right}
        result["checks"] = {
            "entropy_distractor_valid": valid,
            "entropy_gap": float(left["entropy"] - right["entropy"]),
            "risk_gap_informative_minus_distractor": float(right["joint_risk"] - left["joint_risk"]),
        }
        result["status"] = "PASS" if valid else "FAIL"
    elif mdp.reward_family == "covariance_reversal":
        variance_gap = float(abs(float(left["value_variance"]) - float(right["value_variance"])))
        cross_sum = float(abs(float(left["cross_term"]) + float(right["cross_term"])))
        left_covariance = np.asarray(left["value_gradient_covariance"], dtype=np.float64)
        right_covariance = np.asarray(right["value_gradient_covariance"], dtype=np.float64)
        covariance_difference_norm = float(np.linalg.norm(left_covariance - right_covariance))
        covariance_nonzero = bool(np.linalg.norm(left_covariance) > tolerance)
        opposite = bool(float(left["cross_term"]) * float(right["cross_term"]) < 0.0)
        left_score = np.asarray(left["accumulated_score"], dtype=np.float64)
        right_score = np.asarray(right["accumulated_score"], dtype=np.float64)
        score_sum_norm = float(np.linalg.norm(left_score + right_score))
        score_norm_gap = float(abs(np.linalg.norm(left_score) - np.linalg.norm(right_score)))
        left_no_cross = float(left["joint_risk"] - left["cross_term"])
        right_no_cross = float(right["joint_risk"] - right["cross_term"])
        no_cross_gap = float(abs(left_no_cross - right_no_cross))
        valid = bool(
            variance_gap <= tolerance
            and cross_sum <= tolerance
            and covariance_difference_norm <= tolerance
            and covariance_nonzero
            and opposite
            and np.linalg.norm(np.asarray(left["local_features"]) - np.asarray(right["local_features"])) <= tolerance
            and score_sum_norm <= tolerance
            and score_norm_gap <= tolerance
            and no_cross_gap <= tolerance
        )
        result["nodes"] = {"left": left, "right": right}
        result["checks"] = {
            "covariance_reversal_valid": valid,
            "equal_value_variance_abs_gap": variance_gap,
            "opposite_cross_term_abs_sum": cross_sum,
            "same_covariance_vector_difference_norm": covariance_difference_norm,
            "covariance_vector_nonzero": covariance_nonzero,
            "opposite_nonzero_signs": opposite,
            "local_feature_difference_norm": float(
                np.linalg.norm(np.asarray(left["local_features"]) - np.asarray(right["local_features"]))
            ),
            "opposite_accumulated_score_vector_sum_norm": score_sum_norm,
            "equal_accumulated_score_norm_abs_gap": score_norm_gap,
            "equal_no_cross_risk_abs_gap": no_cross_gap,
        }
        result["status"] = "PASS" if valid else "FAIL"
    else:
        result["nodes"] = {"audit": left}
    return result


def _accumulated_prefix_score(mdp: FiniteMDP, prefix: tuple[int, ...]) -> np.ndarray:
    return np.sum([mdp.score(prefix[:depth], action) for depth, action in enumerate(prefix)], axis=0) if prefix else np.zeros(mdp.parameter_size, dtype=np.float64)


def joint_risk_for_audit(moments: JointMoments | JointTraceMoments, accumulated_score: np.ndarray) -> float:
    """Local import-free spelling of the registered joint trace risk."""

    return float(
        (moments.gradient_variance_trace if isinstance(moments, JointTraceMoments) else np.trace(moments.gradient_covariance))
        + 2.0 * accumulated_score @ moments.value_gradient_covariance
        + moments.value_variance * accumulated_score @ accumulated_score
    )
