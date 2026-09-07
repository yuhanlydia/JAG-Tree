"""Deterministic Phase-0 JAG experiment and auditable live tree planner."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
import math
import os
from typing import Any, Callable

import numpy as np

from ..metrics import cosine_similarity
from ..runtime import named_rng
from .allocator import (
    FrontierItem,
    OracleScope,
    OracleUnavailableError,
    exact_frontier_plan,
    greedy_frontier_plan,
    joint_risk,
    joint_risk_no_cross,
)
from .estimator import TreeNode, leaf_equal_estimate, recursive_estimate
from .mdp import FiniteMDP, exact_target, make_phase0_mdp, structural_audit


_REGISTERED_FAMILIES = ("root_only", "suffix_only", "entropy_distractor", "covariance_reversal")
_REGISTERED_ARMS = (
    "flat_iid",
    "uniform_tree",
    "leaf_equal_naive",
    "entropy_tree",
    "value_variance_tree",
    "gradient_only",
    "joint_no_cross",
    "jag_oracle",
    "jag_learned",
)
_ADAPTIVE_ARMS = frozenset(
    {
        "leaf_equal_naive",
        "entropy_tree",
        "value_variance_tree",
        "gradient_only",
        "joint_no_cross",
        "jag_learned",
    }
)


def _data(config: Any) -> dict[str, Any]:
    value = config.data if hasattr(config, "data") else config
    if not isinstance(value, Mapping):
        raise TypeError("resolved_config must be a mapping or ResolvedConfig")
    return dict(value)


def _finite_or_none(value: float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, not bool or a coerced value")
    return int(value)


def _arm_metadata(arm: str, baseline: str = "zero") -> dict[str, Any]:
    if arm == "leaf_equal_naive":
        return {
            "jag_status": "NON_JAG_NEGATIVE_CONTROL",
            "estimator": "leaf_equal",
            "allocation_rule": "live_joint_marginal",
            "moment_source": "exact_future_oracle_ablation",
            "moment_estimator": "recursive_sampled_continuation",
            "moment_continuation_branching": 1,
            "oracle_ablation": True,
            "deployable": False,
            "uses_same_tree_outcomes_for_allocation": False,
            "negative_control_mechanism": "leaf_equal_weighting",
            "negative_control_scope": "weighting-only; allocation never reads sampled outcomes",
            "baseline": baseline,
            "baseline_frozen_before_evaluation": True,
            "baseline_evaluation_frequency": "once_per_sibling_group",
            "baseline_reads_evaluation_outcomes": False,
            "unbiased_claim": False,
        }
    allocation_rule = {
        "flat_iid": "independent_roots",
        "uniform_tree": "balanced_live_frontier",
        "entropy_tree": "live_entropy_marginal",
        "value_variance_tree": "live_value_variance_marginal",
        "gradient_only": "live_gradient_variance_marginal",
        "joint_no_cross": "live_joint_no_cross_marginal",
        "jag_oracle": "exact_live_frontier_joint_risk",
        "jag_learned": "lagged_ridge_live_joint_marginal",
    }[arm]
    exact_moment_arms = {"value_variance_tree", "gradient_only", "joint_no_cross", "jag_oracle"}
    moment_source = (
        "lagged_calibration_record"
        if arm == "jag_learned"
        else "exact_future_oracle_ablation"
        if arm in exact_moment_arms
        else "policy_entropy"
        if arm == "entropy_tree"
        else "none"
    )
    uses_moment_estimator = arm in exact_moment_arms or arm == "jag_learned"
    return {
        "jag_status": "JAG" if arm in {"jag_oracle", "jag_learned"} else "BASELINE",
        "estimator": "recursive",
        "allocation_rule": allocation_rule,
        "moment_source": moment_source,
        "moment_estimator": "recursive_sampled_continuation" if uses_moment_estimator else "not_applicable",
        "moment_continuation_branching": 1 if uses_moment_estimator else None,
        "oracle_ablation": arm in exact_moment_arms,
        "deployable": arm not in exact_moment_arms,
        "baseline": baseline,
        "baseline_frozen_before_evaluation": True,
        "baseline_evaluation_frequency": "once_per_sibling_group",
        "baseline_reads_evaluation_outcomes": False,
        "unbiased_claim": True,
    }


def _canonical_plan(plan: Mapping[str, int]) -> list[list[Any]]:
    return [[node_id, int(plan[node_id])] for node_id in sorted(plan)]


def _plan_hash(plan: Mapping[str, int]) -> str:
    payload = json.dumps(_canonical_plan(plan), ensure_ascii=False, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _allocation_summary(plans: list[dict[str, int]], budget: int) -> dict[str, Any]:
    histogram: dict[str, dict[str, Any]] = {}
    for plan in plans:
        edge_count = int(sum(plan.values()))
        if edge_count != budget:
            raise AssertionError(f"allocation has {edge_count} edges, expected {budget}")
        digest = _plan_hash(plan)
        canonical = _canonical_plan(plan)
        if digest not in histogram:
            histogram[digest] = {"count": 0, "edge_count": edge_count, "canonical_plan": canonical}
        elif histogram[digest]["canonical_plan"] != canonical:
            raise AssertionError("canonical plan hash collision")
        histogram[digest]["count"] += 1
    return {
        "replication_count": len(plans),
        "unique_plan_count": len(histogram),
        "total_edge_samples": int(len(plans) * budget),
        "plan_histogram": {digest: histogram[digest] for digest in sorted(histogram)},
    }


def _allocation_evidence(traces: list[list[dict[str, Any]]], plans: list[dict[str, int]], budget: int) -> dict[str, Any]:
    events = [event for trace in traces for event in trace]
    marginal_events = [event for event in events if event.get("kind") == "marginal_increment"]
    oracle_events = [event for event in events if event.get("kind") == "exact_frontier_solution"]
    errors = [float(event["score_identity_error"]) for event in marginal_events]
    trace_hashes = Counter(
        sha256(json.dumps(trace, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        for trace in traces
    )
    return {
        "planner": "levelwise_live_frontier",
        "score_formula": "w_squared*risk/(cost*b*(b+1))",
        "branch_counts_frozen_before_child_draws": bool(
            all(event.get("branch_counts_frozen_before_child_draws", True) for event in events)
        ),
        "exact_edge_budget_every_replication": bool(all(sum(plan.values()) == budget for plan in plans)),
        "decision_event_count": len(marginal_events),
        "max_score_identity_error": float(max(errors, default=0.0)),
        "transport_conditioned_event_count": int(
            sum(bool(event.get("ancestor_branching")) for event in marginal_events)
        ),
        "oracle_frontier_solve_count": len(oracle_events),
        "oracle_max_optimality_gap": float(
            max((float(event.get("optimality_gap", 0.0)) for event in oracle_events), default=0.0)
        ),
        "trace_hash_histogram": {digest: int(trace_hashes[digest]) for digest in sorted(trace_hashes)},
    }


def _summary(
    samples: np.ndarray,
    target: np.ndarray,
    budget: int,
    arm: str,
    plans: list[dict[str, int]],
    traces: list[list[dict[str, Any]]],
    paired_replicates: list[dict[str, Any]],
    baseline: str,
    status: str = "OK",
) -> dict[str, Any]:
    mean = np.mean(samples, axis=0)
    bias_vector = mean - target
    absolute_bias = float(np.linalg.norm(bias_vector))
    target_norm = float(np.linalg.norm(target))
    mse = float(np.mean(np.sum((samples - target) ** 2, axis=1)))
    if target_norm == 0.0:
        z_score: float | None = None
        z_status = "ZERO_TARGET_NORM"
    else:
        projection = samples @ (target / target_norm)
        standard_error = float(np.std(projection, ddof=0) / np.sqrt(len(projection)))
        disagreement = float(np.mean(projection) - target_norm)
        if standard_error == 0.0 and disagreement != 0.0:
            z_score = None
            z_status = "ZERO_STANDARD_ERROR_NONZERO_DISAGREEMENT"
        else:
            z_score = 0.0 if standard_error == 0.0 else float(disagreement / standard_error)
            z_status = "OK"
    relative_bias = 0.0 if target_norm == 0.0 and absolute_bias == 0.0 else None if target_norm == 0.0 else absolute_bias / target_norm
    return {
        "arm": arm,
        "budget": int(budget),
        "sample_count": int(len(samples)),
        "bias": absolute_bias,
        "absolute_bias": absolute_bias,
        "relative_bias": _finite_or_none(relative_bias) if relative_bias is not None else None,
        "bias_z_score": _finite_or_none(z_score) if z_score is not None else None,
        "bias_z_score_status": z_status,
        "mse": mse,
        "budget_times_mse": float(budget * mse),
        "cosine": float(cosine_similarity(mean, target)),
        "allocation": _allocation_summary(plans, budget),
        "allocation_evidence": _allocation_evidence(traces, plans, budget),
        "paired_replicates": paired_replicates,
        "arm_metadata": _arm_metadata(arm, baseline),
        "negative_control": arm == "leaf_equal_naive",
        "status": status,
        "status_reason": None,
    }


def _unavailable_summary(budget: int, arm: str, status: str, reason: str, baseline: str = "zero") -> dict[str, Any]:
    return {
        "arm": arm,
        "budget": int(budget),
        "sample_count": 0,
        "bias": None,
        "absolute_bias": None,
        "relative_bias": None,
        "bias_z_score": None,
        "bias_z_score_status": "UNAVAILABLE",
        "mse": None,
        "budget_times_mse": None,
        "cosine": None,
        "allocation": _allocation_summary([], int(budget)),
        "allocation_evidence": {
            "planner": "exact_live_frontier",
            "score_formula": "w_squared*risk/(cost*b*(b+1))",
            "branch_counts_frozen_before_child_draws": True,
            "exact_edge_budget_every_replication": False,
            "decision_event_count": 0,
            "max_score_identity_error": 0.0,
            "transport_conditioned_event_count": 0,
            "oracle_frontier_solve_count": 0,
            "oracle_max_optimality_gap": 0.0,
            "trace_hash_histogram": {},
        },
        "paired_replicates": [],
        "arm_metadata": _arm_metadata(arm, baseline),
        "negative_control": arm == "leaf_equal_naive",
        "status": status,
        "status_reason": reason,
    }


def _accumulated_score(mdp: FiniteMDP, prefix: tuple[int, ...]) -> np.ndarray:
    if not prefix:
        return np.zeros(mdp.parameter_size, dtype=np.float64)
    return np.sum([mdp.score(prefix[:depth], action) for depth, action in enumerate(prefix)], axis=0)


@dataclass(frozen=True)
class CalibrationRecord:
    """Immutable, versioned labels from completed earlier environments."""

    schema_version: str
    family: str
    horizon: int
    actions: int
    environment_seeds: tuple[int, ...]
    environment_ids: tuple[str, ...]
    environment_hashes: tuple[str, ...]
    row_environment_ids: tuple[str, ...]
    features: tuple[tuple[float, ...], ...]
    targets: tuple[float, ...]
    value_targets: tuple[float, ...]
    risk_label_baseline: str
    completed: bool
    data_hash: str
    provenance_hash: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "family": self.family,
            "horizon": self.horizon,
            "actions": self.actions,
            "environment_seeds": list(self.environment_seeds),
            "environment_ids": list(self.environment_ids),
            "environment_hashes": list(self.environment_hashes),
            "row_environment_ids": list(self.row_environment_ids),
            "features": [list(row) for row in self.features],
            "targets": list(self.targets),
            "value_targets": list(self.value_targets),
            "risk_label_baseline": self.risk_label_baseline,
            "completed": self.completed,
        }

    def computed_hash(self) -> str:
        canonical = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(canonical.encode("utf-8")).hexdigest()

    def computed_provenance_hash(self) -> str:
        payload = {
            "builder": "coding_opsd.jag.build_calibration_record",
            "schema_version": self.schema_version,
            "family": self.family,
            "horizon": self.horizon,
            "actions": self.actions,
            "environment_ids": list(self.environment_ids),
            "environment_hashes": list(self.environment_hashes),
            "risk_label_baseline": self.risk_label_baseline,
            "completed": self.completed,
            "data_hash": self.data_hash,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(("jag-calibration-provenance-v1:" + canonical).encode("utf-8")).hexdigest()


class RidgePriority:
    def __init__(
        self,
        coefficients: np.ndarray,
        calibration_seeds: tuple[int, ...] = (),
        calibration_record: CalibrationRecord | None = None,
    ):
        coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if coefficients.shape != (5,) or not np.isfinite(coefficients).all():
            raise ValueError("ridge priority requires five finite coefficients")
        self.coefficients = coefficients.copy()
        self.coefficients.setflags(write=False)
        self.calibration_seeds = calibration_seeds
        self.calibration_record = calibration_record

    def predict(self, mdp: FiniteMDP, prefix: tuple[int, ...]) -> float:
        return float(_priority_features(mdp, prefix) @ self.coefficients)


class RidgeValueBaseline:
    """A frozen value baseline fit only on completed lagged environments."""

    def __init__(self, coefficients: np.ndarray, calibration_record: CalibrationRecord):
        coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if coefficients.shape != (5,) or not np.isfinite(coefficients).all():
            raise ValueError("ridge value baseline requires five finite coefficients")
        self.coefficients = coefficients.copy()
        self.coefficients.setflags(write=False)
        self.calibration_record = calibration_record

    def predict(self, mdp: FiniteMDP, prefix: tuple[int, ...]) -> float:
        return float(_priority_features(mdp, prefix) @ self.coefficients)


def _ridge_coefficients(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    ridge = 1e-6 * np.eye(features.shape[1])
    return np.linalg.solve(features.T @ features + ridge, features.T @ targets)


def _environment_id(family: str, horizon: int, actions: int, seed: int) -> str:
    return f"{family}:h{horizon}:a{actions}:seed{seed}"


def _environment_hash(mdp: FiniteMDP) -> str:
    payload = {
        "family": mdp.reward_family,
        "horizon": mdp.horizon,
        "actions": mdp.actions,
        "seed": mdp.reward_seed,
        "parameters": mdp.parameters.tolist(),
        "root_rewards": mdp._root_rewards.tolist(),
        "suffix_rewards": mdp._suffix_rewards.tolist(),
        "entropy_signal": mdp._entropy_signal.tolist(),
        "covariance_signal": mdp._covariance_signal.tolist(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _priority_features(mdp: FiniteMDP, prefix: tuple[int, ...]) -> np.ndarray:
    probabilities = mdp.action_probabilities(prefix)
    return np.array(
        [
            1.0,
            len(prefix) / mdp.horizon,
            -np.sum(probabilities * np.log(probabilities)),
            sum(prefix) / (mdp.horizon * max(1, mdp.actions - 1)),
            float(np.max(probabilities)),
        ],
        dtype=np.float64,
    )


def build_calibration_record(
    family: str,
    horizon: int,
    actions: int,
    seeds: tuple[int, ...],
    risk_label_baseline: str = "zero",
) -> CalibrationRecord:
    """Materialize labels from completed environments into one fixed record.

    ``lagged_cross_fitted_value`` uses leave-one-environment-out value fits for
    risk labels.  The eventual evaluation baseline is separately fit from all
    completed rows; neither operation reads the evaluation environment.
    """

    horizon = _strict_int(horizon, "calibration horizon")
    actions = _strict_int(actions, "calibration actions")
    seeds = tuple(_strict_int(value, "calibration seed") for value in seeds)
    if not seeds or horizon < 1 or actions < 1:
        raise ValueError("calibration record needs seeds and positive dimensions")
    if len(set(seeds)) != len(seeds):
        raise ValueError("calibration record seeds must be unique")
    if any(seed < 0 for seed in seeds):
        raise ValueError("calibration record seeds must be nonnegative")
    if risk_label_baseline not in {"zero", "lagged_cross_fitted_value"}:
        raise ValueError("unsupported calibration baseline")
    environments: list[tuple[int, str, FiniteMDP, list[tuple[int, ...]], list[tuple[float, ...]], list[float]]] = []
    for calibration_seed in seeds:
        mdp = make_phase0_mdp(horizon, actions, family, calibration_seed)
        prefixes: list[tuple[int, ...]] = []
        rows: list[tuple[float, ...]] = []
        values: list[float] = []
        for depth in range(horizon):
            for prefix in product(range(actions), repeat=depth):
                prefixes.append(prefix)
                rows.append(tuple(float(value) for value in _priority_features(mdp, prefix)))
                values.append(float(mdp._conditional_value(prefix)))
        environments.append(
            (
                int(calibration_seed),
                _environment_id(family, horizon, actions, int(calibration_seed)),
                mdp,
                prefixes,
                rows,
                values,
            )
        )

    all_rows = [row for _, _, _, _, rows, _ in environments for row in rows]
    all_values = [value for _, _, _, _, _, values in environments for value in values]
    row_environment_ids = [environment_id for _, environment_id, _, prefixes, _, _ in environments for _ in prefixes]
    risk_labels: list[float] = []
    for held_seed, _, mdp, prefixes, _, _ in environments:
        if risk_label_baseline == "zero":
            baseline_for_prefix: float | Callable[[tuple[int, ...]], float] = 0.0
        else:
            training_rows = [
                row
                for seed_value, _, _, _, rows, _ in environments
                if seed_value != held_seed
                for row in rows
            ]
            training_values = [
                value
                for seed_value, _, _, _, _, values in environments
                if seed_value != held_seed
                for value in values
            ]
            # A one-environment smoke record cannot cross-fit.  Its risk labels
            # use the preregistered zero fallback and the record says so below;
            # the evaluation baseline is still a lagged fit, never evaluation data.
            if training_rows:
                fold_coefficients = _ridge_coefficients(
                    np.asarray(training_rows, dtype=np.float64),
                    np.asarray(training_values, dtype=np.float64),
                )
                baseline_for_prefix = lambda prefix, m=mdp, c=fold_coefficients: float(
                    _priority_features(m, prefix) @ c
                )
            else:
                baseline_for_prefix = 0.0
        for prefix in prefixes:
            moments = mdp.conditional_moments(
                prefix,
                baseline=baseline_for_prefix,
                continuation_branching=1,
                full_covariance=False,
            )
            risk_labels.append(max(0.0, joint_risk(moments, _accumulated_score(mdp, prefix))))

    label_protocol = (
        risk_label_baseline
        if risk_label_baseline == "zero" or len(environments) > 1
        else "lagged_cross_fitted_value:single_environment_zero_label_fallback"
    )
    provisional = CalibrationRecord(
        schema_version="jag.calibration.v1",
        family=family,
        horizon=int(horizon),
        actions=int(actions),
        environment_seeds=tuple(int(item) for item in seeds),
        environment_ids=tuple(item[1] for item in environments),
        environment_hashes=tuple(_environment_hash(item[2]) for item in environments),
        row_environment_ids=tuple(row_environment_ids),
        features=tuple(all_rows),
        targets=tuple(float(value) for value in risk_labels),
        value_targets=tuple(float(value) for value in all_values),
        risk_label_baseline=label_protocol,
        completed=True,
        data_hash="",
        provenance_hash="",
    )
    with_data_hash = CalibrationRecord(**{**provisional.__dict__, "data_hash": provisional.computed_hash()})
    return CalibrationRecord(
        **{**with_data_hash.__dict__, "provenance_hash": with_data_hash.computed_provenance_hash()}
    )


@lru_cache(maxsize=128)
def _trusted_calibration_record(
    family: str,
    horizon: int,
    actions: int,
    seeds: tuple[int, ...],
    risk_label_baseline: str,
) -> CalibrationRecord:
    """Rebuild canonical finite-MDP evidence without calling validation."""

    return build_calibration_record(
        family,
        horizon,
        actions,
        seeds,
        risk_label_baseline=risk_label_baseline,
    )


def _validate_calibration_record(record: CalibrationRecord, evaluation_seed: int) -> tuple[np.ndarray, np.ndarray]:
    if record.schema_version != "jag.calibration.v1":
        raise ValueError("calibration record has unsupported fixed schema")
    if not record.completed:
        raise ValueError("calibration record must be completed")
    if record.data_hash != record.computed_hash():
        raise ValueError("calibration record hash mismatch")
    if record.provenance_hash != record.computed_provenance_hash():
        raise ValueError("calibration record provenance mismatch")
    if record.family not in _REGISTERED_FAMILIES or record.horizon < 1 or record.actions < 1:
        raise ValueError("calibration record environment declaration is invalid")
    allowed_baseline_protocols = {
        "zero",
        "lagged_cross_fitted_value",
        "lagged_cross_fitted_value:single_environment_zero_label_fallback",
    }
    if record.risk_label_baseline not in allowed_baseline_protocols:
        raise ValueError("calibration record baseline protocol is invalid")
    if (
        record.risk_label_baseline.endswith("single_environment_zero_label_fallback")
        and len(record.environment_seeds) != 1
    ):
        raise ValueError("calibration record baseline protocol does not match environment count")
    if not record.environment_seeds or len(set(record.environment_seeds)) != len(record.environment_seeds):
        raise ValueError("calibration record environment seeds are invalid")
    if any(calibration_seed >= evaluation_seed for calibration_seed in record.environment_seeds):
        raise ValueError("calibration environments must be completed earlier seeds")
    expected_ids = tuple(
        _environment_id(record.family, record.horizon, record.actions, seed)
        for seed in record.environment_seeds
    )
    if record.environment_ids != expected_ids:
        raise ValueError("calibration record environment IDs do not match its declaration")
    expected_hashes = tuple(
        _environment_hash(make_phase0_mdp(record.horizon, record.actions, record.family, seed))
        for seed in record.environment_seeds
    )
    if record.environment_hashes != expected_hashes:
        raise ValueError("calibration record environment fingerprints do not match")
    rows_per_environment = sum(record.actions ** depth for depth in range(record.horizon))
    expected_row_ids = tuple(
        environment_id
        for environment_id in expected_ids
        for _ in range(rows_per_environment)
    )
    if record.row_environment_ids != expected_row_ids:
        raise ValueError("calibration record row provenance is invalid")
    if not (
        len(record.features)
        == len(record.targets)
        == len(record.value_targets)
        == len(record.row_environment_ids)
    ) or not record.features:
        raise ValueError("calibration record has incompatible feature/target rows")
    design = np.asarray(record.features, dtype=np.float64)
    response = np.asarray(record.targets, dtype=np.float64)
    values = np.asarray(record.value_targets, dtype=np.float64)
    if (
        design.ndim != 2
        or design.shape[1] != 5
        or not np.isfinite(design).all()
        or not np.isfinite(response).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError("calibration record contains invalid values")
    builder_baseline = (
        "lagged_cross_fitted_value"
        if record.risk_label_baseline.startswith("lagged_cross_fitted_value")
        else "zero"
    )
    trusted = _trusted_calibration_record(
        record.family,
        record.horizon,
        record.actions,
        tuple(record.environment_seeds),
        builder_baseline,
    )
    trusted_fields = {
        "schema_version": "schema",
        "environment_ids": "environment IDs",
        "environment_hashes": "environment fingerprints",
        "row_environment_ids": "row provenance",
        "features": "features",
        "targets": "risk targets",
        "value_targets": "value targets",
        "risk_label_baseline": "baseline protocol",
        "completed": "completion status",
        "data_hash": "data hash",
        "provenance_hash": "provenance hash",
    }
    for field, label in trusted_fields.items():
        if getattr(record, field) != getattr(trusted, field):
            raise ValueError(f"calibration record trusted rebuild mismatch: {label}")
    return design, response


def fit_calibration_record(record: CalibrationRecord, evaluation_seed: int) -> RidgePriority:
    """Fit strictly from a verified record; never regenerate current labels."""

    design, response = _validate_calibration_record(record, evaluation_seed)
    return RidgePriority(
        _ridge_coefficients(design, response),
        record.environment_seeds,
        record,
    )


def fit_value_baseline(record: CalibrationRecord, evaluation_seed: int) -> RidgeValueBaseline:
    """Fit the evaluation baseline solely from the frozen record value column."""

    design, _ = _validate_calibration_record(record, evaluation_seed)
    values = np.asarray(record.value_targets, dtype=np.float64)
    return RidgeValueBaseline(_ridge_coefficients(design, values), record)


def _node_priority(
    mdp: FiniteMDP,
    prefix: tuple[int, ...],
    arm: str,
    learned: RidgePriority | None = None,
    baseline: float | Callable[[tuple[int, ...]], float] = 0.0,
) -> float:
    """A frozen, prefix-only single-child risk for one unopened node."""

    if len(prefix) >= mdp.horizon:
        raise ValueError("terminal prefixes cannot be allocated")
    if arm == "jag_learned":
        if learned is None:
            raise ValueError("jag_learned requires a lagged predictor")
        return max(0.0, learned.predict(mdp, prefix))
    if arm == "entropy_tree":
        probabilities = mdp.action_probabilities(prefix)
        return float(-np.sum(probabilities * np.log(probabilities)))
    moments = mdp.conditional_moments(
        prefix,
        baseline=baseline,
        continuation_branching=1,
        full_covariance=False,
    )
    if arm == "value_variance_tree":
        return float(max(0.0, moments.value_variance))
    if arm == "gradient_only":
        return float(max(0.0, moments.gradient_variance_trace))
    score = _accumulated_score(mdp, prefix)
    if arm == "joint_no_cross":
        return float(max(0.0, joint_risk_no_cross(moments, score)))
    if arm in {"jag_oracle", "leaf_equal_naive"}:
        return float(max(0.0, joint_risk(moments, score)))
    raise ValueError(f"arm {arm!r} has no adaptive node priority")


@dataclass
class _LiveNode:
    node_id: str
    node: TreeNode
    ancestor_branching: tuple[int, ...]


class _KeyedActionRNG:
    """Common-random-number categorical draws keyed independently of arm."""

    def __init__(self, seed: int, family: str, budget: int, replication: int):
        self.seed = int(seed)
        self.namespace = f"jag-crn:{family}:{int(budget)}:{int(replication)}"
        self.stream_id = sha256(f"{self.seed}:{self.namespace}".encode("utf-8")).hexdigest()
        self.draw_index = 0

    def choice(self, actions: int, *, p: np.ndarray) -> int:
        probabilities = np.asarray(p, dtype=np.float64)
        if probabilities.shape != (actions,) or not np.isclose(np.sum(probabilities), 1.0):
            raise ValueError("categorical probabilities are invalid")
        uniform = float(named_rng(self.seed, f"{self.namespace}:draw:{self.draw_index}").random())
        self.draw_index += 1
        return int(min(actions - 1, np.searchsorted(np.cumsum(probabilities), uniform, side="right")))


def _level_extra_budget(slack: int, remaining_height: int) -> int:
    """Reserve opportunities for deeper live frontiers while spending exactly."""

    if slack <= 0:
        return 0
    if remaining_height == 1:
        return slack
    feasible_units = slack // remaining_height
    if feasible_units == 0:
        return 0
    units_now = max(1, math.ceil(feasible_units / remaining_height))
    return int(units_now * remaining_height)


def _feasible_level_increments(
    depth: int,
    frontier_count: int,
    slack: int,
    horizon: int,
    branchable_depth_count: int,
    max_branching: int | None,
    cache: dict[tuple[int, int, int], tuple[int, ...]],
) -> tuple[int, ...]:
    """Return current-level increments that admit an exact feasible suffix."""

    key = (depth, frontier_count, slack)
    if key in cache:
        return cache[key]
    if slack < 0:
        return ()
    remaining_height = horizon - depth
    if depth >= branchable_depth_count:
        candidates = range(1)
    else:
        capacity = slack // remaining_height
        if max_branching is not None:
            capacity = min(capacity, frontier_count * (max_branching - 1))
        candidates = range(int(capacity) + 1)
    feasible: list[int] = []
    for increments in candidates:
        next_slack = slack - increments * remaining_height
        if depth == horizon - 1:
            suffix_exists = next_slack == 0
        else:
            suffix_exists = bool(
                _feasible_level_increments(
                    depth + 1,
                    frontier_count + increments,
                    next_slack,
                    horizon,
                    branchable_depth_count,
                    max_branching,
                    cache,
                )
            )
        if suffix_exists:
            feasible.append(increments)
    result = tuple(feasible)
    cache[key] = result
    return result


def _choose_level_increments(
    arm: str,
    depth: int,
    slack: int,
    remaining_height: int,
    feasible: tuple[int, ...],
) -> int:
    if not feasible:
        raise ValueError("budget is not representable under branching and branchable-depth constraints")
    if arm == "flat_iid":
        desired = slack // remaining_height if depth == 0 else 0
    elif arm == "uniform_tree" and depth == 0 and remaining_height > 1:
        desired = 0
    else:
        desired = _level_extra_budget(slack, remaining_height) // remaining_height
    return min(feasible, key=lambda value: (abs(value - desired), value))


def _balanced_plan(items: list[FrontierItem], extra_budget: int) -> tuple[dict[str, int], list[dict[str, Any]]]:
    branching = {item.node_id: item.branching for item in items}
    remaining = int(extra_budget)
    events: list[dict[str, Any]] = []
    while items and remaining >= min(item.cost for item in items):
        eligible = [
            item
            for item in items
            if item.cost <= remaining
            and (item.max_branching is None or branching[item.node_id] < item.max_branching)
        ]
        if not eligible:
            break
        selected = min(eligible, key=lambda item: (branching[item.node_id], item.node_id))
        before = branching[selected.node_id]
        events.append(
            {
                "kind": "balanced_increment",
                "node_id": selected.node_id,
                "before": before,
                "after": before + 1,
                "cost": float(selected.cost),
            }
        )
        branching[selected.node_id] += 1
        remaining -= int(selected.cost)
    if remaining != 0:
        raise AssertionError("balanced level planner left an unspendable quota")
    return branching, events


def _scope_reason(mdp: FiniteMDP, budget: int, scope: OracleScope) -> str | None:
    if mdp.horizon > scope.max_horizon:
        return f"horizon {mdp.horizon} exceeds exact oracle max_horizon={scope.max_horizon}"
    if budget > scope.max_budget:
        return f"budget {budget} exceeds exact oracle max_budget={scope.max_budget}"
    return None


def _sample_budget_tree(
    mdp: FiniteMDP,
    budget: int,
    arm: str,
    rng: Any,
    learned: RidgePriority | None = None,
    oracle_scope: OracleScope | None = None,
    baseline: float | Callable[[tuple[int, ...]], float] = 0.0,
    max_branching: int | None = None,
    branchable_depth_count: int | None = None,
) -> tuple[TreeNode, dict[str, int]]:
    """Plan and sample a complete genealogy with exactly ``budget`` edges.

    Planning is levelwise.  At a level, every prefix is already known but none
    of its outgoing child outcomes has been drawn.  All branch counts are
    computed and frozen together; only then are replacement children sampled.
    """

    if arm not in _REGISTERED_ARMS:
        raise ValueError(f"unsupported JAG Phase-0 arm {arm!r}")
    if budget < mdp.horizon:
        raise ValueError("budget must fund at least one complete trajectory")
    if max_branching is not None and max_branching < 1:
        raise ValueError("max_branching must be positive")
    depth_count = mdp.horizon if branchable_depth_count is None else int(branchable_depth_count)
    if depth_count < 0 or depth_count > mdp.horizon:
        raise ValueError("branchable_depth_count must be between zero and horizon")
    feasibility_cache: dict[tuple[int, int, int], tuple[int, ...]] = {}
    initial_slack = budget - mdp.horizon
    if not _feasible_level_increments(
        0, 1, initial_slack, mdp.horizon, depth_count, max_branching, feasibility_cache
    ):
        raise ValueError("budget is not representable under branching and branchable-depth constraints")
    scope = oracle_scope or OracleScope()
    if arm == "jag_oracle":
        reason = _scope_reason(mdp, budget, scope)
        if reason is not None:
            raise OracleUnavailableError(reason)

    root = TreeNode(np.zeros(mdp.parameter_size), prefix=())
    frontier = [_LiveNode("n000000", root, ())]
    plan: dict[str, int] = {}
    allocated = 0
    next_id = 1
    priority_cache: dict[tuple[int, ...], float] = {}

    def priority(prefix: tuple[int, ...]) -> float:
        if prefix not in priority_cache:
            priority_cache[prefix] = _node_priority(mdp, prefix, arm, learned, baseline)
        return priority_cache[prefix]

    for depth in range(mdp.horizon):
        remaining_height = mdp.horizon - depth
        mandatory = len(frontier) * remaining_height
        slack = int(budget - allocated - mandatory)
        if slack < 0:
            raise AssertionError("tree budget ledger underflow")
        feasible = _feasible_level_increments(
            depth,
            len(frontier),
            slack,
            mdp.horizon,
            depth_count,
            max_branching,
            feasibility_cache,
        )
        level_increments = _choose_level_increments(arm, depth, slack, remaining_height, feasible)
        level_budget = int(level_increments * remaining_height)

        if arm in {"flat_iid", "uniform_tree"}:
            item_risks = {entry.node_id: 1.0 for entry in frontier}
        else:
            item_risks = {entry.node_id: priority(entry.node.prefix) for entry in frontier}
        items = [
            FrontierItem(
                node_id=entry.node_id,
                parent_id=None,
                branching=1,
                risk=float(item_risks[entry.node_id]),
                cost=float(remaining_height),
                ancestor_branching=entry.ancestor_branching,
                max_branching=max_branching,
            )
            for entry in frontier
        ]

        if arm == "jag_oracle":
            planned = exact_frontier_plan(items, level_budget, scope)
            counts = planned.branching
            raw_events = list(planned.events)
            if planned.remaining_budget != 0.0:
                raise AssertionError("exact frontier oracle left an unspendable quota")
        elif arm in _ADAPTIVE_ARMS:
            planned = greedy_frontier_plan(items, level_budget)
            counts = planned.branching
            raw_events = list(planned.events)
            if planned.remaining_budget != 0.0:
                raise AssertionError("adaptive level planner left an unspendable quota")
        else:
            counts, raw_events = _balanced_plan(items, level_budget)

        by_id = {entry.node_id: entry for entry in frontier}
        for event in raw_events:
            event["depth"] = depth
            event["branch_counts_frozen_before_child_draws"] = True
            event.setdefault("kind", "marginal_increment")
            node_id = event.get("node_id")
            if node_id in by_id:
                event["prefix"] = list(by_id[str(node_id)].node.prefix)
            for candidate in event.get("candidates", []):
                candidate_id = str(candidate["node_id"])
                candidate["prefix"] = list(by_id[candidate_id].node.prefix)
            root.planning_events.append(event)
        root.planning_events.append(
            {
                "kind": "level_plan",
                "depth": depth,
                "remaining_height": remaining_height,
                "slack_before": slack,
                "level_budget": level_budget,
                "level_increments": level_increments,
                "max_branching": max_branching,
                "depth_is_branchable": depth < depth_count,
                "branch_counts_frozen_before_child_draws": True,
                "frontier": [
                    {
                        "node_id": entry.node_id,
                        "prefix": list(entry.node.prefix),
                        "ancestor_branching": list(entry.ancestor_branching),
                        "risk": float(item_risks[entry.node_id]),
                        "branching": int(counts[entry.node_id]),
                    }
                    for entry in frontier
                ],
            }
        )

        for entry in frontier:
            count = int(counts[entry.node_id])
            entry.node.planned_branching = count
            plan[entry.node_id] = count

        next_frontier: list[_LiveNode] = []
        for entry in frontier:
            count = int(counts[entry.node_id])
            for _ in range(count):
                action = int(rng.choice(mdp.actions, p=mdp.action_probabilities(entry.node.prefix)))
                child_prefix = entry.node.prefix + (action,)
                child = TreeNode(mdp.score(entry.node.prefix, action), prefix=child_prefix)
                entry.node.children.append(child)
                child_id = f"n{next_id:06d}"
                next_id += 1
                if len(child_prefix) == mdp.horizon:
                    child.reward = mdp.reward(child_prefix)
                else:
                    next_frontier.append(_LiveNode(child_id, child, (count,) + entry.ancestor_branching))
        allocated += sum(int(value) for value in counts.values())
        frontier = next_frontier

    if allocated != budget or frontier:
        raise AssertionError("tree did not consume its exact matched edge budget")
    return root, plan


def _parse_oracle_scope(phase0: Mapping[str, Any]) -> OracleScope:
    if "oracle_scope" not in phase0:
        return OracleScope()
    raw = phase0["oracle_scope"]
    if not isinstance(raw, Mapping):
        raise TypeError("phase0.oracle_scope must be a mapping")
    allowed = {"max_horizon", "max_budget", "max_frontier_nodes", "max_states"}
    unknown = set(raw) - allowed
    if unknown:
        names = ", ".join(sorted((repr(name) for name in unknown)))
        raise ValueError(f"phase0.oracle_scope contains unknown keys: {names}")
    missing = allowed - set(raw)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"phase0.oracle_scope is missing required keys: {names}")
    return OracleScope(
        max_horizon=_strict_int(raw["max_horizon"], "phase0.oracle_scope.max_horizon"),
        max_budget=_strict_int(raw["max_budget"], "phase0.oracle_scope.max_budget"),
        max_frontier_nodes=_strict_int(
            raw["max_frontier_nodes"], "phase0.oracle_scope.max_frontier_nodes"
        ),
        max_states=_strict_int(raw["max_states"], "phase0.oracle_scope.max_states"),
    )


def _audit_prefix(mdp: FiniteMDP) -> tuple[int, ...]:
    if mdp.reward_family == "covariance_reversal" and mdp.horizon >= 7:
        return (0, 0, 1, 1, 0, 1)
    if mdp.reward_family == "entropy_distractor" and mdp.horizon >= 2:
        return (1,)
    return tuple(0 for _ in range(max(0, mdp.horizon - 2)))


def _sample_prefix_chain(mdp: FiniteMDP, prefix: tuple[int, ...], rng: Any) -> TreeNode:
    root = TreeNode(np.zeros(mdp.parameter_size), prefix=prefix)
    current = root
    current_prefix = prefix
    while len(current_prefix) < mdp.horizon:
        action = int(rng.choice(mdp.actions, p=mdp.action_probabilities(current_prefix)))
        child_prefix = current_prefix + (action,)
        child = TreeNode(mdp.score(current_prefix, action), prefix=child_prefix)
        current.children = [child]
        current.planned_branching = 1
        current = child
        current_prefix = child_prefix
    current.reward = mdp.reward(current_prefix)
    return root


def _held_out_covariance_audit(
    mdp: FiniteMDP,
    seed: int,
    replications: int,
    baseline_prefix: float | Callable[[tuple[int, ...]], float],
    baseline_node: float | Callable[[TreeNode], float],
) -> dict[str, Any]:
    """Compare predicted and independently realized joint contribution covariance."""

    if replications < 2:
        raise ValueError("covariance_audit_replications must be at least two")
    prefix = _audit_prefix(mdp)
    predicted = mdp.conditional_moments(
        prefix,
        baseline=baseline_prefix,
        continuation_branching=1,
        full_covariance=True,
    )
    joint_samples: list[np.ndarray] = []
    for replication in range(replications):
        audit_rng = named_rng(seed, f"jag-heldout-covariance:{mdp.reward_family}:{replication}")
        value, gradient = recursive_estimate(
            _sample_prefix_chain(mdp, prefix, audit_rng),
            baseline=baseline_node,
        )
        joint_samples.append(np.concatenate(([value], gradient)))
    sample_matrix = np.stack(joint_samples)
    realized_joint_mean = np.mean(sample_matrix, axis=0)
    centered = sample_matrix - realized_joint_mean
    realized_covariance = np.asarray(centered.T @ centered / replications, dtype="<f8")
    predicted_covariance = np.asarray(
        np.block(
            [
                [
                    np.asarray([[predicted.value_variance]], dtype=np.float64),
                    predicted.value_gradient_covariance[None, :],
                ],
                [
                    predicted.value_gradient_covariance[:, None],
                    predicted.gradient_covariance,
                ],
            ]
        ),
        dtype="<f8",
    )
    difference = realized_covariance - predicted_covariance
    numerator = float(np.sum(difference * difference))
    denominator = float(np.sum(predicted_covariance * predicted_covariance))
    relative_error = None if denominator == 0.0 else float(np.sqrt(numerator / denominator))
    predicted_hash = sha256(predicted_covariance.tobytes(order="C")).hexdigest()
    realized_hash = sha256(realized_covariance.tobytes(order="C")).hexdigest()
    return {
        "status": "OK" if denominator > 0.0 else "ZERO_PREDICTED_COVARIANCE",
        "prefix": list(prefix),
        "sample_count": int(replications),
        "held_out_from_rollout_and_calibration": True,
        "object": "joint_node_contribution_[value,gradient]_covariance",
        "joint_dimension": int(predicted_covariance.shape[0]),
        "covariance_dtype": "<f8",
        "covariance_normalization": "population_1_over_n",
        "predicted_covariance": predicted_covariance.tolist(),
        "realized_covariance": realized_covariance.tolist(),
        "realized_joint_mean": np.asarray(realized_joint_mean, dtype="<f8").tolist(),
        "numerator": numerator,
        "numerator_definition": "squared_frobenius(realized_covariance-predicted_covariance)",
        "denominator": denominator,
        "denominator_definition": "squared_frobenius(predicted_covariance)",
        "relative_frobenius_error": _finite_or_none(relative_error) if relative_error is not None else None,
        "predicted_covariance_hash": predicted_hash,
        "realized_covariance_hash": realized_hash,
    }


def _run_family(
    arguments: tuple[
        str,
        int,
        int,
        tuple[str, ...],
        tuple[int, ...],
        int,
        int,
        tuple[int, ...],
        int,
        OracleScope,
        str,
        int,
        int,
        int,
    ]
) -> tuple[str, dict[str, Any]]:
    (
        family,
        horizon,
        actions,
        arms,
        budgets,
        replications,
        seed,
        calibration_seeds,
        calibration_horizon,
        oracle_scope,
        baseline_mode,
        max_branching,
        branchable_depth_count,
        covariance_audit_replications,
    ) = arguments
    mdp = make_phase0_mdp(horizon, actions, family, seed)
    target = exact_target(mdp)
    calibration_record = (
        build_calibration_record(
            family,
            calibration_horizon,
            actions,
            calibration_seeds,
            risk_label_baseline=baseline_mode,
        )
        if "jag_learned" in arms or baseline_mode == "lagged_cross_fitted_value"
        else None
    )
    learned = (
        fit_calibration_record(calibration_record, seed)
        if calibration_record is not None and "jag_learned" in arms
        else None
    )
    value_baseline = (
        fit_value_baseline(calibration_record, seed)
        if calibration_record is not None and baseline_mode == "lagged_cross_fitted_value"
        else None
    )
    if value_baseline is None:
        baseline_prefix: float | Callable[[tuple[int, ...]], float] = 0.0
        baseline_node: float | Callable[[TreeNode], float] = 0.0
    else:
        baseline_prefix = lambda prefix: value_baseline.predict(mdp, prefix)
        baseline_node = lambda node: value_baseline.predict(mdp, node.prefix)
    family_rows = []
    for arm in arms:
        for budget in budgets:
            if arm == "jag_oracle":
                reason = _scope_reason(mdp, int(budget), oracle_scope)
                if reason is not None:
                    family_rows.append(
                        _unavailable_summary(
                            int(budget), arm, "UNAVAILABLE_ORACLE_SCOPE", reason, baseline_mode
                        )
                    )
                    continue
            samples: list[np.ndarray] = []
            plans: list[dict[str, int]] = []
            traces: list[list[dict[str, Any]]] = []
            paired_replicates: list[dict[str, Any]] = []
            unavailable_reason: str | None = None
            for replication in range(replications):
                sampling_rng = _KeyedActionRNG(seed, family, int(budget), replication)
                try:
                    root, plan = _sample_budget_tree(
                        mdp,
                        int(budget),
                        arm,
                        sampling_rng,
                        learned,
                        oracle_scope,
                        baseline_prefix,
                        max_branching,
                        branchable_depth_count,
                    )
                except OracleUnavailableError as exc:
                    unavailable_reason = str(exc)
                    break
                estimator = leaf_equal_estimate if arm == "leaf_equal_naive" else recursive_estimate
                _, gradient = estimator(root, baseline=baseline_node)
                samples.append(gradient)
                plans.append(plan)
                traces.append(root.planning_events)
                delta = np.asarray(gradient - target.gradient, dtype=np.float64)
                paired_replicates.append(
                    {
                        "replication": int(replication),
                        "crn_stream_id": sampling_rng.stream_id,
                        "gradient_hash": sha256(np.asarray(gradient, dtype=np.float64).tobytes(order="C")).hexdigest(),
                        "gradient_delta_hash": sha256(delta.tobytes(order="C")).hexdigest(),
                        "squared_error": float(delta @ delta),
                    }
                )
            if unavailable_reason is not None:
                family_rows.append(
                    _unavailable_summary(
                        int(budget), arm, "UNAVAILABLE_ORACLE_SCOPE", unavailable_reason, baseline_mode
                    )
                )
                continue
            status = (
                "EXACT_FRONTIER_ORACLE"
                if arm == "jag_oracle"
                else "LAGGED_RIDGE"
                if arm == "jag_learned"
                else "NON_JAG_NEGATIVE_CONTROL"
                if arm == "leaf_equal_naive"
                else "OK"
            )
            family_rows.append(
                _summary(
                    np.stack(samples),
                    target.gradient,
                    int(budget),
                    arm,
                    plans,
                    traces,
                    paired_replicates,
                    baseline_mode,
                    status,
                )
            )
    return family, {
        "exact": {
            "probability_sum": target.probability_sum,
            "expected_reward": target.expected_reward,
            "gradient": target.gradient.tolist(),
        },
        "structural_audit": structural_audit(mdp, target),
        "covariance_audit": _held_out_covariance_audit(
            mdp,
            seed,
            covariance_audit_replications,
            baseline_prefix,
            baseline_node,
        ),
        "predictor_provenance": None
        if calibration_record is None
        else {
            "schema_version": calibration_record.schema_version,
            "completed": calibration_record.completed,
            "calibration_seeds": list(calibration_record.environment_seeds),
            "environment_ids": list(calibration_record.environment_ids),
            "row_count": len(calibration_record.targets),
            "data_hash": calibration_record.data_hash,
            "provenance_hash": calibration_record.provenance_hash,
            "environment_hashes": list(calibration_record.environment_hashes),
            "risk_label_baseline": calibration_record.risk_label_baseline,
            "evaluation_seed": int(seed),
            "oracle_labels_from_evaluation_environment": False,
            "evaluation_baseline_fit_from_completed_record_only": value_baseline is not None,
        },
        "summaries": family_rows,
    }


def _comparison(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return _finite_or_none(numerator / denominator)


def _paired_error_delta(first: Mapping[str, Any] | None, second: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if first is None or second is None:
        return None
    first_by_stream = {item["crn_stream_id"]: item for item in first.get("paired_replicates", [])}
    second_by_stream = {item["crn_stream_id"]: item for item in second.get("paired_replicates", [])}
    shared = sorted(set(first_by_stream) & set(second_by_stream))
    if not shared:
        return None
    deltas = [
        float(first_by_stream[stream]["squared_error"])
        - float(second_by_stream[stream]["squared_error"])
        for stream in shared
    ]
    payload = json.dumps(deltas, separators=(",", ":"), allow_nan=False)
    return {
        "definition": "first_squared_error-minus-second_squared_error_on_shared_crn_stream",
        "paired_count": len(shared),
        "mean_delta": float(np.mean(deltas)),
        "delta_hash": sha256(payload.encode("utf-8")).hexdigest(),
    }


def _gate_payload(
    config: Mapping[str, Any],
    seed: int,
    families: tuple[str, ...],
    arms: tuple[str, ...],
    budgets: tuple[int, ...],
    replications: int,
    results: Mapping[str, Any],
) -> dict[str, Any]:
    rows = {
        (family, str(row["arm"]), int(row["budget"])): row
        for family, family_result in results.items()
        for row in family_result["summaries"]
    }
    relative_bias = [
        {"family": family, "arm": arm, "budget": budget, "value": rows[(family, arm, budget)]["relative_bias"]}
        for family in families
        for arm in arms
        for budget in budgets
        if (family, arm, budget) in rows
    ]
    comparisons: list[dict[str, Any]] = []
    unbiased_baselines = {"flat_iid", "uniform_tree", "entropy_tree", "value_variance_tree", "gradient_only", "joint_no_cross"}
    for family in families:
        for budget in budgets:
            available_baseline_mse = [
                float(rows[(family, arm, budget)]["mse"])
                for arm in arms
                if arm in unbiased_baselines
                and (family, arm, budget) in rows
                and rows[(family, arm, budget)]["mse"] is not None
            ]
            oracle = rows.get((family, "jag_oracle", budget))
            uniform = rows.get((family, "uniform_tree", budget))
            learned = rows.get((family, "jag_learned", budget))
            oracle_mse = None if oracle is None else oracle["mse"]
            learned_mse = None if learned is None else learned["mse"]
            uniform_mse = None if uniform is None else uniform["mse"]
            best = min(available_baseline_mse) if available_baseline_mse else None
            oracle_reduction = None
            if best not in {None, 0.0} and oracle_mse is not None:
                oracle_reduction = _finite_or_none((float(best) - float(oracle_mse)) / float(best))
            gap_closed = None
            if oracle_mse is not None and learned_mse is not None and uniform_mse is not None:
                gap_closed = _comparison(float(uniform_mse) - float(learned_mse), float(uniform_mse) - float(oracle_mse))
            comparisons.append(
                {
                    "family": family,
                    "budget": int(budget),
                    "oracle_vs_best_unbiased_relative_mse_reduction": oracle_reduction,
                    "learned_oracle_uniform_gap_closed": gap_closed,
                    "learned_minus_uniform_paired_squared_error": _paired_error_delta(learned, uniform),
                }
            )
    for budget in budgets:
        full = rows.get(("covariance_reversal", "jag_oracle", budget))
        no_cross = rows.get(("covariance_reversal", "joint_no_cross", budget))
        full_mse = None if full is None else full["mse"]
        no_cross_mse = None if no_cross is None else no_cross["mse"]
        gain = None
        if full_mse is not None and no_cross_mse not in {None, 0.0}:
            gain = _finite_or_none((float(no_cross_mse) - float(full_mse)) / float(no_cross_mse))
        comparisons.append(
            {
                "family": "covariance_reversal",
                "budget": int(budget),
                "full_joint_vs_no_cross_relative_mse_gain": gain,
                "full_joint_minus_no_cross_paired_squared_error": _paired_error_delta(full, no_cross),
            }
        )

    available_rows = [row for row in rows.values() if row["sample_count"] > 0]
    required_keys = {
        (family, arm, budget)
        for family in _REGISTERED_FAMILIES
        for arm in _REGISTERED_ARMS
        for budget in (64, 128, 256)
    }
    available_keys = {key for key, row in rows.items() if row["sample_count"] == replications}
    diagnostics = {
        family: results[family]["structural_audit"].get("status")
        for family in ("entropy_distractor", "covariance_reversal")
        if family in results
    }
    completeness_checks = {
        "registered_families_present": set(_REGISTERED_FAMILIES).issubset(families),
        "registered_arms_present": set(_REGISTERED_ARMS).issubset(arms),
        "registered_budgets_present": {64, 128, 256}.issubset(budgets),
        "registered_replications_present": replications >= 100_000,
        "all_registered_cells_have_samples": required_keys.issubset(available_keys),
        "fifty_environment_seed_results_present": False,
        "diagnostic_structures_pass": diagnostics == {"entropy_distractor": "PASS", "covariance_reversal": "PASS"},
        "registered_covariance_coverage_gate_implemented": False,
    }
    profile = str(dict(config.get("runtime", {})).get("profile", "formal"))
    configured_seeds = [int(item) for item in config.get("seeds", [])]
    missing = [name for name, passed in completeness_checks.items() if not passed]
    active_seed_declared = not configured_seeds or int(seed) in configured_seeds
    failed_requested_diagnostics = [
        family
        for family in ("entropy_distractor", "covariance_reversal")
        if family in families and results[family]["structural_audit"].get("status") != "PASS"
    ]
    invalid_reasons = []
    if not active_seed_declared:
        invalid_reasons.append("active seed is not declared by config.seeds")
    if failed_requested_diagnostics:
        invalid_reasons.append(
            "requested structural diagnostics failed or were inapplicable: "
            + ", ".join(failed_requested_diagnostics)
        )
    status = "INVALID" if invalid_reasons else "INCOMPLETE"
    reason = (
        "; ".join(invalid_reasons)
        if invalid_reasons
        else
        "smoke profile is never formal evidence"
        if profile == "smoke"
        else "missing registered evidence: " + ", ".join(missing)
        if missing
        else "formal gate aggregation across 50 seeds is outside a single-seed runner"
    )
    return {
        "status": status,
        "reason": reason,
        "inputs": {
            "thresholds": dict(dict(config.get("phase0", {})).get("gate", {})),
            "relative_bias": relative_bias,
            "bias_z_score": [
                {
                    "family": family,
                    "arm": arm,
                    "budget": budget,
                    "value": rows[(family, arm, budget)]["bias_z_score"],
                    "status": rows[(family, arm, budget)]["bias_z_score_status"],
                }
                for family in families
                for arm in arms
                for budget in budgets
                if (family, arm, budget) in rows
            ],
            "held_out_covariance_audit": [
                {"family": family, **results[family]["covariance_audit"]}
                for family in families
            ],
            "registered_comparisons": comparisons,
        },
        "evidence": {
            "active_seed": int(seed),
            "configured_seed_ids": configured_seeds,
            "active_seed_declared": active_seed_declared,
            "observed_seed_count": 1,
            "required_seed_count": 50,
            "rollout_replications_per_cell": int(replications),
            "available_cell_count": len(available_rows),
            "required_cell_count": len(required_keys),
            "checks": completeness_checks,
            "missing": missing,
        },
    }


def run_jag_phase0(resolved_config: Any, seed: int) -> dict[str, Any]:
    """Run Phase-0 arms and return finite, JSON-serializable evidence."""

    config = _data(resolved_config)
    seed = _strict_int(seed, "evaluation seed")
    if seed < 0:
        raise ValueError("evaluation seed must be nonnegative")
    raw_declared_seeds = config.get("seeds", [])
    if not isinstance(raw_declared_seeds, (list, tuple)):
        raise TypeError("config.seeds must be a sequence of integers")
    declared_seeds = tuple(_strict_int(item, "declared seed") for item in raw_declared_seeds)
    if len(set(declared_seeds)) != len(declared_seeds):
        raise ValueError("declared seeds must be unique")
    if any(item < 0 for item in declared_seeds):
        raise ValueError("declared seeds must be nonnegative")
    phase0 = dict(config.get("phase0", {}))
    horizon = _strict_int(phase0.get("horizon", 8), "phase0.horizon")
    actions = _strict_int(phase0.get("actions", 4), "phase0.actions")
    families = tuple(str(item) for item in phase0.get("reward_families", _REGISTERED_FAMILIES))
    arms = tuple(str(item) for item in phase0.get("arms", _REGISTERED_ARMS))
    budgets = tuple(_strict_int(item, "phase0 budget") for item in phase0.get("budgets", [64, 128, 256]))
    replications = _strict_int(phase0.get("rollout_replications", 100), "phase0.rollout_replications")
    covariance_audit_replications = _strict_int(
        phase0.get("covariance_audit_replications", 32),
        "phase0.covariance_audit_replications",
    )
    raw_estimator = config.get("estimator", {})
    if not isinstance(raw_estimator, Mapping):
        raise TypeError("estimator must be a mapping")
    estimator = dict(raw_estimator)
    baseline_mode = str(estimator.get("primary_baseline", "zero"))
    max_branching = _strict_int(estimator.get("max_branching", max(budgets, default=1)), "estimator.max_branching")
    branchable_depth_count = _strict_int(
        estimator.get("branchable_depth_count", horizon), "estimator.branchable_depth_count"
    )
    if horizon < 1 or actions < 1 or replications < 1 or covariance_audit_replications < 2:
        raise ValueError("horizon, actions, and rollout_replications must be positive")
    if baseline_mode not in {"zero", "lagged_cross_fitted_value"}:
        raise ValueError("estimator.primary_baseline is unsupported")
    if max_branching < 1:
        raise ValueError("estimator.max_branching must be positive")
    if branchable_depth_count < 0 or branchable_depth_count > horizon:
        raise ValueError("estimator.branchable_depth_count must be between zero and horizon")
    if len(set(families)) != len(families) or len(set(arms)) != len(arms) or len(set(budgets)) != len(budgets):
        raise ValueError("families, arms, and budgets must not contain duplicates")
    if any(arm not in _REGISTERED_ARMS for arm in arms):
        raise ValueError("phase0.arms contains an unsupported arm")
    if any(budget < horizon for budget in budgets):
        raise ValueError("every budget must fund at least one complete trajectory")
    for budget in budgets:
        feasibility_cache: dict[tuple[int, int, int], tuple[int, ...]] = {}
        if not _feasible_level_increments(
            0,
            1,
            int(budget - horizon),
            horizon,
            branchable_depth_count,
            max_branching,
            feasibility_cache,
        ):
            raise ValueError(
                f"budget {budget} is not representable under max_branching={max_branching} "
                f"and branchable_depth_count={branchable_depth_count}"
            )
    oracle_scope = _parse_oracle_scope(phase0)
    calibration_seeds = tuple(
        _strict_int(item, "calibration seed")
        for item in phase0.get("learned_calibration_seeds", [seed - 2, seed - 1])
    )
    calibration_horizon = _strict_int(
        phase0.get("learned_calibration_horizon", horizon), "phase0.learned_calibration_horizon"
    )
    calibration_required = "jag_learned" in arms or baseline_mode == "lagged_cross_fitted_value"
    if calibration_required:
        if not calibration_seeds or calibration_horizon < 1:
            raise ValueError("lagged calibration seeds and horizon must be positive")
        if len(set(calibration_seeds)) != len(calibration_seeds):
            raise ValueError("lagged calibration seeds must be unique")
        if any(calibration_seed < 0 for calibration_seed in calibration_seeds):
            raise ValueError(
                "lagged calibration seeds must be nonnegative; low evaluation seeds require explicit earlier seeds"
            )
        if any(calibration_seed >= seed for calibration_seed in calibration_seeds):
            raise ValueError("lagged calibration seeds must be strictly earlier than the evaluation seed")
    jobs = [
        (
            family,
            horizon,
            actions,
            arms,
            budgets,
            replications,
            int(seed),
            calibration_seeds,
            calibration_horizon,
            oracle_scope,
            baseline_mode,
            max_branching,
            branchable_depth_count,
            covariance_audit_replications,
        )
        for family in families
    ]
    if len(jobs) == 1:
        result_pairs = [_run_family(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as executor:
            result_pairs = list(executor.map(_run_family, jobs))
    results = dict(result_pairs)
    output: dict[str, Any] = {
        "seed": int(seed),
        "estimator_policy": {
            "primary_baseline": baseline_mode,
            "max_branching": max_branching,
            "branchable_depth_count": branchable_depth_count,
            "branchable_depths": list(range(branchable_depth_count)),
        },
        "oracle_scope": {
            "target": "exact_each_observed_unopened_level_frontier",
            "global_tree_oracle_claimed": False,
            "max_horizon": oracle_scope.max_horizon,
            "max_budget": oracle_scope.max_budget,
            "max_frontier_nodes": oracle_scope.max_frontier_nodes,
            "max_states": oracle_scope.max_states,
        },
        "families": results,
    }
    output["gate"] = _gate_payload(config, int(seed), families, arms, budgets, replications, results)
    json.dumps(output, allow_nan=False)
    return output
