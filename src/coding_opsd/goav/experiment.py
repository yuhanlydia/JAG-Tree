"""Small deterministic CPU-only synthetic Phase-0 GOAV experiment."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

import numpy as np

from coding_opsd.metrics import cosine_similarity, effective_sample_size
from coding_opsd.runtime import named_rng

from .estimator import aipw_gradient, design_risk, exact_realized_design_mse, loo_influence
from .noise import oracle_joint_posterior, simulate_synthetic_oracle
from .solver import bayes_voi_scores, bernoulli_design, full_audit_design, poisson_neyman_design, score_design, solve_goav
from .subsets import SubsetDesign, persist_sampled_request, sample_subset


def _nested(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _epsilon_g(targets: np.ndarray) -> float:
    """Freeze the registered one-percent dev-median squared-gradient stabilizer."""

    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or not np.isfinite(values).all():
        raise ValueError("targets must be a non-empty finite task-by-gradient matrix")
    squared_norms = np.einsum("tp,tp->t", values, values)
    return max(0.01 * float(np.median(squared_norms)), float(np.finfo(np.float64).eps))


def _standardized_design_bias(task_estimates: list[np.ndarray], targets: np.ndarray, epsilon_g: float) -> float:
    """Average task-wise design bias after taking each task's design expectation."""

    target_values = np.asarray(targets, dtype=np.float64)
    if len(task_estimates) != len(target_values) or epsilon_g <= 0.0 or not np.isfinite(epsilon_g):
        raise ValueError("task estimates, targets, and epsilon_g are incompatible")
    standardized: list[float] = []
    for estimates, target in zip(task_estimates, target_values, strict=True):
        values = np.asarray(estimates, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1:] != target.shape:
            raise ValueError("each task must have one or more matching gradient estimates")
        bias = np.mean(values, axis=0) - target
        standardized.append(float(np.linalg.norm(bias) / np.sqrt(np.dot(target, target) + epsilon_g)))
    return float(np.mean(standardized))


def _array_hash(values: np.ndarray) -> str:
    return sha256(np.asarray(values, dtype="<f8").tobytes(order="C")).hexdigest()


def _synthetic_scores(
    seed: int,
    tasks: int,
    candidates: int,
    sketch_dimension: int,
    score_rank: int,
    sketch_seed: int,
) -> np.ndarray:
    """Create a low-rank score family in a fixed Rademacher sketch space."""

    if min(tasks, candidates, sketch_dimension, score_rank) < 1:
        raise ValueError("synthetic score dimensions must be positive")
    if score_rank > sketch_dimension:
        raise ValueError("synthetic score rank cannot exceed sketch dimension")
    coefficient_rng = named_rng(seed, "goav_phase0_score_permutations")
    base = np.full(candidates, -1.0)
    base[0] = float(candidates - 1)
    coefficients = np.empty((tasks, candidates, score_rank), dtype=np.float64)
    for task in range(tasks):
        for component in range(score_rank):
            coefficients[task, :, component] = coefficient_rng.permutation(base)
    projection_rng = named_rng(sketch_seed, "goav_gradient_projection")
    projection = projection_rng.choice(
        (-1.0, 1.0), size=(score_rank, sketch_dimension)
    ) / np.sqrt(float(sketch_dimension))
    return np.einsum("tkr,rp->tkp", coefficients, projection)


def _freeze_epsilon_g(
    phase0: Mapping[str, Any],
    candidates: int,
    tests: int,
    cluster_ids: np.ndarray,
    false_negative: float,
    false_positive: float,
    rho: float,
    sketch_dimension: int,
    score_rank: int | None,
    sketch_seed: int,
) -> tuple[float, dict[str, Any]]:
    """Freeze epsilon from an explicit scalar or an evaluation-independent synthetic dev panel."""

    explicit = phase0.get("epsilon_G")
    if explicit is not None:
        epsilon = float(explicit)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("explicit epsilon_G must be finite and positive")
        return epsilon, {"source": "explicit", "input_hash": _array_hash(np.array([epsilon]))}

    dev_tasks = int(phase0.get("epsilon_g_dev_tasks", max(4, int(phase0.get("tasks_min", 4)))))
    dev_seed = int(phase0.get("epsilon_g_dev_seed", 730_241))
    if dev_tasks < 1:
        raise ValueError("epsilon_g_dev_tasks must be positive")
    dev_labels, _ = simulate_synthetic_oracle(
        dev_tasks,
        candidates,
        cluster_ids,
        false_negative,
        false_positive,
        rho,
        dev_seed,
    )
    if score_rank is None:
        dev_scores = named_rng(dev_seed, "goav_epsilon_dev_scores").normal(
            size=(dev_tasks, candidates, sketch_dimension)
        )
        score_rng_label = "goav_epsilon_dev_scores"
    else:
        dev_scores = _synthetic_scores(
            dev_seed,
            dev_tasks,
            candidates,
            sketch_dimension,
            score_rank,
            sketch_seed,
        )
        score_rng_label = "goav_phase0_score_permutations"
    dev_targets = np.asarray(
        [loo_influence(dev_scores[index]) @ dev_labels[index].astype(np.float64) for index in range(dev_tasks)]
    )
    epsilon = _epsilon_g(dev_targets)
    provenance = {
        "source": "synthetic_dev_panel",
        "seed": dev_seed,
        "tasks": dev_tasks,
        "candidates": candidates,
        "tests": tests,
        "panel_rng_label": "goav_synthetic_oracle",
        "score_rng_label": score_rng_label,
        "input_hash": _array_hash(dev_targets),
        "formula": "max(0.01*median_squared_gradient_norm,float64_epsilon)",
    }
    return epsilon, provenance


def _arm_design(
    arm: str,
    covariance: np.ndarray,
    influence: np.ndarray,
    mu: np.ndarray,
    evidence: np.ndarray,
    costs: np.ndarray,
    budget_fraction: float,
    floor: float,
    steps: int,
    learning_rate: float,
    restarts: int,
) -> SubsetDesign | None:
    budget = float(budget_fraction * costs.sum())
    if arm in {"cheap_only", "deterministic_topk_invalid"}:
        return None
    if arm in {"uniform_subset_ht", "uniform_subset_aipw"}:
        return bernoulli_design(np.full(len(costs), budget_fraction), costs)
    if arm == "entropy_subset_aipw":
        clipped = np.clip(mu, 1e-15, 1.0 - 1e-15)
        entropy = -clipped * np.log(clipped) - (1.0 - clipped) * np.log1p(-clipped)
        return score_design(entropy, budget, floor, costs)
    if arm == "killrate_subset_aipw":
        kill_rate = np.mean(evidence == 0, axis=1)
        return score_design(kill_rate, budget, floor, costs)
    if arm == "poisson_neyman_aipw":
        return poisson_neyman_design(covariance, influence, costs, budget, floor)
    if arm == "bayes_voi_aipw":
        return score_design(bayes_voi_scores(covariance, influence), budget, floor, costs)
    if arm == "goav_exact_subset_aipw":
        return solve_goav(
            covariance,
            influence,
            costs,
            budget_fraction,
            floor,
            steps=steps,
            lr=learning_rate,
            restarts=restarts,
        )
    if arm == "full_audit":
        return full_audit_design(len(costs), costs)
    raise ValueError(f"unknown GOAV arm: {arm}")


def run_goav_phase0(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Run declared synthetic smoke arms with paired groups and audit draws."""

    phase0 = _nested(config, "phase0")
    acquisition = _nested(config, "acquisition")
    gradient_target = _nested(config, "gradient_target")
    noise = _nested(_nested(config, "oracle_noise_model"), "medium_noise")
    runtime = _nested(config, "runtime")
    solver_config = _nested(acquisition, "solver")
    tasks = int(phase0.get("tasks_min", 4))
    candidates = int(phase0.get("group_size", 8))
    tests = int(phase0.get("tests_per_task_min", 4))
    coverage_tests = int(phase0.get("coverage_tests_per_candidate_primary", tests))
    draws = int(phase0.get("subset_draws_per_group_design", 8))
    budget_fraction = float(phase0.get("primary_budget_fraction", acquisition.get("primary_budget_fraction", 0.1)))
    floor = float(acquisition.get("primary_inclusion_floor", 0.02))
    steps = int(phase0.get("solver_steps", solver_config.get("steps", 200)))
    learning_rate = float(phase0.get("solver_learning_rate", solver_config.get("learning_rate", 0.05)))
    registered_initializers = solver_config.get("initialization", ())
    default_restarts = len(registered_initializers) if isinstance(registered_initializers, (list, tuple)) and registered_initializers else 8
    restarts = int(phase0.get("solver_restarts", default_restarts))
    arms = list(
        phase0.get(
            "arms",
            [
                "cheap_only",
                "uniform_subset_ht",
                "uniform_subset_aipw",
                "entropy_subset_aipw",
                "killrate_subset_aipw",
                "poisson_neyman_aipw",
                "bayes_voi_aipw",
                "goav_exact_subset_aipw",
                "deterministic_topk_invalid",
                "full_audit",
            ],
        )
    )
    if min(tasks, candidates, tests, coverage_tests, draws) < 1:
        raise ValueError("tasks, candidates, tests, and draws must be positive")
    if coverage_tests > tests:
        raise ValueError("primary coverage tests cannot exceed the fixed test pool")
    cluster_size = int(noise.get("cluster_size", 2))
    cluster_ids = np.arange(tests) // cluster_size
    false_negative = float(noise.get("false_negative", 0.1))
    false_positive = float(noise.get("false_positive", 0.2))
    rho = float(noise.get("flip_icc", 0.6))
    sketch_dimension = int(
        gradient_target.get("sketch_dimension", max(2, candidates // 2))
    )
    if sketch_dimension < 1:
        raise ValueError("gradient sketch dimension must be positive")
    score_rank_raw = phase0.get("synthetic_score_rank")
    score_rank = None if score_rank_raw is None else int(score_rank_raw)
    sketch_seed = int(gradient_target.get("sketch_seed", 9517))
    epsilon_g, epsilon_provenance = _freeze_epsilon_g(
        phase0,
        candidates,
        tests,
        cluster_ids,
        false_negative,
        false_positive,
        rho,
        sketch_dimension,
        score_rank,
        sketch_seed,
    )
    labels, evidences = simulate_synthetic_oracle(
        tasks, candidates, cluster_ids, false_negative, false_positive, rho, seed
    )
    primary_evidences = evidences[:, :, :coverage_tests]
    primary_cluster_ids = cluster_ids[:coverage_tests]
    if score_rank is None:
        score_rng = named_rng(seed, "goav_phase0_scores")
        scores = score_rng.normal(size=(tasks, candidates, sketch_dimension))
    else:
        scores = _synthetic_scores(
            seed,
            tasks,
            candidates,
            sketch_dimension,
            score_rank,
            sketch_seed,
        )
    costs = np.ones(candidates, dtype=np.float64)
    task_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for task in range(tasks):
        _, mu, covariance = oracle_joint_posterior(
            primary_evidences[task],
            false_negative,
            false_positive,
            rho,
            primary_cluster_ids,
        )
        influence = loo_influence(scores[task])
        target = influence @ labels[task].astype(np.float64)
        task_data.append((mu, covariance, influence, target))
    targets = np.asarray([data[3] for data in task_data])
    events: list[dict[str, Any]] = []
    accumulators: dict[str, dict[str, Any]] = {
        arm: {"estimates": [[] for _ in range(tasks)], "design_expectations": [], "cosines": [], "weights": [], "costs": [], "expected": [], "analytic": [], "realized": [], "support": 0}
        for arm in arms
    }

    for task, (mu, covariance, influence, target) in enumerate(task_data):
        # Freeze and persist every outcome-blind task/arm design before any
        # trusted label from this task enters an audit event or realized-risk
        # calculation.  This makes non-sequential acquisition auditable from
        # the event stream rather than merely an implementation convention.
        designs: dict[str, SubsetDesign | None] = {}
        for arm in arms:
            design = _arm_design(
                arm,
                covariance,
                influence,
                mu,
                primary_evidences[task],
                costs,
                budget_fraction,
                floor,
                steps,
                learning_rate,
                restarts,
            )
            designs[arm] = design
            accumulator = accumulators[arm]
            if arm == "deterministic_topk_invalid":
                count = max(1, int(round(budget_fraction * candidates)))
                accumulator["expected"].append(float(count))
                accumulator["support"] += 1
            if design is not None:
                events.append(
                    {
                        "event": "design_logged",
                        "task": task,
                        "arm": arm,
                        "design_hash": design.design_hash,
                        "probabilities": design.probabilities.tolist(),
                        "pi": design.pi.tolist(),
                        "pi2": design.pi2.tolist(),
                        "pi_hash": _array_hash(design.pi),
                        "pi2_hash": _array_hash(design.pi2),
                    }
                )
                accumulator["expected"].append(design.expected_cost)
                if not design.full_audit:
                    accumulator["support"] += int(np.any(design.probabilities <= 0.0) or np.any(design.pi < floor - 1e-12))
                accumulator["analytic"].append(design_risk(covariance, influence, design))

        residual = labels[task].astype(np.float64) - mu
        for arm in arms:
            design = designs[arm]
            accumulator = accumulators[arm]
            if design is not None:
                accumulator["realized"].append(exact_realized_design_mse(residual, influence, design))
            if arm == "cheap_only":
                design_expectation = influence @ mu
            elif arm == "deterministic_topk_invalid":
                count = max(1, int(round(budget_fraction * candidates)))
                selected = np.argsort(-mu)[:count]
                pseudo = mu.copy()
                pseudo[selected] = labels[task, selected]
                design_expectation = influence @ pseudo
            else:
                # HT/AIPW are exactly unbiased under the logged full-support
                # distribution.  Use their design expectation for bias rather
                # than treating finite evaluation draws as systematic bias.
                design_expectation = target.copy()
            accumulator["design_expectations"].append(design_expectation)
            arm_draws = 1 if arm in {"cheap_only", "deterministic_topk_invalid", "full_audit"} else draws
            for draw in range(arm_draws):
                if arm == "cheap_only":
                    estimate = influence @ mu
                    realized_cost = 0.0
                elif arm == "deterministic_topk_invalid":
                    count = max(1, int(round(budget_fraction * candidates)))
                    selected = np.argsort(-mu)[:count]
                    pseudo = mu.copy()
                    pseudo[selected] = labels[task, selected]
                    estimate = influence @ pseudo
                    realized_cost = float(count)
                else:
                    assert design is not None
                    request_rng = named_rng(seed, f"goav_audit:{task}:{arm}:{draw}")
                    request = sample_subset(design, request_rng)
                    persisted = persist_sampled_request(request, labels[task].astype(np.float64))
                    events.append({"event": "audit_request", "task": task, "arm": arm, "draw": draw, **persisted})
                    observed = {index: float(labels[task, index]) for index in request.selected_indices}
                    means = np.zeros(candidates) if arm == "uniform_subset_ht" else mu
                    estimate = aipw_gradient(influence, means, observed, design.pi)
                    realized_cost = float(np.dot(request.mask, costs))
                    accumulator["weights"].extend((1.0 / design.pi[list(request.selected_indices)]).tolist())
                accumulator["estimates"][task].append(estimate)
                accumulator["cosines"].append(cosine_similarity(estimate, target))
                accumulator["costs"].append(realized_cost)

    rows: list[dict[str, Any]] = []
    for arm in arms:
        accumulator = accumulators[arm]
        task_estimates = [np.asarray(values) for values in accumulator["estimates"]]
        design_expectations = [
            np.asarray([value]) for value in accumulator["design_expectations"]
        ]
        squared_error = 0.0
        stabilized_target = 0.0
        for estimates, target in zip(task_estimates, targets, strict=True):
            errors = estimates - target[None, :]
            squared_error += float(np.einsum("np,np->", errors, errors))
            stabilized_target += len(estimates) * float(np.dot(target, target) + epsilon_g)
        weights = np.asarray(accumulator["weights"], dtype=np.float64)
        expected_costs = accumulator["expected"]
        rows.append(
            {
                "arm": arm,
                "nMSE": float(squared_error / stabilized_target),
                "gradient_cosine": float(np.mean(accumulator["cosines"])),
                "standardized_bias": _standardized_design_bias(design_expectations, targets, epsilon_g),
                "kish_ess": float(effective_sample_size(weights)) if weights.size else 0.0,
                "weight_p50": float(np.quantile(weights, 0.50)) if weights.size else 0.0,
                "weight_p95": float(np.quantile(weights, 0.95)) if weights.size else 0.0,
                "weight_p99": float(np.quantile(weights, 0.99)) if weights.size else 0.0,
                "expected_cost": float(np.mean(expected_costs)) if expected_costs else 0.0,
                "realized_cost": float(np.mean(accumulator["costs"])),
                "support_violations": int(accumulator["support"]),
                "exact_risk": float(np.mean(accumulator["analytic"])) if accumulator["analytic"] else 0.0,
                "realized_risk": float(np.mean(accumulator["realized"])) if accumulator["realized"] else 0.0,
            }
        )
    return {
        "rows": rows,
        "epsilon_G": epsilon_g,
        "epsilon_G_provenance": epsilon_provenance,
        "solver": {"steps": steps, "learning_rate": learning_rate, "restarts": restarts},
        "gates": {"status": "INCOMPLETE" if runtime.get("profile", "smoke") == "smoke" else "INCOMPLETE", "formal_evidence": False},
        "events": events,
    }
