"""Exact contract tests for the Phase-0 GOAV implementation."""

from __future__ import annotations

import unittest

import numpy as np

from coding_opsd.goav.estimator import (
    aipw_gradient,
    aipw_pseudolabel,
    design_risk,
    direct_loo_gradient,
    exact_realized_design_mse,
    loo_influence,
)
from coding_opsd.goav.experiment import _epsilon_g, _standardized_design_bias, _synthetic_scores, run_goav_phase0
from coding_opsd.goav.noise import oracle_joint_posterior, shared_error_moments, simulate_synthetic_oracle
from coding_opsd.goav.solver import (
    bayes_voi_scores,
    bernoulli_design,
    full_audit_design,
    poisson_neyman_design,
    score_design,
    solve_goav,
)
from coding_opsd.goav.subsets import (
    SubsetDesign,
    enumerate_subsets,
    inclusion_probabilities,
    persist_sampled_request,
    sample_subset,
)
from coding_opsd.runtime import named_rng


class SubsetTests(unittest.TestCase):
    def test_enumeration_uses_candidate_zero_as_least_significant_bit(self) -> None:
        subsets = enumerate_subsets(8)
        self.assertEqual(subsets.shape, (256, 8))
        self.assertEqual(subsets.dtype, np.bool_)
        np.testing.assert_array_equal(subsets[0], np.zeros(8, dtype=bool))
        np.testing.assert_array_equal(subsets[1], [1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(subsets[2], [0, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(subsets[-1], np.ones(8, dtype=bool))

    def test_inclusion_probabilities_are_exact_and_reject_invalid_distributions(self) -> None:
        subsets = enumerate_subsets(2)
        probabilities = np.array([0.1, 0.2, 0.3, 0.4])
        pi, pi2 = inclusion_probabilities(probabilities, subsets)
        np.testing.assert_allclose(pi, [0.6, 0.7], atol=1e-15)
        np.testing.assert_allclose(pi2, [[0.6, 0.4], [0.4, 0.7]], atol=1e-15)
        self.assertGreaterEqual(pi2[0, 1], pi.sum() - 1.0)
        self.assertLessEqual(pi2[0, 1], min(pi))
        with self.assertRaises(ValueError):
            inclusion_probabilities(probabilities * 0.9, subsets)
        with self.assertRaises(ValueError):
            inclusion_probabilities([0.1, -0.1, 0.5, 0.5], subsets)

    def test_sample_record_contains_only_labels_selected_by_logged_design(self) -> None:
        design = bernoulli_design([0.35, 0.55, 0.75], [1.0, 1.0, 1.0])
        request = sample_subset(design, np.random.default_rng(14))
        record = persist_sampled_request(request, np.array([10.0, 20.0, 30.0]))
        self.assertEqual(record["design_hash"], design.design_hash)
        selected = set(np.flatnonzero(request.mask).tolist())
        self.assertEqual(set(map(int, record["selected_labels"])), selected)
        self.assertNotIn("labels", record)
        self.assertNotIn("unselected_labels", record)

    def test_full_audit_is_deterministic_without_full_support(self) -> None:
        design = full_audit_design(3, [1.0, 2.0, 3.0])
        self.assertTrue(design.full_audit)
        self.assertEqual(np.count_nonzero(design.probabilities), 1)
        np.testing.assert_array_equal(design.pi, np.ones(3))
        self.assertAlmostEqual(design.expected_cost, 6.0)

    def test_subset_design_constructor_cannot_bypass_probability_invariants(self) -> None:
        with self.assertRaises(ValueError):
            SubsetDesign(
                probabilities=np.array([0.5, 0.5, 0.0, 0.0]),
                pi=np.array([0.5, 0.0]),
                pi2=np.array([[0.5, 0.0], [0.0, 0.0]]),
                costs=np.ones(2),
                expected_cost=0.5,
                design_hash="not-the-probability-hash",
            )

        valid = bernoulli_design([0.4, 0.6], [1.0, 2.0])
        with self.assertRaises(ValueError):
            SubsetDesign(
                probabilities=valid.probabilities,
                pi=valid.pi,
                pi2=valid.pi2,
                costs=valid.costs,
                expected_cost=valid.expected_cost + 1.0,
                design_hash=valid.design_hash,
            )
        with self.assertRaises(ValueError):
            valid.costs[0] = 9.0


class EstimatorTests(unittest.TestCase):
    def test_linear_loo_influence_matches_direct_leave_one_out_gradient(self) -> None:
        scores = np.array([[1.0, -2.0], [3.0, 0.5], [-1.0, 4.0], [2.0, 1.5]])
        labels = np.array([1.0, 0.0, 1.0, 1.0])
        influence = loo_influence(scores)
        self.assertEqual(influence.shape, (2, 4))
        np.testing.assert_allclose(influence @ labels, direct_loo_gradient(scores, labels), atol=2e-15)

    def test_aipw_is_finite_group_unbiased_under_full_subset_enumeration(self) -> None:
        scores = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])
        labels = np.array([1.0, 0.0, 1.0])
        mu = np.array([0.25, 0.6, 0.4])
        design = bernoulli_design([0.2, 0.55, 0.8], np.ones(3))
        estimate = np.zeros(2)
        subsets = enumerate_subsets(3)
        for probability, mask in zip(design.probabilities, subsets, strict=True):
            observed = {int(i): labels[i] for i in np.flatnonzero(mask)}
            estimate += probability * aipw_gradient(loo_influence(scores), mu, observed, design.pi)
        np.testing.assert_allclose(estimate, direct_loo_gradient(scores, labels), atol=2e-14)

        indexed = labels.copy()
        indexed[1] = np.nan
        pseudolabel = aipw_pseudolabel(mu, indexed, design.pi)
        self.assertEqual(pseudolabel[1], mu[1])
        with self.assertRaises(ValueError):
            aipw_pseudolabel(mu, {0: 1.0}, [0.0, 0.5, 0.5])
        with self.assertRaises(ValueError):
            aipw_pseudolabel(mu, {0: 1.0}, [np.inf, 0.5, 0.5])
        with self.assertRaises(ValueError):
            aipw_pseudolabel(mu, {0: 1.0}, [1.01, 0.5, 0.5])
        with self.assertRaises(ValueError):
            aipw_pseudolabel(mu, {0: np.nan}, design.pi)

    def test_analytic_design_risk_matches_explicit_realized_mse(self) -> None:
        residual = np.array([0.7, -0.2, 0.4])
        covariance = np.outer(residual, residual)
        influence = np.array([[1.0, 2.0, -0.5], [0.25, -1.0, 1.5]])
        design = SubsetDesign.from_probabilities(
            [0.08, 0.12, 0.09, 0.11, 0.14, 0.16, 0.13, 0.17],
            [1.0, 1.0, 1.0],
        )
        analytic = design_risk(covariance, influence, design)
        explicit = exact_realized_design_mse(residual, influence, design)
        self.assertAlmostEqual(analytic, explicit, places=12)

    def test_risk_fails_closed_for_nonsymmetric_or_non_psd_inputs(self) -> None:
        influence = np.eye(2)
        design = bernoulli_design([0.4, 0.6], [1.0, 1.0])
        with self.assertRaises(ValueError):
            design_risk([[1.0, 0.4], [0.2, 1.0]], influence, design)
        with self.assertRaises(ValueError):
            design_risk([[1.0, 2.0], [2.0, 1.0]], influence, design)
        with self.assertRaises(ValueError):
            design_risk(np.eye(2), influence, design, metric=[[1.0, 0.3], [0.1, 1.0]])
        with self.assertRaises(ValueError):
            design_risk(np.eye(2), influence, design, metric=[[1.0, 2.0], [2.0, 1.0]])

    def test_raw_inclusions_and_realized_mse_fail_closed(self) -> None:
        covariance = np.eye(2)
        influence = np.eye(2)
        invalid_joint = (
            [[0.5, np.nan], [np.nan, 0.5]],
            [[0.5, 0.2], [0.1, 0.5]],
            [[0.4, 0.2], [0.2, 0.5]],
            [[0.8, 0.1], [0.1, 0.8]],
        )
        for pi2 in invalid_joint:
            with self.subTest(pi2=pi2), self.assertRaises(ValueError):
                design_risk(covariance, influence, [0.5, 0.5] if pi2 != invalid_joint[-1] else [0.8, 0.8], pi2)
        with self.assertRaises(ValueError):
            design_risk(covariance, influence, [np.nan, 0.5], np.eye(2) * 0.5)

        design = bernoulli_design([0.4, 0.6], [1.0, 1.0])
        with self.assertRaises(ValueError):
            exact_realized_design_mse([0.2, -0.1], [[1.0, np.inf], [0.0, 1.0]], design)
        for metric in ([[1.0, 0.2], [0.1, 1.0]], [[1.0, 2.0], [2.0, 1.0]], [[1.0, np.nan], [np.nan, 1.0]]):
            with self.subTest(metric=metric), self.assertRaises(ValueError):
                exact_realized_design_mse([0.2, -0.1], influence, design, metric=metric)


class DesignTests(unittest.TestCase):
    def test_bayes_voi_uses_joint_posterior_update_geometry_not_neyman_diagonal(self) -> None:
        covariance = np.array(
            [[0.90367347, 0.44100498, 0.41236286], [0.44100498, 2.71713827, 1.38858902], [0.41236286, 1.38858902, 0.75893006]]
        )
        influence = np.array([[0.2941325, 0.02842224, 0.54671299], [-0.73645409, -0.16290995, -0.48211931]])
        neyman = np.diag(covariance) * np.einsum("pk,pk->k", influence, influence)
        voi = bayes_voi_scores(covariance, influence)
        np.testing.assert_allclose(voi, [1.25067264, 1.10337262, 1.49401139], atol=2e-8)
        self.assertEqual(int(np.argmax(neyman)), 0)
        self.assertEqual(int(np.argmax(voi)), 2)

    def test_independent_neyman_design_has_poisson_off_diagonal_factors_and_budget(self) -> None:
        covariance = np.array([[1.0, 0.4, -0.2], [0.4, 3.0, 0.1], [-0.2, 0.1, 0.5]])
        influence = np.array([[1.0, 0.5, 2.0], [0.0, 1.0, -0.5]])
        costs = np.array([1.0, 2.0, 1.5])
        design = poisson_neyman_design(covariance, influence, costs, budget=1.4, floor=0.08)
        off_diagonal = ~np.eye(3, dtype=bool)
        np.testing.assert_allclose(design.pi2[off_diagonal], np.outer(design.pi, design.pi)[off_diagonal], atol=2e-14)
        self.assertAlmostEqual(design.expected_cost, 1.4, places=10)
        self.assertGreaterEqual(design.pi.min(), 0.08 - 1e-12)
        self.assertTrue(np.all(design.probabilities > 0.0))

    def test_neyman_zero_risk_candidates_absorb_degenerate_budget_feasibly(self) -> None:
        design = poisson_neyman_design(np.diag([1.0, 0.0, 0.0]), np.eye(3), np.ones(3), budget=2.0, floor=0.05)
        self.assertAlmostEqual(design.expected_cost, 2.0, places=10)
        self.assertTrue(np.all(design.probabilities > 0.0))
        self.assertGreaterEqual(design.pi.min(), 0.05 - 1e-12)

    def test_neyman_serialized_marginals_never_round_below_floor(self) -> None:
        importance = np.array(
            [
                1.2720385804619193e-07,
                6.942489960665621e-07,
                4.9126103595996306e-08,
                4.997615601657286e-08,
                1.9205621600458867e-07,
                1.1061605076750985e-08,
                3.1023119042338155e-08,
                6.946482473615158e-05,
            ]
        )
        design = poisson_neyman_design(
            np.diag(importance), np.eye(8), np.ones(8), budget=.8, floor=.02
        )
        self.assertGreaterEqual(float(design.pi.min()), .02)

    def test_score_design_uses_nonnegative_scores_and_meets_budget(self) -> None:
        design = score_design([0.0, 0.5, 2.0, 1.0], budget=1.2, floor=0.05)
        self.assertAlmostEqual(design.expected_cost, 1.2, places=10)
        self.assertGreaterEqual(design.pi.min(), 0.05 - 1e-12)
        self.assertGreater(design.pi[2], design.pi[1])
        with self.assertRaises(ValueError):
            score_design([0.0, -0.1], budget=0.5, floor=0.05)

    def test_goav_solver_is_deterministic_feasible_and_never_worse_than_uniform_initializer(self) -> None:
        covariance = np.array(
            [[1.0, 0.85, -0.3, 0.1], [0.85, 1.3, -0.25, 0.0], [-0.3, -0.25, 0.8, 0.35], [0.1, 0.0, 0.35, 0.7]]
        )
        influence = np.array([[1.0, -0.7, 0.25, 0.4], [0.2, 0.5, -1.0, 0.3]])
        costs = np.ones(4)
        first = solve_goav(covariance, influence, costs, 0.3, 0.04, steps=12, lr=0.05, restarts=3)
        second = solve_goav(covariance, influence, costs, 0.3, 0.04, steps=12, lr=0.05, restarts=3)
        np.testing.assert_array_equal(first.probabilities, second.probabilities)
        self.assertAlmostEqual(float(first.probabilities.sum()), 1.0, places=13)
        self.assertTrue(np.all(first.probabilities > 0.0))
        self.assertAlmostEqual(first.expected_cost, 1.2, delta=1e-4)
        self.assertGreaterEqual(first.pi.min(), 0.04 - 1e-10)
        initializer = bernoulli_design(np.full(4, 0.3), costs)
        self.assertLessEqual(design_risk(covariance, influence, first), design_risk(covariance, influence, initializer) + 1e-10)


class OracleAndExperimentTests(unittest.TestCase):
    def test_synthetic_gradient_geometry_honors_sketch_dimension_and_rank(self) -> None:
        scores = _synthetic_scores(17, 3, 8, 31, 1, 9517)
        self.assertEqual(scores.shape, (3, 8, 31))
        for task_scores in scores:
            self.assertLessEqual(np.linalg.matrix_rank(task_scores), 1)
            np.testing.assert_allclose(
                np.sort(np.linalg.norm(task_scores, axis=1)),
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0],
            )
        self.assertGreater(
            len({int(np.argmax(np.linalg.norm(task, axis=1))) for task in scores}),
            1,
        )
        np.testing.assert_array_equal(
            scores, _synthetic_scores(17, 3, 8, 31, 1, 9517)
        )

    def test_bias_is_normalized_per_task_before_aggregation_with_frozen_epsilon(self) -> None:
        targets = np.array([[1.0, 0.0], [3.0, 0.0]])
        epsilon_g = _epsilon_g(targets)
        self.assertAlmostEqual(epsilon_g, 0.05)
        task_estimates = [np.array([[2.0, 0.0], [2.0, 0.0]]), np.array([[2.0, 0.0], [2.0, 0.0]])]
        expected = 0.5 * (1.0 / np.sqrt(1.05) + 1.0 / np.sqrt(9.05))
        self.assertAlmostEqual(_standardized_design_bias(task_estimates, targets, epsilon_g), expected)

    def test_oracle_joint_posterior_is_normalized_and_psd(self) -> None:
        evidence = np.array([[1, 1, 0, 1], [0, 1, 0, 0], [1, 0, 1, 1]], dtype=int)
        log_q, mu, covariance = oracle_joint_posterior(evidence, 0.1, 0.2, 0.6, [0, 0, 1, 1])
        self.assertAlmostEqual(float(np.exp(log_q).sum()), 1.0, places=13)
        self.assertEqual(mu.shape, (3,))
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-14)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(covariance).min()), -1e-12)

    def test_rho_zero_uses_fixed_rates_and_shared_beta_moments_match_registered_icc(self) -> None:
        _, mu, _ = oracle_joint_posterior([[1]], 0.1, 0.2, 0.0, [0])
        self.assertAlmostEqual(mu[0], 0.6, places=13)
        mean, variance, covariance, correlation = shared_error_moments(0.2, 0.6)
        self.assertAlmostEqual(mean, 0.2)
        self.assertAlmostEqual(variance, 0.16)
        self.assertAlmostEqual(covariance, 0.096)
        self.assertAlmostEqual(correlation, 0.6)
        self.assertEqual(shared_error_moments(0.2, 0.0)[2:], (0.0, 0.0))

    def test_synthetic_noise_preserves_registered_fn_fp_marginals(self) -> None:
        labels, evidence = simulate_synthetic_oracle(
            groups=20_000,
            candidates=4,
            cluster_ids=[0, 0, 1, 1],
            false_negative=0.1,
            false_positive=0.2,
            rho=0.6,
            seed=91,
        )
        expanded = labels[:, :, None]
        fn_rate = np.mean(evidence[expanded.repeat(4, axis=2)] == 0)
        fp_rate = np.mean(evidence[(~expanded.astype(bool)).repeat(4, axis=2)] == 1)
        self.assertAlmostEqual(float(fn_rate), 0.1, delta=0.012)
        self.assertAlmostEqual(float(fp_rate), 0.2, delta=0.012)

    def test_phase0_runner_is_repeatedly_deterministic_and_marks_smoke_incomplete(self) -> None:
        config = {
            "runtime": {"profile": "smoke"},
            "phase0": {
                "tasks_min": 2,
                "group_size": 4,
                "tests_per_task_min": 4,
                "subset_draws_per_group_design": 2,
                "primary_budget_fraction": 0.25,
                "epsilon_g_dev_tasks": 5,
                "epsilon_g_dev_seed": 811,
                "arms": ["uniform_subset_aipw", "poisson_neyman_aipw", "goav_exact_subset_aipw", "full_audit"],
            },
            "acquisition": {
                "primary_inclusion_floor": 0.04,
                "solver": {"steps": 3, "learning_rate": 0.04, "initialization": ["uniform", "poisson_neyman"]},
            },
            "oracle_noise_model": {
                "medium_noise": {"false_negative": 0.1, "false_positive": 0.2, "flip_icc": 0.6, "cluster_size": 2}
            },
        }
        first = run_goav_phase0(config, seed=7)
        second = run_goav_phase0(config, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["gates"]["status"], "INCOMPLETE")
        self.assertEqual(first["solver"], {"steps": 3, "learning_rate": 0.04, "restarts": 2})
        self.assertGreater(first["epsilon_G"], 0.0)
        self.assertEqual(first["epsilon_G_provenance"], second["epsilon_G_provenance"])
        changed_evaluation = run_goav_phase0(config, seed=8)
        self.assertEqual(first["epsilon_G"], changed_evaluation["epsilon_G"])
        self.assertEqual(first["epsilon_G_provenance"], changed_evaluation["epsilon_G_provenance"])
        self.assertEqual({row["arm"] for row in first["rows"]}, set(config["phase0"]["arms"]))
        self.assertTrue(first["events"])
        logged_hashes: set[tuple[int, str, str]] = set()
        seen_audit_for_task: set[int] = set()
        draws_by_task_arm: dict[tuple[int, str], list[int]] = {}
        for event in first["events"]:
            if event["event"] == "design_logged":
                self.assertNotIn(event["task"], seen_audit_for_task)
                self.assertEqual(len(event["pi_hash"]), 64)
                self.assertEqual(len(event["pi2_hash"]), 64)
                logged_hashes.add((event["task"], event["arm"], event["design_hash"]))
            if event["event"] == "audit_request":
                seen_audit_for_task.add(event["task"])
                self.assertNotIn("labels", event)
                self.assertNotIn("unselected_labels", event)
                self.assertIn((event["task"], event["arm"], event["design_hash"]), logged_hashes)
                self.assertIsInstance(event["draw"], int)
                draws_by_task_arm.setdefault((event["task"], event["arm"]), []).append(event["draw"])
                design = next(
                    item for item in first["events"]
                    if item["event"] == "design_logged"
                    and item["task"] == event["task"]
                    and item["arm"] == event["arm"]
                )
                replay = named_rng(
                    7,
                    f"goav_audit:{event['task']}:{event['arm']}:{event['draw']}",
                ).choice(len(design["probabilities"]), p=design["probabilities"])
                self.assertEqual(event["subset_index"], int(replay))
        for task in range(2):
            for arm in config["phase0"]["arms"]:
                expected = [0] if arm == "full_audit" else [0, 1]
                self.assertEqual(draws_by_task_arm[(task, arm)], expected)

    def test_phase0_runner_uses_registered_gradient_sketch_dimension(self) -> None:
        base = {
            "runtime": {"profile": "smoke"},
            "phase0": {
                "tasks_min": 3,
                "group_size": 4,
                "tests_per_task_min": 4,
                "subset_draws_per_group_design": 3,
                "primary_budget_fraction": .25,
                "epsilon_G": .01,
                "arms": ["uniform_subset_aipw"],
            },
            "acquisition": {"primary_inclusion_floor": .04},
            "oracle_noise_model": {
                "medium_noise": {
                    "false_negative": .1,
                    "false_positive": .2,
                    "flip_icc": .6,
                    "cluster_size": 2,
                }
            },
        }
        low = {**base, "gradient_target": {"sketch_dimension": 2}}
        high = {**base, "gradient_target": {"sketch_dimension": 9}}
        self.assertNotEqual(
            run_goav_phase0(low, seed=13)["rows"],
            run_goav_phase0(high, seed=13)["rows"],
        )

    def test_phase0_runner_limits_primary_cheap_coverage_to_registered_count(self) -> None:
        config = {
            "runtime": {"profile": "smoke"},
            "gradient_target": {"sketch_dimension": 8, "sketch_seed": 9517},
            "phase0": {
                "tasks_min": 3,
                "group_size": 4,
                "tests_per_task_min": 8,
                "coverage_tests_per_candidate_primary": 4,
                "synthetic_score_rank": 1,
                "subset_draws_per_group_design": 3,
                "primary_budget_fraction": .25,
                "epsilon_G": .01,
                "arms": ["uniform_subset_aipw"],
            },
            "acquisition": {"primary_inclusion_floor": .04},
            "oracle_noise_model": {
                "medium_noise": {
                    "false_negative": .1,
                    "false_positive": .2,
                    "flip_icc": .6,
                    "cluster_size": 4,
                }
            },
        }
        limited = run_goav_phase0(config, seed=21)
        full = run_goav_phase0(
            {
                **config,
                "phase0": {
                    **config["phase0"],
                    "coverage_tests_per_candidate_primary": 8,
                },
            },
            seed=21,
        )
        self.assertNotEqual(limited["rows"], full["rows"])

    def test_deterministic_topk_is_flagged_invalid_and_reports_its_cost(self) -> None:
        config = {
            "runtime": {"profile": "smoke"},
            "phase0": {
                "tasks_min": 2,
                "group_size": 4,
                "tests_per_task_min": 2,
                "subset_draws_per_group_design": 1,
                "primary_budget_fraction": 0.25,
                "arms": ["deterministic_topk_invalid"],
                "epsilon_g_dev_tasks": 2,
                "epsilon_g_dev_seed": 44,
            },
            "acquisition": {"primary_inclusion_floor": 0.04},
            "oracle_noise_model": {
                "medium_noise": {"false_negative": 0.1, "false_positive": 0.2, "flip_icc": 0.0, "cluster_size": 1}
            },
        }
        row = run_goav_phase0(config, seed=5)["rows"][0]
        self.assertEqual(row["expected_cost"], 1.0)
        self.assertEqual(row["realized_cost"], 1.0)
        self.assertGreater(row["support_violations"], 0)


if __name__ == "__main__":
    unittest.main()
