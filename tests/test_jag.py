"""Behavioral tests for the JAG finite-MDP reference implementation."""

from __future__ import annotations

import itertools
import json
from hashlib import sha256
import time
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from coding_opsd.jag import (
    AllocationItem,
    FiniteMDP,
    JointMoments,
    OracleScope,
    OracleUnavailableError,
    TreeNode,
    exact_frontier_plan,
    exact_target,
    greedy_allocate,
    integer_oracle,
    joint_risk,
    joint_risk_no_cross,
    leaf_equal_estimate,
    marginal_score,
    make_phase0_mdp,
    recursive_estimate,
    run_jag_phase0,
    sample_balanced_tree,
    transport_weight,
)
from coding_opsd.jag.allocator import FrontierItem, greedy_frontier_plan
from coding_opsd.jag.experiment import (
    RidgePriority,
    _sample_budget_tree,
    build_calibration_record,
    fit_calibration_record,
)


class FiniteMDPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.theta = np.array([0.2, -0.3])
        self.weights = np.array([[0.4, -0.1, 0.2, 0.0, 0.1, -0.2], [-0.2, 0.3, -0.1, 0.1, -0.3, 0.2]])
        self.mdp = FiniteMDP(2, 2, self.theta, self.weights, "suffix_only", reward_seed=7)

    def test_enumerates_all_paths_and_normalizes_probability(self) -> None:
        paths = list(self.mdp.paths())
        self.assertEqual(paths, list(itertools.product(range(2), repeat=2)))
        self.assertAlmostEqual(sum(self.mdp.path_probability(path) for path in paths), 1.0, places=12)
        self.assertAlmostEqual(exact_target(self.mdp).probability_sum, 1.0, places=12)

    def test_score_has_zero_probability_weighted_mean_at_each_prefix(self) -> None:
        for prefix in [(), (0,), (1,)]:
            mean = sum(self.mdp.action_probability(prefix, action) * self.mdp.score(prefix, action) for action in range(2))
            np.testing.assert_allclose(mean, np.zeros(self.mdp.parameter_size), atol=1e-12)

    def test_exact_gradient_matches_central_finite_difference(self) -> None:
        target = exact_target(self.mdp)
        epsilon = 1e-6
        flattened = self.mdp.parameters.copy()
        numerical = np.empty_like(flattened)
        for index in range(flattened.size):
            plus, minus = flattened.copy(), flattened.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numerical[index] = (self.mdp.with_parameters(plus).expected_reward() - self.mdp.with_parameters(minus).expected_reward()) / (2 * epsilon)
        np.testing.assert_allclose(target.gradient, numerical, atol=1e-6)

    def test_formal_horizon_exact_oracle_finishes_within_smoke_budget(self) -> None:
        mdp = FiniteMDP(8, 4, np.zeros(4), np.zeros((4, 10)), "root_only", reward_seed=4)
        started = time.perf_counter()
        target = exact_target(mdp)
        self.assertLess(time.perf_counter() - started, 8.0)
        self.assertAlmostEqual(target.probability_sum, 1.0, places=12)


class EstimatorTests(unittest.TestCase):
    def test_one_child_chain_is_trajectory_reinforce_with_prefix_baselines(self) -> None:
        root = TreeNode(edge_score=np.zeros(2))
        middle = TreeNode(edge_score=np.array([1.0, -2.0]))
        leaf = TreeNode(edge_score=np.array([-3.0, 4.0]), reward=5.0)
        root.children = [middle]
        middle.children = [leaf]
        baselines = {id(root): 1.5, id(middle): 2.0}
        value, gradient = recursive_estimate(root, lambda node: baselines.get(id(node), 0.0))
        self.assertEqual(value, 5.0)
        expected = (5.0 - 1.5) * middle.edge_score + (5.0 - 2.0) * leaf.edge_score
        np.testing.assert_allclose(gradient, expected)

    def test_sibling_permutation_does_not_change_recursive_estimate(self) -> None:
        left = TreeNode(edge_score=np.array([1.0, 0.0]), reward=2.0)
        right = TreeNode(edge_score=np.array([0.0, 1.0]), reward=4.0)
        root = TreeNode(edge_score=np.zeros(2), children=[left, right])
        first = recursive_estimate(root, lambda _: 1.0)
        root.children.reverse()
        second = recursive_estimate(root, lambda _: 1.0)
        self.assertEqual(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])

    def test_callable_baseline_is_evaluated_once_for_a_sibling_group(self) -> None:
        root = TreeNode(edge_score=np.zeros(1), children=[TreeNode(edge_score=np.ones(1), reward=2.0), TreeNode(edge_score=np.ones(1), reward=4.0)])
        calls: list[int] = []
        def baseline(_: TreeNode) -> float:
            calls.append(1)
            return float(len(calls))
        first = recursive_estimate(root, baseline)
        self.assertEqual(len(calls), 1)
        root.children.reverse()
        calls.clear()
        second = recursive_estimate(root, baseline)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])

    def test_recursive_value_is_not_leaf_equal_when_branching_is_unequal(self) -> None:
        left = TreeNode(edge_score=np.zeros(1), reward=6.0)
        right = TreeNode(edge_score=np.zeros(1), children=[TreeNode(edge_score=np.zeros(1), reward=0.0), TreeNode(edge_score=np.zeros(1), reward=4.0)])
        root = TreeNode(edge_score=np.zeros(1), children=[left, right])
        recursive_value, _ = recursive_estimate(root, 0.0)
        self.assertEqual(recursive_value, 4.0)
        self.assertNotEqual(recursive_value, (6.0 + 0.0 + 4.0) / 3.0)
        self.assertEqual(leaf_equal_estimate(root)[0], (6.0 + 0.0 + 4.0) / 3.0)

    def test_leaf_equal_callable_baseline_is_frozen_once_per_sibling_group(self) -> None:
        root = TreeNode(
            edge_score=np.zeros(1),
            children=[
                TreeNode(edge_score=np.ones(1), reward=2.0),
                TreeNode(edge_score=-np.ones(1), reward=4.0),
            ],
        )
        calls: list[tuple[int, ...]] = []

        def baseline(node: TreeNode) -> float:
            calls.append(node.prefix)
            return 1.0

        value, gradient = leaf_equal_estimate(root, baseline)
        self.assertEqual(value, 3.0)
        self.assertEqual(calls, [()])
        np.testing.assert_allclose(gradient, [-1.0])


class AllocationTests(unittest.TestCase):
    def test_joint_risk_matches_population_covariance_of_value_score_sum(self) -> None:
        values = np.array([0.0, 2.0])
        gradients = np.array([[1.0, 0.0], [3.0, 2.0]])
        accumulated_score = np.array([2.0, -1.0])
        moments = JointMoments.from_samples(values, gradients)
        direct = np.trace(np.cov(gradients + values[:, None] * accumulated_score, rowvar=False, bias=True))
        self.assertAlmostEqual(joint_risk(moments, accumulated_score), direct)
        self.assertNotEqual(joint_risk(moments, accumulated_score), joint_risk_no_cross(moments, accumulated_score))

    def test_transport_weight_and_marginal_score_square_the_transport(self) -> None:
        self.assertEqual(transport_weight([2, 4]), 1 / 8)
        self.assertAlmostEqual(marginal_score(8.0, 3, 2.0, [2, 4]), (1 / 8) ** 2 * 8.0 / (2.0 * 3 * 4))

    def test_greedy_matches_literal_integer_oracle_with_stable_node_id_ties(self) -> None:
        items = [
            AllocationItem("b", branching=1, risk=8.0, cost=1.0),
            AllocationItem("a", branching=1, risk=8.0, cost=1.0),
        ]
        greedy = greedy_allocate(items, total_extra=2)
        oracle = integer_oracle(items, total_extra=2)
        self.assertEqual(greedy, {"a": 2, "b": 2})
        self.assertEqual(oracle, {"a": 2, "b": 2})

    def test_covariance_reversal_changes_full_joint_allocation_but_not_no_cross_ranking(self) -> None:
        moments = JointMoments.from_samples([0.0, 2.0], [[0.0, 0.0], [2.0, 0.0]])
        full = greedy_allocate(
            [
                AllocationItem("a", 1, joint_risk(moments, [-1.0, 0.0])),
                AllocationItem("b", 1, joint_risk(moments, [1.0, 0.0])),
            ],
            total_extra=1,
        )
        no_cross = greedy_allocate(
            [
                AllocationItem("a", 1, joint_risk_no_cross(moments, [-1.0, 0.0])),
                AllocationItem("b", 1, joint_risk_no_cross(moments, [1.0, 0.0])),
            ],
            total_extra=1,
        )
        self.assertEqual(full, {"a": 1, "b": 2})
        self.assertEqual(no_cross, {"a": 2, "b": 1})

    def test_greedy_recomputes_descendant_transport_after_ancestor_increment(self) -> None:
        items = [
            AllocationItem("ancestor", 1, risk=80.0),
            AllocationItem("descendant", 1, risk=80.0, parent_id="ancestor"),
            AllocationItem("peer", 1, risk=30.0),
        ]
        self.assertEqual(greedy_allocate(items, total_extra=2), {"ancestor": 2, "descendant": 1, "peer": 2})

    def test_integer_oracle_breaks_one_extra_tie_by_lowest_node_id(self) -> None:
        items = [AllocationItem("b", 1, 4.0), AllocationItem("a", 1, 4.0)]
        self.assertEqual(integer_oracle(items, total_extra=1), {"a": 2, "b": 1})

    def test_frontier_planner_uses_registered_marginal_not_raw_risk(self) -> None:
        items = [
            FrontierItem("high-risk", None, 1, risk=10.0, cost=10.0),
            FrontierItem("efficient", None, 1, risk=9.0, cost=1.0),
        ]
        plan = greedy_frontier_plan(items, extra_budget=1)
        self.assertEqual(plan.branching, {"efficient": 2, "high-risk": 1})
        self.assertEqual(plan.events[0]["node_id"], "efficient")

    def test_frontier_planner_uses_fixed_ancestor_transport_and_records_identity(self) -> None:
        items = [
            FrontierItem("raw-high", None, 1, risk=10.0, cost=1.0, ancestor_branching=(4,)),
            FrontierItem("transport-high", None, 1, risk=2.0, cost=1.0, ancestor_branching=(1,)),
        ]
        plan = greedy_frontier_plan(items, extra_budget=1)
        event = plan.events[0]
        self.assertEqual(event["node_id"], "transport-high")
        self.assertAlmostEqual(event["score"], 2.0 / 2.0)
        self.assertEqual(event["ancestor_branching"], [1])
        self.assertAlmostEqual(event["score_identity_error"], 0.0)

    def test_exact_frontier_oracle_matches_literal_objective_with_actual_ancestry(self) -> None:
        items = [
            FrontierItem("a", None, 1, risk=16.0, cost=2.0, ancestor_branching=(2,)),
            FrontierItem("b", None, 1, risk=40.0, cost=2.0, ancestor_branching=(4,)),
        ]
        result = exact_frontier_plan(items, 4, OracleScope(max_horizon=3, max_budget=12, max_frontier_nodes=4, max_states=100))
        candidates = []
        for a, b in itertools.product(range(1, 4), repeat=2):
            if 2 * (a - 1) + 2 * (b - 1) <= 4:
                objective = (1 / 2) ** 2 * 16 / a + (1 / 4) ** 2 * 40 / b
                candidates.append((objective, (-a, -b), {"a": a, "b": b}))
        expected = min(candidates, key=lambda item: (item[0], item[1]))
        self.assertEqual(result.branching, expected[2])
        self.assertAlmostEqual(result.objective_after, expected[0])
        self.assertEqual(result.events[0]["optimality_gap"], 0.0)

    def test_exact_frontier_oracle_enforces_its_own_budget_scope(self) -> None:
        with self.assertRaisesRegex(OracleUnavailableError, "max_budget"):
            exact_frontier_plan(
                [FrontierItem("a", None, 1, risk=1.0, cost=1.0)],
                13,
                OracleScope(max_horizon=3, max_budget=12, max_frontier_nodes=4, max_states=100),
            )


class SamplingAndExperimentTests(unittest.TestCase):
    def test_balanced_tree_draws_replacement_children_after_branching_is_fixed(self) -> None:
        mdp = FiniteMDP(1, 2, [0.0, 0.0], np.zeros((2, 6)), "root_only", reward_seed=3)
        root = sample_balanced_tree(mdp, branching=3, rng=np.random.default_rng(4))
        self.assertEqual(len(root.children), 3)
        self.assertTrue(all(child.reward is not None for child in root.children))

    def test_smoke_runner_is_seeded_serializable_and_formally_incomplete(self) -> None:
        config = {
            "runtime": {"profile": "smoke"},
            "phase0": {
                "horizon": 2,
                "actions": 2,
                "reward_families": ["root_only"],
                "arms": ["flat_iid", "uniform_tree"],
                "budgets": [4],
                "rollout_replications": 6,
                "covariance_audit_replications": 16,
            },
        }
        first = run_jag_phase0(config, seed=12)
        second = run_jag_phase0(config, seed=12)
        self.assertEqual(first, second)
        self.assertEqual(first["gate"]["status"], "INCOMPLETE")
        json.dumps(first, allow_nan=False)
        summary = first["families"]["root_only"]["summaries"][0]
        self.assertEqual(summary["sample_count"], 6)
        self.assertEqual(summary["allocation"]["replication_count"], 6)
        self.assertEqual(sum(item["count"] for item in summary["allocation"]["plan_histogram"].values()), 6)
        self.assertEqual(summary["allocation"]["total_edge_samples"], 24)
        self.assertIn("relative_bias", first["gate"]["inputs"])
        self.assertIn("registered_comparisons", first["gate"]["inputs"])
        covariance_audit = first["families"]["root_only"]["covariance_audit"]
        self.assertGreater(covariance_audit["denominator"], 0)
        self.assertGreaterEqual(covariance_audit["numerator"], 0)
        predicted = np.asarray(covariance_audit["predicted_covariance"], dtype="<f8")
        realized = np.asarray(covariance_audit["realized_covariance"], dtype="<f8")
        self.assertEqual(predicted.shape, realized.shape)
        self.assertEqual(predicted.shape, (covariance_audit["joint_dimension"],) * 2)
        numerator = float(np.sum((realized - predicted) ** 2))
        denominator = float(np.sum(predicted ** 2))
        self.assertAlmostEqual(covariance_audit["numerator"], numerator)
        self.assertAlmostEqual(covariance_audit["denominator"], denominator)
        self.assertAlmostEqual(
            covariance_audit["relative_frobenius_error"],
            float(np.sqrt(numerator / denominator)),
        )
        self.assertEqual(covariance_audit["covariance_dtype"], "<f8")
        self.assertEqual(
            covariance_audit["predicted_covariance_hash"],
            sha256(predicted.tobytes(order="C")).hexdigest(),
        )
        self.assertEqual(
            covariance_audit["realized_covariance_hash"],
            sha256(realized.tobytes(order="C")).hexdigest(),
        )
        self.assertIn("held_out_covariance_audit", first["gate"]["inputs"])
        self.assertNotIn("covariance_interval_coverage_90", summary)
        self.assertFalse(
            first["gate"]["evidence"]["checks"]["registered_covariance_coverage_gate_implemented"]
        )

    def test_runner_uses_distinct_matched_budget_genealogies_for_declared_arms(self) -> None:
        config = {
            "phase0": {
                "horizon": 3,
                "actions": 2,
                "reward_families": ["covariance_reversal"],
                "arms": ["flat_iid", "uniform_tree", "joint_no_cross", "jag_oracle"],
                "budgets": [6],
                "rollout_replications": 4,
            },
        }
        rows = run_jag_phase0(config, seed=18)["families"]["covariance_reversal"]["summaries"]
        by_arm = {row["arm"]: row for row in rows}
        self.assertNotEqual(by_arm["flat_iid"]["allocation"]["plan_histogram"], by_arm["uniform_tree"]["allocation"]["plan_histogram"])
        self.assertEqual(by_arm["joint_no_cross"]["allocation"]["total_edge_samples"], 24)
        self.assertEqual(by_arm["jag_oracle"]["allocation"]["total_edge_samples"], 24)
        self.assertEqual(by_arm["jag_oracle"]["status"], "EXACT_FRONTIER_ORACLE")
        self.assertAlmostEqual(by_arm["jag_oracle"]["allocation_evidence"]["oracle_max_optimality_gap"], 0.0)

    def test_learned_priority_does_not_read_current_environment_moments(self) -> None:
        mdp = FiniteMDP(2, 2, [0.0, 0.0], np.zeros((2, 6)), "root_only", reward_seed=2)
        mdp.conditional_moments = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("current moments read"))  # type: ignore[method-assign]
        root, plan = _sample_budget_tree(mdp, 4, "jag_learned", np.random.default_rng(2), RidgePriority(np.ones(5)))
        self.assertEqual(sum(plan.values()), 4)
        self.assertEqual(recursive_estimate(root)[0], recursive_estimate(root)[0])

    def test_lagged_predictor_trains_from_completed_hashed_record_and_rejects_tampering_or_current_seed(self) -> None:
        record = build_calibration_record("root_only", 2, 2, (2, 3))
        predictor = fit_calibration_record(record, evaluation_seed=4)
        self.assertEqual(predictor.calibration_record.data_hash, record.data_hash)
        self.assertTrue(predictor.calibration_record.completed)
        self.assertFalse(predictor.coefficients.flags.writeable)
        changed_targets = (record.targets[0] + 1.0,) + record.targets[1:]
        with self.assertRaisesRegex(ValueError, "hash"):
            fit_calibration_record(replace(record, targets=changed_targets), evaluation_seed=4)
        changed = replace(record, targets=changed_targets)
        self_hashed = replace(changed, data_hash=changed.computed_hash())
        with self.assertRaisesRegex(ValueError, "provenance"):
            fit_calibration_record(self_hashed, evaluation_seed=4)
        fully_self_hashed = replace(
            self_hashed,
            provenance_hash=self_hashed.computed_provenance_hash(),
        )
        with self.assertRaisesRegex(ValueError, "trusted rebuild"):
            fit_calibration_record(fully_self_hashed, evaluation_seed=4)
        invalid_protocol = replace(record, risk_label_baseline="invented")
        invalid_protocol = replace(invalid_protocol, data_hash=invalid_protocol.computed_hash())
        invalid_protocol = replace(
            invalid_protocol,
            provenance_hash=invalid_protocol.computed_provenance_hash(),
        )
        with self.assertRaisesRegex(ValueError, "baseline protocol"):
            fit_calibration_record(invalid_protocol, evaluation_seed=4)
        wrong_environment = replace(record, environment_hashes=("0" * 64,) + record.environment_hashes[1:])
        wrong_environment = replace(wrong_environment, data_hash=wrong_environment.computed_hash())
        wrong_environment = replace(
            wrong_environment,
            provenance_hash=wrong_environment.computed_provenance_hash(),
        )
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            fit_calibration_record(wrong_environment, evaluation_seed=4)
        with self.assertRaisesRegex(ValueError, "earlier"):
            fit_calibration_record(record, evaluation_seed=3)

    def test_oracle_scope_parser_rejects_coercions_and_unknown_keys(self) -> None:
        base = {
            "runtime": {"profile": "smoke"},
            "phase0": {
                "horizon": 2,
                "actions": 2,
                "reward_families": ["root_only"],
                "arms": ["flat_iid"],
                "budgets": [2],
                "rollout_replications": 1,
                "covariance_audit_replications": 2,
            },
        }
        valid_scope = {
            "max_horizon": 3,
            "max_budget": 12,
            "max_frontier_nodes": 8,
            "max_states": 100000,
        }
        invalid_scopes = [
            ({**valid_scope, "max_horizon": True}, "integer"),
            ({**valid_scope, "max_budget": 12.0}, "integer"),
            ({**valid_scope, "max_states": "100000"}, "integer"),
            ({**valid_scope, "surprise": 1}, "unknown"),
            ({key: value for key, value in valid_scope.items() if key != "max_states"}, "missing"),
        ]
        for scope, message in invalid_scopes:
            with self.subTest(scope=scope):
                config = {**base, "phase0": {**base["phase0"], "oracle_scope": scope}}
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    run_jag_phase0(config, seed=12)

    def test_live_runner_planner_uses_transport_instead_of_raw_risk(self) -> None:
        class ScriptedRNG:
            def __init__(self) -> None:
                self.values = iter([0, 1, 0, 0, 0, 0] + [0] * 20)

            def choice(self, _: int, *, p: np.ndarray) -> int:
                self.probabilities = p
                return next(self.values)

        mdp = FiniteMDP(3, 2, np.zeros(2), np.zeros((2, 6)), "root_only", reward_seed=2)

        def risk(
            _: FiniteMDP,
            prefix: tuple[int, ...],
            __: str,
            learned: RidgePriority | None = None,
            baseline: object = 0.0,
        ) -> float:
            _ = learned, baseline
            if prefix == (0,):
                return 100.0
            if prefix == (1,):
                return 1.0
            if len(prefix) == 2 and prefix[0] == 0:
                return 10.0
            if len(prefix) == 2 and prefix[0] == 1:
                return 2.0
            return 1.0

        with patch("coding_opsd.jag.experiment._node_priority", side_effect=risk):
            root, plan = _sample_budget_tree(mdp, 12, "joint_no_cross", ScriptedRNG())  # type: ignore[arg-type]
        self.assertEqual(sum(plan.values()), 12)
        depth_two = [event for event in root.planning_events if event["depth"] == 2 and event["kind"] == "marginal_increment"]
        self.assertGreaterEqual(len(depth_two), 1)
        self.assertEqual(depth_two[0]["prefix"], [1, 0])
        self.assertEqual(depth_two[0]["risk"], 2.0)
        self.assertTrue(any(event["risk"] == 10.0 for event in depth_two[0]["candidates"]))
        self.assertAlmostEqual(depth_two[0]["score_identity_error"], 0.0)

    def test_oracle_is_unavailable_outside_explicit_scope_instead_of_falling_back(self) -> None:
        config = {
            "runtime": {"profile": "smoke"},
            "phase0": {
                "horizon": 4,
                "actions": 2,
                "reward_families": ["root_only"],
                "arms": ["jag_oracle"],
                "budgets": [8],
                "rollout_replications": 2,
                "oracle_scope": {"max_horizon": 3, "max_budget": 12, "max_frontier_nodes": 8, "max_states": 10000},
            },
        }
        row = run_jag_phase0(config, seed=18)["families"]["root_only"]["summaries"][0]
        self.assertEqual(row["status"], "UNAVAILABLE_ORACLE_SCOPE")
        self.assertEqual(row["sample_count"], 0)
        self.assertEqual(row["allocation"]["replication_count"], 0)
        self.assertIn("horizon", row["status_reason"])

    def test_leaf_equal_is_explicit_non_jag_negative_control_and_all_plans_are_counted(self) -> None:
        config = {
            "runtime": {"profile": "smoke"},
            "phase0": {
                "horizon": 3,
                "actions": 2,
                "reward_families": ["root_only"],
                "arms": ["leaf_equal_naive"],
                "budgets": [7],
                "rollout_replications": 5,
            },
        }
        row = run_jag_phase0(config, seed=21)["families"]["root_only"]["summaries"][0]
        self.assertTrue(row["negative_control"])
        self.assertEqual(row["arm_metadata"]["jag_status"], "NON_JAG_NEGATIVE_CONTROL")
        self.assertFalse(row["arm_metadata"]["uses_same_tree_outcomes_for_allocation"])
        self.assertEqual(row["arm_metadata"]["negative_control_mechanism"], "leaf_equal_weighting")
        self.assertEqual(row["arm_metadata"]["moment_estimator"], "recursive_sampled_continuation")
        self.assertEqual(row["allocation"]["replication_count"], 5)
        self.assertEqual(row["allocation"]["total_edge_samples"], 35)
        self.assertEqual(sum(entry["count"] for entry in row["allocation"]["plan_histogram"].values()), 5)

    def test_runner_enforces_and_reports_baseline_branching_and_branchable_depths(self) -> None:
        config = {
            "runtime": {"profile": "smoke"},
            "estimator": {
                "primary_baseline": "zero",
                "max_branching": 2,
                "branchable_depth_count": 1,
            },
            "phase0": {
                "horizon": 3,
                "actions": 2,
                "reward_families": ["root_only"],
                "arms": ["flat_iid"],
                "budgets": [6],
                "rollout_replications": 3,
            },
        }
        result = run_jag_phase0(config, seed=21)
        self.assertEqual(
            result["estimator_policy"],
            {
                "primary_baseline": "zero",
                "max_branching": 2,
                "branchable_depth_count": 1,
                "branchable_depths": [0],
            },
        )
        row = result["families"]["root_only"]["summaries"][0]
        for entry in row["allocation"]["plan_histogram"].values():
            self.assertLessEqual(max(branching for _, branching in entry["canonical_plan"]), 2)
        self.assertEqual(row["arm_metadata"]["baseline"], "zero")
        self.assertTrue(row["arm_metadata"]["baseline_frozen_before_evaluation"])
        self.assertEqual(row["arm_metadata"]["baseline_evaluation_frequency"], "once_per_sibling_group")
        self.assertFalse(row["arm_metadata"]["baseline_reads_evaluation_outcomes"])
        impossible = {**config, "phase0": {**config["phase0"], "budgets": [4]}}
        with self.assertRaisesRegex(ValueError, "representable"):
            run_jag_phase0(impossible, seed=21)

    def test_runner_uses_paired_keyed_uniform_streams_and_persists_replicate_evidence(self) -> None:
        config = {
            "runtime": {"profile": "smoke"},
            "estimator": {"primary_baseline": "zero", "max_branching": 4, "branchable_depth_count": 2},
            "phase0": {
                "horizon": 2,
                "actions": 2,
                "reward_families": ["suffix_only"],
                "arms": ["flat_iid", "uniform_tree"],
                "budgets": [4],
                "rollout_replications": 4,
                "covariance_audit_replications": 16,
            },
        }
        rows = run_jag_phase0(config, seed=21)["families"]["suffix_only"]["summaries"]
        streams = [[item["crn_stream_id"] for item in row["paired_replicates"]] for row in rows]
        self.assertEqual(streams[0], streams[1])
        self.assertEqual([item["replication"] for item in rows[0]["paired_replicates"]], list(range(4)))
        self.assertTrue(all(item["gradient_hash"] for row in rows for item in row["paired_replicates"]))
        other_rows = run_jag_phase0(config, seed=22)["families"]["suffix_only"]["summaries"]
        other_streams = [item["crn_stream_id"] for item in other_rows[0]["paired_replicates"]]
        self.assertNotEqual(streams[0], other_streams)

    def test_undeclared_seed_and_inapplicable_requested_structure_are_invalid(self) -> None:
        undeclared = {
            "runtime": {"profile": "smoke"},
            "seeds": [20],
            "phase0": {"horizon": 2, "actions": 2, "reward_families": ["root_only"], "arms": ["flat_iid"], "budgets": [2], "rollout_replications": 1},
        }
        self.assertEqual(run_jag_phase0(undeclared, seed=21)["gate"]["status"], "INVALID")
        bad_structure = {
            "runtime": {"profile": "smoke"},
            "phase0": {"horizon": 3, "actions": 2, "reward_families": ["covariance_reversal"], "arms": ["flat_iid"], "budgets": [3], "rollout_replications": 1},
        }
        self.assertEqual(run_jag_phase0(bad_structure, seed=21)["gate"]["status"], "INVALID")

    def test_runner_rejects_invalid_or_nonlagged_calibration_seed_declarations(self) -> None:
        base = {
            "runtime": {"profile": "smoke"},
            "estimator": {
                "primary_baseline": "lagged_cross_fitted_value",
                "max_branching": 2,
                "branchable_depth_count": 2,
            },
            "phase0": {
                "horizon": 2,
                "actions": 2,
                "reward_families": ["root_only"],
                "arms": ["jag_learned"],
                "budgets": [2],
                "rollout_replications": 1,
                "covariance_audit_replications": 2,
            },
        }
        for invalid_seeds, message in [
            ([1, 1], "unique"),
            ([True], "integer"),
            ([3], "earlier"),
            ([-1], "nonnegative"),
        ]:
            with self.subTest(seeds=invalid_seeds):
                config = {**base, "phase0": {**base["phase0"], "learned_calibration_seeds": invalid_seeds}}
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    run_jag_phase0(config, seed=3)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            run_jag_phase0(base, seed=1)
        boolean_setting = {**base, "estimator": {**base["estimator"], "max_branching": True}}
        with self.assertRaisesRegex(TypeError, "integer"):
            run_jag_phase0(boolean_setting, seed=3)
        for declared_seeds, message in [([True], "integer"), ([3, 3], "unique")]:
            with self.subTest(declared_seeds=declared_seeds):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    run_jag_phase0({**base, "seeds": declared_seeds}, seed=3)


class DiagnosticRewardTests(unittest.TestCase):
    def test_entropy_distractor_has_high_entropy_but_lower_risk_than_informative_node_across_seeds(self) -> None:
        for seed in (3, 17, 101):
            with self.subTest(seed=seed):
                audit = make_phase0_mdp(3, 4, "entropy_distractor", seed).structural_audit()
                nodes = audit["nodes"]
                self.assertGreater(nodes["distractor"]["entropy"], nodes["informative"]["entropy"])
                self.assertLess(nodes["distractor"]["joint_risk"], nodes["informative"]["joint_risk"])
                self.assertTrue(audit["checks"]["entropy_distractor_valid"])

    def test_covariance_reversal_has_equal_variance_and_opposite_measured_cross_terms_across_seeds(self) -> None:
        for seed in (3, 17, 101):
            with self.subTest(seed=seed):
                audit = make_phase0_mdp(8, 4, "covariance_reversal", seed).structural_audit()
                left, right = audit["nodes"]["left"], audit["nodes"]["right"]
                self.assertAlmostEqual(left["value_variance"], right["value_variance"], places=12)
                self.assertLess(left["cross_term"] * right["cross_term"], 0.0)
                self.assertAlmostEqual(left["cross_term"] + right["cross_term"], 0.0, places=12)
                left_covariance = np.asarray(left["value_gradient_covariance"])
                right_covariance = np.asarray(right["value_gradient_covariance"])
                self.assertGreater(np.linalg.norm(left_covariance), 0.0)
                np.testing.assert_allclose(left_covariance, -right_covariance, atol=1e-12)
                self.assertAlmostEqual(audit["checks"]["local_feature_difference_norm"], 0.0, places=12)
                self.assertAlmostEqual(audit["checks"]["accumulated_score_difference_norm"], 0.0, places=12)
                self.assertTrue(audit["checks"]["covariance_reversal_valid"])


class ConditionalMomentTests(unittest.TestCase):
    def test_conditional_moments_include_the_childs_downstream_gradient(self) -> None:
        mdp = FiniteMDP(2, 2, [0.0, 0.0], np.zeros((2, 6)), "suffix_only", reward_seed=11)
        contributions = []
        values = []
        for action in range(2):
            suffixes = [(action, next_action) for next_action in range(2)]
            child_value = sum(0.5 * mdp.reward(path) for path in suffixes)
            child_gradient = sum(0.5 * mdp.reward(path) * mdp.score((action,), path[1]) for path in suffixes)
            values.append(child_value)
            contributions.append(child_gradient + child_value * mdp.score((), action))
        expected = JointMoments.from_samples(values, np.asarray(contributions))
        actual = mdp.conditional_moments(())
        np.testing.assert_allclose(actual.gradient_mean, expected.gradient_mean)
        # Exact child means recover the contribution mean, but intentionally
        # omit the sampled suffix variance retained by conditional_moments.
        self.assertGreater(
            float(np.trace(actual.gradient_covariance)),
            float(np.trace(expected.gradient_covariance)),
        )

    def test_conditional_moments_match_full_continuation_enumeration_not_conditional_means(self) -> None:
        mdp = FiniteMDP(2, 2, [0.2, -0.3], np.array([[0.4, -0.1, 0.2, 0.0, 0.1, -0.2], [-0.2, 0.3, -0.1, 0.1, -0.3, 0.2]]), "suffix_only", reward_seed=7)
        values, gradients, probabilities = [], [], []
        for path in mdp.paths():
            reward = mdp.reward(path)
            values.append(reward)
            gradients.append(reward * (mdp.score((), path[0]) + mdp.score((path[0],), path[1])))
            probabilities.append(mdp.path_probability(path))
        expected = self._weighted_moments(values, gradients, probabilities)
        actual = mdp.conditional_moments((), baseline=0.0, continuation_branching=1)
        self.assertGreater(float(np.trace(actual.gradient_covariance)), 0.3)
        self.assertAlmostEqual(actual.value_variance, expected.value_variance)
        np.testing.assert_allclose(actual.gradient_mean, expected.gradient_mean, atol=1e-12)
        np.testing.assert_allclose(actual.gradient_covariance, expected.gradient_covariance, atol=1e-12)
        np.testing.assert_allclose(actual.value_gradient_covariance, expected.value_gradient_covariance, atol=1e-12)

    def test_conditional_moments_match_enumeration_under_two_child_suffix_allocation(self) -> None:
        mdp = FiniteMDP(2, 2, [0.2, -0.3], np.array([[0.4, -0.1, 0.2, 0.0, 0.1, -0.2], [-0.2, 0.3, -0.1, 0.1, -0.3, 0.2]]), "suffix_only", reward_seed=7)
        values, gradients, probabilities = [], [], []
        for first, second_a, second_b in itertools.product(range(2), repeat=3):
            probability = mdp.action_probability((), first)
            probability *= mdp.action_probability((first,), second_a)
            probability *= mdp.action_probability((first,), second_b)
            rewards = np.array([mdp.reward((first, second_a)), mdp.reward((first, second_b))])
            value = float(np.mean(rewards))
            downstream = np.mean(
                np.stack(
                    [
                        rewards[0] * mdp.score((first,), second_a),
                        rewards[1] * mdp.score((first,), second_b),
                    ]
                ),
                axis=0,
            )
            gradient = downstream + value * mdp.score((), first)
            values.append(value)
            gradients.append(gradient)
            probabilities.append(probability)
        expected = self._weighted_moments(values, gradients, probabilities)
        actual = mdp.conditional_moments(
            (),
            baseline=0.0,
            continuation_branching={(0,): 2, (1,): 2},
        )
        self.assertAlmostEqual(actual.value_variance, expected.value_variance)
        np.testing.assert_allclose(actual.gradient_mean, expected.gradient_mean, atol=1e-12)
        np.testing.assert_allclose(actual.gradient_covariance, expected.gradient_covariance, atol=1e-12)
        np.testing.assert_allclose(actual.value_gradient_covariance, expected.value_gradient_covariance, atol=1e-12)

    @staticmethod
    def _weighted_moments(values: list[float], gradients: list[np.ndarray], probabilities: list[float]) -> JointMoments:
        value_array = np.asarray(values, dtype=np.float64)
        gradient_array = np.asarray(gradients, dtype=np.float64)
        probability_array = np.asarray(probabilities, dtype=np.float64)
        probability_array /= np.sum(probability_array)
        value_mean = float(probability_array @ value_array)
        gradient_mean = probability_array @ gradient_array
        centered_values = value_array - value_mean
        centered_gradients = gradient_array - gradient_mean
        return JointMoments(
            value_mean,
            gradient_mean,
            float(probability_array @ (centered_values ** 2)),
            (centered_gradients.T * probability_array) @ centered_gradients,
            probability_array @ (centered_values[:, None] * centered_gradients),
        )


if __name__ == "__main__":
    unittest.main()
