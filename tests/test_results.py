from __future__ import annotations

import copy
from hashlib import sha256
import json
import unittest

import numpy as np

from coding_opsd.config import config_hash
from coding_opsd.jag.experiment import run_jag_phase0
from coding_opsd.results import NormalizedRun, ResultNormalizationError, normalize_result


JAG_CONFIG = {
    "direction": "jag_tree",
    "runtime": {"profile": "formal"},
    "seeds": [101],
    "output": {"identity": "jag-test"},
}
PBPF_CONFIG = {
    "direction": "predictive_belief_particle_filter",
    "runtime": {"profile": "formal"},
    "seeds": [101],
    "output": {"identity": "pbpf-test"},
}
GOAV_CONFIG = {
    "direction": "gradient_optimal_active_verification",
    "runtime": {"profile": "formal"},
    "seeds": [101],
    "output": {"identity": "goav-test"},
    "phase0": {
        "tasks_min": 1,
        "group_size": 2,
        "subset_draws_per_group_design": 1,
        "primary_budget_fraction": 0.6,
        "arms": ["goav_exact_subset_aipw"],
    },
    "acquisition": {"primary_inclusion_floor": 0.1},
}
PBPF_SMOKE_CONFIG = {**PBPF_CONFIG, "runtime": {"profile": "smoke"}}
GOAV_SMOKE_CONFIG = {**GOAV_CONFIG, "runtime": {"profile": "smoke"}}


def expected_run_id(config: dict, seed: int = 101) -> str:
    return f'{config["output"]["identity"]}-seed{seed}-{config_hash(config)[:8]}'


def array_hash(values: list) -> str:
    return sha256(np.asarray(values, dtype="<f8").tobytes(order="C")).hexdigest()


def pbpf_result() -> dict:
    return {
        "seed": 101,
        "status": "FAIL",
        "rows": [
            {
                "episode": 0,
                "prefix": 4,
                "horizon": "all_remaining",
                "arm": "exact_bayes",
                "nll": 0.25,
                "brier": 0.1,
                "posterior_kl": 0.0,
                "ess": None,
                "resampling": 0,
                "hpd_true_inclusion": 1.0,
            },
            {
                "episode": 0,
                "prefix": 4,
                "horizon": "all_remaining",
                "arm": "particle_16",
                "nll": None,
                "brier": None,
                "posterior_kl": None,
                "ess": 0.0,
                "pre_resampling_ess": 1.5,
                "pre_resampling_ess_history": [4.0, 1.5],
                "resampling": 1,
                "resampling_rate": 0.25,
                "unique_ancestor_ratio": None,
                "post_rejuvenation_unique_state_ratio": None,
                "resampling_events": [
                    {
                        "position": 2,
                        "pre_resampling_ess": 1.5,
                        "unique_ancestor_ratio": 0.25,
                        "post_rejuvenation_unique_state_ratio": 0.5,
                    }
                ],
                "particles_per_correct_equivalence_class": 3,
                "hpd_true_inclusion": 0.0,
                "diagnostic": "PARTICLE_COLLAPSE",
            },
        ],
        "gate_status": "FAIL",
        "gate_inputs": {
            "hpd_nominal": 0.9,
            "by_prefix_horizon": {"4": {"all_remaining": {"p16_collapse_count": 1}}},
            "prefix_4": {
                "paired_exact_vs_prior_inputs": [0.3, 0.2],
                "paired_exact_vs_map_inputs": [0.4, 0.1],
                "paired_p16_to_exact_inputs": [0.05],
                "hpd_coverage_inputs": [1.0, 0.0],
                "p16_collapse_count": 1,
            },
            "exact_vs_prior_nll_delta": 0.25,
            "exact_mixture_vs_map_nll_delta": 0.25,
            "p16_to_exact_nll_gap": 0.05,
            "map_gap_fraction_closed": 0.8,
        },
        "rejected_arms": ["bag_of_tests"],
    }


def goav_result() -> dict:
    probabilities = [0.1, 0.3, 0.3, 0.3]
    pi = [0.6, 0.6]
    pi2 = [[0.6, 0.3], [0.3, 0.6]]
    design_hash = array_hash(probabilities)
    return {
        "rows": [
            {
                "arm": "goav_exact_subset_aipw",
                "nMSE": 0.2,
                "gradient_cosine": 0.9,
                "standardized_bias": 0.03,
                "kish_ess": 3.5,
                "weight_p50": 2.0,
                "weight_p95": 3.0,
                "weight_p99": 3.2,
                "expected_cost": 1.0,
                "realized_cost": 1.0,
                "support_violations": 0,
                "exact_risk": 0.4,
                "realized_risk": 0.45,
            }
        ],
        "epsilon_G": 0.01,
        "epsilon_G_provenance": {"source": "explicit", "input_hash": "e" * 64},
        "solver": {"steps": 20, "learning_rate": 0.05, "restarts": 2},
        "gates": {"status": "INCOMPLETE", "formal_evidence": False},
        "events": [
            {
                "event": "design_logged",
                "task": 0,
                "arm": "goav_exact_subset_aipw",
                "design_hash": design_hash,
                "probabilities": probabilities,
                "pi": pi,
                "pi2": pi2,
                "pi_hash": array_hash(pi),
                "pi2_hash": array_hash(pi2),
            },
            {
                "event": "audit_request",
                "task": 0,
                "arm": "goav_exact_subset_aipw",
                "draw": 0,
                "subset_index": 1,
                "selected_indices": [0],
                "design_hash": design_hash,
                "inclusion_probabilities": pi,
                "selected_labels": {"0": 1.0},
            },
        ],
    }


def jag_result() -> dict:
    empty_allocation = {
        "replication_count": 0,
        "unique_plan_count": 0,
        "total_edge_samples": 0,
        "plan_histogram": {},
    }
    allocation = {
        "replication_count": 2,
        "unique_plan_count": 1,
        "total_edge_samples": 128,
        "plan_histogram": {
            "a" * 64: {"count": 2, "edge_count": 64, "canonical_plan": [["n000000", 64]]}
        },
    }
    unavailable_evidence = {
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
    }
    available_evidence = {**unavailable_evidence, "exact_edge_budget_every_replication": True}
    predicted_covariance = [
        [0.30, 0.02, -0.01],
        [0.02, 0.20, 0.03],
        [-0.01, 0.03, 0.10],
    ]
    realized_covariance = [
        [0.32, 0.01, -0.02],
        [0.01, 0.18, 0.04],
        [-0.02, 0.04, 0.11],
    ]
    predicted_array = np.asarray(predicted_covariance, dtype="<f8")
    realized_array = np.asarray(realized_covariance, dtype="<f8")
    covariance_numerator = float(np.sum((realized_array - predicted_array) ** 2))
    covariance_denominator = float(np.sum(predicted_array**2))
    covariance_audit = {
        "status": "OK",
        "prefix": [0],
        "sample_count": 32,
        "held_out_from_rollout_and_calibration": True,
        "object": "joint_node_contribution_[value,gradient]_covariance",
        "joint_dimension": 3,
        "covariance_dtype": "<f8",
        "covariance_normalization": "population_1_over_n",
        "predicted_covariance": predicted_covariance,
        "realized_covariance": realized_covariance,
        "realized_joint_mean": [0.5, 0.1, -0.1],
        "numerator": covariance_numerator,
        "numerator_definition": "squared_frobenius(realized_covariance-predicted_covariance)",
        "denominator": covariance_denominator,
        "denominator_definition": "squared_frobenius(predicted_covariance)",
        "relative_frobenius_error": float(
            np.sqrt(covariance_numerator / covariance_denominator)
        ),
        "predicted_covariance_hash": array_hash(predicted_covariance),
        "realized_covariance_hash": array_hash(realized_covariance),
    }
    common_metadata = {
        "jag_status": "JAG",
        "estimator": "recursive",
        "allocation_rule": "exact_live_frontier_joint_risk",
        "moment_source": "exact_future_oracle_ablation",
        "moment_estimator": "recursive_sampled_continuation",
        "moment_continuation_branching": 1,
        "oracle_ablation": True,
        "deployable": False,
        "baseline": "lagged_cross_fitted_value",
        "baseline_frozen_before_evaluation": True,
        "baseline_evaluation_frequency": "once_per_sibling_group",
        "baseline_reads_evaluation_outcomes": False,
        "unbiased_claim": True,
    }
    return {
        "seed": 101,
        "estimator_policy": {
            "primary_baseline": "lagged_cross_fitted_value",
            "max_branching": 64,
            "branchable_depth_count": 2,
            "branchable_depths": [0, 1],
        },
        "oracle_scope": {
            "target": "exact_each_observed_unopened_level_frontier",
            "global_tree_oracle_claimed": False,
            "max_horizon": 3,
            "max_budget": 12,
            "max_frontier_nodes": 8,
            "max_states": 100000,
        },
        "families": {
            "root_only": {
                "exact": {"probability_sum": 1.0, "expected_reward": 0.5, "gradient": [0.1, -0.1]},
                "structural_audit": {
                    "family": "root_only",
                    "reward_min": 0.0,
                    "reward_max": 1.0,
                    "status": "NOT_APPLICABLE",
                    "cross_term_prefix": "0",
                    "cross_term_comparator_prefix": "1",
                    "measured_cross_term": 0.0,
                    "tolerance": 1e-10,
                    "nodes": {
                        "audit": {
                            "prefix": [0],
                            "local_features": [0.1, 0.2],
                            "accumulated_score": [0.0, 0.0],
                            "entropy": 0.69,
                            "value_variance": 0.1,
                            "cross_term": 0.0,
                            "joint_risk": 0.2,
                            "value_gradient_covariance": [0.0, 0.0],
                            "value_gradient_covariance_norm": 0.0,
                        }
                    },
                    "checks": {},
                },
                "covariance_audit": covariance_audit,
                "predictor_provenance": {
                    "schema_version": "jag.calibration.v1",
                    "completed": True,
                    "calibration_seeds": [99, 100],
                    "environment_ids": ["root_only:h2:a2:seed99", "root_only:h2:a2:seed100"],
                    "environment_hashes": ["e" * 64, "f" * 64],
                    "row_count": 6,
                    "data_hash": "1" * 64,
                    "provenance_hash": "2" * 64,
                    "risk_label_baseline": "lagged_cross_fitted_value",
                    "evaluation_seed": 101,
                    "oracle_labels_from_evaluation_environment": False,
                    "evaluation_baseline_fit_from_completed_record_only": True,
                },
                "summaries": [
                    {
                        "arm": "jag_oracle",
                        "budget": 64,
                        "sample_count": 0,
                        "bias": None,
                        "absolute_bias": None,
                        "relative_bias": None,
                        "bias_z_score": None,
                        "bias_z_score_status": "UNAVAILABLE",
                        "mse": None,
                        "budget_times_mse": None,
                        "cosine": None,
                        "allocation": empty_allocation,
                        "allocation_evidence": unavailable_evidence,
                        "paired_replicates": [],
                        "arm_metadata": common_metadata,
                        "negative_control": False,
                        "status": "UNAVAILABLE_ORACLE_SCOPE",
                        "status_reason": "horizon exceeds exact oracle scope",
                    },
                    {
                        "arm": "uniform_tree",
                        "budget": 64,
                        "sample_count": 2,
                        "bias": 0.01,
                        "absolute_bias": 0.01,
                        "relative_bias": 0.02,
                        "bias_z_score": 0.1,
                        "bias_z_score_status": "OK",
                        "mse": 0.2,
                        "budget_times_mse": 12.8,
                        "cosine": 0.99,
                        "allocation": allocation,
                        "allocation_evidence": available_evidence,
                        "paired_replicates": [
                            {
                                "replication": 0,
                                "crn_stream_id": "jag-crn:root_only:b64:r0",
                                "gradient_hash": "3" * 64,
                                "gradient_delta_hash": "4" * 64,
                                "squared_error": 0.2,
                            },
                            {
                                "replication": 1,
                                "crn_stream_id": "jag-crn:root_only:b64:r1",
                                "gradient_hash": "5" * 64,
                                "gradient_delta_hash": "6" * 64,
                                "squared_error": 0.2,
                            },
                        ],
                        "arm_metadata": {
                            "jag_status": "BASELINE",
                            "estimator": "recursive",
                            "allocation_rule": "balanced_live_frontier",
                            "moment_source": "none",
                            "moment_estimator": "not_applicable",
                            "moment_continuation_branching": None,
                            "oracle_ablation": False,
                            "deployable": True,
                            "baseline": "lagged_cross_fitted_value",
                            "baseline_frozen_before_evaluation": True,
                            "baseline_evaluation_frequency": "once_per_sibling_group",
                            "baseline_reads_evaluation_outcomes": False,
                            "unbiased_claim": True,
                        },
                        "negative_control": False,
                        "status": "OK",
                        "status_reason": None,
                    },
                ],
            }
        },
        "gate": {
            "status": "INCOMPLETE",
            "reason": "formal registered seed/replication/budget matrix not established",
            "inputs": {
                "thresholds": {},
                "relative_bias": [
                    {"family": "root_only", "arm": "jag_oracle", "budget": 64, "value": None},
                    {"family": "root_only", "arm": "uniform_tree", "budget": 64, "value": 0.02},
                ],
                "bias_z_score": [
                    {
                        "family": "root_only",
                        "arm": "jag_oracle",
                        "budget": 64,
                        "value": None,
                        "status": "UNAVAILABLE",
                    },
                    {
                        "family": "root_only",
                        "arm": "uniform_tree",
                        "budget": 64,
                        "value": 0.1,
                        "status": "OK",
                    },
                ],
                "held_out_covariance_audit": [{"family": "root_only", **covariance_audit}],
                "registered_comparisons": [
                    {
                        "family": "root_only",
                        "budget": 64,
                        "oracle_vs_best_unbiased_relative_mse_reduction": None,
                        "learned_oracle_uniform_gap_closed": None,
                        "learned_minus_uniform_paired_squared_error": None,
                    },
                    {
                        "family": "covariance_reversal",
                        "budget": 64,
                        "full_joint_vs_no_cross_relative_mse_gain": None,
                        "full_joint_minus_no_cross_paired_squared_error": None,
                    },
                ],
            },
            "evidence": {
                "active_seed": 101,
                "configured_seed_ids": [101],
                "active_seed_declared": True,
                "observed_seed_count": 1,
                "required_seed_count": 50,
                "rollout_replications_per_cell": 2,
                "available_cell_count": 1,
                "required_cell_count": 108,
                "checks": {
                    "registered_families_present": False,
                    "registered_arms_present": False,
                    "registered_budgets_present": False,
                    "registered_replications_present": False,
                    "all_registered_cells_have_samples": False,
                    "fifty_environment_seed_results_present": False,
                    "diagnostic_structures_pass": False,
                    "registered_covariance_coverage_gate_implemented": False,
                },
                "missing": ["registered_families_present"],
            },
        },
    }


class ResultNormalizationTests(unittest.TestCase):
    def test_pbpf_normalizes_null_mask_gate_arrays_and_preserves_row_order(self) -> None:
        normalized = normalize_result(
            "pbpf", "predictive_belief_particle_filter", PBPF_CONFIG, 101, expected_run_id(PBPF_CONFIG), pbpf_result()
        )
        self.assertIsInstance(normalized, NormalizedRun)
        self.assertEqual(normalized.metrics["status"], "FAIL")
        self.assertEqual(normalized.gates["source_status"], "FAIL")
        self.assertTrue(normalized.gates["formal_evidence"])
        self.assertEqual(normalized.rows[0]["row_index"], 0)
        self.assertEqual(normalized.rows[1]["row_index"], 1)
        self.assertEqual(normalized.rows[1]["record_type"], "forecast_diagnostic")
        matrix = normalized.arrays["pbpf_row_numeric"]
        present = normalized.arrays["pbpf_row_numeric_present"]
        self.assertEqual(matrix.shape, (2, 6))
        self.assertEqual(present.shape, matrix.shape)
        self.assertEqual(matrix.dtype, np.dtype("float64"))
        self.assertEqual(present.dtype, np.dtype("bool"))
        self.assertFalse(present[1, 0])
        self.assertEqual(matrix[1, 0], 0.0)
        self.assertEqual(normalized.metrics["collapse_counts"], {"total": 1, "by_arm": {"particle_16": 1}})
        self.assertEqual(normalized.arrays["pbpf_gate_prefix4_paired_p16_to_exact"].shape, (1,))
        self.assertEqual(normalized.events, ())

    def test_smoke_forces_incomplete_but_retains_source_state(self) -> None:
        raw = pbpf_result()
        raw["status"] = raw["gate_status"] = "PASS"
        normalized = normalize_result(
            "pbpf", "predictive_belief_particle_filter", PBPF_SMOKE_CONFIG, 101, expected_run_id(PBPF_SMOKE_CONFIG), raw
        )
        self.assertEqual(normalized.metrics["status"], "INCOMPLETE")
        self.assertEqual(normalized.gates["status"], "INCOMPLETE")
        self.assertEqual(normalized.gates["source_status"], "PASS")
        self.assertFalse(normalized.gates["formal_evidence"])
        self.assertEqual(normalized.gates["reason"], "smoke_profile_below_formal_registration")

    def test_smoke_never_hides_invalid_source_state(self) -> None:
        raw = pbpf_result()
        raw["status"] = raw["gate_status"] = "INVALID"
        normalized = normalize_result(
            "pbpf", "predictive_belief_particle_filter", PBPF_SMOKE_CONFIG, 101, expected_run_id(PBPF_SMOKE_CONFIG), raw
        )
        self.assertEqual(normalized.metrics["status"], "INVALID")
        self.assertEqual(normalized.gates["status"], "INVALID")
        self.assertFalse(normalized.gates["formal_evidence"])
        self.assertEqual(normalized.gates["reason"], "source_result_invalid")

    def test_explicit_formal_evidence_status_matrix_is_conservative(self) -> None:
        cases = [
            ("formal", "PASS", None, "PASS", True, False),
            ("formal", "FAIL", None, "FAIL", True, False),
            ("formal", "INCOMPLETE", None, "INCOMPLETE", False, False),
            ("formal", "INVALID", None, "INVALID", False, False),
            ("formal", "PASS", True, "PASS", True, False),
            ("formal", "PASS", False, "INCOMPLETE", False, False),
            ("formal", "FAIL", True, "FAIL", True, False),
            ("formal", "FAIL", False, "INCOMPLETE", False, False),
            ("formal", "INCOMPLETE", False, "INCOMPLETE", False, False),
            ("formal", "INCOMPLETE", True, None, None, True),
            ("formal", "INVALID", False, "INVALID", False, False),
            ("formal", "INVALID", True, None, None, True),
            ("smoke", "PASS", None, "INCOMPLETE", False, False),
            ("smoke", "FAIL", None, "INCOMPLETE", False, False),
            ("smoke", "INCOMPLETE", None, "INCOMPLETE", False, False),
            ("smoke", "INVALID", None, "INVALID", False, False),
            ("smoke", "PASS", True, "INCOMPLETE", False, False),
            ("smoke", "PASS", False, "INCOMPLETE", False, False),
            ("smoke", "FAIL", True, "INCOMPLETE", False, False),
            ("smoke", "FAIL", False, "INCOMPLETE", False, False),
            ("smoke", "INCOMPLETE", False, "INCOMPLETE", False, False),
            ("smoke", "INCOMPLETE", True, None, None, True),
            ("smoke", "INVALID", False, "INVALID", False, False),
            ("smoke", "INVALID", True, None, None, True),
        ]
        for profile, source_status, source_evidence, final_status, formal_evidence, rejects in cases:
            with self.subTest(profile=profile, source_status=source_status, source_evidence=source_evidence):
                if source_evidence is None:
                    config = copy.deepcopy(PBPF_CONFIG)
                    config["runtime"]["profile"] = profile
                    raw = pbpf_result()
                    raw["status"] = raw["gate_status"] = source_status
                    arguments = (
                        "pbpf",
                        "predictive_belief_particle_filter",
                        config,
                        101,
                        expected_run_id(config),
                        raw,
                    )
                else:
                    config = copy.deepcopy(GOAV_CONFIG)
                    config["runtime"]["profile"] = profile
                    raw = goav_result()
                    raw["gates"] = {"status": source_status, "formal_evidence": source_evidence}
                    arguments = (
                        "goav",
                        "gradient_optimal_active_verification",
                        config,
                        101,
                        expected_run_id(config),
                        raw,
                    )
                if rejects:
                    with self.assertRaises(ResultNormalizationError):
                        normalize_result(*arguments)
                    continue
                normalized = normalize_result(*arguments)
                self.assertEqual(normalized.metrics["status"], final_status)
                self.assertEqual(normalized.gates["status"], final_status)
                self.assertEqual(normalized.gates["source_status"], source_status)
                self.assertEqual(normalized.gates["formal_evidence"], formal_evidence)
                if profile == "formal" and source_status in {"PASS", "FAIL"} and source_evidence is False:
                    self.assertEqual(normalized.gates["reason"], "source_formal_evidence_false")

    def test_pbpf_rejects_missing_extra_status_disagreement_seed_and_nonfinite(self) -> None:
        cases = []
        missing = pbpf_result()
        del missing["rows"]
        cases.append(missing)
        extra = pbpf_result()
        extra["guessed_metric"] = 1
        cases.append(extra)
        disagreement = pbpf_result()
        disagreement["gate_status"] = "PASS"
        cases.append(disagreement)
        wrong_seed = pbpf_result()
        wrong_seed["seed"] = 202
        cases.append(wrong_seed)
        nonfinite = pbpf_result()
        nonfinite["rows"][0]["nll"] = float("inf")
        cases.append(nonfinite)
        malformed_diagnostic = pbpf_result()
        malformed_diagnostic["rows"][1]["pre_resampling_ess_history"] = [4.0, "not-a-number"]
        cases.append(malformed_diagnostic)
        overflowing_scalar = pbpf_result()
        overflowing_scalar["rows"][0]["nll"] = 10**10_000
        cases.append(overflowing_scalar)
        overflowing_array = pbpf_result()
        overflowing_array["gate_inputs"]["prefix_4"]["paired_p16_to_exact_inputs"] = [10**10_000]
        cases.append(overflowing_array)
        non_utf8 = pbpf_result()
        non_utf8["rows"][0]["arm"] = "\ud800"
        cases.append(non_utf8)
        non_utf8_key = pbpf_result()
        non_utf8_key["gate_inputs"]["\ud800"] = 1
        cases.append(non_utf8_key)
        empty = pbpf_result()
        empty["rows"] = []
        cases.append(empty)
        for case_index, raw in enumerate(cases):
            with self.subTest(case_index=case_index):
                with self.assertRaises(ResultNormalizationError):
                    normalize_result("pbpf", "predictive_belief_particle_filter", PBPF_CONFIG, 101, expected_run_id(PBPF_CONFIG), raw)

    def test_goav_normalizes_rows_design_arrays_and_native_event_sequence(self) -> None:
        normalized = normalize_result(
            "goav", "gradient_optimal_active_verification", GOAV_CONFIG, 101, expected_run_id(GOAV_CONFIG), goav_result()
        )
        self.assertEqual(normalized.rows[0]["record_type"], "arm_summary")
        self.assertEqual([event["sequence"] for event in normalized.events], [0, 1])
        self.assertEqual(normalized.events[0]["event"], "design_logged")
        self.assertEqual(normalized.events[1]["event"], "audit_request")
        self.assertEqual(normalized.metrics["design_event_count"], 1)
        self.assertEqual(normalized.metrics["audit_event_count"], 1)
        probability_keys = [key for key in normalized.arrays if key.endswith("_probabilities")]
        self.assertEqual(len(probability_keys), 1)
        self.assertEqual(normalized.arrays[probability_keys[0]].shape, (4,))
        self.assertEqual(normalized.arrays["goav_row_numeric"].shape, (1, 12))

    def test_goav_rejects_audit_before_matching_design(self) -> None:
        raw = goav_result()
        raw["events"] = list(reversed(raw["events"]))
        with self.assertRaisesRegex(ResultNormalizationError, "preceding design"):
            normalize_result("goav", "gradient_optimal_active_verification", GOAV_CONFIG, 101, expected_run_id(GOAV_CONFIG), raw)

    def test_goav_calls_full_semantic_validator_on_normalized_events(self) -> None:
        mutations = []
        wrong_hash = goav_result()
        wrong_hash["events"][0]["design_hash"] = wrong_hash["events"][1]["design_hash"] = "d" * 64
        mutations.append(wrong_hash)
        wrong_pi = goav_result()
        wrong_pi["events"][0]["pi"] = [0.4, 0.5]
        wrong_pi["events"][0]["pi_hash"] = array_hash([0.4, 0.5])
        wrong_pi["events"][1]["inclusion_probabilities"] = [0.4, 0.5]
        mutations.append(wrong_pi)
        wrong_subset = goav_result()
        wrong_subset["events"][1]["subset_index"] = 2
        mutations.append(wrong_subset)
        no_events = goav_result()
        no_events["events"] = []
        mutations.append(no_events)
        no_rows = goav_result()
        no_rows["rows"] = []
        mutations.append(no_rows)
        for raw in mutations:
            with self.subTest(raw=raw):
                with self.assertRaises(ResultNormalizationError):
                    normalize_result(
                        "goav",
                        "gradient_optimal_active_verification",
                        GOAV_CONFIG,
                        101,
                        expected_run_id(GOAV_CONFIG),
                        raw,
                    )

    def test_smoke_does_not_hide_malformed_source_gate_metadata(self) -> None:
        extras = (
            ("reason", {"not": "a reason"}),
            ("reason", "unregistered custom reason"),
            ("inputs", {}),
        )
        for field, value in extras:
            with self.subTest(field=field, value=value):
                raw = goav_result()
                raw["gates"][field] = value
                with self.assertRaises(ResultNormalizationError):
                    normalize_result(
                        "goav",
                        "gradient_optimal_active_verification",
                        GOAV_SMOKE_CONFIG,
                        101,
                        expected_run_id(GOAV_SMOKE_CONFIG),
                        raw,
                    )

    def test_jag_flattens_families_and_keeps_unavailable_values_as_null(self) -> None:
        normalized = normalize_result("jag", "jag_tree", JAG_CONFIG, 101, expected_run_id(JAG_CONFIG), jag_result())
        self.assertEqual(normalized.metrics["family_order"], ["root_only"])
        self.assertEqual(len(normalized.rows), 2)
        self.assertIsNone(normalized.rows[0]["bias_z_score"])
        self.assertEqual(normalized.rows[0]["bias_z_score_status"], "UNAVAILABLE")
        self.assertEqual(normalized.rows[0]["status_reason"], "horizon exceeds exact oracle scope")
        self.assertEqual(normalized.rows[1]["allocation"]["total_edge_samples"], 128)
        self.assertEqual(normalized.rows[1]["paired_replicates"][0]["replication"], 0)
        self.assertEqual(normalized.gates["inputs"]["relative_bias"][1]["value"], 0.02)
        self.assertEqual(normalized.gates["evidence"]["active_seed"], 101)
        self.assertEqual(normalized.metrics["estimator_policy"]["branchable_depths"], [0, 1])
        self.assertEqual(normalized.metrics["oracle_scope"]["max_states"], 100000)
        self.assertEqual(
            normalized.metrics["family_metadata"]["root_only"]["covariance_audit"]["sample_count"],
            32,
        )
        self.assertEqual(
            normalized.metrics["family_metadata"]["root_only"]["predictor_provenance"]["evaluation_seed"],
            101,
        )
        gradient_keys = [key for key in normalized.arrays if key.startswith("jag_exact_gradient_")]
        self.assertEqual(len(gradient_keys), 1)
        self.assertEqual(normalized.arrays[gradient_keys[0]].shape, (2,))
        self.assertEqual(normalized.arrays[gradient_keys[0]].dtype, np.dtype("float64"))
        self.assertNotEqual(normalized.arrays[gradient_keys[0]].dtype.kind, "O")
        np.testing.assert_array_equal(normalized.arrays[gradient_keys[0]], [0.1, -0.1])
        covariance_metadata = normalized.metrics["family_metadata"]["root_only"][
            "covariance_audit"
        ]
        predicted_key = covariance_metadata["predicted_covariance_array_key"]
        realized_key = covariance_metadata["realized_covariance_array_key"]
        self.assertNotIn("predicted_covariance", covariance_metadata)
        self.assertNotIn("realized_covariance", covariance_metadata)
        self.assertEqual(covariance_metadata["joint_dimension"], 3)
        self.assertEqual(covariance_metadata["covariance_dtype"], "<f8")
        self.assertEqual(normalized.arrays[predicted_key].shape, (3, 3))
        self.assertEqual(normalized.arrays[realized_key].shape, (3, 3))
        self.assertEqual(normalized.arrays[predicted_key].dtype, np.dtype("<f8"))
        self.assertEqual(normalized.arrays[realized_key].dtype, np.dtype("<f8"))
        self.assertEqual(
            array_hash(normalized.arrays[predicted_key].tolist()),
            covariance_metadata["predicted_covariance_hash"],
        )
        self.assertEqual(
            array_hash(normalized.arrays[realized_key].tolist()),
            covariance_metadata["realized_covariance_hash"],
        )
        self.assertEqual(
            normalized.gates["inputs"]["held_out_covariance_audit"][0][
                "predicted_covariance"
            ],
            jag_result()["families"]["root_only"]["covariance_audit"][
                "predicted_covariance"
            ],
        )
        self.assertIn("allocation_evidence", normalized.metrics)

    def test_jag_adapter_accepts_the_current_runner_contract(self) -> None:
        config = {
            "direction": "jag_tree",
            "runtime": {"profile": "smoke"},
            "seeds": [12],
            "output": {"identity": "jag-actual-test"},
            "phase0": {
                "horizon": 2,
                "actions": 2,
                "reward_families": ["root_only"],
                "arms": ["flat_iid", "uniform_tree"],
                "budgets": [4],
                "rollout_replications": 2,
                "covariance_audit_replications": 4,
            },
        }
        raw = run_jag_phase0(config, seed=12)
        normalized = normalize_result(
            "jag", "jag_tree", config, 12, expected_run_id(config, seed=12), raw
        )
        covariance = normalized.metrics["family_metadata"]["root_only"]["covariance_audit"]
        self.assertEqual(normalized.metrics["status"], "INCOMPLETE")
        self.assertEqual(len(normalized.rows), 2)
        self.assertEqual(len(normalized.arrays), 3)
        self.assertEqual(
            normalized.arrays[covariance["predicted_covariance_array_key"]].shape,
            (covariance["joint_dimension"], covariance["joint_dimension"]),
        )

    def test_jag_rejects_incoherent_covariance_artifacts(self) -> None:
        cases = []
        wrong_shape = jag_result()
        wrong_shape["families"]["root_only"]["covariance_audit"][
            "predicted_covariance"
        ] = [[0.3, 0.0], [0.0, 0.2]]
        cases.append(wrong_shape)
        wrong_hash = jag_result()
        wrong_hash["families"]["root_only"]["covariance_audit"][
            "predicted_covariance_hash"
        ] = "0" * 64
        cases.append(wrong_hash)
        wrong_numerator = jag_result()
        wrong_numerator["families"]["root_only"]["covariance_audit"]["numerator"] += 1.0
        cases.append(wrong_numerator)
        wrong_dtype = jag_result()
        wrong_dtype["families"]["root_only"]["covariance_audit"][
            "covariance_dtype"
        ] = "float64"
        cases.append(wrong_dtype)
        gate_disagrees = jag_result()
        gate_disagrees["gate"]["inputs"]["held_out_covariance_audit"][0] = copy.deepcopy(
            gate_disagrees["gate"]["inputs"]["held_out_covariance_audit"][0]
        )
        gate_disagrees["gate"]["inputs"]["held_out_covariance_audit"][0][
            "realized_joint_mean"
        ][0] += 0.1
        cases.append(gate_disagrees)
        for case_index, raw in enumerate(cases):
            with self.subTest(case_index=case_index):
                with self.assertRaises(ResultNormalizationError):
                    normalize_result(
                        "jag", "jag_tree", JAG_CONFIG, 101, expected_run_id(JAG_CONFIG), raw
                    )

    def test_jag_rejects_unversioned_extras_and_infinity(self) -> None:
        raw = jag_result()
        raw["mystery"] = {"source": "future"}
        with self.assertRaises(ResultNormalizationError):
            normalize_result("jag", "jag_tree", JAG_CONFIG, 101, expected_run_id(JAG_CONFIG), raw)
        raw = jag_result()
        raw["runner_provenance_v1"] = {"source": "future"}
        with self.assertRaises(ResultNormalizationError):
            normalize_result("jag", "jag_tree", JAG_CONFIG, 101, expected_run_id(JAG_CONFIG), raw)
        raw = jag_result()
        raw["families"]["root_only"]["calibration_provenance_v1"] = {"source": "future"}
        with self.assertRaises(ResultNormalizationError):
            normalize_result("jag", "jag_tree", JAG_CONFIG, 101, expected_run_id(JAG_CONFIG), raw)
        raw = jag_result()
        raw["gate"]["formal_evidence"] = False
        with self.assertRaises(ResultNormalizationError):
            normalize_result("jag", "jag_tree", JAG_CONFIG, 101, expected_run_id(JAG_CONFIG), raw)
        raw = jag_result()
        raw["families"]["root_only"]["summaries"][1]["bias_z_score"] = float("inf")
        with self.assertRaises(ResultNormalizationError):
            normalize_result("jag", "jag_tree", JAG_CONFIG, 101, expected_run_id(JAG_CONFIG), raw)

    def test_jag_rejects_unknown_nested_schema_and_incoherent_evidence(self) -> None:
        mutations = []
        extra_policy = jag_result()
        extra_policy["estimator_policy"]["future_policy"] = "guess"
        mutations.append(extra_policy)
        extra_covariance = jag_result()
        extra_covariance["families"]["root_only"]["covariance_audit"]["matrix"] = [[1.0]]
        mutations.append(extra_covariance)
        extra_predictor = jag_result()
        extra_predictor["families"]["root_only"]["predictor_provenance"]["future_hash"] = "f" * 64
        mutations.append(extra_predictor)
        extra_pair = jag_result()
        extra_pair["families"]["root_only"]["summaries"][1]["paired_replicates"][0]["loss"] = 1.0
        mutations.append(extra_pair)
        missing_pair = jag_result()
        missing_pair["families"]["root_only"]["summaries"][1]["paired_replicates"].pop()
        mutations.append(missing_pair)
        extra_metadata = jag_result()
        extra_metadata["families"]["root_only"]["summaries"][1]["arm_metadata"]["future"] = False
        mutations.append(extra_metadata)
        extra_gate_input = jag_result()
        extra_gate_input["gate"]["inputs"]["coverage_90"] = []
        mutations.append(extra_gate_input)
        extra_gate_evidence = jag_result()
        extra_gate_evidence["gate"]["evidence"]["registered_scale"] = False
        mutations.append(extra_gate_evidence)
        for case_index, raw in enumerate(mutations):
            with self.subTest(case_index=case_index):
                with self.assertRaises(ResultNormalizationError):
                    normalize_result(
                        "jag", "jag_tree", JAG_CONFIG, 101, expected_run_id(JAG_CONFIG), raw
                    )

    def test_numeric_arrays_reject_boolean_and_string_coercion(self) -> None:
        pbpf = pbpf_result()
        pbpf["gate_inputs"]["prefix_4"]["paired_p16_to_exact_inputs"] = [True]
        jag = jag_result()
        jag["families"]["root_only"]["exact"]["gradient"] = ["0.1", -0.1]
        cases = [
            (
                "pbpf",
                "predictive_belief_particle_filter",
                PBPF_CONFIG,
                expected_run_id(PBPF_CONFIG),
                pbpf,
            ),
            ("jag", "jag_tree", JAG_CONFIG, expected_run_id(JAG_CONFIG), jag),
        ]
        for alias, direction, config, run_id, raw in cases:
            with self.subTest(alias=alias):
                with self.assertRaises(ResultNormalizationError):
                    normalize_result(alias, direction, config, 101, run_id, raw)

    def test_generated_array_keys_must_fit_the_sealed_artifact_contract(self) -> None:
        raw = jag_result()
        family = "x" * 200
        raw["families"] = {family: raw["families"].pop("root_only")}
        raw["families"][family]["structural_audit"]["family"] = family
        with self.assertRaises(ResultNormalizationError):
            normalize_result("jag", "jag_tree", JAG_CONFIG, 101, expected_run_id(JAG_CONFIG), raw)

    def test_direction_pairs_are_exact_and_run_id_profile_are_validated(self) -> None:
        wrong_direction_config = copy.deepcopy(JAG_CONFIG)
        wrong_direction_config["direction"] = "gradient_optimal_active_verification"
        missing_runtime_config = copy.deepcopy(JAG_CONFIG)
        missing_runtime_config["runtime"] = {}
        unsafe_identity_config = copy.deepcopy(JAG_CONFIG)
        unsafe_identity_config["output"]["identity"] = "../escaped"
        boolean_seed_config = copy.deepcopy(JAG_CONFIG)
        boolean_seed_config["seeds"] = [True]
        float_seed_config = copy.deepcopy(JAG_CONFIG)
        float_seed_config["seeds"] = [101.0]
        negative_seed_config = copy.deepcopy(JAG_CONFIG)
        negative_seed_config["seeds"] = [-1]
        duplicate_seed_config = copy.deepcopy(JAG_CONFIG)
        duplicate_seed_config["seeds"] = [101, 101]
        long_identity_config = copy.deepcopy(JAG_CONFIG)
        long_identity_config["output"]["identity"] = "x" * 176
        invalid_calls = [
            ("jag", "gradient_optimal_active_verification", JAG_CONFIG, 101, expected_run_id(JAG_CONFIG), jag_result()),
            ("unknown", "jag_tree", JAG_CONFIG, 101, expected_run_id(JAG_CONFIG), jag_result()),
            ("jag", "jag_tree", missing_runtime_config, 101, "unused", jag_result()),
            ("jag", "jag_tree", wrong_direction_config, 101, "unused", jag_result()),
            ("jag", "jag_tree", JAG_CONFIG, True, expected_run_id(JAG_CONFIG), jag_result()),
            ("jag", "jag_tree", JAG_CONFIG, -1, "unused", jag_result()),
            ("jag", "jag_tree", JAG_CONFIG, 202, expected_run_id(JAG_CONFIG, 202), jag_result()),
            ("jag", "jag_tree", unsafe_identity_config, 101, "unused", jag_result()),
            ("jag", "jag_tree", boolean_seed_config, 101, "unused", jag_result()),
            ("jag", "jag_tree", float_seed_config, 101, "unused", jag_result()),
            ("jag", "jag_tree", negative_seed_config, 101, "unused", jag_result()),
            ("jag", "jag_tree", duplicate_seed_config, 101, "unused", jag_result()),
            ("jag", "jag_tree", long_identity_config, 101, "unused", jag_result()),
            ("jag", "jag_tree", JAG_CONFIG, 101, "wrong-run-id", jag_result()),
            ("jag", "jag_tree", JAG_CONFIG, 101, "", jag_result()),
        ]
        for arguments in invalid_calls:
            with self.subTest(arguments=arguments[:5]):
                with self.assertRaises(ResultNormalizationError):
                    normalize_result(*arguments)

        boundary_config = copy.deepcopy(JAG_CONFIG)
        boundary_config["output"]["identity"] = "x" * 175
        boundary_run_id = expected_run_id(boundary_config)
        self.assertEqual(len(boundary_run_id), 192)
        normalized = normalize_result(
            "jag", "jag_tree", boundary_config, 101, boundary_run_id, jag_result()
        )
        self.assertEqual(normalized.rows[0]["run_id"], boundary_run_id)

    def test_normalization_is_deterministic_json_safe_and_does_not_alias_input(self) -> None:
        raw = goav_result()
        untouched = copy.deepcopy(raw)
        first = normalize_result("goav", "gradient_optimal_active_verification", GOAV_CONFIG, 101, expected_run_id(GOAV_CONFIG), raw)
        second = normalize_result("goav", "gradient_optimal_active_verification", GOAV_CONFIG, 101, expected_run_id(GOAV_CONFIG), raw)
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.gates, second.gates)
        self.assertEqual(first.rows, second.rows)
        self.assertEqual(first.events, second.events)
        self.assertEqual(set(first.arrays), set(second.arrays))
        for key in first.arrays:
            np.testing.assert_array_equal(first.arrays[key], second.arrays[key])
            self.assertNotEqual(first.arrays[key].dtype.kind, "O")
            self.assertTrue(np.isfinite(first.arrays[key]).all())
        for payload in (first.metrics, first.gates, first.rows, first.events):
            json.dumps(payload, allow_nan=False)
        raw["rows"][0]["nMSE"] = 999.0
        raw["events"][0]["probabilities"][0] = 999.0
        self.assertEqual(first.rows[0]["nMSE"], untouched["rows"][0]["nMSE"])
        self.assertEqual(first.events[0]["probabilities"], untouched["events"][0]["probabilities"])


if __name__ == "__main__":
    unittest.main()
