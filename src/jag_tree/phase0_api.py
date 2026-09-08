"""Verified finite-MDP JAG operators retained as the Phase-0 oracle."""

from .allocator import AllocationItem, FrontierItem, FrontierPlan, JointMoments, OracleScope, OracleUnavailableError, exact_frontier_plan, greedy_allocate, greedy_frontier_plan, integer_oracle, joint_risk, joint_risk_no_cross, marginal_score, transport_weight
from .estimator import TreeNode, leaf_equal_estimate, recursive_estimate
from .mdp import ExactTarget, FiniteMDP, exact_target, make_phase0_mdp, sample_balanced_tree, sample_chain, structural_audit
from .phase0 import CalibrationRecord, build_calibration_record, fit_calibration_record, run_jag_phase0

__all__ = [
    "AllocationItem", "CalibrationRecord", "ExactTarget", "FiniteMDP", "FrontierItem", "FrontierPlan", "JointMoments", "OracleScope", "OracleUnavailableError", "TreeNode",
    "build_calibration_record", "exact_frontier_plan", "exact_target", "fit_calibration_record", "greedy_allocate", "greedy_frontier_plan", "integer_oracle", "joint_risk",
    "joint_risk_no_cross", "leaf_equal_estimate", "marginal_score", "recursive_estimate", "make_phase0_mdp", "run_jag_phase0", "sample_balanced_tree", "sample_chain", "structural_audit", "transport_weight",
]
