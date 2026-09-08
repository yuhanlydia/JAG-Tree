from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from jag_tree.allocator import AllocationNode, allocate
from jag_tree.audit import FrozenMomentPredictor, MomentPrediction, audit_replay
from jag_tree.bank import TreeBank
from jag_tree.baselines import arm_risk
from jag_tree.estimator import TreeNode, recursive_credit
from jag_tree.gradient import GradientSpec, ScoreExample, score_sketch
from jag_tree.phase0_api import JointMoments
from jag_tree.schema import TaskRecord, TreeNode as BankNode


def _moments(cross: float = 0.5) -> JointMoments:
    return JointMoments(0.0, np.zeros(2), 4.0, np.diag([3.0, 2.0]), np.array([cross, 0.0]))


def test_recursive_credit_counts_each_edge_once() -> None:
    leaf = TreeNode(np.array([2.0]), reward=3.0)
    middle = TreeNode(np.array([5.0]), children=[leaf])
    root = TreeNode(np.zeros(1), children=[middle])
    value, gradient = recursive_credit(root, baseline=1.0)
    assert value == 3.0
    np.testing.assert_allclose(gradient, [(3 - 1) * 5 + (3 - 1) * 2])


def test_literal_arm_scores_include_joint_cross_covariance() -> None:
    node = AllocationNode("n", branching=1, cost=2.0, ancestor_score=np.array([2.0, 0.0]), entropy=0.25, trace_score=7.0)
    moments = _moments()
    assert arm_risk(node, moments, "uniform") == 1.0
    assert arm_risk(node, moments, "entropy") == 0.25
    assert arm_risk(node, moments, "value_variance") == 4.0
    assert arm_risk(node, moments, "gradient_only") == 5.0
    assert arm_risk(node, moments, "trace_score") == 7.0
    assert arm_risk(node, moments, "jag_no_cross") == 21.0
    assert arm_risk(node, moments, "jag_full") == 23.0
    assert arm_risk(node, moments, "oracle_moments") == 23.0


@pytest.mark.parametrize("arm", ["uniform", "entropy", "value_variance", "gradient_only", "trace_score", "jag_no_cross", "jag_full", "oracle_moments"])
def test_allocate_is_permutation_invariant_for_every_arm(arm: str) -> None:
    nodes = [AllocationNode("b", 1, 1.0, np.array([0.0, 0.0]), 0.5, 2.0), AllocationNode("a", 1, 1.0, np.array([1.0, 0.0]), 1.0, 3.0)]
    moments = {"a": _moments(), "b": _moments(-0.5)}
    assert allocate(nodes, moments, budget=1, arm=arm) == allocate(reversed(nodes), moments, budget=1, arm=arm)


def test_allocate_rejects_outcome_leak_and_unknown_arm() -> None:
    leaked = {"node_id": "n", "branching": 1, "cost": 1.0, "ancestor_score": [0.0], "entropy": 1.0, "trace_score": 1.0, "outcome": "pass"}
    with pytest.raises(ValueError, match="outcome"):
        allocate([leaked], {"n": JointMoments(0.0, np.zeros(1), 1.0, np.eye(1), np.zeros(1))}, 1, "jag_full")
    with pytest.raises(ValueError, match="unknown"):
        allocate([AllocationNode("n", 1, 1.0, np.zeros(1))], {"n": JointMoments(0.0, np.zeros(1), 1.0, np.eye(1), np.zeros(1))}, 1, "future_arm")


def test_tiny_torch_score_sketch_matches_registered_direct_gradient() -> None:
    torch = pytest.importorskip("torch")
    class TinyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(3))

        def forward(self, input_ids):  # type: ignore[no-untyped-def]
            return self.logits.reshape(1, 1, 3).expand(input_ids.shape[0], input_ids.shape[1], 3)

    model = TinyPolicy()
    example = ScoreExample({"input_ids": torch.tensor([[2, 0]])}, edge_start=1, edge_length=1)
    result = score_sketch(model, [example], GradientSpec(("logits",), dimension=2, seed=3, keep_exact=True))
    np.testing.assert_allclose(result.exact, [[2 / 3, -1 / 3, -1 / 3]], rtol=1e-6, atol=1e-6)
    assert result.sketch.shape == (1, 2)
    np.testing.assert_allclose(result.sketch, result.exact @ result.projection)


def _audit_bank(tmp_path, rewards=(1.0, 0.0, 1.0, 0.0)) -> TreeBank:  # type: ignore[no-untyped-def]
    task = TaskRecord("audit", "Solve", "", ("assert True",), "fixture", "audit", "frozen_audit")
    root = BankNode("root", "audit", None, 0, "prompt", (), 0, "policy", "template")
    leaves = [BankNode(f"l{i}", "audit", "root", 1, f"code-{i}", (i, i + 1), 2, "policy", "template", reward=reward, edge_logprob=-i, edge_entropy=float(i + 1)) for i, reward in enumerate(rewards)]
    scores = {"l0": np.array([2.0, 0.0]), "l1": np.array([-2.0, 0.0]), "l2": np.array([0.0, 3.0]), "l3": np.array([0.0, -3.0])}
    return TreeBank.create(tmp_path, [task], [root, *leaves], generation_identity="generation", score_arrays=scores)


def _predictor() -> FrozenMomentPredictor:
    return FrozenMomentPredictor("calibration-sha", {
        "l0": MomentPrediction(9.0, 1.0, np.array([-3.0, 0.0]), 1.0, 2.0),
        "l1": MomentPrediction(1.0, 8.0, np.array([-10.0, 0.0]), 2.0, 1.0),
        "l2": MomentPrediction(4.0, 4.0, np.array([0.0, -2.0]), 3.0, 9.0),
        "l3": MomentPrediction(2.0, 2.0, np.array([0.0, -10.0]), 4.0, 3.0),
    })


def test_frozen_predictor_covariance_cannot_reenable_writes() -> None:
    covariance = _predictor().predict("l0").value_gradient_covariance
    with pytest.raises(ValueError):
        covariance.setflags(write=True)


def test_frozen_allocation_is_reward_invariant_and_never_exceeds_available_edges(tmp_path) -> None:  # type: ignore[no-untyped-def]
    first = audit_replay(_audit_bank(tmp_path / "first"), "value_variance", 7, budget=8, predictor=_predictor())
    flipped = audit_replay(_audit_bank(tmp_path / "flipped", rewards=(0.0, 1.0, 0.0, 1.0)), "value_variance", 7, budget=8, predictor=_predictor())
    assert first.selected_leaf_ids == flipped.selected_leaf_ids
    assert first.ledger.generation_tokens <= 8
    assert sum(first.allocation.values()) <= 4
    assert {"gradient_mse", "gradient_bias_norm", "gradient_cosine"} <= first.metrics.keys()


def test_replay_arms_use_distinct_frozen_scores_and_retain_outputs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bank = _audit_bank(tmp_path / "bank")
    predictor = _predictor()
    arms = ["entropy", "value_variance", "gradient_only", "trace_score", "jag_no_cross", "jag_full"]
    replays = {arm: audit_replay(bank, arm, 13, budget=4, predictor=predictor) for arm in arms}
    assert len({tuple((row["leaf_id"], round(float(row["proposal_probability"]), 6)) for row in replay.rows) for replay in replays.values()}) >= 4
    assert all(replay.rows and replay.ledger.generation_tokens <= 4 for replay in replays.values())
    assert all({"proposal_probability", "target_probability", "importance_weight", "rng_uniform"} <= replay.rows[0].keys() for replay in replays.values())


def test_stochastic_replay_varies_by_seed_and_converges_to_finite_target(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bank = _audit_bank(tmp_path / "mc")
    first = audit_replay(bank, "uniform", 1, budget=40_000, predictor=_predictor())
    second = audit_replay(bank, "uniform", 2, budget=40_000, predictor=_predictor())
    assert tuple(row["leaf_id"] for row in first.rows[:20]) != tuple(row["leaf_id"] for row in second.rows[:20])
    assert first.metrics["draw_count"] == 20_000
    assert first.metrics["gradient_bias_norm"] < 0.06
    assert first.metrics["gradient_mcse"] > 0
    assert first.metrics["gradient_mse"] == pytest.approx(first.metrics["per_draw_gradient_mse"] / 20_000)


def test_oracle_uses_contribution_magnitude_not_inverse_cost_when_draw_count_is_fixed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    task = TaskRecord("oracle", "Solve", "", ("assert True",), "fixture", "oracle", "frozen_audit")
    root = BankNode("r", "oracle", None, 0, "prompt", (), 0, "policy", "template")
    left = BankNode("a", "oracle", "r", 1, "A", (1,), 1, "policy", "template", reward=1.0)
    right = BankNode("b", "oracle", "r", 1, "B", (2, 3, 4, 5), 4, "policy", "template", reward=1.0)
    bank = TreeBank.create(tmp_path / "oracle", [task], [root, left, right], generation_identity="generation", score_arrays={"a": np.ones(1), "b": np.ones(1)})
    replay = audit_replay(bank, "oracle_moments", 9, budget=80)
    assert replay.metrics["per_draw_gradient_mse"] == pytest.approx(0.0)
    assert {float(row["proposal_probability"]) for row in replay.rows} == {0.5}
