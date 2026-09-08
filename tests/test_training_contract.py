from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from jag_tree.artifacts import ArtifactError, verify_artifacts, write_artifacts
from jag_tree.ledger import RunLedger
from jag_tree.loss import jag_loss
from jag_tree.cli import main
from jag_tree.config import load_experiment
from jag_tree.rollout import FakePolicyBackend
from jag_tree.sandbox import FakeSandbox
from jag_tree.schema import TaskRecord, TreeNode
from jag_tree.bank import TreeBank
from jag_tree.trainer import TrainingBatch, recursive_training_targets, replay_edge_coefficients, run_experiment, train_one_update


def test_jag_loss_matches_explicit_unique_edge_reinforce_gradient() -> None:
    torch = pytest.importorskip("torch")
    logprobs = torch.tensor([0.2, -0.4, 0.7], requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.bool)
    credit = torch.tensor([2.0, -1.0])
    weights = torch.tensor([0.5, 2.0])
    loss = jag_loss(logprobs, mask, credit, weights)
    loss.backward()
    np.testing.assert_allclose(logprobs.grad.numpy(), [-1.0, -1.0, 2.0])


def test_jag_loss_rejects_mismatched_masks_and_repeated_edges() -> None:
    with pytest.raises(ValueError, match="shape"):
        jag_loss(np.zeros(2), np.ones((1, 3)), np.ones(1), np.ones(1))
    with pytest.raises(ValueError, match="repeated"):
        jag_loss(np.zeros(2), np.ones((2, 2)), np.ones(2), np.ones(2))


def test_one_update_rejects_extra_epochs_and_budget_drift() -> None:
    batch = TrainingBatch(np.zeros(2), np.eye(2, dtype=bool), np.ones(2), np.ones(2))
    with pytest.raises(ValueError, match="exactly one"):
        train_one_update(None, None, batch, optimizer_epochs=2)
    expected = RunLedger(generation_tokens=100, programs=10, verification_cpu_seconds=5, optimizer_steps=1)
    drifted = RunLedger(generation_tokens=103, programs=10, verification_cpu_seconds=5, optimizer_steps=1)
    with pytest.raises(ValueError, match="budget drift"):
        train_one_update(None, None, batch, expected_ledger=expected, actual_ledger=drifted)


def test_recursive_training_targets_match_uneven_tree_global_reinforce_denominator(tmp_path: Path) -> None:
    tasks = [
        TaskRecord("a", "A", "", ("assert True",), "fixture", "a", "train"),
        TaskRecord("b", "B", "", ("assert True",), "fixture", "b", "train"),
    ]
    nodes = [
        TreeNode("ar", "a", None, 0, "prompt-a", (), 0, "policy", "template"),
        TreeNode("a1", "a", "ar", 1, "A", (1,), 1, "policy", "template", reward=1.0),
        TreeNode("br", "b", None, 0, "prompt-b", (), 0, "policy", "template"),
        TreeNode("bm", "b", "br", 1, "prefix", (2,), 1, "policy", "template"),
        TreeNode("b0", "b", "bm", 2, "B0", (3,), 1, "policy", "template", reward=0.0),
        TreeNode("b1", "b", "bm", 2, "B1", (4,), 1, "policy", "template", reward=1.0),
    ]
    scores = {"a1": np.array([1.0]), "bm": np.array([4.0]), "b0": np.array([2.0]), "b1": np.array([3.0])}
    bank = TreeBank.create(tmp_path / "bank", tasks, nodes, generation_identity="generation", score_arrays=scores)
    targets = recursive_training_targets(bank, baseline="independent")
    assert targets.edge_ids == ("a1", "bm", "b0", "b1")
    np.testing.assert_allclose(targets.credit, [1.0, 0.5, 0.0, 1.0])
    np.testing.assert_allclose(targets.weights, [0.5, 0.5, 0.25, 0.25])
    assert float(sum(scores[node][0] * credit * weight for node, credit, weight in zip(targets.edge_ids, targets.credit, targets.weights))) == pytest.approx(2.25)
    assert targets.edge_mask.sum(axis=0).tolist() == [1, 1, 1, 1]


def test_recursive_training_targets_support_leave_one_out_without_leaf_equal_credit(tmp_path: Path) -> None:
    task = TaskRecord("a", "A", "", ("assert True",), "fixture", "a", "train")
    nodes = [
        TreeNode("r", "a", None, 0, "prompt", (), 0, "policy", "template"),
        TreeNode("x", "a", "r", 1, "X", (1,), 1, "policy", "template", reward=0.0),
        TreeNode("y", "a", "r", 1, "Y", (2,), 1, "policy", "template", reward=1.0),
        TreeNode("z", "a", "r", 1, "Z", (3,), 1, "policy", "template", reward=3.0),
    ]
    bank = TreeBank.create(tmp_path / "loo", [task], nodes, generation_identity="generation", score_arrays={node.node_id: np.ones(1) for node in nodes[1:]})
    targets = recursive_training_targets(bank, baseline="loo")
    np.testing.assert_allclose(targets.credit, [-2.0, -0.5, 2.5])
    np.testing.assert_allclose(targets.weights, [1 / 3, 1 / 3, 1 / 3])


def test_online_coefficients_preserve_replay_multiplicity_and_importance(tmp_path: Path) -> None:
    task = TaskRecord("a", "A", "", ("assert True",), "fixture", "a", "train")
    nodes = [
        TreeNode("r", "a", None, 0, "prompt", (), 0, "policy", "template"),
        TreeNode("x", "a", "r", 1, "X", (1,), 1, "policy", "template", reward=1.0),
        TreeNode("y", "a", "r", 1, "Y", (2,), 1, "policy", "template", reward=1.0),
    ]
    bank = TreeBank.create(tmp_path / "replay", [task], nodes, generation_identity="generation", score_arrays={"x": np.ones(1), "y": np.ones(1)})
    rows = (
        {"leaf_id": "x", "path_edge_ids": ("x",), "importance_weight": 0.5},
        {"leaf_id": "x", "path_edge_ids": ("x",), "importance_weight": 0.5},
        {"leaf_id": "y", "path_edge_ids": ("y",), "importance_weight": 1.5},
        {"leaf_id": "x", "path_edge_ids": ("x",), "importance_weight": 1.5},
    )
    coefficients = replay_edge_coefficients(bank, rows)
    assert coefficients == pytest.approx({"x": 0.625, "y": 0.375})
    assert sum(coefficients.values()) == pytest.approx(1.0)


def test_artifacts_are_create_once_and_tamper_evident(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_artifacts(run, {"status": "INCOMPLETE", "seed": 7}, rows=[{"task_id": "a", "reward": 1.0}])
    assert verify_artifacts(run)["status"] == "INCOMPLETE"
    with pytest.raises(FileExistsError):
        write_artifacts(run, {"status": "INCOMPLETE"}, rows=[])
    (run / "result.json").write_text('{"status":"PASS"}\n', encoding="utf-8")
    with pytest.raises(ArtifactError, match="checksum"):
        verify_artifacts(run)


def test_pass_artifacts_fail_closed_without_complete_gates_and_exact_inventory(tmp_path: Path) -> None:
    incomplete = tmp_path / "false-pass"
    write_artifacts(incomplete, {"status": "PASS"}, rows=[])
    with pytest.raises(ArtifactError, match="unsupported"):
        verify_artifacts(incomplete)
    valid = tmp_path / "valid"
    evidence_rows = [{"gate": gate, "statistic": "score", "value": 1.0, "threshold": 1.0, "comparison": ">=", "passed": True} for gate in ("e0", "e1", "statistics")]
    evidence_text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in evidence_rows)
    task = TaskRecord("pass", "P", "", ("assert True",), "fixture", "pass", "frozen_audit")
    bank = TreeBank.create(tmp_path / "pass-bank", [task], [TreeNode("root", "pass", None, 0, "prompt", (), 0, "policy", "template")], generation_identity="generation")
    config_payload = {"experiment": "registered"}
    config_digest = sha256(json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    bank_digest = sha256((bank.path / "manifest.json").read_bytes()).hexdigest()
    write_artifacts(valid, {"status": "PASS", "evidence_complete": True, "gates": {"e0": "PASS", "e1": "PASS", "statistics": "PASS"}, "infrastructure_failures": 0}, rows=evidence_rows, manifest={"config": config_payload, "config_sha256": config_digest, "bank_path": str(bank.path), "bank_sha256": bank_digest, "rows_sha256": sha256(evidence_text.encode()).hexdigest()})
    with pytest.raises(ArtifactError, match="unsupported"):
        verify_artifacts(valid)
    checksums = json.loads((valid / "checksums.json").read_text())
    checksums["untracked.txt"] = "0" * 64
    (valid / "checksums.json").write_text(json.dumps(checksums))
    with pytest.raises(ArtifactError, match="inventory"):
        verify_artifacts(valid)


def test_dry_run_stops_before_backends_and_fake_smoke_writes_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"task_id": "a", "statement": "A", "starter_code": "", "tests": ["assert True"], "source": "fixture", "lineage_group": "a", "role": "frozen_audit"}) + "\n", encoding="utf-8")
    revision = "d18ecad881a290f0c792f1c2b5dbee1820d4a424"
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(f"model: {{name: qwen25_coder_1_5b, revision: {revision}}}\nformal: false\nhardware: cpu\ndatasets: []\nallocator: jag_full\nstage: frozen\ntasks: {{kind: jsonl, path: {tasks}, role: frozen_audit}}\narms: [uniform, jag_full]\nbudget: 0\n", encoding="utf-8")
    config = load_experiment(config_path)
    dry = run_experiment(config, output_root=tmp_path / "dry", seed=7, dry_run=True)
    assert dry["dry_run"] is True
    assert main(["run", str(config_path), "--output-root", str(tmp_path / "unused"), "--seed", "7", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    result = run_experiment(config, output_root=tmp_path / "real", seed=7, backend=FakePolicyBackend(revision), sandbox=FakeSandbox())
    assert verify_artifacts(Path(str(result["run_path"]))) ["status"] == "INCOMPLETE"

    online_path = tmp_path / "online.yaml"
    online_path.write_text(config_path.read_text().replace("stage: frozen", "stage: online").replace("budget: 0", "budget: 4"), encoding="utf-8")
    with pytest.raises(ValueError, match="training-capable"):
        run_experiment(load_experiment(online_path), output_root=tmp_path / "online", seed=7, backend=FakePolicyBackend(revision), sandbox=FakeSandbox())

    class TrainingFake(FakePolicyBackend):
        def __init__(self, revision: str) -> None:
            super().__init__(revision)
            self.training_calls: list[Path] = []

        def train_bank_one_update(self, bank_path: Path, *, replay_rows, expected_ledger) -> float:  # type: ignore[no-untyped-def]
            assert replay_rows and all(row["path_edge_ids"] for row in replay_rows)
            assert expected_ledger.generation_tokens <= 64
            self.training_calls.append(bank_path)
            return 0.25

    online_path.write_text(online_path.read_text() + "training: {formal_updates: 2}\n", encoding="utf-8")
    training_backend = TrainingFake(revision)
    online = run_experiment(load_experiment(online_path), output_root=tmp_path / "online-real", seed=7, backend=training_backend, sandbox=FakeSandbox())
    assert online["updates_completed"] == 2
    assert len(training_backend.training_calls) == 2
    assert len(set(training_backend.training_calls)) == 2
