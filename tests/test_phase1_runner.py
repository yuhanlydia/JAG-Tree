from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from jag_tree.benchmarks import BenchmarkSpec, load_tasks
from jag_tree.models import model_plan
from jag_tree.phase1 import run_frozen_audit
from jag_tree.rollout import FakePolicyBackend, GenerationRequest, build_genealogy
from jag_tree.sandbox import FakeSandbox, SandboxOutcome
from jag_tree.schema import TreeNode
from jag_tree.statistics import evaluation_summary, holm_adjust, paired_bootstrap


def _dataset(path: Path) -> None:
    rows = [
        {"task_id": "a", "statement": "A", "starter_code": "", "tests": ["assert True"], "source": "fixture", "lineage_group": "a", "role": "frozen_audit"},
        {"task_id": "b", "statement": "B", "starter_code": "", "tests": ["assert True"], "source": "fixture", "lineage_group": "b", "role": "frozen_audit"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_jsonl_loader_fails_closed_on_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    _dataset(path)
    assert [task.task_id for task in load_tasks(BenchmarkSpec("jsonl", "frozen_audit", path=path))] == ["a", "b"]
    path.write_text('{"task_id":"broken"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="required"):
        load_tasks(BenchmarkSpec("jsonl", "frozen_audit", path=path))


def test_frozen_audit_generates_one_common_bank_for_all_arms(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _dataset(tasks)
    backend = FakePolicyBackend("0123456789abcdef0123456789abcdef01234567")
    sandbox = FakeSandbox({("a", "candidate-a"): SandboxOutcome.PASS, ("b", "candidate-b"): SandboxOutcome.WRONG})
    config = {
        "tasks": {"kind": "jsonl", "path": str(tasks), "role": "frozen_audit"},
        "model": {"name": "qwen25_coder_7b", "revision": backend.revision},
        "output_root": str(tmp_path / "run"),
        "seed": 19,
        "budget": 2,
        "arms": ["uniform", "jag_full"],
    }
    result = run_frozen_audit(config, backend, sandbox)
    assert backend.calls == [("a", 19), ("b", 19)]
    assert result.model_revision == backend.revision
    assert result.arm_bank_sha256["uniform"] == result.arm_bank_sha256["jag_full"]
    assert set(result.arm_replays) == {"uniform", "jag_full"}
    assert all(replay.rows for replay in result.arm_replays.values())
    bank_scores = result.arm_replays["uniform"].rows
    assert bank_scores[0]["calibration_identity"].startswith("fake-calibration:")
    assert result.task_rows == ({"task_id": "a", "reward": 1.0}, {"task_id": "b", "reward": 0.0})
    rerun = run_frozen_audit(config, FakePolicyBackend(backend.revision), sandbox)
    assert rerun.bank_path == result.bank_path
    tasks.write_text(tasks.read_text().splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="task manifest"):
        run_frozen_audit(config, FakePolicyBackend(backend.revision), sandbox)
    _dataset(tasks)
    config["generation"] = {"root_samples": 2}
    with pytest.raises(ValueError, match="generation identity"):
        run_frozen_audit(config, FakePolicyBackend(backend.revision), sandbox)


def test_frozen_audit_executes_only_leaf_programs(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _dataset(tasks)

    class BranchedBackend(FakePolicyBackend):
        def generate_tree(self, task, seed, request=None):  # type: ignore[no-untyped-def]
            self.calls.append((task.task_id, seed))
            root = TreeNode(f"{task.task_id}:r", task.task_id, None, 0, "prompt", (), 0, self.revision, "v1")
            prefix = TreeNode(f"{task.task_id}:p", task.task_id, root.node_id, 1, "prefix", (1,), 1, self.revision, "v1")
            leaf = TreeNode(f"{task.task_id}:l", task.task_id, prefix.node_id, 2, f"candidate-{task.task_id}", (1, 2), 2, self.revision, "v1")
            return root, prefix, leaf

    revision = "0123456789abcdef0123456789abcdef01234567"
    result = run_frozen_audit({"tasks": {"kind": "jsonl", "path": str(tasks), "role": "frozen_audit"}, "model": {"name": "qwen25_coder_7b", "revision": revision}, "output_root": str(tmp_path / "branched"), "seed": 19, "budget": 0, "arms": ["uniform"]}, BranchedBackend(revision), FakeSandbox({("a", "candidate-a"): SandboxOutcome.PASS, ("b", "candidate-b"): SandboxOutcome.PASS}))
    assert result.task_rows == ({"task_id": "a", "reward": 1.0}, {"task_id": "b", "reward": 1.0})


def test_frozen_audit_aborts_on_infrastructure_failure_instead_of_scoring_wrong(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _dataset(tasks)
    revision = "0123456789abcdef0123456789abcdef01234567"
    with pytest.raises(RuntimeError, match="infrastructure"):
        run_frozen_audit(
            {"tasks": {"kind": "jsonl", "path": str(tasks), "role": "frozen_audit"}, "model": {"name": "qwen25_coder_7b", "revision": revision}, "output_root": str(tmp_path / "infra"), "seed": 19, "budget": 2, "arms": ["uniform"]},
            FakePolicyBackend(revision),
            FakeSandbox({("a", "candidate-a"): SandboxOutcome.INFRASTRUCTURE_FAILURE}),
        )


def test_paired_bootstrap_uses_task_means_and_is_deterministic() -> None:
    rows = [
        {"task_id": "a", "treatment": 2.0, "baseline": 1.0},
        {"task_id": "a", "treatment": 4.0, "baseline": 1.0},
        {"task_id": "b", "treatment": 0.0, "baseline": 1.0},
    ]
    first = paired_bootstrap(rows, seed=7, resamples=200)
    second = paired_bootstrap(rows, seed=7, resamples=200)
    assert first == second
    assert first.estimate == pytest.approx(0.5)  # mean of task deltas: mean(1,3)=2; b=-1


def test_connected_evaluation_statistics_use_task_seed_hierarchy_and_holm() -> None:
    rows = [
        {"task_id": task, "seed": seed, "candidate": candidate, "passed": task == "a" and candidate == 7}
        for task in ("a", "b") for seed in (1, 2) for candidate in range(8)
    ]
    summary = evaluation_summary(rows)
    assert summary["task_count"] == 2
    assert summary["seed_count"] == 2
    assert summary["pass_at_1"] == 0.0625
    assert summary["pass_at_8"] == 0.5
    assert 0 < summary["effective_sample_size"] <= 4
    adjusted = holm_adjust({"jag": 0.01, "entropy": 0.04, "uniform": 0.2})
    assert adjusted["jag"] == pytest.approx(0.03)
    assert adjusted["entropy"] == pytest.approx(0.08)


def test_model_planning_records_revision_without_optional_imports() -> None:
    before = set(sys.modules)
    plan = model_plan("qwen25_coder_7b", "0123456789abcdef0123456789abcdef01234567", "16gb")
    assert plan["hf_id"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert plan["revision"].startswith("012345")
    assert "transformers" not in set(sys.modules) - before
    assert "datasets" not in set(sys.modules) - before


def test_genealogy_expands_every_registered_branch_and_counts_incremental_edges() -> None:
    task = load_tasks(BenchmarkSpec("jsonl", "frozen_audit", path=Path(__file__).parent / "fixtures/tasks.jsonl"))[0]
    request = GenerationRequest(root_samples=4, branch_depths=(2, 4, 6), children_per_branch=3, max_new_tokens=8)

    def sample_edge(parent_tokens: tuple[int, ...], count: int, seed: int) -> tuple[int, ...]:
        return tuple(seed % 997 for _ in range(count))

    nodes = build_genealogy(task, "prompt", "policy-sha", "template-v1", 11, request, sample_edge, lambda tokens: ",".join(map(str, tokens)))
    parent_ids = {node.parent_id for node in nodes if node.parent_id is not None}
    leaves = [node for node in nodes if node.node_id not in parent_ids]
    assert len(leaves) == 4 * 3**3
    assert sum(node.unique_tokens for node in nodes) == 4 * 2 + 12 * 2 + 36 * 2 + 108 * 2
    assert all(node.unique_tokens == len(node.token_ids) for node in nodes)
    assert all(len(node.token_ids) == 2 for node in nodes if node.parent_id is not None)


def test_genealogy_never_branches_after_terminal_edge() -> None:
    task = load_tasks(BenchmarkSpec("jsonl", "frozen_audit", path=Path(__file__).parent / "fixtures/tasks.jsonl"))[0]
    calls = []
    request = GenerationRequest(root_samples=4, branch_depths=(2, 4), children_per_branch=3, max_new_tokens=6)

    def sample_edge(parent_tokens: tuple[int, ...], count: int, seed: int) -> tuple[int, ...]:
        calls.append((parent_tokens, count, seed))
        return (0,)

    nodes = build_genealogy(task, "prompt", "policy", "template", 7, request, sample_edge, str, lambda edge: edge[-1] == 0)
    assert len(calls) == 4
    assert len(nodes) == 5
