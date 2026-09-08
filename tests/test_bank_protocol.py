from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from jag_tree.bank import BankError, TreeBank, verify_bank
from jag_tree.ledger import RunLedger
from jag_tree.manifest import ManifestError, build_manifest
from jag_tree.sandbox import ContainerSandbox, FakeSandbox, SandboxLimits, SandboxOutcome, SubprocessSandbox, container_command
from jag_tree.schema import TaskRecord, TreeNode, code_fingerprint, test_fingerprint as fingerprint_tests


def _task(task_id: str = "t1", role: str = "train", lineage: str = "family-a") -> TaskRecord:
    return TaskRecord(task_id, "Add two numbers", "def add(a,b):\n    return a + b\n", ("assert add(1, 2) == 3",), "fixture", lineage, role)


def test_manifest_rejects_lineage_crossing_split_roles_and_hashes_canonically() -> None:
    with pytest.raises(ManifestError, match="split roles"):
        build_manifest([_task(), _task("t2", "frozen_audit")], split_seed=7)
    first = build_manifest([_task()], split_seed=7)
    second = build_manifest([replace(_task(), starter_code="def add(a, b): return a+b")], split_seed=7)
    assert first.sha256 == second.sha256
    assert code_fingerprint("x = 1 # comment\n") == code_fingerprint("x=1")
    assert fingerprint_tests(["assert f( 1 ) == 2"]) == fingerprint_tests(["assert f(1)==2"])


def test_manifest_derives_duplicate_components_instead_of_trusting_lineage_labels() -> None:
    first = _task("a", "train", "declared-a")
    duplicate = replace(first, task_id="b", role="frozen_audit", lineage_group="declared-b")
    with pytest.raises(ManifestError, match="content-derived"):
        build_manifest([first, duplicate], split_seed=7)


def test_tree_bank_is_create_once_and_verifies_parent_versions_and_tampering(tmp_path: Path) -> None:
    root = TreeNode("root", "t1", None, 0, "p", (), 0, "policy-sha", "template-v1")
    leaf = TreeNode("leaf", "t1", "root", 1, "answer", (1, 2), 2, "policy-sha", "template-v1", reward=1.0)
    path = tmp_path / "bank"
    TreeBank.create(path, [_task()], [root, leaf], score_arrays={"leaf": np.array([1.0, -1.0])}, generation_identity="generation-sha")
    verified = verify_bank(path)
    assert verified.manifest["task_count"] == 1
    np.testing.assert_array_equal(verified.score_arrays["leaf"], [1.0, -1.0])
    with pytest.raises(ValueError, match="read-only"):
        verified.score_arrays["leaf"][0] = 9
    with pytest.raises(ValueError):
        verified.score_arrays["leaf"].setflags(write=True)
    with pytest.raises(TypeError):
        verified.manifest["files"]["nodes.jsonl"] = "changed"  # type: ignore[index]
    with pytest.raises(FileExistsError):
        TreeBank.create(path, [_task()], [root, leaf], generation_identity="generation-sha")
    bad = tmp_path / "bad"
    with pytest.raises(BankError, match="parent"):
        TreeBank.create(bad, [_task()], [replace(leaf, parent_id="missing")], generation_identity="generation-sha")
    nodes_path = path / "nodes.jsonl"
    nodes_path.write_text(nodes_path.read_text().replace("answer", "tampered"), encoding="utf-8")
    with pytest.raises(BankError, match="checksum"):
        verify_bank(path)


def test_bank_requires_exact_checksum_inventory_and_binds_generation_identity(tmp_path: Path) -> None:
    root = TreeNode("root", "t1", None, 0, "p", (), 0, "policy-sha", "template-v1")
    path = tmp_path / "bank"
    bank = TreeBank.create(path, [_task()], [root], generation_identity="generation-sha")
    assert bank.manifest["generation_identity"] == "generation-sha"
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"].pop("scores.npz")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BankError, match="inventory"):
        verify_bank(path)


def test_ledgers_compare_each_budget_and_keep_infrastructure_failures_separate() -> None:
    left = RunLedger()
    right = RunLedger()
    left.record_generation(100)
    right.record_generation(101)
    left.record_verification(programs=10, cpu_seconds=20)
    right.record_verification(programs=10, cpu_seconds=20.8)
    left.record_optimization(gpu_hours=1.0, steps=2)
    right.record_optimization(gpu_hours=1.0, steps=2)
    assert left.compare(right, token_tolerance=0.01, cpu_tolerance=0.05).matched
    right.record_generation(2)
    assert not left.compare(right, token_tolerance=0.01, cpu_tolerance=0.05).matched
    left.record_outcome(SandboxOutcome.INFRASTRUCTURE_FAILURE)
    assert left.infrastructure_failures == 1
    assert left.programs == 10


def test_fake_sandbox_is_deterministic_and_subprocess_requires_opt_in(tmp_path: Path) -> None:
    sandbox = FakeSandbox({("task", "ok"): SandboxOutcome.PASS})
    limits = SandboxLimits(timeout_seconds=1, memory_mb=64)
    task = replace(_task(), task_id="task")
    assert sandbox.run(task, "ok", limits).outcome is SandboxOutcome.PASS
    assert sandbox.run(task, "other", limits).outcome is SandboxOutcome.WRONG
    with pytest.raises(PermissionError, match="trusted fixture"):
        SubprocessSandbox(trusted_task_ids=frozenset()).run(task, "print(1)", limits)


def test_trusted_subprocess_fails_closed_without_external_verifier() -> None:
    task = _task()
    sandbox = SubprocessSandbox(trusted_task_ids=frozenset({task.task_id}))
    limits = SandboxLimits(timeout_seconds=1, memory_mb=128)
    assert sandbox.run(task, "def add(a,b): return a+b", limits).outcome is SandboxOutcome.INFRASTRUCTURE_FAILURE
    assert sandbox.run(task, "import os; os._exit(0)", limits).outcome is SandboxOutcome.INFRASTRUCTURE_FAILURE
    assert sandbox.run(task, "print('__JAG_TESTS_PASSED__'); import os; os._exit(0)", limits).outcome is SandboxOutcome.INFRASTRUCTURE_FAILURE
    frame_attack = "import os,sys\nm=next(x for x in sys._getframe().f_back.f_code.co_consts if isinstance(x,str) and x.startswith('__JAG_TESTS_PASSED__:'))\nprint(m,flush=True)\nos._exit(0)"
    assert sandbox.run(replace(task, tests=("assert False",)), frame_attack, limits).outcome is SandboxOutcome.INFRASTRUCTURE_FAILURE
    exit_code_attack = "import os,sys\nc=next(x for x in sys._getframe().f_back.f_code.co_consts if type(x) is int and 40<=x<120)\nos._exit(c)"
    assert sandbox.run(replace(task, tests=("assert False",)), exit_code_attack, limits).outcome is SandboxOutcome.INFRASTRUCTURE_FAILURE


def test_container_command_disables_network_and_applies_resource_limits() -> None:
    task = _task()
    limits = SandboxLimits(timeout_seconds=3, memory_mb=256)
    command = container_command("docker", "python@sha256:" + "a" * 64, task, "def add(a,b): return a+b", limits)
    assert "--network=none" in command
    assert "--memory=256m" in command
    assert "--read-only" in command
    assert "--pids-limit=64" in command
    assert "def add" not in " ".join(command)
    container = ContainerSandbox("python@sha256:" + "a" * 64)
    assert container.run(task, "import os; os._exit(0)", limits).outcome is SandboxOutcome.INFRASTRUCTURE_FAILURE
    with pytest.raises(ValueError, match="digest"):
        ContainerSandbox("python:latest")
