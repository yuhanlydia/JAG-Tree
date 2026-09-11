"""Frozen-bank generation and common-random-number allocator replay."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .audit import AuditReplay, FrozenMomentPredictor, audit_replay
from .bank import TreeBank, verify_bank
from .benchmarks import BenchmarkSpec, load_tasks
from .config import ExperimentConfig, thaw
from .manifest import build_manifest
from .ledger import RunLedger
from .rollout import GenerationRequest, PolicyBackend
from .sandbox import SandboxBackend, SandboxLimits, SandboxOutcome


@dataclass(frozen=True)
class FrozenAuditResult:
    bank_path: Path
    model_revision: str
    arm_bank_sha256: dict[str, str]
    task_rows: tuple[dict[str, object], ...]
    arm_replays: Mapping[str, AuditReplay]
    execution_ledger: RunLedger


def _mapping(config: ExperimentConfig | Mapping[str, Any]) -> dict[str, Any]:
    return thaw(config.data) if isinstance(config, ExperimentConfig) else dict(config)


def _bank_identity(path: Path) -> str:
    return sha256((path / "manifest.json").read_bytes()).hexdigest()


def run_frozen_audit(config: ExperimentConfig | Mapping[str, Any], backend: PolicyBackend, sandbox: SandboxBackend) -> FrozenAuditResult:
    data = _mapping(config)
    task_spec = data.get("tasks")
    if not isinstance(task_spec, Mapping):
        raise ValueError("frozen audit requires a tasks mapping")
    spec = BenchmarkSpec(str(task_spec["kind"]), str(task_spec["role"]), Path(task_spec["path"]) if task_spec.get("path") else None, str(task_spec["revision"]) if task_spec.get("revision") else None, str(task_spec.get("split", "test")), task_spec.get("field_map", {}))
    tasks = load_tasks(spec)
    # Seal and validate duplicate/split components before any policy call.
    current_manifest = build_manifest(tasks, int(data["seed"]))
    model = data.get("model")
    if not isinstance(model, Mapping) or str(model.get("revision")) != backend.revision:
        raise ValueError("backend model revision differs from experiment")
    root = Path(data["output_root"])
    bank_path = root / "candidate_bank"
    request_data = data.get("generation", {})
    generation_identity = sha256(json.dumps(request_data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if bank_path.exists():
        bank = verify_bank(bank_path)
        if current_manifest.sha256 != bank.manifest["task_manifest_sha256"]:
            raise ValueError("task manifest differs from existing candidate bank")
        if generation_identity != bank.manifest["generation_identity"]:
            raise ValueError("generation identity differs from existing candidate bank")
    else:
        request = GenerationRequest(int(request_data.get("root_samples", 1)), tuple(request_data.get("branch_depths", ())), int(request_data.get("children_per_branch", 1)), int(request_data.get("max_new_tokens", 256)), float(request_data.get("temperature", 1.0)), float(request_data.get("top_p", 1.0)))
        nodes = []
        execution_ledger = RunLedger()
        for task_index, task in enumerate(tasks, 1):
            logging.getLogger(__name__).info("generating task %d/%d: %s", task_index, len(tasks), task.task_id)
            generated = backend.generate_tree(task, int(data["seed"]), request)
            for generated_node in generated:
                if generated_node.parent_id is not None:
                    execution_ledger.record_generation(generated_node.unique_tokens)
            parent_ids = {node.parent_id for node in generated if node.parent_id is not None}
            for node in generated:
                if node.parent_id is not None and node.node_id not in parent_ids:
                    result = sandbox.run(task, node.text, SandboxLimits(float(data.get("timeout_seconds", 10)), int(data.get("memory_mb", 512))))
                    if result.outcome is SandboxOutcome.INFRASTRUCTURE_FAILURE:
                        raise RuntimeError(f"sandbox infrastructure failure for task {task.task_id}: {result.stderr}")
                    execution_ledger.record_verification(programs=1, cpu_seconds=result.cpu_seconds)
                    reward = 1.0 if result.outcome is SandboxOutcome.PASS else 0.0
                    node = replace(node, reward=reward)
                nodes.append(node)
        score_arrays: dict[str, np.ndarray] = {}
        by_task = {task.task_id: task for task in tasks}
        for task_index, task_id in enumerate(sorted(by_task), 1):
            logging.getLogger(__name__).info("scoring gradients %d/%d: %s", task_index, len(by_task), task_id)
            task_nodes = tuple(node for node in nodes if node.task_id == task_id)
            score_arrays.update({key: np.asarray(value, dtype=np.float64) for key, value in backend.score_tree(by_task[task_id], task_nodes).items()})
        required_scores = {node.node_id for node in nodes if node.parent_id is not None}
        if set(score_arrays) != required_scores:
            raise ValueError("policy backend must score every and only non-root generated edge")
        dimensions = {array.reshape(-1).size for array in score_arrays.values()}
        if len(dimensions) != 1 or next(iter(dimensions), 0) < 1 or not all(np.isfinite(array).all() for array in score_arrays.values()):
            raise ValueError("policy-score gradients must be finite with one common positive dimension")
        bank = TreeBank.create(bank_path, tasks, nodes, generation_identity=generation_identity, score_arrays=score_arrays, split_seed=int(data["seed"]))
    if bank_path.exists() and 'execution_ledger' not in locals():
        execution_ledger = RunLedger()
    if any(node.policy_revision != backend.revision for node in bank.nodes):
        raise ValueError("bank model revision differs from backend")
    identity = _bank_identity(bank.path)
    arm_identities = {}
    predictor = backend.frozen_predictor(bank.nodes)
    if not isinstance(predictor, FrozenMomentPredictor):
        raise ValueError("backend must provide a typed frozen calibration predictor")
    replays: dict[str, AuditReplay] = {}
    for arm in data["arms"]:
        replays[str(arm)] = audit_replay(bank, str(arm), int(data["seed"]), budget=int(data["budget"]), predictor=predictor)
        arm_identities[str(arm)] = identity
    rewards: dict[str, list[float]] = {}
    for node in bank.nodes:
        if node.reward is not None:
            rewards.setdefault(node.task_id, []).append(node.reward)
    rows = tuple({"task_id": task_id, "reward": float(np.mean(rewards[task_id]))} for task_id in sorted(rewards))
    return FrozenAuditResult(bank.path, backend.revision, arm_identities, rows, replays, execution_ledger)
