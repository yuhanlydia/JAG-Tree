"""One-update training contract, QLoRA setup, and staged launcher."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from .artifacts import write_artifacts
from .bank import TreeBank, verify_bank
from .config import ExperimentConfig, thaw, validate_experiment
from .ledger import RunLedger
from .loss import jag_loss
from .phase1 import run_frozen_audit
from .rollout import PolicyBackend
from .sandbox import SandboxBackend


@dataclass(frozen=True)
class TrainingBatch:
    logprobs: Any
    edge_mask: Any
    credit: Any
    weights: Any


@dataclass(frozen=True)
class RecursiveTrainingTargets:
    """One credit row per unique conditional edge in genealogy order."""

    edge_ids: tuple[str, ...]
    credit: Any
    weights: Any
    edge_mask: Any


def replay_edge_coefficients(bank: TreeBank | str | Path, replay_rows: tuple[Mapping[str, object], ...]) -> dict[str, float]:
    """Aggregate the audited importance-weighted draw estimator onto unique edges."""

    loaded = bank if isinstance(bank, TreeBank) else verify_bank(bank)
    if not replay_rows:
        raise ValueError("online update requires stochastic replay rows")
    leaves = {node.node_id: node for node in loaded.nodes if node.reward is not None}
    known_edges = {node.node_id for node in loaded.nodes if node.parent_id is not None}
    coefficients: dict[str, float] = {}
    for row in replay_rows:
        leaf = leaves.get(str(row.get("leaf_id")))
        path = row.get("path_edge_ids")
        if leaf is None or not isinstance(path, (tuple, list)) or not path:
            raise ValueError("replay row must reference a sealed leaf and nonempty path")
        contribution = float(row["importance_weight"]) * float(leaf.reward) / len(replay_rows)
        for edge_id in path:
            edge = str(edge_id)
            if edge not in known_edges:
                raise ValueError("replay path contains unknown edge")
            coefficients[edge] = coefficients.get(edge, 0.0) + contribution
    return coefficients


def recursive_training_targets(bank: TreeBank | str | Path, *, baseline: str = "loo") -> RecursiveTrainingTargets:
    """Build recursively transported REINFORCE targets for an uneven forest.

    ``independent`` is the registered zero predictor (independent of the current
    sibling outcomes). ``loo`` uses the other siblings and is therefore also
    independent of the sampled edge whose score it multiplies.
    """

    import numpy as np

    loaded = bank if isinstance(bank, TreeBank) else verify_bank(bank)
    if baseline not in {"independent", "loo"}:
        raise ValueError("baseline must be 'independent' or 'loo'")
    by_id = {node.node_id: node for node in loaded.nodes}
    children: dict[str, list[Any]] = {}
    for node in loaded.nodes:
        if node.parent_id is not None:
            children.setdefault(node.parent_id, []).append(node)
    for rows in children.values():
        rows.sort(key=lambda node: node.node_id)

    value_cache: dict[str, float] = {}

    def subtree_value(node: Any) -> float:
        if node.node_id in value_cache:
            return value_cache[node.node_id]
        if node.node_id not in children:
            if node.reward is None:
                raise ValueError(f"training leaf {node.node_id} has no sealed reward")
            value = float(node.reward)
        else:
            value = float(np.mean([subtree_value(child) for child in children[node.node_id]]))
        value_cache[node.node_id] = value
        return value

    roots = sorted((node for node in loaded.nodes if node.parent_id is None), key=lambda node: node.node_id)
    if not roots:
        raise ValueError("training bank has no roots")
    rows: list[tuple[str, float, float]] = []

    def visit(parent: Any, transport: float) -> None:
        siblings = children.get(parent.node_id, [])
        if not siblings:
            return
        values = np.asarray([subtree_value(child) for child in siblings], dtype=np.float64)
        child_weight = transport / len(siblings)
        for index, child in enumerate(siblings):
            if baseline == "independent" or len(siblings) == 1:
                control = 0.0
            else:
                control = float((values.sum() - values[index]) / (len(siblings) - 1))
            rows.append((child.node_id, float(values[index] - control), child_weight))
            visit(child, child_weight)

    for root in roots:
        visit(root, 1.0 / len(roots))
    if not rows:
        raise ValueError("training bank has no conditional edges")
    edge_ids = tuple(row[0] for row in rows)
    credit = np.asarray([row[1] for row in rows], dtype=np.float64)
    weights = np.asarray([row[2] for row in rows], dtype=np.float64)
    edge_mask = np.eye(len(rows), dtype=bool)
    return RecursiveTrainingTargets(edge_ids, credit, weights, edge_mask)


@dataclass(frozen=True)
class QLoRAProfile:
    rank: int
    alpha: int
    max_response_tokens: int
    sequential_group: int
    max_updates: int
    quantized: bool


QLORA_PROFILES = {
    "16gb": QLoRAProfile(16, 32, 1024, 2, 1, True),
    "24gb": QLoRAProfile(32, 64, 1536, 4, 20, True),
    "4x24gb": QLoRAProfile(32, 64, 2048, 8, 200, True),
    "h200_formal": QLoRAProfile(32, 64, 2048, 8, 800, False),
}


def train_one_update(model: Any, optimizer: Any, batch: TrainingBatch, *, optimizer_epochs: int = 1, expected_ledger: RunLedger | None = None, actual_ledger: RunLedger | None = None) -> float:
    if optimizer_epochs != 1:
        raise ValueError("JAG requires exactly one optimizer epoch per rollout batch")
    if (expected_ledger is None) != (actual_ledger is None):
        raise ValueError("both expected and actual ledgers are required")
    if expected_ledger is not None and actual_ledger is not None:
        comparison = expected_ledger.compare(actual_ledger, token_tolerance=0.01, cpu_tolerance=0.05)
        if not comparison.matched:
            raise ValueError(f"budget drift exceeds tolerance: {comparison}")
    if optimizer is None:
        raise ValueError("optimizer is required for a training update")
    optimizer.zero_grad(set_to_none=True)
    loss = jag_loss(batch.logprobs, batch.edge_mask, batch.credit, batch.weights)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def prepare_qlora_model(model_id: str, revision: str, hardware: str) -> Any:
    if hardware not in QLORA_PROFILES:
        raise ValueError(f"unsupported QLoRA hardware profile: {hardware}")
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("QLoRA training requires the optional training extra") from exc
    profile = QLORA_PROFILES[hardware]
    kwargs: dict[str, Any] = {"revision": revision, "device_map": "auto", "torch_dtype": torch.bfloat16}
    if profile.quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if profile.quantized:
        model = prepare_model_for_kbit_training(model)
    return get_peft_model(model, LoraConfig(r=profile.rank, lora_alpha=profile.alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))


def select_checkpoint_by_token_aulc(rows: list[Mapping[str, float]]) -> str:
    if not rows:
        raise ValueError("checkpoint selection requires dev rows")
    grouped: dict[str, list[Mapping[str, float]]] = {}
    for row in rows:
        grouped.setdefault(str(row["checkpoint"]), []).append(row)
    scores = {}
    for checkpoint, values in grouped.items():
        ordered = sorted(values, key=lambda row: row["tokens"])
        area = 0.0
        for left, right in zip(ordered, ordered[1:]):
            area += (right["tokens"] - left["tokens"]) * (left["score"] + right["score"]) / 2
        span = ordered[-1]["tokens"] - ordered[0]["tokens"]
        scores[checkpoint] = area / span if span else ordered[0]["score"]
    return min(scores, key=lambda checkpoint: (-scores[checkpoint], checkpoint))


def run_experiment(config: ExperimentConfig, *, output_root: str | Path, seed: int, dry_run: bool = False, backend: PolicyBackend | None = None, sandbox: SandboxBackend | None = None) -> dict[str, object]:
    validate_experiment(config)
    data = thaw(config.data)
    configured_seeds = tuple(int(value) for value in data.get("seeds", (seed,)))
    if int(seed) not in configured_seeds:
        raise ValueError(f"seed {seed} is not registered in config seeds {configured_seeds}")
    stage = str(data.get("stage", "frozen"))
    training = data.get("training", {})
    updates = int(training.get("updates", training.get("formal_updates", 1))) if stage == "online" else 0
    if updates < 0:
        raise ValueError("training update count must be nonnegative")
    plan = {"config_sha256": config.sha256, "mode": stage, "seed": int(seed), "output_root": str(Path(output_root)), "dry_run": bool(dry_run), "planned_updates": updates}
    if dry_run:
        return plan
    if "baseline_matrix" in data or "ablations" in data:
        raise ValueError("matrix-only configuration has no executable dispatch")
    if data.get("formal_activation") and data.get("formal") is not True:
        raise ValueError("formal activation config cannot execute with formal:false")
    if stage == "online" and int(data.get("budget", 0)) <= 0:
        raise ValueError("online execution requires a positive allocation budget")
    if stage == "online" and data.get("model", {}).get("name") != "qwen25_coder_1_5b" and data.get("formal") is not True:
        raise ValueError("non-plumbing online model execution requires activated formal gates")
    if data.get("formal") is True:
        from .artifacts import verify_artifacts

        prerequisites = data["prerequisites"]
        for gate in ("e0", "e1"):
            artifact_path = prerequisites.get(f"{gate}_artifact_path")
            if not artifact_path:
                raise ValueError(f"formal execution requires checksum-bound {gate.upper()} artifact path")
            verified = verify_artifacts(artifact_path)
            if verified.get("status") != "PASS":
                raise ValueError(f"formal prerequisite {gate.upper()} is not verified PASS")
            actual = sha256((Path(artifact_path) / "checksums.json").read_bytes()).hexdigest()
            if actual != prerequisites[f"{gate}_artifact_sha256"]:
                raise ValueError(f"formal prerequisite {gate.upper()} checksum mismatch")
    if backend is None or sandbox is None:
        raise ValueError("execution requires explicit policy and sandbox backends")
    if stage == "online" and not callable(getattr(backend, "train_bank_one_update", None)):
        raise ValueError("online execution requires a training-capable policy backend")
    data["seed"] = int(seed)
    bank_root = Path(output_root) / f"bank-{config.sha256[:12]}-seed{seed}"
    repetitions = updates if stage == "online" else 1
    losses: list[float] = []
    output_rows: list[dict[str, object]] = []
    audit = None
    for update in range(repetitions):
        data["output_root"] = str(bank_root / f"update-{update:04d}") if stage == "online" else str(bank_root)
        audit = run_frozen_audit(data, backend, sandbox)
        output_rows.extend({**row, "update": update} for row in audit.task_rows)
        for arm, replay in audit.arm_replays.items():
            output_rows.extend({**row, "update": update, "row_type": "audit_replay"} for row in replay.rows)
        if stage == "online":
            replay = audit.arm_replays[str(data["allocator"])]
            if not replay.selected_edge_ids:
                raise ValueError("online allocation selected no trainable edges within budget")
            expected_training = RunLedger(generation_tokens=sum(node.unique_tokens for node in verify_bank(audit.bank_path).nodes if node.node_id in replay.selected_edge_ids), optimizer_steps=1)
            losses.append(float(backend.train_bank_one_update(audit.bank_path, replay_rows=replay.rows, expected_ledger=expected_training)))  # type: ignore[attr-defined]
    if audit is None:
        raise ValueError("online execution requires at least one update")
    run_path = Path(output_root) / f"run-{config.sha256[:12]}-seed{seed}"
    arm_evidence = {
        arm: {
            "status": "COMPLETE" if replay.metrics["draw_count"] >= 2 else "INCOMPLETE",
            "metrics": {key: value for key, value in replay.metrics.items() if __import__("math").isfinite(value)},
            "ledger": vars(replay.ledger),
        }
        for arm, replay in audit.arm_replays.items()
    }
    result = {**plan, "status": "INCOMPLETE", "evidence_complete": False, "model_revision": audit.model_revision, "bank_path": str(audit.bank_path), "training_losses": losses, "updates_completed": len(losses), "arm_evidence": arm_evidence}
    write_artifacts(run_path, result, rows=output_rows, manifest={"config_sha256": config.sha256, "sources": [str(path) for path in config.source_paths]})
    return {**result, "run_path": str(run_path)}
