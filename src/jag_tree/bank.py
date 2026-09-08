"""Atomic, create-once JSONL/NPZ candidate tree banks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from .manifest import build_manifest
from .schema import TaskRecord, TreeNode


class BankError(ValueError):
    """Candidate bank violates its immutable schema or checksums."""


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _deep_freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _validate(tasks: tuple[TaskRecord, ...], nodes: tuple[TreeNode, ...]) -> None:
    task_ids = {task.task_id for task in tasks}
    if len(task_ids) != len(tasks):
        raise BankError("task IDs must be unique")
    by_id = {node.node_id: node for node in nodes}
    if len(by_id) != len(nodes):
        raise BankError("node IDs must be unique")
    for node in nodes:
        if node.task_id not in task_ids:
            raise BankError(f"node {node.node_id} references unknown task")
        if node.parent_id is None:
            if node.depth != 0:
                raise BankError("root depth must be zero")
            continue
        parent = by_id.get(node.parent_id)
        if parent is None or parent.task_id != node.task_id:
            raise BankError(f"node {node.node_id} has invalid parent")
        if node.depth != parent.depth + 1:
            raise BankError("child depth must increment parent depth")
        if (node.policy_revision, node.template_version) != (parent.policy_revision, parent.template_version):
            raise BankError("parent and child policy/template versions differ")


@dataclass(frozen=True)
class TreeBank:
    path: Path
    tasks: tuple[TaskRecord, ...]
    nodes: tuple[TreeNode, ...]
    score_arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, object]

    @classmethod
    def create(cls, path: str | Path, tasks: Iterable[TaskRecord], nodes: Iterable[TreeNode], *, generation_identity: str, score_arrays: Mapping[str, np.ndarray] | None = None, split_seed: int = 0) -> "TreeBank":
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"bank already exists: {target}")
        task_rows, node_rows = tuple(tasks), tuple(nodes)
        if not generation_identity:
            raise BankError("generation identity must be non-empty")
        arrays = {str(key): np.asarray(value) for key, value in (score_arrays or {}).items()}
        _validate(task_rows, node_rows)
        if not set(arrays).issubset({node.node_id for node in node_rows}):
            raise BankError("score array keys must be tree node IDs")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        try:
            (staging / "tasks.jsonl").write_text("".join(_json(task.to_dict()) + "\n" for task in sorted(task_rows, key=lambda row: row.task_id)), encoding="utf-8")
            (staging / "nodes.jsonl").write_text("".join(_json(node.to_dict()) + "\n" for node in sorted(node_rows, key=lambda row: row.node_id)), encoding="utf-8")
            np.savez_compressed(staging / "scores.npz", **arrays)
            task_manifest = build_manifest(task_rows, split_seed)
            files = {name: _file_sha(staging / name) for name in ("tasks.jsonl", "nodes.jsonl", "scores.npz")}
            manifest = {"schema_version": 1, "task_count": len(task_rows), "node_count": len(node_rows), "split_seed": split_seed, "task_manifest_sha256": task_manifest.sha256, "generation_identity": generation_identity, "policy_revisions": sorted({node.policy_revision for node in node_rows}), "template_versions": sorted({node.template_version for node in node_rows}), "files": files}
            (staging / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
            (staging / "COMPLETE").write_text("complete\n", encoding="utf-8")
            os.replace(staging, target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return verify_bank(target)


def verify_bank(path: str | Path) -> TreeBank:
    root = Path(path)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if not (root / "COMPLETE").is_file():
            raise BankError("bank is incomplete")
        if set(manifest.get("files", {})) != {"tasks.jsonl", "nodes.jsonl", "scores.npz"}:
            raise BankError("checksum inventory must exactly match bank payloads")
        for name, expected in manifest["files"].items():
            if _file_sha(root / name) != expected:
                raise BankError(f"checksum mismatch for {name}")
        tasks = tuple(TaskRecord.from_dict(json.loads(line)) for line in (root / "tasks.jsonl").read_text(encoding="utf-8").splitlines())
        nodes = tuple(TreeNode.from_dict(json.loads(line)) for line in (root / "nodes.jsonl").read_text(encoding="utf-8").splitlines())
        _validate(tasks, nodes)
        if (len(tasks), len(nodes)) != (manifest["task_count"], manifest["node_count"]):
            raise BankError("manifest count mismatch")
        task_manifest = build_manifest(tasks, int(manifest["split_seed"]))
        if task_manifest.sha256 != manifest["task_manifest_sha256"]:
            raise BankError("task manifest identity mismatch")
        with np.load(root / "scores.npz", allow_pickle=False) as archive:
            arrays = {}
            for key in archive.files:
                source = np.asarray(archive[key])
                # Immutable bytes own the backing memory, so callers cannot
                # escape read-only status with ``setflags(write=True)``.
                array = np.frombuffer(source.tobytes(), dtype=source.dtype).reshape(source.shape)
                arrays[key] = array
        if sorted({node.policy_revision for node in nodes}) != manifest.get("policy_revisions") or sorted({node.template_version for node in nodes}) != manifest.get("template_versions"):
            raise BankError("policy or template identity mismatch")
        return TreeBank(root, tasks, nodes, MappingProxyType(arrays), _deep_freeze(manifest))
    except BankError:
        raise
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise BankError(f"invalid bank: {exc}") from exc
