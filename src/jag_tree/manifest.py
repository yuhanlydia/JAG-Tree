"""Lineage-safe split manifests with canonical identities."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Iterable

from .schema import TaskRecord


class ManifestError(ValueError):
    """Task split manifest violates identity or lineage isolation."""


@dataclass(frozen=True)
class TaskManifest:
    records: tuple[dict[str, object], ...]
    split_seed: int
    sha256: str


def build_manifest(records: Iterable[TaskRecord], split_seed: int) -> TaskManifest:
    ordered = tuple(sorted(records, key=lambda record: record.task_id))
    if len({record.task_id for record in ordered}) != len(ordered):
        raise ManifestError("task IDs must be unique")
    parents = list(range(len(ordered)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    normalized_statements = [tuple(re.findall(r"[a-z0-9]+", record.statement.lower())) for record in ordered]
    identities = [record.canonical_identity() for record in ordered]

    def statement_near(left: int, right: int) -> bool:
        left_tokens, right_tokens = set(normalized_statements[left]), set(normalized_statements[right])
        if not left_tokens or not right_tokens:
            return False
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) >= 0.85

    content_edges: set[tuple[int, int]] = set()
    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            same_declared = ordered[left].lineage_group == ordered[right].lineage_group
            same_statement = statement_near(left, right)
            same_code = bool(ordered[left].starter_code.strip() and identities[left]["code_sha256"] == identities[right]["code_sha256"])
            same_tests = bool(ordered[left].tests and identities[left]["tests_sha256"] == identities[right]["tests_sha256"])
            if same_declared or same_statement or same_code or same_tests:
                union(left, right)
            if same_statement or same_code or same_tests:
                content_edges.add((left, right))
    component_roles: dict[int, set[str]] = {}
    for index, record in enumerate(ordered):
        component_roles.setdefault(find(index), set()).add(record.role)
    crossing_roots = {root for root, roles in component_roles.items() if len(roles) > 1}
    if crossing_roots:
        content_crossing = any(find(left) in crossing_roots for left, _ in content_edges)
        label = "content-derived duplicate components" if content_crossing else "lineage groups"
        raise ManifestError(f"{label} cross split roles")
    rows = tuple(record.canonical_identity() for record in ordered)
    payload = {"records": rows, "split_seed": int(split_seed)}
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return TaskManifest(rows, int(split_seed), digest)
