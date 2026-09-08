"""Canonical task and genealogy schemas used by frozen banks."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from hashlib import sha256
import io
import json
import tokenize
from typing import Any, Iterable


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def code_fingerprint(code: str) -> str:
    """Hash normalized Python structure, ignoring formatting and comments."""

    try:
        normalized = ast.dump(ast.parse(code), annotate_fields=True, include_attributes=False)
    except SyntaxError:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        normalized = "".join(token.string for token in tokens if token.type not in {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER})
    return _digest(normalized)


def test_fingerprint(tests: Iterable[str]) -> str:
    normalized = ["".join(str(test).split()) for test in tests]
    return _digest(sorted(normalized))


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    statement: str
    starter_code: str
    tests: tuple[str, ...]
    source: str
    lineage_group: str
    role: str

    def __post_init__(self) -> None:
        if not all((self.task_id, self.statement, self.source, self.lineage_group, self.role)):
            raise ValueError("task fields must be non-empty")
        object.__setattr__(self, "tests", tuple(self.tests))

    def canonical_identity(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "statement": " ".join(self.statement.split()), "code_sha256": code_fingerprint(self.starter_code), "tests_sha256": test_fingerprint(self.tests), "source": self.source, "lineage_group": self.lineage_group, "role": self.role}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskRecord":
        value = dict(value)
        value["tests"] = tuple(value["tests"])
        return cls(**value)


@dataclass(frozen=True)
class TreeNode:
    node_id: str
    task_id: str
    parent_id: str | None
    depth: int
    text: str
    token_ids: tuple[int, ...]
    unique_tokens: int
    policy_revision: str
    template_version: str
    reward: float | None = None
    edge_logprob: float = 0.0
    edge_entropy: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(int(item) for item in self.token_ids))
        if not self.node_id or not self.task_id or self.depth < 0 or self.unique_tokens < 0:
            raise ValueError("invalid tree node identity/count")
        if self.unique_tokens < len(self.token_ids):
            raise ValueError("unique_tokens cannot be smaller than stored token_ids")
        if not all(map(lambda value: isinstance(value, (int, float)), (self.edge_logprob, self.edge_entropy))):
            raise ValueError("edge policy statistics must be numeric")
        if self.edge_entropy < 0:
            raise ValueError("edge entropy must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TreeNode":
        value = dict(value)
        value["token_ids"] = tuple(value["token_ids"])
        return cls(**value)
