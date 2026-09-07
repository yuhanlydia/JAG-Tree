"""Typed runtime-checkable contracts for external Phase-1/2 integrations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PolicyScoreRequest:
    """Identified policy samples and the contexts whose scores are requested."""

    request_id: str
    sample_ids: tuple[str, ...]
    contexts: tuple[str, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyScoreResult:
    """Scores returned for exactly the request's sample identities."""

    request_id: str
    sample_ids: tuple[str, ...]
    scores: FloatArray


@dataclass(frozen=True)
class GradientRequest:
    """Identified samples, parameter blocks, and scalar targets for gradients."""

    request_id: str
    sample_ids: tuple[str, ...]
    block_ids: tuple[str, ...]
    targets: Mapping[str, float]


@dataclass(frozen=True)
class GradientResult:
    """Gradient rows for the ordered sample and parameter-block identities."""

    request_id: str
    sample_ids: tuple[str, ...]
    block_ids: tuple[str, ...]
    gradients: FloatArray


@dataclass(frozen=True)
class Candidate:
    """An immutable candidate identity paired with its content hash and source."""

    candidate_id: str
    content_hash: str
    source: str


@dataclass(frozen=True)
class CandidateGroupRequest:
    group_id: str
    task_id: str


@dataclass(frozen=True)
class CandidateGroup:
    group_id: str
    task_id: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class SandboxLimits:
    wall_time_seconds: float
    memory_mebibytes: int


@dataclass(frozen=True)
class SandboxRequest:
    request_id: str
    candidate_id: str
    test_ids: tuple[str, ...]
    limits: SandboxLimits


@dataclass(frozen=True)
class SandboxOutcome:
    candidate_id: str
    test_id: str
    status: str
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SandboxResult:
    request_id: str
    outcomes: tuple[SandboxOutcome, ...]


@runtime_checkable
class PolicyBackend(Protocol):
    def score(self, request: PolicyScoreRequest) -> PolicyScoreResult: ...


@runtime_checkable
class GradientProvider(Protocol):
    def gradients(self, request: GradientRequest) -> GradientResult: ...


@runtime_checkable
class CandidateBank(Protocol):
    def get_group(self, request: CandidateGroupRequest) -> CandidateGroup: ...


@runtime_checkable
class SandboxBackend(Protocol):
    def evaluate(self, request: SandboxRequest) -> SandboxResult: ...
