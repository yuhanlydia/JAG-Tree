"""Fail-closed generated-code boundaries and deterministic fixture backend."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Mapping, Protocol

from .schema import TaskRecord


class SandboxOutcome(str, Enum):
    PASS = "pass"
    WRONG = "wrong"
    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: float
    memory_mb: int

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.memory_mb < 32:
            raise ValueError("sandbox limits require positive time and at least 32MB")


@dataclass(frozen=True)
class SandboxResult:
    outcome: SandboxOutcome
    cpu_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""


class SandboxBackend(Protocol):
    def run(self, task: TaskRecord, code: str, limits: SandboxLimits) -> SandboxResult: ...


class FakeSandbox:
    def __init__(self, outcomes: Mapping[tuple[str, str], SandboxOutcome] | None = None) -> None:
        self._outcomes = dict(outcomes or {})

    def run(self, task: TaskRecord, code: str, limits: SandboxLimits) -> SandboxResult:
        _ = limits
        return SandboxResult(self._outcomes.get((task.task_id, code), SandboxOutcome.WRONG))


class SubprocessSandbox:
    """Unavailable until an external verifier can attest test completion."""

    def __init__(self, *, trusted_task_ids: frozenset[str]) -> None:
        self.trusted_task_ids = trusted_task_ids

    def run(self, task: TaskRecord, code: str, limits: SandboxLimits) -> SandboxResult:
        del code, limits
        if task.task_id not in self.trusted_task_ids:
            raise PermissionError("local subprocess execution is limited to an explicit trusted fixture")
        return SandboxResult(SandboxOutcome.INFRASTRUCTURE_FAILURE, stderr="trusted subprocess verifier is unavailable; candidate was not executed")


_IMAGE_DIGEST = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}\Z")


def container_command(runtime: str, image: str, task: TaskRecord, code: str, limits: SandboxLimits) -> list[str]:
    if not _IMAGE_DIGEST.fullmatch(image):
        raise ValueError("container image must include an immutable sha256 digest")
    del task, code
    return [runtime, "run", "--rm", "--network=none", f"--memory={limits.memory_mb}m", "--cpus=1", "--pids-limit=64", "--read-only", "--tmpfs=/tmp:rw,noexec,nosuid,size=16m", "--security-opt=no-new-privileges", "--cap-drop=ALL", image, "python", "-I", "-c", "raise SystemExit(125)"]


class ContainerSandbox:
    """OCI execution plan that fails closed without an external verifier."""

    def __init__(self, image: str, *, runtime: str = "docker") -> None:
        if not _IMAGE_DIGEST.fullmatch(image):
            raise ValueError("container image must include an immutable sha256 digest")
        self.image = image
        self.runtime = runtime

    def run(self, task: TaskRecord, code: str, limits: SandboxLimits) -> SandboxResult:
        del task, code, limits
        return SandboxResult(SandboxOutcome.INFRASTRUCTURE_FAILURE, stderr="external OCI test verifier is unavailable; candidate was not executed")
