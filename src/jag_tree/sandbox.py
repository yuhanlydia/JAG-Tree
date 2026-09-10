"""Fail-closed generated-code boundaries and deterministic fixture backend."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import tempfile
import time
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


def _candidate_source(code: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", code, re.IGNORECASE | re.DOTALL)
    return (match.group(1) if match else code).strip() + "\n"


class TrustedLocalSandbox:
    """Run public-test candidates as an unprivileged local process.

    This backend is intended for pilot execution on a dedicated worker.  It drops
    to ``nobody``, clears the environment, uses a private temporary directory and
    applies hard process resource limits.  Formal runs should still prefer an
    independently attested OCI evaluator when the host permits namespaces.
    """

    @staticmethod
    def _limit(memory_mb: int, timeout_seconds: float):
        def apply() -> None:
            resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024,) * 2)
            cpu = max(1, int(math.ceil(timeout_seconds)))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
            resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
            resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            if os.geteuid() == 0:
                os.setgroups([])
                os.setgid(65534)
                os.setuid(65534)
        return apply

    @staticmethod
    def _run(source: Path, limits: SandboxLimits, *, stdin: str = "") -> tuple[int, str, str, float] | None:
        started = time.monotonic()
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        try:
            completed = subprocess.run(
                ["/usr/bin/python3", "-I", str(source)],
                input=stdin,
                text=True,
                capture_output=True,
                cwd=source.parent,
                env={"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8"},
                timeout=limits.timeout_seconds,
                preexec_fn=TrustedLocalSandbox._limit(limits.memory_mb, limits.timeout_seconds),
            )
        except subprocess.TimeoutExpired:
            return None
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
        return completed.returncode, completed.stdout, completed.stderr, max(cpu, time.monotonic() - started if completed.returncode < 0 else cpu)

    def run(self, task: TaskRecord, code: str, limits: SandboxLimits) -> SandboxResult:
        source_text = _candidate_source(code)
        with tempfile.TemporaryDirectory(prefix="jag-eval-") as directory:
            root = Path(directory)
            root.chmod(0o755)
            candidate = root / "candidate.py"
            candidate.write_text(source_text, encoding="utf-8")
            candidate.chmod(0o644)
            parsed = None
            if len(task.tests) == 1:
                try:
                    value = json.loads(task.tests[0])
                    if isinstance(value, dict) and "inputs" in value and "outputs" in value:
                        parsed = value
                except json.JSONDecodeError:
                    pass
            cpu_total = 0.0
            if parsed is not None and not parsed.get("fn_name"):
                inputs, outputs = parsed["inputs"], parsed["outputs"]
                if len(inputs) != len(outputs):
                    return SandboxResult(SandboxOutcome.INFRASTRUCTURE_FAILURE, stderr="test input/output lengths differ")
                for test_input, expected in zip(inputs, outputs):
                    result = self._run(candidate, limits, stdin=str(test_input))
                    if result is None:
                        return SandboxResult(SandboxOutcome.TIMEOUT, cpu_seconds=cpu_total)
                    returncode, stdout, stderr, cpu = result
                    cpu_total += cpu
                    if returncode != 0:
                        return SandboxResult(SandboxOutcome.EXCEPTION, cpu_total, stderr=stderr[-4000:])
                    expected_values = expected if isinstance(expected, list) else [expected]
                    if all(stdout.split() != str(item).split() for item in expected_values):
                        return SandboxResult(SandboxOutcome.WRONG, cpu_total)
                return SandboxResult(SandboxOutcome.PASS, cpu_total)

            harness = root / "harness.py"
            tests = "\n".join(task.tests)
            harness.write_text(
                "namespace = {}\n"
                f"exec({source_text!r}, namespace, namespace)\n"
                "try:\n"
                f"    exec({tests!r}, namespace, namespace)\n"
                "except AssertionError:\n"
                "    raise SystemExit(10)\n",
                encoding="utf-8",
            )
            harness.chmod(0o644)
            result = self._run(harness, limits)
            if result is None:
                return SandboxResult(SandboxOutcome.TIMEOUT)
            returncode, stdout, stderr, cpu = result
            if returncode == 0:
                return SandboxResult(SandboxOutcome.PASS, cpu)
            if returncode == 10:
                return SandboxResult(SandboxOutcome.WRONG, cpu)
            if returncode in {-signal.SIGXCPU, -signal.SIGKILL}:
                return SandboxResult(SandboxOutcome.TIMEOUT, cpu)
            return SandboxResult(SandboxOutcome.EXCEPTION, cpu, stderr=stderr[-4000:])
