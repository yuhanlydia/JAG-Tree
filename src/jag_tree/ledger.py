"""Separate generation, verification, and optimization ledgers."""

from __future__ import annotations

from dataclasses import dataclass

from .sandbox import SandboxOutcome


@dataclass(frozen=True)
class BudgetComparison:
    matched: bool
    token_relative_drift: float
    cpu_relative_drift: float
    program_drift: int
    step_drift: int


@dataclass
class RunLedger:
    generation_tokens: int = 0
    replay_sampling_tokens: int = 0
    programs: int = 0
    verification_cpu_seconds: float = 0.0
    optimization_gpu_hours: float = 0.0
    optimizer_steps: int = 0
    infrastructure_failures: int = 0

    def record_generation(self, unique_tokens: int) -> None:
        if unique_tokens < 0:
            raise ValueError("token count must be nonnegative")
        self.generation_tokens += int(unique_tokens)

    def record_replay_sampling(self, tokens: int) -> None:
        if tokens < 0:
            raise ValueError("replay token count must be nonnegative")
        self.replay_sampling_tokens += int(tokens)

    def record_verification(self, *, programs: int, cpu_seconds: float) -> None:
        if programs < 0 or cpu_seconds < 0:
            raise ValueError("verification costs must be nonnegative")
        self.programs += int(programs)
        self.verification_cpu_seconds += float(cpu_seconds)

    def record_optimization(self, *, gpu_hours: float, steps: int) -> None:
        if gpu_hours < 0 or steps < 0:
            raise ValueError("optimization costs must be nonnegative")
        self.optimization_gpu_hours += float(gpu_hours)
        self.optimizer_steps += int(steps)

    def record_outcome(self, outcome: SandboxOutcome) -> None:
        if outcome is SandboxOutcome.INFRASTRUCTURE_FAILURE:
            self.infrastructure_failures += 1

    def compare(self, other: "RunLedger", *, token_tolerance: float, cpu_tolerance: float) -> BudgetComparison:
        def drift(left: float, right: float) -> float:
            return abs(left - right) / max(abs(left), 1e-12)
        token_drift = drift(self.generation_tokens, other.generation_tokens)
        cpu_drift = drift(self.verification_cpu_seconds, other.verification_cpu_seconds)
        programs = other.programs - self.programs
        steps = other.optimizer_steps - self.optimizer_steps
        return BudgetComparison(token_drift <= token_tolerance + 1e-12 and cpu_drift <= cpu_tolerance + 1e-12 and programs == 0 and steps == 0, token_drift, cpu_drift, programs, steps)
