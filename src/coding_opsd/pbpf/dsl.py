"""Small, immutable finite DSL used by the PBPF mechanism experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache
from hashlib import sha256
from typing import Iterable

import numpy as np


class Outcome(IntEnum):
    PASS = 0
    WRONG = 1
    EXCEPTION = 2
    TIMEOUT = 3


class Bug(IntEnum):
    CORRECT = 0
    PLUS_ONE = 1
    MINUS_ONE = 2
    BOUNDARY_FLIP = 3
    ARGUMENT_PERMUTATION = 4
    MISSING_BRANCH = 5
    INDEX_PLUS_ONE = 6
    EXCEPTION = 7
    # Descriptive aliases retain a compact stable eight-value kernel enum.
    ARGUMENT_ORDER_PERMUTATION = 4
    MISSING_BRANCH_FILTER = 5
    INDEX_SHIFT = 6
    EXCEPTION_INJECTION = 7

    @property
    def sites(self) -> tuple[int, ...]:
        # Stable finite edit sites.  A site denotes an input location where
        # applicable, or a named global edit (site zero) for global transforms.
        if self in (Bug.BOUNDARY_FLIP, Bug.EXCEPTION):
            return tuple(range(64))
        if self is Bug.MISSING_BRANCH:
            return (0, 1)  # even and odd partition respectively
        return (0,)


BUG_PRIOR = (.20, .12, .12, .12, .11, .11, .11, .11)


def _is_integral(value: object) -> bool:
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))


def _integer_tuple(values: Iterable[object], name: str) -> tuple[int, ...]:
    materialized = tuple(values)
    if any(not _is_integral(value) for value in materialized):
        raise ValueError(f"{name} must contain integer values, never booleans or floats")
    return tuple(int(value) for value in materialized)


def _integer_evidence(values: object) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 2:
        raise ValueError("prefix evidence must be a two-dimensional integer array")
    if raw.dtype == np.dtype(bool) or (raw.dtype.kind not in "iu" and any(not _is_integral(value) for value in raw.flat)):
        raise ValueError("prefix evidence must contain integer outcomes, never booleans or floats")
    evidence = raw.astype(int, copy=True)
    if (evidence < int(Outcome.PASS)).any() or (evidence > int(Outcome.TIMEOUT)).any():
        raise ValueError("prefix evidence contains an outcome outside the categorical alphabet")
    return evidence


@dataclass(frozen=True, eq=False)
class Program:
    """An immutable full truth table with explicit execution sentinels."""

    outputs: tuple[int | Outcome, ...]

    def __post_init__(self) -> None:
        if len(self.outputs) != 64:
            raise ValueError("programs must have exactly 64 outputs")
        if any(isinstance(value, Outcome) and value in (Outcome.PASS, Outcome.WRONG) for value in self.outputs):
            raise ValueError("PASS and WRONG are execution outcomes, not program-output sentinels")
        if any(not isinstance(value, Outcome) and not _is_integral(value) for value in self.outputs):
            raise ValueError("program outputs must be integers, never booleans or floats")
        values = tuple(value if isinstance(value, Outcome) else int(value) for value in self.outputs)
        if any(not isinstance(value, (int, Outcome)) or (not isinstance(value, Outcome) and (value < 0 or value >= 64)) for value in values):
            raise ValueError("outputs must be modulo-64 values or Outcome sentinels")
        object.__setattr__(self, "outputs", values)

    def __call__(self, x: int) -> int | Outcome:
        if not _is_integral(x) or not 0 <= int(x) < 64:
            raise ValueError("program inputs must be integer identifiers in 0..63")
        return self.outputs[int(x)]

    def _typed_outputs(self) -> tuple[tuple[type, int], ...]:
        return tuple((type(value), int(value)) for value in self.outputs)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Program) and self._typed_outputs() == other._typed_outputs()

    def __hash__(self) -> int:
        return hash(self._typed_outputs())


def canonical_programs() -> tuple[Program, ...]:
    """Return the 64 canonical (a*x+b) mod 64 truth tables."""

    multipliers = (1, 3, 5, 7, 9, 11, 13, 15)
    offsets = tuple(range(8))
    return tuple(Program(tuple((a * x + b) % 64 for x in range(64))) for a in multipliers for b in offsets)


CANONICAL_PROGRAMS = canonical_programs()
PROGRAMS = CANONICAL_PROGRAMS
HYPOTHESES = CANONICAL_PROGRAMS


def mutate(program: Program, bug: Bug | int, site: int) -> Program:
    """Apply one deterministic finite mutation kernel."""

    bug = Bug(bug)
    if site not in bug.sites:
        raise ValueError(f"invalid site {site} for {bug.name}")
    values = list(program.outputs)
    numeric = lambda value: int(value)
    shifted = lambda value, amount: value if isinstance(value, Outcome) else (numeric(value) + amount) % 64
    if bug is Bug.PLUS_ONE:
        values = [shifted(value, 1) for value in values]
    elif bug is Bug.MINUS_ONE:
        values = [shifted(value, -1) for value in values]
    elif bug is Bug.BOUNDARY_FLIP:
        values[site] = shifted(values[site], 1)
    elif bug is Bug.ARGUMENT_PERMUTATION:
        values = [values[63 - x] for x in range(64)]
    elif bug is Bug.MISSING_BRANCH:
        values = [0 if x % 2 == site else value for x, value in enumerate(values)]
    elif bug is Bug.INDEX_PLUS_ONE:
        values = [values[(x + 1) % 64] for x in range(64)]
    elif bug is Bug.EXCEPTION:
        values[site] = Outcome.EXCEPTION
    return Program(tuple(values))


@lru_cache(maxsize=None)
def inverse_mutations(source: Program, candidate: Program) -> tuple[tuple[Bug, int], ...]:
    """Enumerate every (bug, site) whose deterministic mutation matches."""

    return tuple((bug, site) for bug in Bug for site in bug.sites if mutate(source, bug, site) == candidate)


def deterministic_test_order(episode_id: str, inputs: Iterable[int], order_seed: int) -> tuple[int, ...]:
    """Stable test permutation independent of model/candidate RNG streams."""

    identifiers = _integer_tuple(inputs, "test inputs")
    if not _is_integral(order_seed):
        raise ValueError("order seed must be an integer, never a boolean or float")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("test inputs must be unique")
    return tuple(sorted(identifiers, key=lambda value: sha256(f"{episode_id}|{value}|{int(order_seed)}".encode()).digest()))


def outcome_for(hypothesis: Program, candidate: Program, input_id: int) -> Outcome:
    value = candidate(input_id)
    if isinstance(value, Outcome) and value is Outcome.EXCEPTION:
        return Outcome.EXCEPTION
    if isinstance(value, Outcome) and value is Outcome.TIMEOUT:
        return Outcome.TIMEOUT
    return Outcome.PASS if value == hypothesis(input_id) else Outcome.WRONG


@dataclass(frozen=True)
class PrefixView:
    """Trainer-visible evidence; deliberately contains no grader state."""

    episode_id: str
    candidate_programs: tuple[Program, ...]
    inputs: tuple[int, ...]
    test_order: tuple[int, ...]
    observed_outcomes: np.ndarray

    def __post_init__(self) -> None:
        evidence = _integer_evidence(self.observed_outcomes)
        if evidence.ndim != 2 or evidence.shape[0] != len(self.candidate_programs) or evidence.shape[1] > len(self.test_order):
            raise ValueError("prefix evidence has incompatible shape")
        _integer_tuple(self.inputs, "prefix inputs")
        _integer_tuple(self.test_order, "prefix test order")
        evidence.setflags(write=False)
        object.__setattr__(self, "observed_outcomes", evidence)

    @property
    def observed_count(self) -> int:
        """Number of leading manifest outcomes revealed to the trainer."""

        return int(self.observed_outcomes.shape[1])


@dataclass(frozen=True)
class Episode:
    """Episode with separated trainer-visible candidates and grader-only truth."""

    episode_id: str
    true_h: int
    candidate_programs: tuple[Program, ...]
    inputs: tuple[int, ...] = tuple(range(64))
    order_seed: int = 0
    test_order: tuple[int, ...] = field(init=False)
    outcome_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not _is_integral(self.true_h) or not 0 <= int(self.true_h) < len(CANONICAL_PROGRAMS):
            raise ValueError("true_h must identify a canonical hypothesis")
        candidates = tuple(self.candidate_programs)
        if not candidates:
            raise ValueError("episodes require at least one candidate")
        inputs = _integer_tuple(self.inputs, "inputs")
        if not inputs or any(value < 0 or value >= 64 for value in inputs):
            raise ValueError("inputs must be non-empty identifiers in 0..63")
        if not _is_integral(self.order_seed):
            raise ValueError("order seed must be an integer, never a boolean or float")
        order = deterministic_test_order(self.episode_id, inputs, self.order_seed)
        matrix = np.asarray([[outcome_for(CANONICAL_PROGRAMS[self.true_h], candidate, value) for value in order] for candidate in candidates], dtype=int)
        matrix.setflags(write=False)
        object.__setattr__(self, "candidate_programs", candidates)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "test_order", order)
        object.__setattr__(self, "outcome_matrix", matrix)

    @classmethod
    def from_candidates(cls, episode_id: str, true_h: int, candidates: Iterable[Program], *, inputs: Iterable[int] = range(64), order_seed: int = 0) -> "Episode":
        return cls(episode_id, true_h, tuple(candidates), tuple(inputs), order_seed)

    def prefix_view(self, t: int, outcomes: np.ndarray | None = None) -> PrefixView:
        if not _is_integral(t) or not 0 <= int(t) <= len(self.test_order):
            raise ValueError("prefix length is outside the test order")
        visible = self.outcome_matrix[:, :t] if outcomes is None else outcomes
        if visible.shape != (len(self.candidate_programs), int(t)):
            raise ValueError("supplied prefix outcomes must contain exactly the revealed leading columns")
        return PrefixView(self.episode_id, self.candidate_programs, self.inputs, self.test_order, visible)

    def outcome_at(self, h_id: int, candidate: Program, position: int) -> Outcome:
        if not 0 <= int(position) < len(self.test_order):
            raise ValueError("test position is outside episode order")
        return outcome_for(CANONICAL_PROGRAMS[int(h_id)], candidate, self.test_order[int(position)])
