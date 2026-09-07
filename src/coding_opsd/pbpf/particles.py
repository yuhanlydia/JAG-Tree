"""Deterministic bootstrap-style particle approximation for finite PBPF."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ..metrics import effective_sample_size
from ..runtime import named_rng
from .dsl import Episode, Outcome
from .posterior import candidate_likelihood, compatible_explanations, predictive


class ParticleCollapseError(RuntimeError):
    """Raised when finite particles assign zero mass to observed evidence."""


@dataclass(frozen=True)
class ParticleDiagnostic:
    """One post-observation SMC event, retaining pre-resampling degeneracy."""

    position: int
    pre_resampling_ess: float
    resampled: bool
    unique_ancestor_ratio: float | None
    post_rejuvenation_unique_state_ratio: float | None


def systematic_resample(weights: np.ndarray | Iterable[float], *, seed: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return deterministic systematic-resampling ancestor indices."""

    if (seed is None) == (rng is None):
        raise ValueError("provide exactly one of seed or rng")
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or not len(values) or (values < 0).any() or not np.isfinite(values).all() or values.sum() <= 0:
        raise ValueError("weights must be a finite non-negative non-empty vector")
    values = values / values.sum()
    generator = named_rng(int(seed), "pbpf_systematic_resample") if rng is None else rng
    locations = (generator.random() + np.arange(len(values))) / len(values)
    return np.searchsorted(np.cumsum(values), locations, side="right")


@dataclass(frozen=True)
class ParticleState:
    h: int
    bugs: tuple[int, ...]
    sites: tuple[int, ...]


@dataclass
class ParticleFilter:
    episode: Episode
    states: list[ParticleState]
    log_weights: np.ndarray
    rng: np.random.Generator
    observed: dict[int, np.ndarray] = field(default_factory=dict)
    ess_history: list[float] = field(default_factory=list)
    diagnostic_history: list[ParticleDiagnostic] = field(default_factory=list)
    resampling_count: int = 0

    @classmethod
    def initialize(cls, episode: Episode, *, particles: int = 8, seed: int = 0) -> "ParticleFilter":
        if particles < 1:
            raise ValueError("particles must be positive")
        rng = named_rng(seed, f"pbpf_particles:{episode.episode_id}")
        compatible = [h for h in range(64) if all(candidate_likelihood(h, candidate) > 0 for candidate in episode.candidate_programs)]
        if not compatible:
            raise ValueError("candidate tables are outside the finite mutation model")
        choices = rng.choice(compatible, size=particles, replace=True)
        states = [cls._draw_state(episode, int(h), rng) for h in choices]
        # Uniform proposal over every H compatible with the fully observed
        # candidate tables.  Candidate likelihood is then importance weighted.
        proposal = 1.0 / len(compatible)
        log_weights = np.asarray([
            sum(np.log(candidate_likelihood(state.h, candidate)) for candidate in episode.candidate_programs) - np.log(64.0) - np.log(proposal)
            for state in states
        ], dtype=float)
        filter_ = cls(episode, states, log_weights, rng)
        filter_._normalize_log_weights()
        return filter_

    @staticmethod
    def _draw_state(episode: Episode, h: int, rng: np.random.Generator) -> ParticleState:
        bugs: list[int] = []
        sites: list[int] = []
        for candidate in episode.candidate_programs:
            explanations = compatible_explanations(h, candidate)
            probabilities = np.asarray([item[2] for item in explanations], dtype=float)
            probabilities /= probabilities.sum()
            bug, site, _ = explanations[int(rng.choice(len(explanations), p=probabilities))]
            bugs.append(int(bug))
            sites.append(int(site))
        return ParticleState(int(h), tuple(bugs), tuple(sites))

    @property
    def weights(self) -> np.ndarray:
        return np.exp(self.log_weights)

    @property
    def ess(self) -> float:
        return effective_sample_size(self.weights)

    def _normalize_log_weights(self) -> None:
        maximum = float(np.max(self.log_weights))
        if not np.isfinite(maximum):
            raise ValueError("all particle weights are zero")
        normalized = self.log_weights - maximum
        self.log_weights = normalized - np.log(np.exp(normalized).sum())

    def _consistent(self, h: int) -> bool:
        return all(
            all(int(observed[g]) == int(self.episode.outcome_at(h, candidate, position)) for g, candidate in enumerate(self.episode.candidate_programs))
            for position, observed in self.observed.items()
        )

    def _target_log_mass(self, h: int) -> float:
        if not self._consistent(h):
            return -np.inf
        likelihoods = [candidate_likelihood(h, candidate) for candidate in self.episode.candidate_programs]
        return -np.log(64.0) + sum(np.log(likelihood) for likelihood in likelihoods) if all(likelihoods) else -np.inf

    def _resample_and_rejuvenate(self) -> tuple[float, float]:
        ancestors = systematic_resample(self.weights, rng=self.rng)
        self.states = [self.states[int(index)] for index in ancestors]
        self.log_weights = np.full(len(self.states), -np.log(len(self.states)))
        self.resampling_count += 1
        proposal_h = [h for h in range(64) if all(candidate_likelihood(h, candidate) > 0 for candidate in self.episode.candidate_programs)]
        for index, state in enumerate(self.states):
            proposed_h = int(self.rng.choice(proposal_h))
            current_log = self._target_log_mass(state.h)
            proposal_log = self._target_log_mass(proposed_h)
            if np.log(self.rng.random()) < min(0.0, proposal_log - current_log):
                state = self._draw_state(self.episode, proposed_h, self.rng)
            else:
                # Explanation is conditionally redrawn even when H is retained.
                state = self._draw_state(self.episode, state.h, self.rng)
            self.states[index] = state
        unique_ancestor_ratio = len(set(int(index) for index in ancestors)) / len(ancestors)
        unique_state_ratio = len({(state.h, state.bugs, state.sites) for state in self.states}) / len(self.states)
        return float(unique_ancestor_ratio), float(unique_state_ratio)

    def observe(self, position: int, outcomes: np.ndarray | Iterable[int]) -> None:
        """Consume exactly one manifest outcome column and update in log space."""

        if isinstance(position, (bool, np.bool_)) or not isinstance(position, (int, np.integer)):
            raise ValueError("position must be an integer manifest index, never a boolean or float")
        position = int(position)
        if position != len(self.observed) or not 0 <= position < len(self.episode.test_order):
            raise ValueError("observations must consume the next manifest test in order")
        raw_evidence = np.asarray(outcomes)
        if raw_evidence.dtype == np.dtype(bool) or (raw_evidence.dtype.kind not in "iu" and any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) for value in raw_evidence.flat)):
            raise ValueError("outcomes must contain integer categorical values, never booleans or floats")
        evidence = raw_evidence.astype(int, copy=True)
        if evidence.shape != (len(self.episode.candidate_programs),):
            raise ValueError("outcomes must provide one categorical value per candidate")
        if (evidence < int(Outcome.PASS)).any() or (evidence > int(Outcome.TIMEOUT)).any():
            raise ValueError("outcomes are outside the categorical alphabet")
        self.observed[position] = evidence.copy()
        increments = np.asarray([0.0 if self._consistent(state.h) else -np.inf for state in self.states])
        self.log_weights = self.log_weights + increments
        if not np.isfinite(self.log_weights).any():
            self.ess_history.append(0.0)
            self.diagnostic_history.append(ParticleDiagnostic(position, 0.0, False, None, None))
            raise ParticleCollapseError("all particles assign zero probability to the next manifest outcome")
        else:
            self._normalize_log_weights()
        pre_resampling_ess = self.ess
        self.ess_history.append(pre_resampling_ess)
        if pre_resampling_ess < len(self.states) / 2:
            ancestor_ratio, unique_state_ratio = self._resample_and_rejuvenate()
            self.diagnostic_history.append(ParticleDiagnostic(position, pre_resampling_ess, True, ancestor_ratio, unique_state_ratio))
        else:
            self.diagnostic_history.append(ParticleDiagnostic(position, pre_resampling_ess, False, None, None))

    def h_marginal(self) -> np.ndarray:
        """Empirical particle H marginal without diagnostic smoothing."""

        probabilities = np.zeros(64, dtype=float)
        for state, weight in zip(self.states, self.weights):
            probabilities[state.h] += weight
        return probabilities

    def full_support_h_marginal(self, floor: float = 1e-8) -> np.ndarray:
        """Diagnostic-only full-support H distribution for posterior KL."""

        if floor <= 0 or not np.isfinite(floor):
            raise ValueError("floor must be finite and positive")
        probabilities = np.full(64, floor, dtype=float)
        for state, weight in zip(self.states, self.weights):
            probabilities[state.h] += weight
        return probabilities / probabilities.sum()

    def predict(self, positions: Iterable[int], floor: float = 1e-8) -> np.ndarray:
        return predictive(self.h_marginal(), self.episode, positions, floor)

    @property
    def particles_per_correct_equivalence_class(self) -> int:
        """Count particles in the true canonical-H (unique truth-table) class."""

        return sum(state.h == self.episode.true_h for state in self.states)
