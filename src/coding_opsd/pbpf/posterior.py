"""Exact finite inference operators for PBPF."""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from typing import Iterable

import numpy as np

from .dsl import BUG_PRIOR, CANONICAL_PROGRAMS, Bug, Episode, Outcome, PrefixView, mutate


class ImpossibleEvidenceError(ValueError):
    """Raised when no finite latent state can produce observed evidence."""


@lru_cache(maxsize=1)
def _mutation_index() -> dict[object, dict[int, tuple[tuple[Bug, int, float], ...]]]:
    """Index the small finite kernel once instead of reinverting every table."""

    mutable: dict[object, dict[int, list[tuple[Bug, int, float]]]] = {}
    for h_id, source in enumerate(CANONICAL_PROGRAMS):
        for bug in Bug:
            for site in bug.sites:
                candidate = mutate(source, bug, site)
                mutable.setdefault(candidate, {}).setdefault(h_id, []).append(
                    (bug, site, float(BUG_PRIOR[int(bug)] / len(bug.sites)))
                )
    return {
        candidate: {
            h_id: tuple(explanations)
            for h_id, explanations in by_h.items()
        }
        for candidate, by_h in mutable.items()
    }


@lru_cache(maxsize=None)
def compatible_explanations(h_id: int, candidate) -> tuple[tuple[Bug, int, float], ...]:
    """Return compatible bug/site explanations and their prior probabilities."""

    return _mutation_index().get(candidate, {}).get(int(h_id), ())


@lru_cache(maxsize=None)
def candidate_explanations(
    h_id: int,
    candidate,
    candidate_source_contamination: float = 0.0,
) -> tuple[tuple[int, Bug, int, float], ...]:
    """Marginalize a same-family semantic source and every bug/edit site."""

    contamination = float(candidate_source_contamination)
    if not np.isfinite(contamination) or not 0.0 <= contamination <= 1.0:
        raise ValueError("candidate source contamination must be finite and in [0, 1]")
    h_id = int(h_id)
    family_start = (h_id // 8) * 8
    explanations: list[tuple[int, Bug, int, float]] = []
    for source_h in range(family_start, family_start + 8):
        source_mass = contamination / 8.0
        if source_h == h_id:
            source_mass += 1.0 - contamination
        if source_mass == 0.0:
            continue
        explanations.extend(
            (source_h, bug, site, source_mass * weight)
            for bug, site, weight in compatible_explanations(source_h, candidate)
        )
    return tuple(explanations)


@lru_cache(maxsize=None)
def candidate_likelihood(
    h_id: int,
    candidate,
    candidate_source_contamination: float = 0.0,
) -> float:
    """p(candidate | H), marginalizing source ambiguity, bug, and edit site."""

    return float(
        sum(
            weight
            for _, _, _, weight in candidate_explanations(
                h_id, candidate, candidate_source_contamination
            )
        )
    )


def validate_prefix(episode: Episode, prefix: PrefixView) -> None:
    """Reject evidence not drawn from this episode's full immutable manifest."""

    if prefix.episode_id != episode.episode_id:
        raise ValueError("prefix belongs to a different episode")
    if prefix.test_order != episode.test_order:
        raise ValueError("prefix does not retain this episode's immutable full test order")
    if prefix.candidate_programs != episode.candidate_programs:
        raise ValueError("prefix candidates do not match episode")
    if prefix.inputs != episode.inputs:
        raise ValueError("prefix inputs do not match episode")


def _evidence_possible(episode: Episode, prefix: PrefixView, h_id: int) -> bool:
    for g, candidate in enumerate(episode.candidate_programs):
        for position, observed in enumerate(prefix.observed_outcomes[g]):
            if int(observed) != int(episode.outcome_at(h_id, candidate, position)):
                return False
    return True


def _normalize(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise ImpossibleEvidenceError("candidate programs and observed evidence are inconsistent with all hypotheses")
    return values / total


def exact_posterior(episode: Episode, prefix: PrefixView | int) -> np.ndarray:
    """Compute p(H | candidate tables, fixed-prefix categorical evidence)."""

    if isinstance(prefix, int):
        prefix = episode.prefix_view(prefix)
    validate_prefix(episode, prefix)
    probabilities = np.zeros(len(CANONICAL_PROGRAMS), dtype=float)
    for h_id in range(len(CANONICAL_PROGRAMS)):
        candidate_mass = float(np.prod([candidate_likelihood(h_id, candidate, episode.candidate_source_contamination) for candidate in episode.candidate_programs]))
        probabilities[h_id] = candidate_mass if candidate_mass and _evidence_possible(episode, prefix, h_id) else 0.0
    return _normalize(probabilities)


def sequential_update(episode: Episode, prefix: PrefixView | int) -> np.ndarray:
    """Bayes-update the candidate-conditioned finite prior one test at a time."""

    if isinstance(prefix, int):
        prefix = episode.prefix_view(prefix)
    validate_prefix(episode, prefix)
    probability = np.asarray([np.prod([candidate_likelihood(h, candidate, episode.candidate_source_contamination) for candidate in episode.candidate_programs]) for h in range(64)], dtype=float)
    probability = _normalize(probability)
    for position in range(prefix.observed_outcomes.shape[1]):
        keep = np.asarray([
            all(int(prefix.observed_outcomes[g, position]) == int(episode.outcome_at(h, candidate, position)) for g, candidate in enumerate(episode.candidate_programs))
            for h in range(64)
        ], dtype=float)
        probability = _normalize(probability * keep)
    return probability


def brute_force_h_marginal(episode: Episode, prefix: PrefixView | int) -> np.ndarray:
    """Audit operator: explicitly enumerate candidate explanation products.

    This is intentionally exponential in the candidate count and intended only
    for tiny fixtures.  ``exact_posterior`` uses the equivalent factorization.
    """

    if isinstance(prefix, int):
        prefix = episode.prefix_view(prefix)
    validate_prefix(episode, prefix)
    masses = np.zeros(64, dtype=float)
    for h_id in range(64):
        explanation_sets = [candidate_explanations(h_id, candidate, episode.candidate_source_contamination) for candidate in episode.candidate_programs]
        if not explanation_sets or any(not explanations for explanations in explanation_sets):
            continue
        if not _evidence_possible(episode, prefix, h_id):
            continue
        masses[h_id] = sum(float(np.prod([explanation[3] for explanation in state])) for state in product(*explanation_sets))
    return _normalize(masses)


def predictive(h_prob: np.ndarray | Iterable[float], episode: Episode, positions: Iterable[int], floor: float = 1e-8) -> np.ndarray:
    """Return marginal outcome probabilities shaped ``[candidate,test,4]``."""

    probability = np.asarray(h_prob, dtype=float)
    if probability.shape != (64,) or not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("h_prob must be a finite non-negative 64-vector")
    probability = _normalize(probability.copy())
    requested = tuple(int(position) for position in positions)
    if not requested or any(position < 0 or position >= len(episode.test_order) for position in requested):
        raise ValueError("positions must be non-empty positions in the fixed order")
    result = np.zeros((len(episode.candidate_programs), len(requested), len(Outcome)), dtype=float)
    for h_id, mass in enumerate(probability):
        if mass == 0:
            continue
        for g, candidate in enumerate(episode.candidate_programs):
            for t, position in enumerate(requested):
                result[g, t, int(episode.outcome_at(h_id, candidate, position))] += mass
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("floor must be finite and positive")
    result = np.maximum(result, float(floor))
    return result / result.sum(axis=-1, keepdims=True)


def exact_map(h_prob: np.ndarray | Iterable[float]) -> int:
    """Marginal-H MAP, breaking ties by canonical hypothesis identifier."""

    probability = np.asarray(h_prob, dtype=float)
    if probability.shape != (64,) or not np.isfinite(probability).all() or (probability < 0).any() or not np.isclose(probability.sum(), 1.0, atol=1e-12):
        raise ValueError("h_prob must be a normalized finite non-negative 64-vector")
    return int(np.flatnonzero(probability == probability.max())[0])


def randomized_hpd(prob: np.ndarray | Iterable[float], mass: float = .9) -> np.ndarray:
    """Return randomized HPD inclusion probabilities with exact expected mass."""

    probability = np.asarray(prob, dtype=float)
    if probability.ndim != 1 or probability.size == 0 or (probability < 0).any() or not np.isfinite(probability).all():
        raise ValueError("probability must be a non-empty finite non-negative vector")
    if not 0 < mass <= 1:
        raise ValueError("mass must be in (0, 1]")
    probability = _normalize(probability.copy())
    order = sorted(range(probability.size), key=lambda index: (-probability[index], index))
    included = np.zeros_like(probability)
    remaining = float(mass)
    for index in order:
        if remaining <= 1e-15:
            break
        take = min(1.0, remaining / probability[index])
        included[index] = take
        remaining -= probability[index] * take
    return included
