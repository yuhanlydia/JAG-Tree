"""Exact bitmask-ordered subset distributions and audit requests."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable

import numpy as np


def enumerate_subsets(candidates: int) -> np.ndarray:
    """Return all boolean subsets in integer order, with candidate zero as the LSB."""

    if not isinstance(candidates, (int, np.integer)) or candidates < 1:
        raise ValueError("candidates must be a positive integer")
    if candidates > 20:
        raise ValueError("exact enumeration is limited to at most 20 candidates")
    masks = np.arange(1 << int(candidates), dtype=np.uint64)[:, None]
    bits = np.arange(int(candidates), dtype=np.uint64)[None, :]
    return ((masks >> bits) & 1).astype(bool)


def inclusion_probabilities(
    probabilities: np.ndarray | Iterable[float],
    subsets: np.ndarray | Iterable[Iterable[bool]],
    *,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute and validate exact first- and second-order inclusion probabilities."""

    p = np.asarray(probabilities, dtype=np.float64)
    x = np.asarray(subsets, dtype=bool)
    if p.ndim != 1 or x.ndim != 2 or len(p) != len(x) or x.shape[1] < 1:
        raise ValueError("probabilities and subsets have incompatible shapes")
    if not np.isfinite(p).all() or (p < 0.0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isclose(p.sum(), 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("probabilities must sum to one")
    weighted = p[:, None] * x
    pi = x.T @ p
    pi2 = x.T @ weighted
    if not np.isfinite(pi).all() or not np.isfinite(pi2).all():
        raise ValueError("inclusion probabilities must be finite")
    if not np.allclose(pi2, pi2.T, atol=tolerance, rtol=0.0):
        raise ValueError("second-order inclusion probabilities must be symmetric")
    if not np.allclose(np.diag(pi2), pi, atol=tolerance, rtol=0.0):
        raise ValueError("the second-order diagonal must equal first-order inclusion")
    lower = np.maximum(0.0, pi[:, None] + pi[None, :] - 1.0)
    upper = np.minimum(pi[:, None], pi[None, :])
    if (pi2 < lower - tolerance).any() or (pi2 > upper + tolerance).any():
        raise ValueError("second-order inclusion probabilities violate Frechet bounds")
    return np.asarray(pi, dtype=np.float64), np.asarray(pi2, dtype=np.float64)


def _probability_hash(probabilities: np.ndarray) -> str:
    canonical = np.asarray(probabilities, dtype="<f8")
    return sha256(canonical.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class SubsetDesign:
    """An exact subset distribution and its precomputed audit quantities."""

    probabilities: np.ndarray
    pi: np.ndarray
    pi2: np.ndarray
    costs: np.ndarray
    expected_cost: float
    design_hash: str
    logits: np.ndarray | None = None
    restart: int | None = None
    full_audit: bool = False

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=np.float64).copy()
        pi = np.asarray(self.pi, dtype=np.float64).copy()
        pi2 = np.asarray(self.pi2, dtype=np.float64).copy()
        costs = np.asarray(self.costs, dtype=np.float64).copy()
        logits = None if self.logits is None else np.asarray(self.logits, dtype=np.float64).copy()
        if probabilities.ndim != 1 or probabilities.size < 2 or probabilities.size & (probabilities.size - 1):
            raise ValueError("probabilities must contain exactly 2^K entries")
        candidates = probabilities.size.bit_length() - 1
        computed_pi, computed_pi2 = inclusion_probabilities(probabilities, enumerate_subsets(candidates))
        if pi.shape != (candidates,) or pi2.shape != (candidates, candidates):
            raise ValueError("stored inclusion probabilities have incompatible shapes")
        if not np.allclose(pi, computed_pi, atol=1e-12, rtol=0.0) or not np.allclose(pi2, computed_pi2, atol=1e-12, rtol=0.0):
            raise ValueError("stored inclusion probabilities do not match the subset distribution")
        if costs.shape != (candidates,) or not np.isfinite(costs).all() or (costs <= 0.0).any():
            raise ValueError("stored costs must be a finite positive candidate vector")
        computed_cost = float(np.dot(computed_pi, costs))
        if not np.isfinite(self.expected_cost) or not np.isclose(self.expected_cost, computed_cost, atol=1e-12, rtol=1e-12):
            raise ValueError("expected cost does not match inclusion probabilities and costs")
        if self.design_hash != _probability_hash(probabilities):
            raise ValueError("design hash does not match the probability vector")
        if self.full_audit:
            if np.count_nonzero(probabilities) != 1 or probabilities[-1] != 1.0:
                raise ValueError("the support exemption is reserved for deterministic full audit")
        elif (probabilities <= 0.0).any():
            raise ValueError("randomized subset designs must have full support")
        if logits is not None and logits.shape != probabilities.shape:
            raise ValueError("logits must have one entry per subset")
        for value in (probabilities, pi, pi2, costs, logits):
            if value is not None:
                value.setflags(write=False)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "pi", pi)
        object.__setattr__(self, "pi2", pi2)
        object.__setattr__(self, "costs", costs)
        object.__setattr__(self, "logits", logits)

    @property
    def hash(self) -> str:
        """Alias used by persisted manifests."""

        return self.design_hash

    @classmethod
    def from_probabilities(
        cls,
        probabilities: np.ndarray | Iterable[float],
        costs: np.ndarray | Iterable[float],
        *,
        logits: np.ndarray | Iterable[float] | None = None,
        restart: int | None = None,
        full_audit: bool = False,
    ) -> "SubsetDesign":
        p = np.asarray(probabilities, dtype=np.float64)
        c = np.asarray(costs, dtype=np.float64)
        if c.ndim != 1 or c.size < 1 or not np.isfinite(c).all() or (c <= 0.0).any():
            raise ValueError("costs must be a non-empty finite positive vector")
        subsets = enumerate_subsets(c.size)
        pi, pi2 = inclusion_probabilities(p, subsets)
        expected_cost = float(np.dot(pi, c))
        return cls(p, pi, pi2, c, expected_cost, _probability_hash(p), None if logits is None else np.asarray(logits), restart, full_audit)


@dataclass(frozen=True)
class SampledAuditRequest:
    """A sampled request containing design metadata but no trusted outcomes."""

    subset_index: int
    mask: np.ndarray
    design_hash: str
    inclusion_probabilities: np.ndarray

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool).copy()
        pi = np.asarray(self.inclusion_probabilities, dtype=np.float64).copy()
        mask.setflags(write=False)
        pi.setflags(write=False)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "inclusion_probabilities", pi)

    @property
    def selected_indices(self) -> tuple[int, ...]:
        return tuple(map(int, np.flatnonzero(self.mask)))


def sample_subset(design: SubsetDesign, rng: np.random.Generator) -> SampledAuditRequest:
    """Sample one auditable request from a previously logged design."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    index = int(rng.choice(len(design.probabilities), p=design.probabilities))
    mask = enumerate_subsets(len(design.pi))[index]
    return SampledAuditRequest(index, mask, design.design_hash, design.pi)


def persist_sampled_request(request: SampledAuditRequest, trusted_values: np.ndarray | Iterable[float]) -> dict[str, Any]:
    """Build a persistence-safe record that reveals only requested trusted labels."""

    values = np.asarray(trusted_values, dtype=np.float64)
    if values.ndim != 1 or values.shape != request.mask.shape:
        raise ValueError("trusted values must match the request candidate dimension")
    selected = {str(index): float(values[index]) for index in request.selected_indices}
    return {
        "subset_index": request.subset_index,
        "selected_indices": list(request.selected_indices),
        "design_hash": request.design_hash,
        "inclusion_probabilities": request.inclusion_probabilities.tolist(),
        "selected_labels": selected,
    }
