"""Small statistical utilities shared by mechanism and model-backed runs."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterable, Mapping
from math import comb


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator else 0.0


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    task_count: int
    resamples: int


def paired_bootstrap(rows: Iterable[Mapping[str, object]], seed: int, *, resamples: int = 10_000, confidence: float = 0.95) -> BootstrapInterval:
    """Hierarchically collapse repeated rows to task means, then resample tasks."""

    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap controls")
    grouped: dict[str, list[float]] = {}
    seeded: dict[str, dict[int, list[float]]] = {}
    has_seed = False
    for row in rows:
        try:
            delta = float(row["treatment"]) - float(row["baseline"])
            grouped.setdefault(str(row["task_id"]), []).append(delta)
            if "seed" in row:
                has_seed = True
                seeded.setdefault(str(row["task_id"]), {}).setdefault(int(row["seed"]), []).append(delta)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("bootstrap rows require task_id, treatment, and baseline") from exc
    if not grouped:
        raise ValueError("bootstrap requires at least one task")
    task_deltas = np.asarray([np.mean(grouped[task]) for task in sorted(grouped)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    if has_seed:
        tasks = sorted(grouped)
        seeds = sorted({item for values in seeded.values() for item in values})
        if any(set(seeded.get(task, {})) != set(seeds) for task in tasks):
            raise ValueError("seeded bootstrap requires a complete task by seed grid")
        matrix = np.asarray([[np.mean(seeded[task][item]) for item in seeds] for task in tasks])
        draws = np.empty(resamples)
        for index in range(resamples):
            task_indices = rng.integers(0, len(tasks), size=len(tasks))
            # One shared seed-resampling vector preserves seed-correlated effects.
            seed_indices = rng.integers(0, len(seeds), size=len(seeds))
            draws[index] = float(matrix[task_indices][:, seed_indices].mean())
    else:
        draws = task_deltas[rng.integers(0, len(task_deltas), size=(resamples, len(task_deltas)))].mean(axis=1)
    alpha = (1 - confidence) / 2
    return BootstrapInterval(float(task_deltas.mean()), float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha)), len(task_deltas), int(resamples))


def _pass_at_k(outcomes: list[bool], k: int) -> float:
    n = len(outcomes)
    c = sum(outcomes)
    if n < k:
        raise ValueError(f"Pass@{k} requires at least {k} candidates per task/seed")
    return 1.0 if n - c < k else float(1.0 - comb(n - c, k) / comb(n, k))


def evaluation_summary(rows: Iterable[Mapping[str, object]]) -> dict[str, float | int]:
    """Compute task×seed Pass@k plus task-cluster ICC and effective N."""

    grouped: dict[tuple[str, int], list[tuple[int, bool]]] = {}
    for row in rows:
        try:
            key = (str(row["task_id"]), int(row["seed"]))
            grouped.setdefault(key, []).append((int(row["candidate"]), bool(row["passed"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("evaluation rows require task_id, seed, candidate, and passed") from exc
    if not grouped:
        raise ValueError("evaluation summary requires rows")
    values: dict[tuple[str, int], tuple[float, float]] = {}
    for key, candidates in grouped.items():
        ordered = [passed for _, passed in sorted(candidates)]
        values[key] = (_pass_at_k(ordered, 1), _pass_at_k(ordered, 8))
    tasks = sorted({key[0] for key in values})
    seeds = sorted({key[1] for key in values})
    if set(values) != {(task, seed) for task in tasks for seed in seeds}:
        raise ValueError("evaluation requires a complete task by seed grid")
    pass8 = np.asarray([[values[(task, seed)][1] for seed in seeds] for task in tasks])
    grand = float(pass8.mean())
    between = len(seeds) * float(np.var(pass8.mean(axis=1), ddof=1)) if len(tasks) > 1 else 0.0
    within = float(np.mean(np.var(pass8, axis=1, ddof=1))) if len(seeds) > 1 else 0.0
    icc = max(0.0, min(1.0, (between - within) / max(between + (len(seeds) - 1) * within, 1e-12)))
    total = len(tasks) * len(seeds)
    ess = total / (1.0 + (len(seeds) - 1) * icc)
    return {
        "task_count": len(tasks), "seed_count": len(seeds),
        "pass_at_1": float(np.mean([value[0] for value in values.values()])),
        "pass_at_8": grand, "icc": icc, "effective_sample_size": float(ess),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down family-wise adjusted p-values."""

    if not p_values or any(not 0 <= float(value) <= 1 for value in p_values.values()):
        raise ValueError("Holm correction requires named p-values in [0, 1]")
    ordered = sorted(((name, float(value)) for name, value in p_values.items()), key=lambda item: (item[1], item[0]))
    result: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        result[name] = running
    return result
