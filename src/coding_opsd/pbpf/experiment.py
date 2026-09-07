"""CPU-only finite PBPF Phase-0 experiment runner."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..metrics import brier_score, categorical_nll, paired_bootstrap_ci
from ..runtime import named_rng
from .dsl import BUG_PRIOR, Bug, Episode, canonical_programs, mutate
from .particles import ParticleCollapseError, ParticleFilter
from .posterior import exact_map, exact_posterior, predictive, randomized_hpd


_REGISTERED_FORMAL_TEST_EPISODES = 5_000


def _episode(
    seed: int,
    index: int,
    candidates: int,
    order_seed: int,
    candidate_source_contamination: float = 0.0,
) -> Episode:
    rng = named_rng(seed, f"pbpf_episode:{index}")
    h = int(rng.integers(64))
    programs_by_h = canonical_programs()
    programs = []
    for _ in range(candidates):
        source_h = h
        if candidate_source_contamination > 0.0 and rng.random() < candidate_source_contamination:
            source_h = (h // 8) * 8 + int(rng.integers(8))
        bug = Bug(int(rng.choice(8, p=BUG_PRIOR)))
        programs.append(mutate(programs_by_h[source_h], bug, int(rng.choice(bug.sites))))
    return Episode.from_candidates(
        f"phase0-{index}",
        h,
        programs,
        order_seed=order_seed,
        candidate_source_contamination=candidate_source_contamination,
    )


def _scores(probabilities: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    return categorical_nll(probabilities.reshape(-1, 4), targets.reshape(-1)), brier_score(probabilities.reshape(-1, 4), targets.reshape(-1))


def _resampling_events(filter_: ParticleFilter) -> list[dict[str, float | int | None]]:
    """Serialize every actual resampling event, including before a later collapse."""

    return [
        {"position": event.position, "pre_resampling_ess": event.pre_resampling_ess, "unique_ancestor_ratio": event.unique_ancestor_ratio, "post_rejuvenation_unique_state_ratio": event.post_rejuvenation_unique_state_ratio}
        for event in filter_.diagnostic_history if event.resampled
    ]


def _horizon_positions(prefix: int, total: int, horizon: int | str) -> tuple[int, ...]:
    if horizon == "all_remaining":
        return tuple(range(prefix, total))
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("forecast horizons must be positive integers or 'all_remaining'")
    return tuple(range(prefix, min(total, prefix + horizon)))


def _paired_differences(reference: Mapping[int, float], comparator: Mapping[int, float]) -> list[float]:
    return [float(comparator[index] - reference[index]) for index in sorted(reference.keys() & comparator.keys())]


def _bootstrap_seed(outer_seed: int, label: str) -> int:
    """Derive a stable, independent paired-bootstrap stream from the run seed."""

    return int(named_rng(outer_seed, f"pbpf_bootstrap:{label}").integers(0, np.iinfo(np.int64).max))


def _group_gate_inputs(values: Mapping[str, Any], *, outer_seed: int, label: str = "gate", bootstrap_resamples: int = 10_000) -> dict[str, Any]:
    exact: Mapping[int, float] = values["exact"]
    prior: Mapping[int, float] = values["prior"]
    map_: Mapping[int, float] = values["map"]
    particle16: Mapping[int, float] = values["particle_16"]
    exact_prior = _paired_differences(exact, prior)
    exact_map = _paired_differences(exact, map_)
    exact_p16 = _paired_differences(exact, particle16)
    common_p16 = sorted(exact.keys() & map_.keys() & particle16.keys())
    aligned_map = [float(map_[index] - exact[index]) for index in common_p16]
    aligned_p16 = [float(particle16[index] - exact[index]) for index in common_p16]
    exact_prior_gap = float(np.mean(exact_prior)) if exact_prior else None
    map_gap = float(np.mean(exact_map)) if exact_map else None
    p16_gap = float(np.mean(exact_p16)) if exact_p16 else None
    aligned_map_gap = float(np.mean(aligned_map)) if aligned_map else None
    aligned_p16_gap = float(np.mean(aligned_p16)) if aligned_p16 else None
    hpd = list(values["hpd"].values())
    expected_ids = set(values["expected_episode_ids"])
    p16_collapse_ids = set(values["p16_collapse_ids"])
    p16_missing_ids = expected_ids - set(particle16)
    def interval(differences: list[float], metric: str, *, one_sided: bool = False) -> list[float] | None:
        if not differences or bootstrap_resamples <= 0:
            return None
        lower, upper = paired_bootstrap_ci(differences, np.zeros(len(differences)), seed=_bootstrap_seed(outer_seed, f"{label}:{metric}"), resamples=bootstrap_resamples, confidence_level=.9 if one_sided else .95)
        return [lower, upper]
    p16_one_sided_ci = interval(exact_p16, "p16_to_exact", one_sided=True)
    return {
        "exact_vs_prior_nll_delta": exact_prior_gap,
        "exact_mixture_vs_map_nll_delta": map_gap,
        "p16_to_exact_nll_gap": p16_gap,
        "map_gap_fraction_closed": None if aligned_map_gap in (None, 0.0) or aligned_p16_gap is None else (aligned_map_gap - aligned_p16_gap) / aligned_map_gap,
        "paired_exact_vs_prior_inputs": exact_prior,
        "paired_exact_vs_map_inputs": exact_map,
        "paired_p16_to_exact_inputs": exact_p16,
        "exact_vs_map_paired_ci": interval(exact_map, "exact_vs_map"),
        "p16_to_exact_one_sided_upper_95": None if p16_one_sided_ci is None else p16_one_sided_ci[1],
        "hpd_coverage_inputs": hpd,
        "hpd_coverage_ci": interval([value - .9 for value in hpd], "hpd_coverage"),
        "particle_collapse_count": int(values["collapses"]),
        "p16_collapse_episode_ids": sorted(p16_collapse_ids),
        "p16_missing_episode_ids": sorted(p16_missing_ids),
        "p16_complete_coverage": not p16_missing_ids and not p16_collapse_ids,
        "expected_episode_ids": sorted(expected_ids),
    }


def _formal_status(phase: Mapping[str, Any], prefix_4: Mapping[str, Any], rejected_arms: list[str]) -> str:
    """Evaluate only registered, available formal gate inputs; otherwise fail closed."""

    gate = dict(phase.get("gate", {}))
    required = ("exact_vs_prior_nll_delta", "exact_mixture_vs_map_nll_delta", "p16_to_exact_nll_gap", "map_gap_fraction_closed", "exact_vs_map_paired_ci", "p16_to_exact_one_sided_upper_95", "hpd_coverage_ci")
    if rejected_arms or any(prefix_4.get(key) is None for key in required):
        return "INCOMPLETE"
    if len(prefix_4.get("expected_episode_ids", ())) != _REGISTERED_FORMAL_TEST_EPISODES or not prefix_4.get("p16_complete_coverage", False):
        return "INCOMPLETE"
    checks = [
        prefix_4["exact_vs_prior_nll_delta"] >= float(gate.get("exact_vs_prior_future_nll_nats_per_test", np.inf)),
        prefix_4["exact_mixture_vs_map_nll_delta"] >= float(gate.get("exact_mixture_vs_exact_map_nll_nats_per_candidate_test_min", np.inf)),
        prefix_4["exact_vs_map_paired_ci"][0] > float(gate.get("exact_mixture_vs_exact_map_paired_ci_lower_min", np.inf)),
        prefix_4["p16_to_exact_one_sided_upper_95"] <= float(gate.get("p16_gap_to_exact_nll_one_sided_ci_upper_max", -np.inf)),
        prefix_4["map_gap_fraction_closed"] >= float(gate.get("exact_map_to_exact_mixture_gap_closed_min", np.inf)),
    ]
    nominal = float(gate.get("credible_set_nominal", .9))
    coverage_ci = [value + nominal for value in prefix_4["hpd_coverage_ci"]]
    checks.append(coverage_ci[0] <= nominal <= coverage_ci[1])
    return "PASS" if all(checks) else "FAIL"


def run_pbpf_phase0(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Evaluate exact, MAP, and particle forecasts at fixed prefixes only.

    Every arm makes all suffix predictions from the same prefix posterior; no
    suffix outcome is consumed while evaluating its own forecast.
    """

    phase = dict(config.get("phase0", {}))
    count = int(dict(phase.get("episodes", {})).get("test", 8))
    candidates = int(phase.get("candidates", 4))
    prefixes = tuple(int(value) for value in phase.get("prefixes", (0, 1, 2, 4, 8, 16)))
    horizons = tuple(phase.get("forecast_horizons", (1, 8, "all_remaining")))
    sweep = tuple(int(value) for value in phase.get("particle_sweep", config.get("belief", {}).get("particle_sweep", (1, 4, 8, 16, 32))))
    order_seed = int(phase.get("test_order_seed", config.get("test_protocol", {}).get("order_seed", 61030)))
    candidate_source_contamination = float(phase.get("candidate_source_contamination", 0.0))
    requested_arms = tuple(phase.get("arms", ("exact_bayes", "pbpf", "map")))
    supported_arms = {"exact_bayes", "pbpf", "map", "prior"}
    rejected_arms = [str(arm) for arm in requested_arms if arm not in supported_arms]
    rows: list[dict[str, Any]] = []
    profile = str(dict(config.get("runtime", {})).get("profile", "formal"))
    bootstrap_resamples = 0 if profile == "smoke" else int(dict(config.get("statistics", {})).get("bootstrap_resamples", 10_000))
    groups: dict[tuple[int, int | str], dict[str, Any]] = {}
    for index in range(count):
        episode = _episode(
            seed,
            index,
            candidates,
            order_seed,
            candidate_source_contamination,
        )
        for prefix_size in prefixes:
            if prefix_size >= len(episode.test_order):
                continue
            prefix = episode.prefix_view(prefix_size)
            exact = exact_posterior(episode, prefix)
            hpd = randomized_hpd(exact, .9)
            for horizon in horizons:
                positions = _horizon_positions(prefix_size, len(episode.test_order), horizon)
                if not positions:
                    continue
                targets = episode.outcome_matrix[:, positions]
                values = groups.setdefault((prefix_size, horizon), {"exact": {}, "prior": {}, "map": {}, "particle_16": {}, "hpd": {}, "collapses": 0, "p16_collapse_ids": set(), "expected_episode_ids": set()})
                values["expected_episode_ids"].add(index)
                exact_prediction = predictive(exact, episode, positions)
                nll, brier = _scores(exact_prediction, targets)
                values["exact"][index] = nll
                values["hpd"][index] = float(hpd[episode.true_h])
                if "exact_bayes" in requested_arms:
                    rows.append({"episode": index, "prefix": prefix_size, "horizon": horizon, "arm": "exact_bayes", "nll": nll, "brier": brier, "posterior_kl": 0.0, "ess": None, "resampling": 0, "hpd_true_inclusion": float(hpd[episode.true_h])})
                prior_probability = np.full(64, 1.0 / 64.0)
                nll, brier = _scores(predictive(prior_probability, episode, positions), targets)
                values["prior"][index] = nll
                rows.append({"episode": index, "prefix": prefix_size, "horizon": horizon, "arm": "prior", "nll": nll, "brier": brier, "posterior_kl": None, "ess": None, "resampling": 0, "hpd_true_inclusion": None})
                if "map" in requested_arms:
                    map_probability = np.zeros(64, dtype=float)
                    map_probability[exact_map(exact)] = 1.0
                    nll, brier = _scores(predictive(map_probability, episode, positions), targets)
                    values["map"][index] = nll
                    rows.append({"episode": index, "prefix": prefix_size, "horizon": horizon, "arm": "map", "nll": nll, "brier": brier, "posterior_kl": None, "ess": None, "resampling": 0, "hpd_true_inclusion": float(hpd[episode.true_h])})
                if "pbpf" in requested_arms:
                    for particles in sweep:
                        try:
                            filter_ = ParticleFilter.initialize(episode, particles=particles, seed=seed + index * 1009 + prefix_size)
                            for position in range(prefix_size):
                                filter_.observe(position, episode.outcome_matrix[:, position])
                            estimated = filter_.full_support_h_marginal()
                            nll, brier = _scores(filter_.predict(positions), targets)
                        except ParticleCollapseError:
                            values["collapses"] += 1
                            if particles == 16:
                                values["p16_collapse_ids"].add(index)
                            latest = filter_.diagnostic_history[-1] if filter_.diagnostic_history else None
                            rows.append({"episode": index, "prefix": prefix_size, "horizon": horizon, "arm": f"particle_{particles}", "nll": None, "brier": None, "posterior_kl": None, "ess": 0.0, "pre_resampling_ess": None if latest is None else latest.pre_resampling_ess, "pre_resampling_ess_history": list(filter_.ess_history), "resampling": filter_.resampling_count, "resampling_rate": 0.0 if prefix_size == 0 else filter_.resampling_count / prefix_size, "unique_ancestor_ratio": None, "post_rejuvenation_unique_state_ratio": None, "resampling_events": _resampling_events(filter_), "particles_per_correct_equivalence_class": filter_.particles_per_correct_equivalence_class, "hpd_true_inclusion": float(hpd[episode.true_h]), "diagnostic": "PARTICLE_COLLAPSE"})
                            continue
                        kl = float(np.sum(exact * np.log(np.maximum(exact, 1e-15) / np.maximum(estimated, 1e-15))))
                        if particles == 16:
                            values["particle_16"][index] = nll
                        latest = filter_.diagnostic_history[-1] if filter_.diagnostic_history else None
                        rows.append({
                            "episode": index,
                            "prefix": prefix_size,
                            "horizon": horizon,
                            "arm": f"particle_{particles}",
                            "nll": nll,
                            "brier": brier,
                            "posterior_kl": kl,
                            "ess": None if latest is None else latest.pre_resampling_ess,
                            "pre_resampling_ess": None if latest is None else latest.pre_resampling_ess,
                            "pre_resampling_ess_history": list(filter_.ess_history),
                            "resampling": filter_.resampling_count,
                            "resampling_rate": 0.0 if prefix_size == 0 else filter_.resampling_count / prefix_size,
                            "unique_ancestor_ratio": None if latest is None else latest.unique_ancestor_ratio,
                            "post_rejuvenation_unique_state_ratio": None if latest is None else latest.post_rejuvenation_unique_state_ratio,
                            "resampling_events": _resampling_events(filter_),
                            "particles_per_correct_equivalence_class": filter_.particles_per_correct_equivalence_class,
                            "hpd_true_inclusion": float(hpd[episode.true_h]),
                        })
    by_prefix_horizon: dict[str, dict[str, dict[str, Any]]] = {}
    for (prefix, horizon), values in groups.items():
        gate_resamples = bootstrap_resamples if (prefix, horizon) == (4, "all_remaining") else 0
        by_prefix_horizon.setdefault(str(prefix), {})[str(horizon)] = _group_gate_inputs(values, outer_seed=seed, label=f"prefix={prefix}/horizon={horizon}", bootstrap_resamples=gate_resamples)
    prefix_4 = by_prefix_horizon.get("4", {}).get("all_remaining", {})
    gate_inputs = {"hpd_nominal": .9, "by_prefix_horizon": by_prefix_horizon, "prefix_4": prefix_4, **{key: prefix_4.get(key) for key in ("exact_vs_prior_nll_delta", "exact_mixture_vs_map_nll_delta", "p16_to_exact_nll_gap", "map_gap_fraction_closed")}}
    status = "INCOMPLETE" if profile == "smoke" else _formal_status(phase, prefix_4, rejected_arms)
    return {"seed": int(seed), "status": status, "rows": rows, "gate_status": status, "gate_inputs": gate_inputs, "rejected_arms": rejected_arms}
