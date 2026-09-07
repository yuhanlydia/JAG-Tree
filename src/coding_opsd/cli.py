"""Public CLI for deterministic, auditable Phase-0 reference experiments."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable

import numpy as np

from .config import ConfigError, ResolvedConfig, config_hash, load_config
from .goav.experiment import run_goav_phase0
from .jag.experiment import _feasible_level_increments, run_jag_phase0
from .pbpf.experiment import run_pbpf_phase0
from .results import DIRECTIONS, NormalizedRun, ResultNormalizationError, normalize_result
from .runtime import RunIntegrityError, RunWriter, validate_goav_events, verify_run_dir


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG_PATHS: Mapping[str, Path] = {
    "jag": _REPOSITORY_ROOT / "configs" / "phase0" / "jag_smoke.yaml",
    "pbpf": _REPOSITORY_ROOT / "configs" / "phase0" / "pbpf_smoke.yaml",
    "goav": _REPOSITORY_ROOT / "configs" / "phase0" / "goav_smoke.yaml",
}
_FORMAL_CONFIG_PATHS: Mapping[str, Path] = {
    "jag_tree": _REPOSITORY_ROOT / "configs" / "jag_tree.yaml",
    "predictive_belief_particle_filter": _REPOSITORY_ROOT / "configs" / "pbpf.yaml",
    "gradient_optimal_active_verification": _REPOSITORY_ROOT / "configs" / "goav.yaml",
}
_RUNNERS: Mapping[str, Callable[[Mapping[str, Any], int], dict[str, Any]]] = {
    "jag": run_jag_phase0,
    "pbpf": run_pbpf_phase0,
    "goav": run_goav_phase0,
}
_SAFE_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")
_SCHEMA_VERSIONS = {
    "jag_tree": "0.1",
    "predictive_belief_particle_filter": "0.1",
    "gradient_optimal_active_verification": "0.2",
}
_JAG_FAMILIES = {
    "root_only",
    "suffix_only",
    "entropy_distractor",
    "covariance_reversal",
}
_JAG_ARMS = {
    "flat_iid",
    "uniform_tree",
    "leaf_equal_naive",
    "entropy_tree",
    "value_variance_tree",
    "gradient_only",
    "joint_no_cross",
    "jag_oracle",
    "jag_learned",
}
_GOAV_ARMS = {
    "cheap_only",
    "uniform_subset_ht",
    "uniform_subset_aipw",
    "entropy_subset_aipw",
    "killrate_subset_aipw",
    "poisson_neyman_aipw",
    "bayes_voi_aipw",
    "goav_exact_subset_aipw",
    "deterministic_topk_invalid",
    "full_audit",
}


class CliError(ValueError):
    """An expected, concise command-line failure."""


class _ParserExit(Exception):
    def __init__(self, status: int):
        super().__init__(status)
        self.status = status


class _ArgumentParser(argparse.ArgumentParser):
    """Argument parser whose help and usage paths return through main."""

    def exit(self, status: int = 0, message: str | None = None) -> None:
        if message:
            self._print_message(message, sys.stderr)
        raise _ParserExit(status)

    def error(self, message: str) -> None:
        raise CliError(message)


@dataclass(frozen=True)
class _PreparedRun:
    alias: str
    resolved: ResolvedConfig
    seed: int
    output_parent: Path
    run_id: str

    @property
    def run_dir(self) -> Path:
        return self.output_parent / self.run_id


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{path} must be an integer >= {minimum}")
    return int(value)


def _number(
    value: Any,
    path: str,
    *,
    lower_exclusive: float | None = None,
    upper: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ConfigError(f"{path} must be finite numeric")
    result = float(value)
    if lower_exclusive is not None and result <= lower_exclusive:
        raise ConfigError(f"{path} must be > {lower_exclusive}")
    if upper is not None and result > upper:
        raise ConfigError(f"{path} must be <= {upper}")
    return result


def _unique_list(value: Any, path: str, *, nonempty: bool = True) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ConfigError(f"{path} must be a{' non-empty' if nonempty else ''} list")
    try:
        unique = len(set(value)) == len(value)
    except TypeError as exc:
        raise ConfigError(f"{path} must contain scalar values") from exc
    if not unique:
        raise ConfigError(f"{path} must not contain duplicates")
    return value


def _scientific_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the explicitly non-scientific launch envelope.

    In particular, ``execution`` remains frozen: a formal configuration must
    retain the registered container, network, determinism, and outcome rules.
    Top-level ``seeds`` are checked against ``statistics.training_seeds``
    before being removed from the projection.
    """

    return {
        key: value
        for key, value in config.items()
        if key not in {"runtime", "output", "seeds"}
    }


def _first_difference(actual: Any, expected: Any, path: str = "$") -> str:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            return f"{path} keys (missing={missing}, extra={extra})"
        for key in sorted(actual_keys):
            difference = _first_difference(actual[key], expected[key], f"{path}.{key}")
            if difference:
                return difference
        return ""
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return f"{path} length (got={len(actual)}, registered={len(expected)})"
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            difference = _first_difference(actual_item, expected_item, f"{path}[{index}]")
            if difference:
                return difference
        return ""
    if type(actual) is not type(expected) or _canonical_json(actual) != _canonical_json(expected):
        return f"{path} (got={actual!r}, registered={expected!r})"
    return ""


def _validate_formal_projection(config: Mapping[str, Any], direction: str) -> None:
    """Fail closed unless a formal run exactly matches the checked-in science."""

    runtime = _mapping(config.get("runtime"), "runtime")
    output = _mapping(config.get("output"), "output")
    if set(runtime) != {"profile"}:
        raise ConfigError("formal scientific projection permits only runtime.profile")
    if set(output) != {"identity"}:
        raise ConfigError("formal scientific projection permits only output.identity")

    statistics_value = config.get("statistics")
    if not isinstance(statistics_value, Mapping):
        raise ConfigError(
            "formal scientific projection (formal profile) requires registered statistics"
        )
    statistics = statistics_value
    registered_seeds = _unique_list(
        statistics.get("training_seeds"),
        "statistics.training_seeds",
    )
    if config.get("seeds") != registered_seeds:
        raise ConfigError(
            "formal scientific projection requires top-level seeds to exactly equal "
            "statistics.training_seeds"
        )

    registered = load_config(_FORMAL_CONFIG_PATHS[direction]).data
    actual_projection = _scientific_projection(config)
    registered_projection = _scientific_projection(registered)
    actual_digest = config_hash(actual_projection)
    registered_digest = config_hash(registered_projection)
    if actual_digest != registered_digest:
        difference = _first_difference(actual_projection, registered_projection)
        raise ConfigError(
            "formal scientific projection differs from the checked-in registration at "
            f"{difference}; got sha256={actual_digest}, "
            f"registered sha256={registered_digest}"
        )


def validate_resolved_config(
    resolved: ResolvedConfig,
    *,
    executable: bool = False,
) -> dict[str, Any]:
    """Perform direction-level semantic checks without running an experiment."""

    config = resolved.data
    direction = config.get("direction")
    aliases = {canonical: alias for alias, canonical in DIRECTIONS.items()}
    if direction not in aliases:
        raise ConfigError(f"unknown direction: {direction!r}")
    if config.get("schema_version") != _SCHEMA_VERSIONS[direction]:
        raise ConfigError(
            f"schema_version for {direction} must be {_SCHEMA_VERSIONS[direction]!r}"
        )
    phase0 = _mapping(config.get("phase0"), "phase0")

    runtime_present = "runtime" in config
    output_present = "output" in config
    if runtime_present != output_present:
        raise ConfigError("runtime and output must either both be present or both be absent")
    if not runtime_present:
        if executable:
            raise ConfigError("run requires an executable overlay with runtime.profile and output.identity")
        state = "VALID_PREREGISTRATION"
        profile: str | None = None
    else:
        runtime = _mapping(config["runtime"], "runtime")
        profile = runtime.get("profile")
        if profile not in {"smoke", "formal"}:
            raise ConfigError("runtime.profile must be 'smoke' or 'formal'")
        output = _mapping(config["output"], "output")
        identity = output.get("identity")
        if not isinstance(identity, str) or not _SAFE_IDENTITY.fullmatch(identity) or identity in {".", ".."}:
            raise ConfigError("output.identity must be a safe basename of at most 160 characters")
        seeds = _unique_list(config.get("seeds"), "seeds")
        for index, seed in enumerate(seeds):
            _integer(seed, f"seeds[{index}]")
        execution = _mapping(config.get("execution", {}), "execution")
        if profile == "smoke" and execution.get("container_digest_required", False) is not False:
            raise ConfigError(
                "CPU Phase-0 overlays must set execution.container_digest_required to false"
            )
        state = "VALID"

    if direction == "jag_tree":
        horizon = _integer(phase0.get("horizon"), "phase0.horizon", minimum=1)
        _integer(phase0.get("actions"), "phase0.actions", minimum=1)
        _integer(phase0.get("rollout_replications"), "phase0.rollout_replications", minimum=1)
        _integer(
            phase0.get("covariance_audit_replications", 32),
            "phase0.covariance_audit_replications",
            minimum=2,
        )
        families = _unique_list(phase0.get("reward_families"), "phase0.reward_families")
        if any(not isinstance(item, str) or not item for item in families):
            raise ConfigError("phase0.reward_families must contain non-empty strings")
        unknown_families = sorted(set(families) - _JAG_FAMILIES)
        if unknown_families:
            raise ConfigError(
                f"phase0.reward_families contains unsupported families: {unknown_families}"
            )
        arms = _unique_list(phase0.get("arms"), "phase0.arms")
        unknown_arms = sorted(set(arms) - _JAG_ARMS)
        if unknown_arms:
            raise ConfigError(f"phase0.arms contains unsupported JAG arms: {unknown_arms}")
        budgets = _unique_list(phase0.get("budgets", [64, 128, 256]), "phase0.budgets")
        parsed_budgets = []
        for index, budget in enumerate(budgets):
            parsed = _integer(budget, f"phase0.budgets[{index}]", minimum=1)
            if parsed < horizon:
                raise ConfigError("each phase0 budget must fund at least one complete trajectory")
            parsed_budgets.append(parsed)
        estimator = _mapping(config.get("estimator", {}), "estimator")
        baseline = estimator.get("primary_baseline", "zero")
        if baseline not in {"zero", "lagged_cross_fitted_value"}:
            raise ConfigError("estimator.primary_baseline is unsupported")
        max_branching = _integer(
            estimator.get("max_branching", max(parsed_budgets)),
            "estimator.max_branching",
            minimum=1,
        )
        branchable_depth_count = _integer(
            estimator.get("branchable_depth_count", horizon),
            "estimator.branchable_depth_count",
        )
        if branchable_depth_count > horizon:
            raise ConfigError("estimator.branchable_depth_count cannot exceed phase0.horizon")
        for budget in parsed_budgets:
            if not _feasible_level_increments(
                0,
                1,
                budget - horizon,
                horizon,
                branchable_depth_count,
                max_branching,
                {},
            ):
                raise ConfigError(
                    f"phase0 budget {budget} is not representable under the registered "
                    "branching and branchable-depth constraints"
                )
        oracle_scope = _mapping(phase0.get("oracle_scope", {}), "phase0.oracle_scope")
        for field, default in (
            ("max_horizon", 3),
            ("max_budget", 12),
            ("max_frontier_nodes", 8),
            ("max_states", 100_000),
        ):
            _integer(oracle_scope.get(field, default), f"phase0.oracle_scope.{field}", minimum=1)
        calibration_required = "jag_learned" in arms or baseline == "lagged_cross_fitted_value"
        if calibration_required and runtime_present:
            configured_seeds = list(config["seeds"])
            calibration_values = phase0.get("learned_calibration_seeds")
            if calibration_values is None:
                if min(configured_seeds) < 2:
                    raise ConfigError(
                        "low evaluation seeds require explicit non-negative earlier calibration seeds"
                    )
            else:
                calibration_seeds = _unique_list(
                    calibration_values, "phase0.learned_calibration_seeds"
                )
                parsed_calibration = [
                    _integer(value, f"phase0.learned_calibration_seeds[{index}]")
                    for index, value in enumerate(calibration_seeds)
                ]
                if max(parsed_calibration) >= min(configured_seeds):
                    raise ConfigError(
                        "phase0.learned_calibration_seeds must be strictly earlier than every evaluation seed"
                    )
            _integer(
                phase0.get("learned_calibration_horizon", horizon),
                "phase0.learned_calibration_horizon",
                minimum=1,
            )
    elif direction == "predictive_belief_particle_filter":
        episodes = _mapping(phase0.get("episodes"), "phase0.episodes")
        _integer(episodes.get("test"), "phase0.episodes.test", minimum=1)
        _integer(phase0.get("candidates"), "phase0.candidates", minimum=1)
        prefixes = _unique_list(phase0.get("prefixes"), "phase0.prefixes")
        for index, prefix in enumerate(prefixes):
            if _integer(prefix, f"phase0.prefixes[{index}]") >= 64:
                raise ConfigError("phase0.prefixes must leave at least one of the 64 tests hidden")
        horizons = _unique_list(phase0.get("forecast_horizons"), "phase0.forecast_horizons")
        for index, horizon in enumerate(horizons):
            if horizon != "all_remaining":
                _integer(horizon, f"phase0.forecast_horizons[{index}]", minimum=1)
        sweep = phase0.get(
            "particle_sweep",
            _mapping(config.get("belief", {}), "belief").get("particle_sweep"),
        )
        particles = _unique_list(sweep, "phase0.particle_sweep")
        for index, count in enumerate(particles):
            _integer(count, f"phase0.particle_sweep[{index}]", minimum=1)
        arms = _unique_list(phase0.get("arms"), "phase0.arms")
        if any(not isinstance(item, str) or not item for item in arms):
            raise ConfigError("phase0.arms must contain non-empty strings")
        if "test_order_seed" in phase0:
            _integer(phase0["test_order_seed"], "phase0.test_order_seed")
    else:
        tasks = _integer(phase0.get("tasks_min"), "phase0.tasks_min", minimum=1)
        if (
            "tasks_max" in phase0
            and _integer(phase0["tasks_max"], "phase0.tasks_max", minimum=1) < tasks
        ):
            raise ConfigError("phase0.tasks_max must be >= phase0.tasks_min")
        candidates = _integer(phase0.get("group_size"), "phase0.group_size", minimum=1)
        if candidates > 20:
            raise ConfigError("phase0.group_size must be <= 20 for exact subset enumeration")
        _integer(
            phase0.get("tests_per_task_min"),
            "phase0.tests_per_task_min",
            minimum=1,
        )
        _integer(
            phase0.get("subset_draws_per_group_design"),
            "phase0.subset_draws_per_group_design",
            minimum=1,
        )
        acquisition = _mapping(config.get("acquisition"), "acquisition")
        fraction = phase0.get(
            "primary_budget_fraction",
            acquisition.get("primary_budget_fraction"),
        )
        _number(
            fraction,
            "phase0.primary_budget_fraction",
            lower_exclusive=0.0,
            upper=1.0,
        )
        _number(
            acquisition.get("primary_inclusion_floor"),
            "acquisition.primary_inclusion_floor",
            lower_exclusive=0.0,
            upper=1.0,
        )
        arms = _unique_list(phase0.get("arms"), "phase0.arms")
        unknown_arms = sorted(set(arms) - _GOAV_ARMS)
        if unknown_arms:
            raise ConfigError(f"phase0.arms contains unsupported GOAV arms: {unknown_arms}")
        floor = float(acquisition["primary_inclusion_floor"])
        if any(arm not in {"cheap_only", "deterministic_topk_invalid", "full_audit"} for arm in arms):
            if floor > float(fraction):
                raise ConfigError(
                    "acquisition.primary_inclusion_floor cannot exceed the audit budget fraction"
                )
        solver = _mapping(acquisition.get("solver", {}), "acquisition.solver")
        steps = phase0.get("solver_steps", solver.get("steps", 200))
        restarts = phase0.get("solver_restarts")
        if restarts is None:
            initializers = solver.get("initialization", [])
            if initializers is not None and not isinstance(initializers, list):
                raise ConfigError("acquisition.solver.initialization must be a list")
            restarts = len(initializers) if initializers else 8
        learning_rate = phase0.get("solver_learning_rate", solver.get("learning_rate", 0.05))
        _integer(steps, "phase0.solver_steps")
        _integer(restarts, "phase0.solver_restarts", minimum=1)
        _number(learning_rate, "phase0.solver_learning_rate", lower_exclusive=0.0)
        noise = _mapping(
            _mapping(config.get("oracle_noise_model"), "oracle_noise_model").get("medium_noise"),
            "oracle_noise_model.medium_noise",
        )
        _integer(noise.get("cluster_size"), "oracle_noise_model.medium_noise.cluster_size", minimum=1)
        for key in ("false_negative", "false_positive"):
            rate = _number(noise.get(key), f"oracle_noise_model.medium_noise.{key}")
            if not 0.0 <= rate <= 1.0:
                raise ConfigError(f"oracle_noise_model.medium_noise.{key} must lie in [0, 1]")
        rho = _number(noise.get("flip_icc"), "oracle_noise_model.medium_noise.flip_icc")
        if not 0.0 <= rho < 1.0:
            raise ConfigError("oracle_noise_model.medium_noise.flip_icc must lie in [0, 1)")
        if "epsilon_G" in phase0:
            _number(phase0["epsilon_G"], "phase0.epsilon_G", lower_exclusive=0.0)
        else:
            _integer(
                phase0.get("epsilon_g_dev_tasks", max(4, tasks)),
                "phase0.epsilon_g_dev_tasks",
                minimum=1,
            )
            _integer(
                phase0.get("epsilon_g_dev_seed", 730241),
                "phase0.epsilon_g_dev_seed",
            )

    if profile == "formal":
        _validate_formal_projection(config, direction)
        if executable:
            raise ConfigError(
                "formal profile is registered but unsupported by the finite CPU Phase-0 "
                "reference runner; use a checked-in smoke overlay"
            )

    return {
        "status": state,
        "direction": direction,
        "direction_alias": aliases[direction],
        "profile": profile,
        "config_sha256": resolved.sha256,
    }


def _preflight(
    alias: str,
    config_path: str | Path,
    seed: int,
    output_parent: str | Path,
) -> _PreparedRun:
    if alias not in DIRECTIONS:
        raise CliError(f"unknown direction alias: {alias!r}")
    _integer(seed, "seed")
    resolved = load_config(config_path)
    validation = validate_resolved_config(resolved, executable=True)
    expected_direction = DIRECTIONS[alias]
    if validation["direction"] != expected_direction:
        raise CliError(
            f"direction mismatch: alias {alias!r} requires {expected_direction!r}, "
            f"config declares {validation['direction']!r}"
        )
    configured_seeds = resolved.data["seeds"]
    if seed not in configured_seeds:
        raise CliError(f"seed {seed} is not declared in the executable overlay")
    identity = resolved.data["output"]["identity"]
    run_id = f"{identity}-seed{seed}-{resolved.sha256[:8]}"
    if len(run_id) > 192:
        raise CliError("deterministic run ID exceeds the 192-character artifact limit")
    parent = Path(output_parent)
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise CliError(f"output parent is not a real directory: {parent}")
    prepared = _PreparedRun(alias, resolved, seed, parent, run_id)
    try:
        prepared.run_dir.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise CliError(f"cannot inspect target run directory: {prepared.run_dir}") from exc
    else:
        raise FileExistsError(
            f"run directory already exists and cannot be reused: {prepared.run_dir}"
        )
    return prepared


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _git_provenance() -> dict[str, Any]:
    def command(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=_REPOSITORY_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = command("rev-parse", "HEAD")
    status = command("status", "--porcelain", "--untracked-files=normal")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def _manifest(prepared: _PreparedRun, normalized: NormalizedRun) -> dict[str, Any]:
    config = prepared.resolved.data
    return {
        "schema_version": "phase0.run.v1",
        "run_id": prepared.run_id,
        "direction": DIRECTIONS[prepared.alias],
        "direction_alias": prepared.alias,
        "output_identity": config["output"]["identity"],
        "seed": prepared.seed,
        "profile": config["runtime"]["profile"],
        "status": normalized.gates["status"],
        "config_sha256": prepared.resolved.sha256,
        "source_configs": [
            {"path": str(path), "sha256": digest}
            for path, digest in zip(
                prepared.resolved.source_paths,
                prepared.resolved.source_hashes,
                strict=True,
            )
        ],
        "git": _git_provenance(),
        "environment": {
            "kind": "cpu_phase0",
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "pyyaml": _package_version("PyYAML"),
            "model_backend": None,
            "model_checkpoint": None,
            "container_digest": "not_applicable",
        },
        "row_count": len(normalized.rows),
        "event_count": len(normalized.events),
        "array_keys": sorted(normalized.arrays),
    }


def _report(prepared: _PreparedRun, normalized: NormalizedRun) -> str:
    reason = normalized.gates.get("reason", "unspecified")
    return (
        "# Phase-0 run report\n\n"
        f"- Run ID: {prepared.run_id}\n"
        f"- Direction: {DIRECTIONS[prepared.alias]} ({prepared.alias})\n"
        f"- Seed: {prepared.seed}\n"
        f"- Profile: {prepared.resolved.data['runtime']['profile']}\n"
        f"- Scientific status: {normalized.gates['status']}\n"
        f"- Status reason: {reason}\n"
        f"- Rows/events/arrays: {len(normalized.rows)} / "
        f"{len(normalized.events)} / {len(normalized.arrays)}\n\n"
        "This is a deterministic finite CPU Phase-0 mechanism experiment. "
        "It is not a 7B-model result or a coding-benchmark score. COMPLETE "
        "means that the artifacts are sealed; gates.json is authoritative "
        "for the scientific status.\n"
    )


def _execute(prepared: _PreparedRun) -> dict[str, Any]:
    config = prepared.resolved.data
    raw_result = _RUNNERS[prepared.alias](config, prepared.seed)
    normalized = normalize_result(
        prepared.alias,
        DIRECTIONS[prepared.alias],
        config,
        prepared.seed,
        prepared.run_id,
        raw_result,
    )
    if normalized.metrics["status"] != normalized.gates["status"]:
        raise ResultNormalizationError("normalized metrics and gates statuses disagree")
    if prepared.alias == "goav":
        validate_goav_events(
            normalized.events,
            run_id=prepared.run_id,
            seed=prepared.seed,
            require_envelope=True,
        )
    manifest = _manifest(prepared, normalized)
    with RunWriter(prepared.run_dir, config, manifest) as writer:
        writer.write_metrics(normalized.metrics)
        writer.write_gates(normalized.gates)
        writer.write_events(normalized.events)
        writer.write_rows(normalized.rows)
        writer.write_npz(normalized.arrays)
        writer.write_report(_report(prepared, normalized))
        writer.complete()
    return {
        "direction_alias": prepared.alias,
        "direction": DIRECTIONS[prepared.alias],
        "run_dir": str(prepared.run_dir),
        "status": normalized.gates["status"],
    }


def _write_exclusive(path: str | Path, contents: str) -> None:
    target = Path(path)
    if target.parent.exists() and (target.parent.is_symlink() or not target.parent.is_dir()):
        raise CliError(f"output parent is not a real directory: {target.parent}")
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise FileExistsError(
            f"output file already exists and cannot be replaced: {target}"
        ) from exc
    except OSError as exc:
        raise CliError(f"could not create output file: {target}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="coding-opsd",
        description="Run deterministic CPU Phase-0 coding-RLVR mechanism experiments.",
    )
    parser.add_argument("--debug", action="store_true", help="show a traceback for failures")
    commands = parser.add_subparsers(dest="command")

    resolve = commands.add_parser(
        "resolve",
        help="print or exclusively write a resolved configuration",
    )
    resolve.add_argument("config")
    resolve.add_argument("--output")

    validate = commands.add_parser("validate", help="semantically validate a configuration")
    validate.add_argument("config")

    run = commands.add_parser("run", help="run and seal one Phase-0 direction")
    run.add_argument("direction", choices=tuple(DIRECTIONS))
    run.add_argument("config")
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--output", required=True)

    run_all = commands.add_parser(
        "run-all",
        help="run all checked-in overlays for a profile",
    )
    run_all.add_argument("--profile", choices=("smoke",), required=True)
    run_all.add_argument("--seed", type=int, required=True)
    run_all.add_argument("--output", required=True)

    verify = commands.add_parser(
        "verify",
        help="verify a sealed run directory without modifying it",
    )
    verify.add_argument("run_dir")
    return parser


def _dispatch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "resolve":
        resolved = load_config(args.config)
        encoded = _canonical_json(resolved.data) + "\n"
        if args.output is None:
            sys.stdout.write(encoded)
        else:
            _write_exclusive(args.output, encoded)
        return 0
    if args.command == "validate":
        resolved = load_config(args.config)
        print(_canonical_json(validate_resolved_config(resolved)))
        return 0
    if args.command == "verify":
        print(_canonical_json(verify_run_dir(args.run_dir)))
        return 0
    if args.command == "run":
        prepared = _preflight(args.direction, args.config, args.seed, args.output)
        summary = _execute(prepared)
        print(summary["run_dir"])
        return 1 if summary["status"] == "INVALID" else 0
    if args.command == "run-all":
        prepared = [
            _preflight(alias, config_path, args.seed, args.output)
            for alias, config_path in SMOKE_CONFIG_PATHS.items()
        ]
        wrong_profiles = [
            item.alias
            for item in prepared
            if item.resolved.data["runtime"]["profile"] != args.profile
        ]
        if wrong_profiles:
            raise CliError(
                f"run-all profile {args.profile!r} disagrees with overlays: {wrong_profiles}"
            )
        # Materialize every target before compute so a collision cannot leave
        # a partially executed run-all batch.
        summaries = [_execute(item) for item in prepared]
        print(_canonical_json(summaries))
        return 1 if any(item["status"] == "INVALID" for item in summaries) else 0
    raise CliError(f"unsupported command: {args.command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    debug = "--debug" in arguments
    parser = build_parser()
    try:
        args = parser.parse_args(arguments)
        return _dispatch(args, parser)
    except _ParserExit as exc:
        return exc.status
    except (
        CliError,
        ConfigError,
        ResultNormalizationError,
        RunIntegrityError,
        FileExistsError,
        ValueError,
    ) as exc:
        if debug:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive process boundary
        if debug:
            raise
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


__all__ = ["SMOKE_CONFIG_PATHS", "build_parser", "main", "validate_resolved_config"]
