"""Strict, deterministic normalization of direction-specific Phase-0 results.

The experiment runners intentionally return small direction-native mappings.
This module is the only translation boundary between those mappings and the
shared on-disk run schema.  It uses explicit schemas rather than key guessing
so an upstream result-shape change fails before a run directory is created.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import math
import re
from typing import Any

import numpy as np

from .config import config_hash
from .runtime import RunIntegrityError, validate_goav_events


STATUSES = frozenset({"PASS", "FAIL", "INCOMPLETE", "INVALID"})
DIRECTIONS = {
    "jag": "jag_tree",
    "pbpf": "predictive_belief_particle_filter",
    "goav": "gradient_optimal_active_verification",
}

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_ARRAY_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}\Z")
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9]+")


class ResultNormalizationError(ValueError):
    """Raised when a native result violates its registered result schema."""


@dataclass(frozen=True)
class NormalizedRun:
    """The five scientific payloads consumed by the artifact writer."""

    metrics: dict[str, Any]
    gates: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    arrays: dict[str, np.ndarray]
    events: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _Context:
    alias: str
    direction: str
    profile: str
    seed: int
    run_id: str
    config: dict[str, Any]


def _fail(path: str, message: str) -> ResultNormalizationError:
    return ResultNormalizationError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(path, "must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise _fail(path, "all keys must be strings")
    try:
        for key in value:
            key.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _fail(path, "all keys must be valid UTF-8 strings") from exc
    return value


def _exact_keys(
    value: Mapping[str, Any],
    required: set[str] | frozenset[str],
    path: str,
    *,
    optional: set[str] | frozenset[str] = frozenset(),
) -> None:
    keys = set(value)
    missing = set(required) - keys
    unexpected = keys - set(required) - set(optional)
    if missing:
        raise _fail(path, f"missing keys {sorted(missing)!r}")
    if unexpected:
        raise _fail(path, f"unexpected keys {sorted(unexpected)!r}")


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(path, "must be an integer")
    if minimum is not None and value < minimum:
        raise _fail(path, f"must be at least {minimum}")
    return int(value)


def _number(value: Any, path: str, *, nullable: bool = False) -> float | int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(path, "must be a finite JSON number" + (" or null" if nullable else ""))
    try:
        finite = math.isfinite(float(value))
    except OverflowError as exc:
        raise _fail(path, "must be representable as a finite binary64 number") from exc
    if not finite:
        raise _fail(path, "must be finite; use null plus an explicit diagnostic when undefined")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(path, "must be a non-empty string")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise _fail(path, "must be boolean")
    return value


def _status(value: Any, path: str) -> str:
    status = _string(value, path)
    if status not in STATUSES:
        raise _fail(path, f"must be one of {sorted(STATUSES)!r}")
    return status


def _json_clone(value: Any, path: str) -> Any:
    """Clone strict JSON data while rejecting NaN, infinity, and aliases."""

    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise _fail(path, "contains a string that is not valid UTF-8") from exc
        return value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _fail(path, "contains a non-finite number")
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_json_clone(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        mapping = _mapping(value, path)
        return {key: _json_clone(item, f"{path}.{key}") for key, item in mapping.items()}
    raise _fail(path, f"contains non-JSON value of type {type(value).__name__}")


def _array(values: Any, path: str, *, ndim: int | None = None) -> np.ndarray:
    # Validate as JSON first.  This deliberately rejects object arrays and
    # numpy scalar leakage from direction-native results.
    cloned = _json_clone(values, path)

    def validate_numeric(item: Any, item_path: str) -> None:
        if isinstance(item, list):
            for index, child in enumerate(item):
                validate_numeric(child, f"{item_path}[{index}]")
            return
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise _fail(item_path, "must be a numeric JSON value without coercion")

    validate_numeric(cloned, path)
    try:
        array = np.asarray(cloned, dtype="<f8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise _fail(path, "must contain only numeric values") from exc
    if ndim is not None and array.ndim != ndim:
        raise _fail(path, f"must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise _fail(path, "must contain only finite values")
    result = np.ascontiguousarray(array, dtype="<f8").copy()
    result.setflags(write=False)
    return result


def _bool_array(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.bool_).copy()
    result.setflags(write=False)
    return result


def _slug(value: str, path: str) -> str:
    text = _string(value, path)
    slug = _SAFE_COMPONENT.sub("_", text).strip("_").lower()
    if not slug:
        raise _fail(path, "does not have a safe array-key representation")
    return slug


def _config_mapping(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return _mapping(config, "config")
    try:
        data = config.data
    except (AttributeError, TypeError, ValueError) as exc:
        raise _fail("config", "must be a mapping or expose a mapping data property") from exc
    return _mapping(data, "config.data")


def _context(
    alias: Any, canonical_direction: Any, config: Any, seed: Any, run_id: Any
) -> _Context:
    alias_value = _string(alias, "alias")
    direction_value = _string(canonical_direction, "canonical_direction")
    expected = DIRECTIONS.get(alias_value)
    if expected is None or expected != direction_value:
        raise _fail("direction", "alias and canonical direction do not match a registered pair")
    seed_value = _integer(seed, "seed", minimum=0)
    run_id_value = _string(run_id, "run_id")
    if (
        _SAFE_RUN_ID.fullmatch(run_id_value) is None
        or run_id_value in {".", ".."}
        or len(run_id_value) > 192
    ):
        raise _fail("run_id", "must be a safe basename of at most 192 characters")
    config_data = _config_mapping(config)
    config_direction = _string(config_data.get("direction"), "config.direction")
    if config_direction != direction_value:
        raise _fail("config.direction", "does not match the canonical direction")
    runtime = _mapping(config_data.get("runtime"), "config.runtime")
    profile = _string(runtime.get("profile"), "config.runtime.profile")
    if profile not in {"smoke", "formal"}:
        raise _fail("config.runtime.profile", "must be 'smoke' or 'formal'")
    seeds = config_data.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise _fail("config.seeds", "must be a non-empty list of non-negative integers")
    configured_seeds = [
        _integer(item, f"config.seeds[{index}]", minimum=0) for index, item in enumerate(seeds)
    ]
    if len(set(configured_seeds)) != len(configured_seeds):
        raise _fail("config.seeds", "must not contain duplicates")
    if seed_value not in configured_seeds:
        raise _fail("seed", "is not registered in config.seeds")
    output = _mapping(config_data.get("output"), "config.output")
    identity = _string(output.get("identity"), "config.output.identity")
    if (
        _SAFE_RUN_ID.fullmatch(identity) is None
        or identity in {".", ".."}
        or len(identity) > 192
    ):
        raise _fail("config.output.identity", "must be a safe basename of at most 192 characters")
    try:
        digest = config_hash(config_data)
    except (TypeError, ValueError) as exc:
        raise _fail("config", "must have a canonical SHA-256 identity") from exc
    expected_run_id = f"{identity}-seed{seed_value}-{digest[:8]}"
    if len(expected_run_id) > 192:
        raise _fail("run_id", "deterministic identity exceeds 192 characters")
    if run_id_value != expected_run_id:
        raise _fail("run_id", f"must equal deterministic identity {expected_run_id!r}")
    return _Context(
        alias_value,
        direction_value,
        profile,
        seed_value,
        run_id_value,
        _json_clone(config_data, "config"),
    )


def _scientific_state(
    context: _Context,
    source_status: str,
    *,
    source_formal_evidence: bool | None = None,
    source_reason: str | None = None,
) -> tuple[str, bool, str]:
    if source_formal_evidence is not None and not isinstance(source_formal_evidence, bool):
        raise _fail("formal_evidence", "must be boolean")
    validated_reason = None if source_reason is None else _string(source_reason, "reason")
    if source_formal_evidence is True and source_status not in {"PASS", "FAIL"}:
        raise _fail("formal_evidence", "can be true only for PASS or FAIL source status")
    if source_status == "INVALID":
        return "INVALID", False, "source_result_invalid"
    if context.profile == "smoke":
        return "INCOMPLETE", False, "smoke_profile_below_formal_registration"
    if source_formal_evidence is False and source_status in {"PASS", "FAIL"}:
        return "INCOMPLETE", False, "source_formal_evidence_false"
    if source_formal_evidence is None:
        formal_evidence = source_status in {"PASS", "FAIL"}
    else:
        formal_evidence = source_formal_evidence
    if validated_reason is not None:
        reason = validated_reason
    else:
        reason = {
            "PASS": "source_formal_gate_passed",
            "FAIL": "source_formal_gate_failed",
            "INCOMPLETE": "source_formal_evidence_incomplete",
            "INVALID": "source_result_invalid",
        }[source_status]
    return source_status, formal_evidence, reason


def _gates(
    context: _Context,
    source_status: str,
    inputs: Any,
    *,
    source_formal_evidence: bool | None = None,
    source_reason: str | None = None,
) -> tuple[dict[str, Any], str]:
    normalized_inputs = _json_clone(_mapping(inputs, "gate.inputs"), "gate.inputs")
    final_status, formal_evidence, reason = _scientific_state(
        context,
        source_status,
        source_formal_evidence=source_formal_evidence,
        source_reason=source_reason,
    )
    gates = {
        "schema_version": "phase0.gates.v1",
        "direction": context.direction,
        "seed": context.seed,
        "profile": context.profile,
        "status": final_status,
        "formal_evidence": formal_evidence,
        "source_status": source_status,
        "reason": reason,
        "inputs": normalized_inputs,
    }
    return gates, final_status


def _metrics(
    context: _Context,
    status: str,
    rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    specific: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "schema_version": "phase0.metrics.v1",
        "direction": context.direction,
        "seed": context.seed,
        "profile": context.profile,
        "status": status,
        "row_count": len(rows),
        "event_count": len(events),
        "array_keys": sorted(arrays),
    }
    overlap = set(result) & set(specific)
    if overlap:
        raise _fail("metrics", f"direction metrics collide with common keys {sorted(overlap)!r}")
    result.update(_json_clone(specific, "metrics.direction_specific"))
    return result


def _row(context: _Context, index: int, record_type: str, native: Mapping[str, Any]) -> dict[str, Any]:
    reserved = {"schema_version", "run_id", "direction", "seed", "row_index", "record_type"}
    if reserved & set(native):
        raise _fail(f"rows[{index}]", "contains a reserved normalized-row key")
    return {
        "schema_version": "phase0.row.v1",
        "run_id": context.run_id,
        "direction": context.direction,
        "seed": context.seed,
        "row_index": index,
        "record_type": record_type,
        **_json_clone(native, f"rows[{index}]"),
    }


def _event(context: _Context, index: int, native: Mapping[str, Any]) -> dict[str, Any]:
    reserved = {"schema_version", "run_id", "direction", "seed", "sequence"}
    if reserved & set(native):
        raise _fail(f"events[{index}]", "contains a reserved normalized-event key")
    return {
        "schema_version": "phase0.event.v1",
        "run_id": context.run_id,
        "direction": context.direction,
        "seed": context.seed,
        "sequence": index,
        **_json_clone(native, f"events[{index}]"),
    }


_PBPF_TOP = frozenset({"seed", "status", "rows", "gate_status", "gate_inputs", "rejected_arms"})
_PBPF_ROW_REQUIRED = frozenset(
    {"episode", "prefix", "horizon", "arm", "nll", "brier", "posterior_kl", "ess", "resampling", "hpd_true_inclusion"}
)
_PBPF_ROW_OPTIONAL = frozenset(
    {
        "pre_resampling_ess",
        "pre_resampling_ess_history",
        "resampling_rate",
        "unique_ancestor_ratio",
        "post_rejuvenation_unique_state_ratio",
        "resampling_events",
        "particles_per_correct_equivalence_class",
        "diagnostic",
    }
)
_PBPF_COLUMNS = ("nll", "brier", "posterior_kl", "ess", "resampling", "hpd_true_inclusion")
_PBPF_GATE_ARRAYS = {
    "paired_exact_vs_prior_inputs": "pbpf_gate_prefix4_paired_exact_vs_prior",
    "paired_exact_vs_map_inputs": "pbpf_gate_prefix4_paired_exact_vs_map",
    "paired_p16_to_exact_inputs": "pbpf_gate_prefix4_paired_p16_to_exact",
    "hpd_coverage_inputs": "pbpf_gate_prefix4_hpd_coverage",
}


def _normalize_pbpf(context: _Context, raw_result: Any) -> NormalizedRun:
    raw = _mapping(raw_result, "pbpf_result")
    _exact_keys(raw, _PBPF_TOP, "pbpf_result")
    if _integer(raw["seed"], "pbpf_result.seed") != context.seed:
        raise _fail("pbpf_result.seed", "does not match requested seed")
    source_status = _status(raw["status"], "pbpf_result.status")
    gate_status = _status(raw["gate_status"], "pbpf_result.gate_status")
    if source_status != gate_status:
        raise _fail("pbpf_result", "status and gate_status disagree")
    gate_inputs = _mapping(raw["gate_inputs"], "pbpf_result.gate_inputs")
    rejected_raw = raw["rejected_arms"]
    if not isinstance(rejected_raw, list):
        raise _fail("pbpf_result.rejected_arms", "must be a list")
    rejected_arms = [_string(value, f"pbpf_result.rejected_arms[{index}]") for index, value in enumerate(rejected_raw)]
    if len(set(rejected_arms)) != len(rejected_arms):
        raise _fail("pbpf_result.rejected_arms", "must not contain duplicates")
    native_rows = raw["rows"]
    if not isinstance(native_rows, list):
        raise _fail("pbpf_result.rows", "must be a list")
    if not native_rows:
        raise _fail("pbpf_result.rows", "must contain at least one scientific row")
    numeric = np.zeros((len(native_rows), len(_PBPF_COLUMNS)), dtype="<f8")
    present = np.zeros(numeric.shape, dtype=np.bool_)
    rows: list[dict[str, Any]] = []
    arm_counts: Counter[str] = Counter()
    collapse_counts: Counter[str] = Counter()
    for index, value in enumerate(native_rows):
        native = _mapping(value, f"pbpf_result.rows[{index}]")
        _exact_keys(native, _PBPF_ROW_REQUIRED, f"pbpf_result.rows[{index}]", optional=_PBPF_ROW_OPTIONAL)
        _integer(native["episode"], f"pbpf_result.rows[{index}].episode", minimum=0)
        _integer(native["prefix"], f"pbpf_result.rows[{index}].prefix", minimum=0)
        horizon = native["horizon"]
        if horizon != "all_remaining":
            _integer(horizon, f"pbpf_result.rows[{index}].horizon", minimum=1)
        arm = _string(native["arm"], f"pbpf_result.rows[{index}].arm")
        arm_counts[arm] += 1
        _integer(native["resampling"], f"pbpf_result.rows[{index}].resampling", minimum=0)
        for column_index, column in enumerate(_PBPF_COLUMNS):
            value_number = _number(native[column], f"pbpf_result.rows[{index}].{column}", nullable=column != "resampling")
            if value_number is not None:
                numeric[index, column_index] = float(value_number)
                present[index, column_index] = True
        for column in ("nll", "brier", "posterior_kl", "ess"):
            if native[column] is not None and float(native[column]) < 0.0:
                raise _fail(f"pbpf_result.rows[{index}].{column}", "must be non-negative")
        if native["hpd_true_inclusion"] is not None and not 0.0 <= float(native["hpd_true_inclusion"]) <= 1.0:
            raise _fail(f"pbpf_result.rows[{index}].hpd_true_inclusion", "must lie in [0, 1]")
        for column in (
            "pre_resampling_ess",
            "resampling_rate",
            "unique_ancestor_ratio",
            "post_rejuvenation_unique_state_ratio",
        ):
            if column in native:
                checked = _number(native[column], f"pbpf_result.rows[{index}].{column}", nullable=True)
                if checked is not None and float(checked) < 0.0:
                    raise _fail(f"pbpf_result.rows[{index}].{column}", "must be non-negative")
        if "pre_resampling_ess_history" in native:
            history = native["pre_resampling_ess_history"]
            if not isinstance(history, list):
                raise _fail(f"pbpf_result.rows[{index}].pre_resampling_ess_history", "must be a list")
            for history_index, item in enumerate(history):
                checked = _number(
                    item,
                    f"pbpf_result.rows[{index}].pre_resampling_ess_history[{history_index}]",
                )
                assert checked is not None
                if float(checked) < 0.0:
                    raise _fail(
                        f"pbpf_result.rows[{index}].pre_resampling_ess_history[{history_index}]",
                        "must be non-negative",
                    )
        if "particles_per_correct_equivalence_class" in native:
            _integer(
                native["particles_per_correct_equivalence_class"],
                f"pbpf_result.rows[{index}].particles_per_correct_equivalence_class",
                minimum=0,
            )
        if "resampling_events" in native:
            resampling_events = native["resampling_events"]
            if not isinstance(resampling_events, list):
                raise _fail(f"pbpf_result.rows[{index}].resampling_events", "must be a list")
            diagnostic_keys = {
                "position",
                "pre_resampling_ess",
                "unique_ancestor_ratio",
                "post_rejuvenation_unique_state_ratio",
            }
            for event_index, event_value in enumerate(resampling_events):
                event_path = f"pbpf_result.rows[{index}].resampling_events[{event_index}]"
                resampling_event = _mapping(event_value, event_path)
                _exact_keys(resampling_event, diagnostic_keys, event_path)
                _integer(resampling_event["position"], f"{event_path}.position", minimum=0)
                for column in diagnostic_keys - {"position"}:
                    checked = _number(resampling_event[column], f"{event_path}.{column}", nullable=True)
                    if checked is not None and float(checked) < 0.0:
                        raise _fail(f"{event_path}.{column}", "must be non-negative")
        diagnostic = native.get("diagnostic")
        if diagnostic is not None:
            if diagnostic != "PARTICLE_COLLAPSE":
                raise _fail(f"pbpf_result.rows[{index}].diagnostic", "unknown diagnostic")
            collapse_counts[arm] += 1
        _json_clone(native, f"pbpf_result.rows[{index}]")
        rows.append(_row(context, index, "forecast_diagnostic", native))
    numeric.setflags(write=False)
    arrays: dict[str, np.ndarray] = {
        "pbpf_row_numeric": numeric,
        "pbpf_row_numeric_present": _bool_array(present),
    }
    prefix_4 = gate_inputs.get("prefix_4", {})
    prefix_4_mapping = _mapping(prefix_4, "pbpf_result.gate_inputs.prefix_4")
    for source_key, array_key in _PBPF_GATE_ARRAYS.items():
        if source_key in prefix_4_mapping:
            arrays[array_key] = _array(
                prefix_4_mapping[source_key],
                f"pbpf_result.gate_inputs.prefix_4.{source_key}",
                ndim=1,
            )
    gates, final_status = _gates(context, source_status, gate_inputs)
    specific = {
        "arm_counts": dict(sorted(arm_counts.items())),
        "collapse_counts": {
            "total": sum(collapse_counts.values()),
            "by_arm": dict(sorted(collapse_counts.items())),
        },
        "rejected_arms": list(rejected_arms),
        "diagnostic_column_schema": {
            "matrix_key": "pbpf_row_numeric",
            "present_mask_key": "pbpf_row_numeric_present",
            "columns": list(_PBPF_COLUMNS),
            "missing_value": "zero_with_present_mask",
        },
    }
    metrics = _metrics(context, final_status, rows, (), arrays, specific)
    return NormalizedRun(metrics, gates, tuple(rows), arrays, ())


_GOAV_TOP = frozenset({"rows", "epsilon_G", "epsilon_G_provenance", "solver", "gates", "events"})
_GOAV_ROW_COLUMNS = (
    "nMSE",
    "gradient_cosine",
    "standardized_bias",
    "kish_ess",
    "weight_p50",
    "weight_p95",
    "weight_p99",
    "expected_cost",
    "realized_cost",
    "support_violations",
    "exact_risk",
    "realized_risk",
)
_GOAV_ROW_KEYS = frozenset({"arm", *_GOAV_ROW_COLUMNS})
_GOAV_DESIGN_KEYS = frozenset(
    {"event", "task", "arm", "design_hash", "probabilities", "pi", "pi2", "pi_hash", "pi2_hash"}
)
_GOAV_AUDIT_KEYS = frozenset(
    {"event", "task", "arm", "draw", "subset_index", "selected_indices", "design_hash", "inclusion_probabilities", "selected_labels"}
)


def _hash_string(value: Any, path: str) -> str:
    result = _string(value, path)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise _fail(path, "must be a lowercase SHA-256 hex digest")
    return result


def _normalize_goav(context: _Context, raw_result: Any) -> NormalizedRun:
    raw = _mapping(raw_result, "goav_result")
    _exact_keys(raw, _GOAV_TOP, "goav_result")
    epsilon = _number(raw["epsilon_G"], "goav_result.epsilon_G")
    assert epsilon is not None
    if float(epsilon) <= 0.0:
        raise _fail("goav_result.epsilon_G", "must be positive")
    epsilon_provenance = _mapping(raw["epsilon_G_provenance"], "goav_result.epsilon_G_provenance")
    solver = _mapping(raw["solver"], "goav_result.solver")
    _exact_keys(solver, {"steps", "learning_rate", "restarts"}, "goav_result.solver")
    _integer(solver["steps"], "goav_result.solver.steps", minimum=1)
    _number(solver["learning_rate"], "goav_result.solver.learning_rate")
    if float(solver["learning_rate"]) <= 0.0:
        raise _fail("goav_result.solver.learning_rate", "must be positive")
    _integer(solver["restarts"], "goav_result.solver.restarts", minimum=1)
    native_rows = raw["rows"]
    if not isinstance(native_rows, list):
        raise _fail("goav_result.rows", "must be a list")
    if not native_rows:
        raise _fail("goav_result.rows", "must contain at least one scientific row")
    numeric = np.zeros((len(native_rows), len(_GOAV_ROW_COLUMNS)), dtype="<f8")
    present = np.ones(numeric.shape, dtype=np.bool_)
    rows: list[dict[str, Any]] = []
    arm_indices: dict[str, int] = {}
    support_violations = 0
    for index, value in enumerate(native_rows):
        native = _mapping(value, f"goav_result.rows[{index}]")
        _exact_keys(native, _GOAV_ROW_KEYS, f"goav_result.rows[{index}]")
        arm = _string(native["arm"], f"goav_result.rows[{index}].arm")
        if arm in arm_indices:
            raise _fail(f"goav_result.rows[{index}].arm", "must occur exactly once")
        arm_indices[arm] = len(arm_indices)
        for column_index, column in enumerate(_GOAV_ROW_COLUMNS):
            if column == "support_violations":
                number = _integer(native[column], f"goav_result.rows[{index}].{column}", minimum=0)
                support_violations += number
            else:
                checked = _number(native[column], f"goav_result.rows[{index}].{column}")
                assert checked is not None
                number = checked
            numeric[index, column_index] = float(number)
        rows.append(_row(context, index, "arm_summary", native))
    numeric.setflags(write=False)
    arrays: dict[str, np.ndarray] = {
        "goav_row_numeric": numeric,
        "goav_row_numeric_present": _bool_array(present),
    }
    native_events = raw["events"]
    if not isinstance(native_events, list):
        raise _fail("goav_result.events", "must be a list")
    events: list[dict[str, Any]] = []
    designs: dict[tuple[int, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    design_count = 0
    audit_count = 0
    for index, value in enumerate(native_events):
        native = _mapping(value, f"goav_result.events[{index}]")
        kind = native.get("event")
        if kind == "design_logged":
            _exact_keys(native, _GOAV_DESIGN_KEYS, f"goav_result.events[{index}]")
        elif kind == "audit_request":
            _exact_keys(native, _GOAV_AUDIT_KEYS, f"goav_result.events[{index}]")
        else:
            raise _fail(f"goav_result.events[{index}].event", "must be design_logged or audit_request")
        task = _integer(native["task"], f"goav_result.events[{index}].task", minimum=0)
        arm = _string(native["arm"], f"goav_result.events[{index}].arm")
        if arm not in arm_indices:
            raise _fail(f"goav_result.events[{index}].arm", "does not identify a summarized arm")
        design_hash = _hash_string(native["design_hash"], f"goav_result.events[{index}].design_hash")
        design_key = (task, arm, design_hash)
        if kind == "design_logged":
            if design_key in designs:
                raise _fail(f"goav_result.events[{index}]", "duplicates a design for task, arm, and hash")
            probabilities = _array(native["probabilities"], f"goav_result.events[{index}].probabilities", ndim=1)
            pi = _array(native["pi"], f"goav_result.events[{index}].pi", ndim=1)
            pi2 = _array(native["pi2"], f"goav_result.events[{index}].pi2", ndim=2)
            if probabilities.size < 2 or probabilities.size & (probabilities.size - 1):
                raise _fail(f"goav_result.events[{index}].probabilities", "must have 2^K entries")
            if pi.size < 1 or probabilities.size != 1 << pi.size or pi2.shape != (pi.size, pi.size):
                raise _fail(f"goav_result.events[{index}]", "probabilities, pi, and pi2 have incompatible shapes")
            _hash_string(native["pi_hash"], f"goav_result.events[{index}].pi_hash")
            _hash_string(native["pi2_hash"], f"goav_result.events[{index}].pi2_hash")
            designs[design_key] = (probabilities, pi, pi2)
            arm_index = arm_indices[arm]
            base = f"goav_design_t{task:06d}_a{arm_index:03d}_{_slug(arm, 'goav arm')}"
            for suffix, array in (("probabilities", probabilities), ("pi", pi), ("pi2", pi2)):
                key = f"{base}_{suffix}"
                if key in arrays:
                    raise _fail(f"goav_result.events[{index}]", "collides with an existing design array key")
                arrays[key] = array
            design_count += 1
        else:
            if design_key not in designs:
                raise _fail(
                    f"goav_result.events[{index}]",
                    "audit request has no preceding design with matching task, arm, and design hash",
                )
            _integer(native["draw"], f"goav_result.events[{index}].draw", minimum=0)
            probabilities, pi, _ = designs[design_key]
            subset_index = _integer(native["subset_index"], f"goav_result.events[{index}].subset_index", minimum=0)
            if subset_index >= probabilities.size:
                raise _fail(f"goav_result.events[{index}].subset_index", "is outside the design support")
            selected = native["selected_indices"]
            if not isinstance(selected, list):
                raise _fail(f"goav_result.events[{index}].selected_indices", "must be a list")
            selected_indices = [
                _integer(item, f"goav_result.events[{index}].selected_indices[{item_index}]", minimum=0)
                for item_index, item in enumerate(selected)
            ]
            if any(item >= pi.size for item in selected_indices) or len(set(selected_indices)) != len(selected_indices):
                raise _fail(f"goav_result.events[{index}].selected_indices", "contains duplicate or out-of-range indices")
            inclusion = _array(
                native["inclusion_probabilities"],
                f"goav_result.events[{index}].inclusion_probabilities",
                ndim=1,
            )
            if inclusion.shape != pi.shape:
                raise _fail(f"goav_result.events[{index}].inclusion_probabilities", "does not match the design dimension")
            selected_labels = _mapping(native["selected_labels"], f"goav_result.events[{index}].selected_labels")
            expected_label_keys = {str(item) for item in selected_indices}
            if set(selected_labels) != expected_label_keys:
                raise _fail(f"goav_result.events[{index}].selected_labels", "must contain exactly the selected indices")
            for label_key, label in selected_labels.items():
                _number(label, f"goav_result.events[{index}].selected_labels.{label_key}")
            audit_count += 1
        events.append(_event(context, index, native))
    try:
        validate_goav_events(
            events,
            run_id=context.run_id,
            seed=context.seed,
            require_envelope=True,
            config=context.config,
            row_arms=tuple(arm_indices),
        )
    except RunIntegrityError as exc:
        raise _fail("goav_result.events", str(exc)) from exc
    gates_raw = _mapping(raw["gates"], "goav_result.gates")
    _exact_keys(
        gates_raw,
        {"status", "formal_evidence"},
        "goav_result.gates",
    )
    source_status = _status(gates_raw["status"], "goav_result.gates.status")
    source_evidence = gates_raw["formal_evidence"]
    if not isinstance(source_evidence, bool):
        raise _fail("goav_result.gates.formal_evidence", "must be boolean")
    gates, final_status = _gates(
        context,
        source_status,
        {},
        source_formal_evidence=source_evidence,
    )
    specific = {
        "epsilon_G": epsilon,
        "epsilon_G_provenance": _json_clone(epsilon_provenance, "goav_result.epsilon_G_provenance"),
        "solver": _json_clone(solver, "goav_result.solver"),
        "design_event_count": design_count,
        "audit_event_count": audit_count,
        "support_violation_count": support_violations,
        "numeric_row_schema": {
            "matrix_key": "goav_row_numeric",
            "present_mask_key": "goav_row_numeric_present",
            "columns": list(_GOAV_ROW_COLUMNS),
            "missing_value": "zero_with_present_mask",
        },
    }
    metrics = _metrics(context, final_status, rows, events, arrays, specific)
    return NormalizedRun(metrics, gates, tuple(rows), arrays, tuple(events))


_JAG_TOP = frozenset({"seed", "estimator_policy", "oracle_scope", "families", "gate"})
_JAG_ESTIMATOR_POLICY = frozenset(
    {"primary_baseline", "max_branching", "branchable_depth_count", "branchable_depths"}
)
_JAG_FAMILY = frozenset(
    {"exact", "structural_audit", "covariance_audit", "predictor_provenance", "summaries"}
)
_JAG_EXACT = frozenset({"probability_sum", "expected_reward", "gradient"})
_JAG_SUMMARY_REQUIRED = frozenset(
    {
        "arm",
        "budget",
        "sample_count",
        "bias",
        "absolute_bias",
        "relative_bias",
        "bias_z_score",
        "bias_z_score_status",
        "mse",
        "budget_times_mse",
        "cosine",
        "allocation",
        "allocation_evidence",
        "paired_replicates",
        "arm_metadata",
        "negative_control",
        "status",
        "status_reason",
    }
)
_JAG_NUMERIC = (
    "bias",
    "absolute_bias",
    "relative_bias",
    "bias_z_score",
    "mse",
    "budget_times_mse",
    "cosine",
)
_JAG_ORACLE_SCOPE = frozenset(
    {"target", "global_tree_oracle_claimed", "max_horizon", "max_budget", "max_frontier_nodes", "max_states"}
)
_JAG_ALLOCATION = frozenset(
    {"replication_count", "unique_plan_count", "total_edge_samples", "plan_histogram"}
)
_JAG_ALLOCATION_EVIDENCE = frozenset(
    {
        "planner",
        "score_formula",
        "branch_counts_frozen_before_child_draws",
        "exact_edge_budget_every_replication",
        "decision_event_count",
        "max_score_identity_error",
        "transport_conditioned_event_count",
        "oracle_frontier_solve_count",
        "oracle_max_optimality_gap",
        "trace_hash_histogram",
    }
)
_JAG_PREDICTOR_PROVENANCE = frozenset(
    {
        "schema_version",
        "completed",
        "calibration_seeds",
        "environment_ids",
        "environment_hashes",
        "row_count",
        "data_hash",
        "provenance_hash",
        "risk_label_baseline",
        "evaluation_seed",
        "oracle_labels_from_evaluation_environment",
        "evaluation_baseline_fit_from_completed_record_only",
    }
)
_JAG_COVARIANCE_AUDIT = frozenset(
    {
        "status",
        "prefix",
        "sample_count",
        "held_out_from_rollout_and_calibration",
        "object",
        "joint_dimension",
        "covariance_dtype",
        "covariance_normalization",
        "predicted_covariance",
        "realized_covariance",
        "realized_joint_mean",
        "numerator",
        "numerator_definition",
        "denominator",
        "denominator_definition",
        "relative_frobenius_error",
        "predicted_covariance_hash",
        "realized_covariance_hash",
    }
)
_JAG_PAIRED_REPLICATE = frozenset(
    {"replication", "crn_stream_id", "gradient_hash", "gradient_delta_hash", "squared_error"}
)
_JAG_ARM_METADATA = frozenset(
    {
        "jag_status",
        "estimator",
        "allocation_rule",
        "moment_source",
        "moment_estimator",
        "moment_continuation_branching",
        "oracle_ablation",
        "deployable",
        "baseline",
        "baseline_frozen_before_evaluation",
        "baseline_evaluation_frequency",
        "baseline_reads_evaluation_outcomes",
        "unbiased_claim",
    }
)
_JAG_LEAF_METADATA = frozenset(
    {"uses_same_tree_outcomes_for_allocation", "negative_control_mechanism", "negative_control_scope"}
)
_JAG_STRUCTURAL_BASE = frozenset(
    {"family", "reward_min", "reward_max", "status", "nodes", "checks"}
)
_JAG_STRUCTURAL_MEASURED = frozenset(
    {"cross_term_prefix", "cross_term_comparator_prefix", "measured_cross_term", "tolerance"}
)
_JAG_STRUCTURAL_NODE = frozenset(
    {
        "prefix",
        "local_features",
        "accumulated_score",
        "entropy",
        "value_variance",
        "cross_term",
        "joint_risk",
        "value_gradient_covariance",
        "value_gradient_covariance_norm",
    }
)
_JAG_STRUCTURAL_NODES = {
    "root_only": frozenset({"audit"}),
    "suffix_only": frozenset({"audit"}),
    "entropy_distractor": frozenset({"distractor", "informative"}),
    "covariance_reversal": frozenset({"left", "right"}),
}
_JAG_STRUCTURAL_CHECKS = {
    "root_only": frozenset(),
    "suffix_only": frozenset(),
    "entropy_distractor": frozenset(
        {"entropy_distractor_valid", "entropy_gap", "risk_gap_informative_minus_distractor"}
    ),
    "covariance_reversal": frozenset(
        {
            "covariance_reversal_valid",
            "equal_value_variance_abs_gap",
            "opposite_cross_term_abs_sum",
            "same_covariance_vector_difference_norm",
            "covariance_vector_nonzero",
            "opposite_nonzero_signs",
            "local_feature_difference_norm",
            "opposite_accumulated_score_vector_sum_norm",
            "equal_accumulated_score_norm_abs_gap",
            "equal_no_cross_risk_abs_gap",
        }
    ),
}
_JAG_GATE_INPUTS = frozenset(
    {"thresholds", "relative_bias", "bias_z_score", "held_out_covariance_audit", "registered_comparisons"}
)
_JAG_GATE_EVIDENCE = frozenset(
    {
        "active_seed",
        "configured_seed_ids",
        "active_seed_declared",
        "observed_seed_count",
        "required_seed_count",
        "rollout_replications_per_cell",
        "available_cell_count",
        "required_cell_count",
        "checks",
        "missing",
    }
)
_JAG_GATE_CHECKS = frozenset(
    {
        "registered_families_present",
        "registered_arms_present",
        "registered_budgets_present",
        "registered_replications_present",
        "all_registered_cells_have_samples",
        "fifty_environment_seed_results_present",
        "diagnostic_structures_pass",
        "registered_covariance_coverage_gate_implemented",
    }
)
_JAG_GENERAL_COMPARISON = frozenset(
    {
        "family",
        "budget",
        "oracle_vs_best_unbiased_relative_mse_reduction",
        "learned_oracle_uniform_gap_closed",
        "learned_minus_uniform_paired_squared_error",
    }
)
_JAG_COVARIANCE_COMPARISON = frozenset(
    {
        "family",
        "budget",
        "full_joint_vs_no_cross_relative_mse_gain",
        "full_joint_minus_no_cross_paired_squared_error",
    }
)
_JAG_PAIRED_DELTA = frozenset({"definition", "paired_count", "mean_delta", "delta_hash"})
_JAG_ARM_STATUSES = frozenset(
    {"OK", "EXACT_FRONTIER_ORACLE", "LAGGED_RIDGE", "NON_JAG_NEGATIVE_CONTROL", "UNAVAILABLE_ORACLE_SCOPE"}
)
_JAG_BIAS_STATUSES = frozenset(
    {"OK", "ZERO_TARGET_NORM", "ZERO_STANDARD_ERROR_NONZERO_DISAGREEMENT", "UNAVAILABLE"}
)


def _validate_jag_allocation(value: Any, path: str, *, sample_count: int, budget: int) -> Mapping[str, Any]:
    allocation = _mapping(value, path)
    _exact_keys(allocation, _JAG_ALLOCATION, path)
    replication_count = _integer(allocation["replication_count"], f"{path}.replication_count", minimum=0)
    unique_plan_count = _integer(allocation["unique_plan_count"], f"{path}.unique_plan_count", minimum=0)
    total_edge_samples = _integer(allocation["total_edge_samples"], f"{path}.total_edge_samples", minimum=0)
    if replication_count != sample_count:
        raise _fail(f"{path}.replication_count", "must equal sample_count")
    if total_edge_samples != sample_count * budget:
        raise _fail(f"{path}.total_edge_samples", "must equal sample_count times budget")
    histogram = _mapping(allocation["plan_histogram"], f"{path}.plan_histogram")
    if unique_plan_count != len(histogram):
        raise _fail(f"{path}.unique_plan_count", "must equal plan_histogram size")
    histogram_total = 0
    for digest, entry_value in histogram.items():
        _hash_string(digest, f"{path}.plan_histogram key")
        entry_path = f"{path}.plan_histogram.{digest}"
        entry = _mapping(entry_value, entry_path)
        _exact_keys(entry, {"count", "edge_count", "canonical_plan"}, entry_path)
        count = _integer(entry["count"], f"{entry_path}.count", minimum=1)
        edge_count = _integer(entry["edge_count"], f"{entry_path}.edge_count", minimum=1)
        if edge_count != budget:
            raise _fail(f"{entry_path}.edge_count", "must equal the summary budget")
        canonical_plan = entry["canonical_plan"]
        if not isinstance(canonical_plan, list) or not canonical_plan:
            raise _fail(f"{entry_path}.canonical_plan", "must be a non-empty list")
        canonical_edge_count = 0
        seen_nodes: set[str] = set()
        for item_index, item in enumerate(canonical_plan):
            item_path = f"{entry_path}.canonical_plan[{item_index}]"
            if not isinstance(item, list) or len(item) != 2:
                raise _fail(item_path, "must be [node_id, edge_count]")
            node = _string(item[0], f"{item_path}[0]")
            if node in seen_nodes:
                raise _fail(item_path, "contains a duplicate node id")
            seen_nodes.add(node)
            canonical_edge_count += _integer(item[1], f"{item_path}[1]", minimum=1)
        if canonical_edge_count != edge_count:
            raise _fail(f"{entry_path}.canonical_plan", "edge counts do not sum to edge_count")
        histogram_total += count
    if histogram_total != replication_count:
        raise _fail(f"{path}.plan_histogram", "counts must sum to replication_count")
    return allocation


def _validate_jag_allocation_evidence(value: Any, path: str) -> Mapping[str, Any]:
    evidence = _mapping(value, path)
    _exact_keys(evidence, _JAG_ALLOCATION_EVIDENCE, path)
    _string(evidence["planner"], f"{path}.planner")
    _string(evidence["score_formula"], f"{path}.score_formula")
    for field in ("branch_counts_frozen_before_child_draws", "exact_edge_budget_every_replication"):
        if not isinstance(evidence[field], bool):
            raise _fail(f"{path}.{field}", "must be boolean")
    for field in (
        "decision_event_count",
        "transport_conditioned_event_count",
        "oracle_frontier_solve_count",
    ):
        _integer(evidence[field], f"{path}.{field}", minimum=0)
    for field in ("max_score_identity_error", "oracle_max_optimality_gap"):
        number = _number(evidence[field], f"{path}.{field}")
        assert number is not None
        if float(number) < 0.0:
            raise _fail(f"{path}.{field}", "must be non-negative")
    histogram = _mapping(evidence["trace_hash_histogram"], f"{path}.trace_hash_histogram")
    for digest, count in histogram.items():
        _hash_string(digest, f"{path}.trace_hash_histogram key")
        _integer(count, f"{path}.trace_hash_histogram.{digest}", minimum=1)
    return evidence


def _jag_integer_list(value: Any, path: str, *, minimum: int = 0) -> list[int]:
    if not isinstance(value, list):
        raise _fail(path, "must be a list")
    return [_integer(item, f"{path}[{index}]", minimum=minimum) for index, item in enumerate(value)]


def _validate_jag_estimator_policy(value: Any, path: str) -> Mapping[str, Any]:
    policy = _mapping(value, path)
    _exact_keys(policy, _JAG_ESTIMATOR_POLICY, path)
    _string(policy["primary_baseline"], f"{path}.primary_baseline")
    _integer(policy["max_branching"], f"{path}.max_branching", minimum=1)
    depth_count = _integer(
        policy["branchable_depth_count"], f"{path}.branchable_depth_count", minimum=0
    )
    depths = _jag_integer_list(policy["branchable_depths"], f"{path}.branchable_depths")
    if depths != list(range(depth_count)):
        raise _fail(f"{path}.branchable_depths", "must enumerate range(branchable_depth_count)")
    return policy


def _validate_jag_structural_node(value: Any, path: str) -> None:
    node = _mapping(value, path)
    _exact_keys(node, _JAG_STRUCTURAL_NODE, path)
    _jag_integer_list(node["prefix"], f"{path}.prefix")
    local_features = _array(node["local_features"], f"{path}.local_features", ndim=1)
    accumulated_score = _array(node["accumulated_score"], f"{path}.accumulated_score", ndim=1)
    covariance = _array(
        node["value_gradient_covariance"], f"{path}.value_gradient_covariance", ndim=1
    )
    if local_features.size < 1 or accumulated_score.size < 1:
        raise _fail(path, "feature and score vectors must not be empty")
    if covariance.shape != accumulated_score.shape:
        raise _fail(f"{path}.value_gradient_covariance", "must match accumulated_score shape")
    for field in (
        "entropy",
        "value_variance",
        "cross_term",
        "joint_risk",
        "value_gradient_covariance_norm",
    ):
        number = _number(node[field], f"{path}.{field}")
        assert number is not None
        if field != "cross_term" and float(number) < 0.0:
            raise _fail(f"{path}.{field}", "must be non-negative")


def _validate_jag_structural_audit(value: Any, path: str, *, family: str) -> Mapping[str, Any]:
    audit = _mapping(value, path)
    if family not in _JAG_STRUCTURAL_NODES:
        raise _fail(f"{path}.family", "is not a registered JAG reward family")
    if "reason" in audit:
        _exact_keys(audit, _JAG_STRUCTURAL_BASE | {"reason"}, path)
    else:
        _exact_keys(audit, _JAG_STRUCTURAL_BASE | _JAG_STRUCTURAL_MEASURED, path)
    if audit["family"] != family:
        raise _fail(f"{path}.family", "does not match family key")
    for field in ("reward_min", "reward_max"):
        _number(audit[field], f"{path}.{field}")
    _string(audit["status"], f"{path}.status")
    nodes = _mapping(audit["nodes"], f"{path}.nodes")
    checks = _mapping(audit["checks"], f"{path}.checks")
    if "reason" in audit:
        _string(audit["reason"], f"{path}.reason")
        if nodes or checks:
            raise _fail(path, "a degenerate structural audit must have empty nodes and checks")
        return audit
    _string(audit["cross_term_prefix"], f"{path}.cross_term_prefix")
    _string(audit["cross_term_comparator_prefix"], f"{path}.cross_term_comparator_prefix")
    _number(audit["measured_cross_term"], f"{path}.measured_cross_term")
    tolerance = _number(audit["tolerance"], f"{path}.tolerance")
    assert tolerance is not None
    if float(tolerance) < 0.0:
        raise _fail(f"{path}.tolerance", "must be non-negative")
    _exact_keys(nodes, _JAG_STRUCTURAL_NODES[family], f"{path}.nodes")
    for node_name, node in nodes.items():
        _validate_jag_structural_node(node, f"{path}.nodes.{node_name}")
    expected_checks = _JAG_STRUCTURAL_CHECKS[family]
    _exact_keys(checks, expected_checks, f"{path}.checks")
    for field, item in checks.items():
        if field.endswith("_valid") or field in {"covariance_vector_nonzero", "opposite_nonzero_signs"}:
            _boolean(item, f"{path}.checks.{field}")
        else:
            checked = _number(item, f"{path}.checks.{field}")
            assert checked is not None
            if float(checked) < 0.0:
                raise _fail(f"{path}.checks.{field}", "must be non-negative")
    return audit


def _validate_jag_covariance_audit(
    value: Any, path: str, *, with_family: bool = False
) -> tuple[Mapping[str, Any], np.ndarray, np.ndarray]:
    audit = _mapping(value, path)
    required = _JAG_COVARIANCE_AUDIT | ({"family"} if with_family else set())
    _exact_keys(audit, required, path)
    if with_family:
        _string(audit["family"], f"{path}.family")
    _string(audit["status"], f"{path}.status")
    _jag_integer_list(audit["prefix"], f"{path}.prefix")
    _integer(audit["sample_count"], f"{path}.sample_count", minimum=2)
    _boolean(
        audit["held_out_from_rollout_and_calibration"],
        f"{path}.held_out_from_rollout_and_calibration",
    )
    joint_dimension = _integer(audit["joint_dimension"], f"{path}.joint_dimension", minimum=1)
    if audit["covariance_dtype"] != "<f8":
        raise _fail(f"{path}.covariance_dtype", "must equal '<f8'")
    if audit["covariance_normalization"] != "population_1_over_n":
        raise _fail(
            f"{path}.covariance_normalization",
            "must equal 'population_1_over_n'",
        )
    predicted = _array(audit["predicted_covariance"], f"{path}.predicted_covariance", ndim=2)
    realized = _array(audit["realized_covariance"], f"{path}.realized_covariance", ndim=2)
    expected_shape = (joint_dimension, joint_dimension)
    if predicted.shape != expected_shape or realized.shape != expected_shape:
        raise _fail(
            path,
            "predicted and realized covariance matrices must match joint_dimension",
        )
    realized_joint_mean = _array(
        audit["realized_joint_mean"], f"{path}.realized_joint_mean", ndim=1
    )
    if realized_joint_mean.shape != (joint_dimension,):
        raise _fail(f"{path}.realized_joint_mean", "must match joint_dimension")
    for field in ("object", "numerator_definition", "denominator_definition"):
        _string(audit[field], f"{path}.{field}")
    numerator = _number(audit["numerator"], f"{path}.numerator")
    denominator = _number(audit["denominator"], f"{path}.denominator")
    assert numerator is not None and denominator is not None
    if float(numerator) < 0.0 or float(denominator) < 0.0:
        raise _fail(path, "covariance audit numerator and denominator must be non-negative")
    relative = _number(
        audit["relative_frobenius_error"], f"{path}.relative_frobenius_error", nullable=True
    )
    if (float(denominator) == 0.0) != (relative is None):
        raise _fail(
            f"{path}.relative_frobenius_error",
            "must be null exactly when the denominator is zero",
        )
    if relative is not None and float(relative) < 0.0:
        raise _fail(f"{path}.relative_frobenius_error", "must be non-negative")
    for field in ("predicted_covariance_hash", "realized_covariance_hash"):
        _hash_string(audit[field], f"{path}.{field}")
    expected_hashes = {
        "predicted_covariance_hash": sha256(predicted.tobytes(order="C")).hexdigest(),
        "realized_covariance_hash": sha256(realized.tobytes(order="C")).hexdigest(),
    }
    for field, expected_hash in expected_hashes.items():
        if audit[field] != expected_hash:
            raise _fail(f"{path}.{field}", "does not match the little-endian float64 matrix")
    expected_numerator = float(np.sum((realized - predicted) ** 2))
    expected_denominator = float(np.sum(predicted**2))
    if float(numerator) != expected_numerator:
        raise _fail(f"{path}.numerator", "does not match the covariance matrices")
    if float(denominator) != expected_denominator:
        raise _fail(f"{path}.denominator", "does not match the predicted covariance matrix")
    expected_relative = (
        None
        if expected_denominator == 0.0
        else float(np.sqrt(expected_numerator / expected_denominator))
    )
    if relative != expected_relative:
        raise _fail(
            f"{path}.relative_frobenius_error",
            "does not match the covariance matrices",
        )
    return audit, predicted, realized


def _validate_jag_predictor_provenance(
    value: Any, path: str, *, context: _Context
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    provenance = _mapping(value, path)
    _exact_keys(provenance, _JAG_PREDICTOR_PROVENANCE, path)
    if provenance["schema_version"] != "jag.calibration.v1":
        raise _fail(f"{path}.schema_version", "must equal 'jag.calibration.v1'")
    _boolean(provenance["completed"], f"{path}.completed")
    seeds = _jag_integer_list(provenance["calibration_seeds"], f"{path}.calibration_seeds")
    environment_ids = provenance["environment_ids"]
    environment_hashes = provenance["environment_hashes"]
    if not isinstance(environment_ids, list) or not isinstance(environment_hashes, list):
        raise _fail(path, "environment_ids and environment_hashes must be lists")
    for index, item in enumerate(environment_ids):
        _string(item, f"{path}.environment_ids[{index}]")
    for index, item in enumerate(environment_hashes):
        _hash_string(item, f"{path}.environment_hashes[{index}]")
    if not seeds or len(seeds) != len(environment_ids) or len(seeds) != len(environment_hashes):
        raise _fail(path, "calibration environment provenance lists must be non-empty and aligned")
    _integer(provenance["row_count"], f"{path}.row_count", minimum=1)
    for field in ("data_hash", "provenance_hash"):
        _hash_string(provenance[field], f"{path}.{field}")
    _string(provenance["risk_label_baseline"], f"{path}.risk_label_baseline")
    if _integer(provenance["evaluation_seed"], f"{path}.evaluation_seed", minimum=0) != context.seed:
        raise _fail(f"{path}.evaluation_seed", "does not match requested seed")
    for field in (
        "oracle_labels_from_evaluation_environment",
        "evaluation_baseline_fit_from_completed_record_only",
    ):
        _boolean(provenance[field], f"{path}.{field}")
    return provenance


def _validate_jag_paired_replicates(value: Any, path: str, *, sample_count: int) -> None:
    if not isinstance(value, list):
        raise _fail(path, "must be a list")
    if len(value) != sample_count:
        raise _fail(path, "must contain exactly sample_count records")
    streams: set[str] = set()
    for index, item_value in enumerate(value):
        item_path = f"{path}[{index}]"
        item = _mapping(item_value, item_path)
        _exact_keys(item, _JAG_PAIRED_REPLICATE, item_path)
        if _integer(item["replication"], f"{item_path}.replication", minimum=0) != index:
            raise _fail(f"{item_path}.replication", "must equal its zero-based list position")
        stream = _string(item["crn_stream_id"], f"{item_path}.crn_stream_id")
        if stream in streams:
            raise _fail(f"{item_path}.crn_stream_id", "must be unique within a summary")
        streams.add(stream)
        for field in ("gradient_hash", "gradient_delta_hash"):
            _hash_string(item[field], f"{item_path}.{field}")
        squared_error = _number(item["squared_error"], f"{item_path}.squared_error")
        assert squared_error is not None
        if float(squared_error) < 0.0:
            raise _fail(f"{item_path}.squared_error", "must be non-negative")


def _validate_jag_arm_metadata(value: Any, path: str, *, arm: str) -> Mapping[str, Any]:
    metadata = _mapping(value, path)
    expected = _JAG_ARM_METADATA | (_JAG_LEAF_METADATA if arm == "leaf_equal_naive" else set())
    _exact_keys(metadata, expected, path)
    for field in (
        "jag_status",
        "estimator",
        "allocation_rule",
        "moment_source",
        "moment_estimator",
        "baseline",
        "baseline_evaluation_frequency",
    ):
        _string(metadata[field], f"{path}.{field}")
    continuation = metadata["moment_continuation_branching"]
    if continuation is not None:
        _integer(continuation, f"{path}.moment_continuation_branching", minimum=1)
    for field in (
        "oracle_ablation",
        "deployable",
        "baseline_frozen_before_evaluation",
        "baseline_reads_evaluation_outcomes",
        "unbiased_claim",
    ):
        _boolean(metadata[field], f"{path}.{field}")
    if arm == "leaf_equal_naive":
        _boolean(
            metadata["uses_same_tree_outcomes_for_allocation"],
            f"{path}.uses_same_tree_outcomes_for_allocation",
        )
        for field in ("negative_control_mechanism", "negative_control_scope"):
            _string(metadata[field], f"{path}.{field}")
    return metadata


def _validate_jag_paired_delta(value: Any, path: str) -> None:
    if value is None:
        return
    delta = _mapping(value, path)
    _exact_keys(delta, _JAG_PAIRED_DELTA, path)
    _string(delta["definition"], f"{path}.definition")
    _integer(delta["paired_count"], f"{path}.paired_count", minimum=1)
    _number(delta["mean_delta"], f"{path}.mean_delta")
    _hash_string(delta["delta_hash"], f"{path}.delta_hash")


def _validate_jag_gate_inputs(
    value: Any, path: str
) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    inputs = _mapping(value, path)
    _exact_keys(inputs, _JAG_GATE_INPUTS, path)
    _mapping(inputs["thresholds"], f"{path}.thresholds")
    for field, expected, numeric_field in (
        ("relative_bias", {"family", "arm", "budget", "value"}, "value"),
        ("bias_z_score", {"family", "arm", "budget", "value", "status"}, "value"),
    ):
        records = inputs[field]
        if not isinstance(records, list):
            raise _fail(f"{path}.{field}", "must be a list")
        for index, record_value in enumerate(records):
            item_path = f"{path}.{field}[{index}]"
            record = _mapping(record_value, item_path)
            _exact_keys(record, expected, item_path)
            _string(record["family"], f"{item_path}.family")
            _string(record["arm"], f"{item_path}.arm")
            _integer(record["budget"], f"{item_path}.budget", minimum=1)
            _number(record[numeric_field], f"{item_path}.{numeric_field}", nullable=True)
            if field == "bias_z_score":
                _string(record["status"], f"{item_path}.status")
    held_out = inputs["held_out_covariance_audit"]
    if not isinstance(held_out, list):
        raise _fail(f"{path}.held_out_covariance_audit", "must be a list")
    held_out_by_family: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(held_out):
        item_path = f"{path}.held_out_covariance_audit[{index}]"
        audit, _, _ = _validate_jag_covariance_audit(item, item_path, with_family=True)
        family = audit["family"]
        assert isinstance(family, str)
        if family in held_out_by_family:
            raise _fail(f"{item_path}.family", "must be unique")
        held_out_by_family[family] = audit
    comparisons = inputs["registered_comparisons"]
    if not isinstance(comparisons, list):
        raise _fail(f"{path}.registered_comparisons", "must be a list")
    for index, item_value in enumerate(comparisons):
        item_path = f"{path}.registered_comparisons[{index}]"
        item = _mapping(item_value, item_path)
        keys = frozenset(item)
        if keys == _JAG_GENERAL_COMPARISON:
            for field in (
                "oracle_vs_best_unbiased_relative_mse_reduction",
                "learned_oracle_uniform_gap_closed",
            ):
                _number(item[field], f"{item_path}.{field}", nullable=True)
            _validate_jag_paired_delta(
                item["learned_minus_uniform_paired_squared_error"],
                f"{item_path}.learned_minus_uniform_paired_squared_error",
            )
        elif keys == _JAG_COVARIANCE_COMPARISON:
            _number(
                item["full_joint_vs_no_cross_relative_mse_gain"],
                f"{item_path}.full_joint_vs_no_cross_relative_mse_gain",
                nullable=True,
            )
            _validate_jag_paired_delta(
                item["full_joint_minus_no_cross_paired_squared_error"],
                f"{item_path}.full_joint_minus_no_cross_paired_squared_error",
            )
        else:
            raise _fail(item_path, "must match one registered comparison schema")
        _string(item["family"], f"{item_path}.family")
        _integer(item["budget"], f"{item_path}.budget", minimum=1)
    return inputs, held_out_by_family


def _validate_jag_gate_evidence(
    value: Any, path: str, *, context: _Context
) -> Mapping[str, Any]:
    evidence = _mapping(value, path)
    _exact_keys(evidence, _JAG_GATE_EVIDENCE, path)
    active_seed = _integer(evidence["active_seed"], f"{path}.active_seed", minimum=0)
    if active_seed != context.seed:
        raise _fail(f"{path}.active_seed", "does not match requested seed")
    configured_seeds = _jag_integer_list(
        evidence["configured_seed_ids"], f"{path}.configured_seed_ids"
    )
    if configured_seeds != context.config["seeds"]:
        raise _fail(f"{path}.configured_seed_ids", "does not match config.seeds")
    active_declared = _boolean(evidence["active_seed_declared"], f"{path}.active_seed_declared")
    if active_declared != (active_seed in configured_seeds):
        raise _fail(f"{path}.active_seed_declared", "disagrees with configured_seed_ids")
    for field in (
        "observed_seed_count",
        "required_seed_count",
        "rollout_replications_per_cell",
        "available_cell_count",
        "required_cell_count",
    ):
        _integer(evidence[field], f"{path}.{field}", minimum=0)
    checks = _mapping(evidence["checks"], f"{path}.checks")
    _exact_keys(checks, _JAG_GATE_CHECKS, f"{path}.checks")
    for field, item in checks.items():
        _boolean(item, f"{path}.checks.{field}")
    missing = evidence["missing"]
    if not isinstance(missing, list):
        raise _fail(f"{path}.missing", "must be a list")
    missing_values = [_string(item, f"{path}.missing[{index}]") for index, item in enumerate(missing)]
    if len(set(missing_values)) != len(missing_values):
        raise _fail(f"{path}.missing", "must not contain duplicates")
    return evidence


def _normalize_jag(context: _Context, raw_result: Any) -> NormalizedRun:
    raw = _mapping(raw_result, "jag_result")
    _exact_keys(raw, _JAG_TOP, "jag_result")
    if _integer(raw["seed"], "jag_result.seed") != context.seed:
        raise _fail("jag_result.seed", "does not match requested seed")
    estimator_policy = _validate_jag_estimator_policy(
        raw["estimator_policy"], "jag_result.estimator_policy"
    )
    oracle_scope = _mapping(raw["oracle_scope"], "jag_result.oracle_scope")
    _exact_keys(oracle_scope, _JAG_ORACLE_SCOPE, "jag_result.oracle_scope")
    _string(oracle_scope["target"], "jag_result.oracle_scope.target")
    _boolean(
        oracle_scope["global_tree_oracle_claimed"],
        "jag_result.oracle_scope.global_tree_oracle_claimed",
    )
    for field in ("max_horizon", "max_budget", "max_frontier_nodes", "max_states"):
        _integer(oracle_scope[field], f"jag_result.oracle_scope.{field}", minimum=1)
    families = _mapping(raw["families"], "jag_result.families")
    if not families:
        raise _fail("jag_result.families", "must not be empty")
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    family_order: list[str] = []
    family_metadata: dict[str, Any] = {}
    family_covariance_audits: dict[str, Mapping[str, Any]] = {}
    allocation_evidence: list[dict[str, Any]] = []
    for family_index, (family, value) in enumerate(families.items()):
        family_name = _string(family, f"jag_result.families key {family_index}")
        family_order.append(family_name)
        native_family = _mapping(value, f"jag_result.families.{family_name}")
        _exact_keys(native_family, _JAG_FAMILY, f"jag_result.families.{family_name}")
        exact = _mapping(native_family["exact"], f"jag_result.families.{family_name}.exact")
        _exact_keys(exact, _JAG_EXACT, f"jag_result.families.{family_name}.exact")
        probability_sum = _number(exact["probability_sum"], f"jag_result.families.{family_name}.exact.probability_sum")
        expected_reward = _number(exact["expected_reward"], f"jag_result.families.{family_name}.exact.expected_reward")
        gradient = _array(exact["gradient"], f"jag_result.families.{family_name}.exact.gradient", ndim=1)
        if gradient.size < 1:
            raise _fail(f"jag_result.families.{family_name}.exact.gradient", "must not be empty")
        gradient_key = f"jag_exact_gradient_f{family_index:03d}_{_slug(family_name, 'JAG family')}"
        arrays[gradient_key] = gradient
        audit = _validate_jag_structural_audit(
            native_family["structural_audit"],
            f"jag_result.families.{family_name}.structural_audit",
            family=family_name,
        )
        covariance_audit, predicted_covariance, realized_covariance = _validate_jag_covariance_audit(
            native_family["covariance_audit"],
            f"jag_result.families.{family_name}.covariance_audit",
        )
        if covariance_audit["joint_dimension"] != gradient.size + 1:
            raise _fail(
                f"jag_result.families.{family_name}.covariance_audit.joint_dimension",
                "must equal one plus the exact gradient dimension",
            )
        family_covariance_audits[family_name] = covariance_audit
        covariance_base = (
            f"jag_covariance_f{family_index:03d}_{_slug(family_name, 'JAG family')}"
        )
        predicted_covariance_key = f"{covariance_base}_predicted"
        realized_covariance_key = f"{covariance_base}_realized"
        arrays[predicted_covariance_key] = predicted_covariance
        arrays[realized_covariance_key] = realized_covariance
        predictor = _validate_jag_predictor_provenance(
            native_family["predictor_provenance"],
            f"jag_result.families.{family_name}.predictor_provenance",
            context=context,
        )
        summaries = native_family["summaries"]
        if not isinstance(summaries, list) or not summaries:
            raise _fail(f"jag_result.families.{family_name}.summaries", "must be a non-empty list")
        for summary_index, summary_value in enumerate(summaries):
            path = f"jag_result.families.{family_name}.summaries[{summary_index}]"
            summary = _mapping(summary_value, path)
            _exact_keys(summary, _JAG_SUMMARY_REQUIRED, path)
            arm = _string(summary["arm"], f"{path}.arm")
            budget = _integer(summary["budget"], f"{path}.budget", minimum=1)
            sample_count = _integer(summary["sample_count"], f"{path}.sample_count", minimum=0)
            negative_control = _boolean(summary["negative_control"], f"{path}.negative_control")
            if negative_control != (arm == "leaf_equal_naive"):
                raise _fail(f"{path}.negative_control", "disagrees with the registered arm")
            arm_status = _string(summary["status"], f"{path}.status")
            if arm_status not in _JAG_ARM_STATUSES:
                raise _fail(f"{path}.status", "is not a registered JAG summary status")
            status_reason = summary["status_reason"]
            if status_reason is not None:
                _string(status_reason, f"{path}.status_reason")
            bias_status = _string(summary["bias_z_score_status"], f"{path}.bias_z_score_status")
            if bias_status not in _JAG_BIAS_STATUSES:
                raise _fail(f"{path}.bias_z_score_status", "is not a registered diagnostic status")
            undefined = False
            for column in _JAG_NUMERIC:
                checked = _number(summary[column], f"{path}.{column}", nullable=True)
                undefined = undefined or checked is None
                if checked is not None and column in {
                    "bias",
                    "absolute_bias",
                    "relative_bias",
                    "mse",
                    "budget_times_mse",
                } and float(checked) < 0.0:
                    raise _fail(f"{path}.{column}", "must be non-negative")
                if checked is not None and column == "cosine" and not -1.0 <= float(checked) <= 1.0:
                    raise _fail(f"{path}.cosine", "must lie in [-1, 1]")
            if undefined and status_reason is None and bias_status == "OK":
                raise _fail(path, "undefined statistics require status_reason or non-OK bias_z_score_status")
            if sample_count == 0 and status_reason is None:
                raise _fail(f"{path}.status_reason", "is required when sample_count is zero")
            if arm_status == "UNAVAILABLE_ORACLE_SCOPE" and sample_count != 0:
                raise _fail(f"{path}.sample_count", "must be zero for an unavailable arm")
            if arm_status != "UNAVAILABLE_ORACLE_SCOPE" and sample_count == 0:
                raise _fail(f"{path}.sample_count", "must be positive for an available arm")
            allocation = _validate_jag_allocation(
                summary["allocation"], f"{path}.allocation", sample_count=sample_count, budget=budget
            )
            allocation_details = _validate_jag_allocation_evidence(
                summary["allocation_evidence"], f"{path}.allocation_evidence"
            )
            _validate_jag_paired_replicates(
                summary["paired_replicates"], f"{path}.paired_replicates", sample_count=sample_count
            )
            _validate_jag_arm_metadata(
                summary["arm_metadata"], f"{path}.arm_metadata", arm=arm
            )
            row_native = {"family": family_name, **_json_clone(summary, path)}
            rows.append(_row(context, len(rows), "arm_budget_summary", row_native))
            allocation_evidence.append(
                {
                    "family": family_name,
                    "arm": arm,
                    "budget": budget,
                    "allocation": _json_clone(allocation, f"{path}.allocation"),
                    "allocation_evidence": _json_clone(allocation_details, f"{path}.allocation_evidence"),
                    "arm_status": summary["status"],
                }
            )
        covariance_metadata = _json_clone(
            covariance_audit, f"jag_result.families.{family_name}.covariance_audit"
        )
        del covariance_metadata["predicted_covariance"]
        del covariance_metadata["realized_covariance"]
        covariance_metadata["predicted_covariance_array_key"] = predicted_covariance_key
        covariance_metadata["realized_covariance_array_key"] = realized_covariance_key
        metadata: dict[str, Any] = {
            "probability_sum": probability_sum,
            "expected_reward": expected_reward,
            "structural_audit": _json_clone(audit, f"jag_result.families.{family_name}.structural_audit"),
            "covariance_audit": covariance_metadata,
            "predictor_provenance": _json_clone(predictor, f"jag_result.families.{family_name}.predictor_provenance"),
            "summary_count": len(summaries),
            "exact_gradient_array_key": gradient_key,
        }
        family_metadata[family_name] = metadata
    gate = _mapping(raw["gate"], "jag_result.gate")
    _exact_keys(
        gate,
        {"status", "reason", "inputs", "evidence"},
        "jag_result.gate",
    )
    source_status = _status(gate["status"], "jag_result.gate.status")
    gate_inputs, gate_covariance_audits = _validate_jag_gate_inputs(
        gate["inputs"], "jag_result.gate.inputs"
    )
    if set(gate_covariance_audits) != set(family_covariance_audits):
        raise _fail(
            "jag_result.gate.inputs.held_out_covariance_audit",
            "must contain exactly one audit for every family",
        )
    for family_name, family_audit in family_covariance_audits.items():
        gate_audit = gate_covariance_audits[family_name]
        gate_without_family = {key: item for key, item in gate_audit.items() if key != "family"}
        if gate_without_family != family_audit:
            raise _fail(
                "jag_result.gate.inputs.held_out_covariance_audit",
                f"audit for family {family_name!r} disagrees with the family audit",
            )
    gate_evidence = _validate_jag_gate_evidence(
        gate["evidence"], "jag_result.gate.evidence", context=context
    )
    gates, final_status = _gates(
        context,
        source_status,
        gate_inputs,
        source_reason=gate["reason"],
    )
    gates["evidence"] = _json_clone(
        gate_evidence,
        "jag_result.gate.evidence",
    )
    specific = {
        "family_order": family_order,
        "family_metadata": family_metadata,
        "allocation_evidence": allocation_evidence,
        "oracle_scope": _json_clone(oracle_scope, "jag_result.oracle_scope"),
        "estimator_policy": _json_clone(estimator_policy, "jag_result.estimator_policy"),
    }
    metrics = _metrics(context, final_status, rows, (), arrays, specific)
    return NormalizedRun(metrics, gates, tuple(rows), arrays, ())


def normalize_result(
    alias: str,
    canonical_direction: str,
    config: Any,
    seed: int,
    run_id: str,
    raw_result: Any,
) -> NormalizedRun:
    """Normalize exactly one registered native Phase-0 result.

    Validation completes before the returned object is created.  Callers must
    therefore invoke this function before constructing a run directory.
    """

    context = _context(alias, canonical_direction, config, seed, run_id)
    adapters = {
        "jag": _normalize_jag,
        "pbpf": _normalize_pbpf,
        "goav": _normalize_goav,
    }
    normalized = adapters[context.alias](context, raw_result)
    # Last-line guarantee for every JSON artifact class.  This catches any
    # accidental numpy scalar or non-finite value introduced above.
    for name, value in (
        ("metrics", normalized.metrics),
        ("gates", normalized.gates),
        ("rows", normalized.rows),
        ("events", normalized.events),
    ):
        _json_clone(value, name)
    for key, array in normalized.arrays.items():
        if _SAFE_ARRAY_KEY.fullmatch(key) is None or key.endswith(".npy") or array.dtype.kind == "O":
            raise _fail(f"arrays.{key}", "array key or dtype is unsafe")
        if not np.isfinite(array).all():
            raise _fail(f"arrays.{key}", "contains a non-finite value")
    return normalized


__all__ = ["DIRECTIONS", "NormalizedRun", "ResultNormalizationError", "normalize_result"]
