"""Deterministic random streams and immutable run-directory artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any
import zipfile

import numpy as np


_REQUIRED_ARTIFACTS = frozenset(
    {
        "resolved_config.json",
        "manifest.json",
        "metrics.json",
        "gates.json",
        "events.jsonl",
        "rows.jsonl",
        "arrays.npz",
        "report.md",
    }
)
_FINAL_ARTIFACTS = _REQUIRED_ARTIFACTS | {"checksums.json", "COMPLETE"}
_DIRECTION_ALIASES = {
    "jag_tree": "jag",
    "predictive_belief_particle_filter": "pbpf",
    "gradient_optimal_active_verification": "goav",
}
_SAFE_BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
_SAFE_ARRAY_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_EVENT_ENVELOPE_FIELDS = {"schema_version", "run_id", "direction", "seed", "sequence"}
_ROW_ENVELOPE_FIELDS = {"schema_version", "run_id", "direction", "seed", "row_index", "record_type"}
_MANIFEST_FIELDS = {
    "schema_version",
    "run_id",
    "direction",
    "direction_alias",
    "output_identity",
    "seed",
    "profile",
    "status",
    "config_sha256",
    "source_configs",
    "git",
    "environment",
    "row_count",
    "event_count",
    "array_keys",
}
_METRICS_FIELDS = {
    "schema_version",
    "direction",
    "seed",
    "profile",
    "status",
    "row_count",
    "event_count",
    "array_keys",
}
_GATES_FIELDS = {
    "schema_version",
    "direction",
    "seed",
    "profile",
    "status",
    "formal_evidence",
    "source_status",
    "reason",
    "inputs",
}
_ENVIRONMENT_FIELDS = {
    "kind",
    "python",
    "numpy",
    "scipy",
    "pyyaml",
    "model_backend",
    "model_checkpoint",
    "container_digest",
}
_GOAV_DESIGN_ARMS = {
    "uniform_subset_ht",
    "uniform_subset_aipw",
    "entropy_subset_aipw",
    "killrate_subset_aipw",
    "poisson_neyman_aipw",
    "bayes_voi_aipw",
    "goav_exact_subset_aipw",
    "full_audit",
}
_GOAV_CONTROL_ARMS = {"cheap_only", "deterministic_topk_invalid"}
_GOAV_ALLOWED_ARMS = _GOAV_DESIGN_ARMS | _GOAV_CONTROL_ARMS
_PBPF_COLUMNS = (
    "nll",
    "brier",
    "posterior_kl",
    "ess",
    "resampling",
    "hpd_true_inclusion",
)
_PBPF_ROW_REQUIRED = {
    "episode",
    "prefix",
    "horizon",
    "arm",
    *_PBPF_COLUMNS,
}
_PBPF_ROW_OPTIONAL = {
    "pre_resampling_ess",
    "pre_resampling_ess_history",
    "resampling_rate",
    "unique_ancestor_ratio",
    "post_rejuvenation_unique_state_ratio",
    "resampling_events",
    "particles_per_correct_equivalence_class",
    "diagnostic",
}
_PBPF_GATE_ARRAYS = {
    "paired_exact_vs_prior_inputs": "pbpf_gate_prefix4_paired_exact_vs_prior",
    "paired_exact_vs_map_inputs": "pbpf_gate_prefix4_paired_exact_vs_map",
    "paired_p16_to_exact_inputs": "pbpf_gate_prefix4_paired_p16_to_exact",
    "hpd_coverage_inputs": "pbpf_gate_prefix4_hpd_coverage",
}
_PBPF_GATE_INPUT_FIELDS = {
    "hpd_nominal",
    "by_prefix_horizon",
    "prefix_4",
    "exact_vs_prior_nll_delta",
    "exact_mixture_vs_map_nll_delta",
    "p16_to_exact_nll_gap",
    "map_gap_fraction_closed",
}
_PBPF_GATE_CELL_FIELDS = {
    "exact_mixture_vs_map_nll_delta",
    "exact_vs_map_paired_ci",
    "exact_vs_prior_nll_delta",
    "expected_episode_ids",
    "hpd_coverage_ci",
    "hpd_coverage_inputs",
    "map_gap_fraction_closed",
    "p16_collapse_episode_ids",
    "p16_complete_coverage",
    "p16_missing_episode_ids",
    "p16_to_exact_nll_gap",
    "p16_to_exact_one_sided_upper_95",
    "paired_exact_vs_map_inputs",
    "paired_exact_vs_prior_inputs",
    "paired_p16_to_exact_inputs",
    "particle_collapse_count",
}
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
_JAG_SUMMARY_FIELDS = {
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
_JAG_ALLOCATION_FIELDS = {
    "replication_count", "unique_plan_count", "total_edge_samples", "plan_histogram"
}
_JAG_ALLOCATION_EVIDENCE_FIELDS = {
    "planner", "score_formula", "branch_counts_frozen_before_child_draws",
    "exact_edge_budget_every_replication", "decision_event_count",
    "max_score_identity_error", "transport_conditioned_event_count",
    "oracle_frontier_solve_count", "oracle_max_optimality_gap", "trace_hash_histogram",
}
_JAG_PAIRED_REPLICATE_FIELDS = {
    "replication", "crn_stream_id", "gradient_hash", "gradient_delta_hash", "squared_error"
}
_JAG_ARM_METADATA_FIELDS = {
    "jag_status", "estimator", "allocation_rule", "moment_source", "moment_estimator",
    "moment_continuation_branching", "oracle_ablation", "deployable", "baseline",
    "baseline_frozen_before_evaluation", "baseline_evaluation_frequency",
    "baseline_reads_evaluation_outcomes", "unbiased_claim",
}
_JAG_LEAF_METADATA_FIELDS = {
    "uses_same_tree_outcomes_for_allocation", "negative_control_mechanism",
    "negative_control_scope",
}
_JAG_PREDICTOR_FIELDS = {
    "schema_version", "completed", "calibration_seeds", "environment_ids",
    "environment_hashes", "row_count", "data_hash", "provenance_hash",
    "risk_label_baseline", "evaluation_seed", "oracle_labels_from_evaluation_environment",
    "evaluation_baseline_fit_from_completed_record_only",
}
_JAG_GATE_INPUT_FIELDS = {
    "thresholds", "relative_bias", "bias_z_score", "held_out_covariance_audit",
    "registered_comparisons",
}
_JAG_GATE_EVIDENCE_FIELDS = {
    "active_seed", "configured_seed_ids", "active_seed_declared", "observed_seed_count",
    "required_seed_count", "rollout_replications_per_cell", "available_cell_count",
    "required_cell_count", "checks", "missing",
}
_JAG_GATE_CHECK_FIELDS = {
    "registered_families_present", "registered_arms_present", "registered_budgets_present",
    "registered_replications_present", "all_registered_cells_have_samples",
    "fifty_environment_seed_results_present", "diagnostic_structures_pass",
    "registered_covariance_coverage_gate_implemented",
}
_DIRECTION_METRICS_FIELDS = {
    "predictive_belief_particle_filter": {
        "arm_counts",
        "collapse_counts",
        "rejected_arms",
        "diagnostic_column_schema",
    },
    "gradient_optimal_active_verification": {
        "epsilon_G",
        "epsilon_G_provenance",
        "solver",
        "design_event_count",
        "audit_event_count",
        "support_violation_count",
        "numeric_row_schema",
    },
    "jag_tree": {
        "family_order",
        "family_metadata",
        "allocation_evidence",
        "oracle_scope",
        "estimator_policy",
    },
}
_CONFIG_SCHEMA_BY_DIRECTION = {
    "jag_tree": "0.1",
    "predictive_belief_particle_filter": "0.1",
    "gradient_optimal_active_verification": "0.2",
}


class RunIntegrityError(ValueError):
    """Raised when a run cannot be sealed or fails read-only verification."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def named_rng(seed: int, label: str) -> np.random.Generator:
    """Create a call-order-independent RNG stream from an outer seed and label."""

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    if not isinstance(label, str):
        raise TypeError("label must be a string")
    payload = _canonical_json({"label": label, "seed": int(seed)}).encode("utf-8")
    entropy = int.from_bytes(sha256(payload).digest(), byteorder="big")
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _safe_artifact_name(name: str) -> str:
    if not isinstance(name, str) or not name or Path(name).name != name or not _SAFE_BASENAME.fullmatch(name):
        raise ValueError("artifact names must be safe basenames")
    return name


def _safe_identity(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_BASENAME.fullmatch(value) or value in {".", ".."}:
        raise RunIntegrityError(f"{field} must be a safe non-empty basename")
    return value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_regular_unsymlinked(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return not stat.S_ISLNK(metadata.st_mode) and stat.S_ISREG(metadata.st_mode)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RunIntegrityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise RunIntegrityError(f"non-finite JSON value: {value}")


def _assert_finite_json(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RunIntegrityError("JSON contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _assert_finite_json(item)
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise RunIntegrityError("JSON object keys must be strings")
        for item in value.values():
            _assert_finite_json(item)
        return
    raise RunIntegrityError(f"unsupported JSON value type: {type(value).__name__}")


def _parse_canonical_json(raw: bytes, name: str) -> Any:
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
        _assert_finite_json(value)
        canonical = (_canonical_json(value) + "\n").encode("utf-8")
    except RunIntegrityError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise RunIntegrityError(f"{name} is not valid canonical JSON") from exc
    if raw != canonical:
        raise RunIntegrityError(f"{name} is not encoded as canonical JSON with one trailing newline")
    return value


def _parse_canonical_jsonl(raw: bytes, name: str) -> list[Any]:
    if raw == b"":
        return []
    if not raw.endswith(b"\n"):
        raise RunIntegrityError(f"{name} must end with a newline")
    records: list[Any] = []
    for line_number, line in enumerate(raw.splitlines(keepends=True), start=1):
        if line == b"\n":
            raise RunIntegrityError(f"{name} contains a blank line at {line_number}")
        records.append(_parse_canonical_json(line, f"{name}:{line_number}"))
    return records


def _normalise_array(key: str, value: Any) -> np.ndarray:
    if not isinstance(key, str) or not _SAFE_ARRAY_KEY.fullmatch(key) or key.endswith(".npy"):
        raise ValueError(f"unsafe NPZ member key: {key!r}")
    array = np.asarray(value)
    if array.dtype.kind not in "buifc":
        raise TypeError(f"array {key!r} must have a numeric or boolean dtype")
    if array.dtype.kind in "fc" and not np.isfinite(array).all():
        raise ValueError(f"array {key!r} contains a non-finite value")
    little_endian_dtype = array.dtype.newbyteorder("<")
    return np.ascontiguousarray(array.astype(little_endian_dtype, copy=False))


def _deterministic_npz_bytes(arrays: Mapping[str, Any]) -> bytes:
    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must be a mapping")
    normalized: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if key in normalized:
            raise ValueError(f"duplicate NPZ member key: {key!r}")
        normalized[key] = _normalise_array(key, value)

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.comment = b""
        for key in sorted(normalized):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, normalized[key], version=(2, 0), allow_pickle=False)
            member = zipfile.ZipInfo(f"{key}.npy", date_time=_FIXED_ZIP_TIME)
            member.compress_type = zipfile.ZIP_STORED
            member.create_system = 3
            member.external_attr = 0o100600 << 16
            member.comment = b""
            member.extra = b""
            archive.writestr(member, payload.getvalue(), compress_type=zipfile.ZIP_STORED)
    return output.getvalue()


def _parse_deterministic_npz(raw: bytes) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw), mode="r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise RunIntegrityError("arrays.npz contains duplicate members")
            if names != sorted(names):
                raise RunIntegrityError("arrays.npz members are not sorted")
            keys: list[str] = []
            for member in members:
                if (
                    member.is_dir()
                    or not member.filename.endswith(".npy")
                    or member.compress_type != zipfile.ZIP_STORED
                    or member.date_time != _FIXED_ZIP_TIME
                ):
                    raise RunIntegrityError("arrays.npz contains a non-canonical member")
                key = member.filename[:-4]
                if not _SAFE_ARRAY_KEY.fullmatch(key) or key.endswith(".npy"):
                    raise RunIntegrityError(f"arrays.npz contains unsafe member {member.filename!r}")
                keys.append(key)
        arrays: dict[str, np.ndarray] = {}
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if archive.files != keys:
                raise RunIntegrityError("arrays.npz member index is inconsistent")
            for key in keys:
                arrays[key] = _normalise_array(key, archive[key])
        if _deterministic_npz_bytes(arrays) != raw:
            raise RunIntegrityError("arrays.npz is not in deterministic canonical form")
        return arrays
    except RunIntegrityError:
        raise
    except (OSError, EOFError, ValueError, TypeError, zipfile.BadZipFile) as exc:
        raise RunIntegrityError("arrays.npz is malformed or unsafe") from exc


def _array_hash(values: np.ndarray) -> str:
    return sha256(np.asarray(values, dtype="<f8").tobytes(order="C")).hexdigest()


def _strict_float_array(value: Any, name: str, dimensions: int) -> np.ndarray:
    def check_scalars(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                check_scalars(child)
            return
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise RunIntegrityError(f"{name} must not coerce booleans or strings")
        try:
            if not math.isfinite(float(item)):
                raise RunIntegrityError(f"{name} contains a non-finite number")
        except (TypeError, ValueError, OverflowError) as exc:
            raise RunIntegrityError(f"{name} contains an invalid number") from exc

    if not isinstance(value, list):
        raise RunIntegrityError(f"{name} must be a JSON array")
    check_scalars(value)
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RunIntegrityError(f"{name} must be a rectangular numeric array") from exc
    if array.ndim != dimensions or not np.isfinite(array).all():
        raise RunIntegrityError(f"{name} must be a finite {dimensions}-dimensional array")
    return array


def _validate_common_event_envelope(
    event: Mapping[str, Any], position: int, run_id: str, direction: str, seed: int
) -> None:
    if not _EVENT_ENVELOPE_FIELDS <= set(event):
        raise RunIntegrityError(f"event {position} is missing its common envelope")
    if event["schema_version"] != "phase0.event.v1":
        raise RunIntegrityError(f"event {position} has an unknown schema_version")
    event_run_id = _safe_identity(event["run_id"], f"event {position} run_id")
    event_seed = _require_int(event["seed"], f"event {position} seed")
    if event_seed < 0:
        raise RunIntegrityError(f"event {position} seed must be non-negative")
    if (
        event_run_id != run_id
        or event["direction"] != direction
        or event_seed != seed
    ):
        raise RunIntegrityError(f"event {position} identity disagrees with the manifest")
    if _require_int(event["sequence"], f"event {position} sequence") != position:
        raise RunIntegrityError(f"event {position} has a non-canonical sequence")


def validate_goav_events(
    events: Iterable[Mapping[str, Any]],
    *,
    run_id: str | None = None,
    seed: int | None = None,
    require_envelope: bool = False,
    config: Mapping[str, Any] | None = None,
    row_arms: Iterable[str] | None = None,
) -> None:
    """Validate GOAV designs, audit support, ordering, counts, and deterministic replay."""

    records = list(events)
    strict = config is not None
    if strict and (run_id is None or seed is None or not require_envelope):
        raise RunIntegrityError("config-aware GOAV validation requires run_id, seed, and envelopes")
    observed_run_id = run_id
    observed_seed = seed
    envelope_mode: bool | None = None

    tasks: int | None = None
    candidates: int | None = None
    draws_per_design: int | None = None
    budget_fraction: float | None = None
    inclusion_floor: float | None = None
    configured_arms: set[str] | None = None
    expected_designs: set[tuple[int, str]] | None = None
    if strict:
        config_map = _require_mapping(config, "GOAV config")
        if config_map.get("direction") != "gradient_optimal_active_verification":
            raise RunIntegrityError("GOAV config has the wrong direction")
        phase0 = _require_mapping(config_map.get("phase0"), "GOAV config phase0")
        tasks = _require_int(phase0.get("tasks_min"), "GOAV tasks_min")
        candidates = _require_int(phase0.get("group_size"), "GOAV group_size")
        draws_per_design = _require_int(
            phase0.get("subset_draws_per_group_design"), "GOAV subset draws"
        )
        acquisition = _require_mapping(config_map.get("acquisition"), "GOAV config acquisition")
        budget_value = phase0.get(
            "primary_budget_fraction", acquisition.get("primary_budget_fraction")
        )
        floor_value = acquisition.get("primary_inclusion_floor")
        if (
            isinstance(budget_value, bool)
            or not isinstance(budget_value, (int, float))
            or not math.isfinite(float(budget_value))
            or not 0.0 < float(budget_value) <= 1.0
            or isinstance(floor_value, bool)
            or not isinstance(floor_value, (int, float))
            or not math.isfinite(float(floor_value))
            or not 0.0 < float(floor_value) <= 1.0
        ):
            raise RunIntegrityError("GOAV budget fraction and inclusion floor must be finite in (0,1]")
        budget_fraction = float(budget_value)
        inclusion_floor = float(floor_value)
        arms = phase0.get("arms")
        if (
            tasks < 1
            or candidates < 1
            or candidates > 20
            or draws_per_design < 1
            or not isinstance(arms, list)
            or not arms
            or any(not isinstance(arm, str) or not arm for arm in arms)
            or len(set(arms)) != len(arms)
        ):
            raise RunIntegrityError("GOAV phase0 task/candidate/draw/arm registration is invalid")
        configured_arms = set(arms)
        expected_designs = {
            (task, arm)
            for task in range(tasks)
            for arm in arms
            if arm in _GOAV_DESIGN_ARMS
        }
        if row_arms is None or set(row_arms) != configured_arms:
            raise RunIntegrityError("GOAV row arms disagree with configured arms")

    designs: dict[tuple[int, str], tuple[str, np.ndarray, np.ndarray]] = {}
    audit_draws: dict[tuple[int, str], set[int]] = {}
    tasks_with_audits: set[int] = set()
    trusted_labels: dict[tuple[int, int], int] = {}
    for position, raw_event in enumerate(records):
        if not isinstance(raw_event, Mapping):
            raise RunIntegrityError(f"GOAV event {position} must be an object")
        event = dict(raw_event)
        envelope_fields = _EVENT_ENVELOPE_FIELDS & set(event)
        has_envelope = bool(envelope_fields)
        if has_envelope and envelope_fields != _EVENT_ENVELOPE_FIELDS:
            raise RunIntegrityError(f"GOAV event {position} has an incomplete common envelope")
        if require_envelope and not has_envelope:
            raise RunIntegrityError(f"GOAV event {position} is missing its common envelope")
        if envelope_mode is None:
            envelope_mode = has_envelope
        elif envelope_mode != has_envelope:
            raise RunIntegrityError("GOAV events inconsistently use the common envelope")
        if has_envelope:
            event_run_id = _safe_identity(event["run_id"], f"GOAV event {position} run_id")
            event_seed = _require_int(event["seed"], f"GOAV event {position} seed")
            if observed_run_id is None:
                observed_run_id = event_run_id
            if observed_seed is None:
                observed_seed = event_seed
            assert observed_run_id is not None and observed_seed is not None
            _validate_common_event_envelope(
                event,
                position,
                observed_run_id,
                "gradient_optimal_active_verification",
                observed_seed,
            )

        task = _require_int(event.get("task"), f"GOAV event {position} task")
        arm = event.get("arm")
        design_hash = event.get("design_hash")
        if task < 0 or (tasks is not None and task >= tasks):
            raise RunIntegrityError(f"GOAV event {position} has an out-of-range task")
        if not isinstance(arm, str) or not arm:
            raise RunIntegrityError(f"GOAV event {position} has an invalid arm")
        if configured_arms is not None and arm not in configured_arms:
            raise RunIntegrityError(f"GOAV event {position} uses an unconfigured arm")
        if row_arms is not None and arm not in set(row_arms):
            raise RunIntegrityError(f"GOAV event {position} arm is absent from rows")
        if not isinstance(design_hash, str) or not _SHA256.fullmatch(design_hash):
            raise RunIntegrityError(f"GOAV event {position} has an invalid design hash")
        identity = (task, arm)

        if event.get("event") == "design_logged":
            allowed = {
                "event",
                "task",
                "arm",
                "design_hash",
                "probabilities",
                "pi",
                "pi2",
                "pi_hash",
                "pi2_hash",
            } | (_EVENT_ENVELOPE_FIELDS if has_envelope else set())
            if set(event) != allowed or arm not in _GOAV_DESIGN_ARMS:
                raise RunIntegrityError("GOAV design event has missing, unexpected, or invalid fields")
            if identity in designs or (strict and task in tasks_with_audits):
                raise RunIntegrityError("GOAV design is duplicated or logged after a task audit")
            probabilities = _strict_float_array(event["probabilities"], "probabilities", 1)
            pi = _strict_float_array(event["pi"], "pi", 1)
            pi2 = _strict_float_array(event["pi2"], "pi2", 2)
            subset_count = len(probabilities)
            if subset_count < 2 or subset_count & (subset_count - 1):
                raise RunIntegrityError("GOAV probabilities must contain exactly 2^K entries")
            design_candidates = subset_count.bit_length() - 1
            if (
                design_candidates > 20
                or (candidates is not None and design_candidates != candidates)
                or pi.shape != (design_candidates,)
                or pi2.shape != (design_candidates, design_candidates)
            ):
                raise RunIntegrityError("GOAV design dimensions disagree with the registration")
            if not np.isclose(probabilities.sum(), 1.0, atol=1e-15, rtol=1e-15):
                raise RunIntegrityError("GOAV probabilities must sum to one")
            subset_indices = np.arange(subset_count, dtype=np.uint64)[:, None]
            bits = np.arange(design_candidates, dtype=np.uint64)[None, :]
            subsets = ((subset_indices >> bits) & 1).astype(bool)
            expected_pi = subsets.T @ probabilities
            expected_pi2 = subsets.T @ (probabilities[:, None] * subsets)
            if not np.array_equal(pi, expected_pi) or not np.array_equal(pi2, expected_pi2):
                raise RunIntegrityError("GOAV pi/pi2 do not exactly match the bitmask distribution")
            if arm == "full_audit":
                expected_probabilities = np.zeros(subset_count, dtype=np.float64)
                expected_probabilities[-1] = 1.0
                if (
                    not np.array_equal(probabilities, expected_probabilities)
                    or not np.array_equal(pi, np.ones(design_candidates))
                    or not np.array_equal(pi2, np.ones((design_candidates, design_candidates)))
                ):
                    raise RunIntegrityError("GOAV full_audit must select only the all-candidate subset")
            elif (probabilities <= 0.0).any() or (pi <= 0.0).any():
                raise RunIntegrityError("GOAV randomized designs require strict full support and positive pi")
            elif strict:
                assert budget_fraction is not None and inclusion_floor is not None
                expected_inclusions = design_candidates * budget_fraction
                if not np.isclose(pi.sum(), expected_inclusions, atol=1e-12, rtol=1e-12):
                    raise RunIntegrityError("GOAV design inclusion mass disagrees with the registered budget")
                if (pi < inclusion_floor).any():
                    raise RunIntegrityError("GOAV design violates the registered inclusion floor")
            if design_hash != _array_hash(probabilities):
                raise RunIntegrityError("GOAV design hash does not match probabilities")
            if event["pi_hash"] != _array_hash(pi) or event["pi2_hash"] != _array_hash(pi2):
                raise RunIntegrityError("GOAV inclusion-probability hash mismatch")
            designs[identity] = (design_hash, probabilities, pi)
            audit_draws[identity] = set()
            continue

        if event.get("event") == "audit_request":
            allowed = {
                "event",
                "task",
                "arm",
                "design_hash",
                "subset_index",
                "selected_indices",
                "inclusion_probabilities",
                "selected_labels",
            }
            if strict or "draw" in event:
                allowed.add("draw")
            allowed |= _EVENT_ENVELOPE_FIELDS if has_envelope else set()
            if set(event) != allowed or identity not in designs:
                raise RunIntegrityError("GOAV audit has invalid fields or precedes its design")
            tasks_with_audits.add(task)
            stored_hash, probabilities, pi = designs[identity]
            if design_hash != stored_hash:
                raise RunIntegrityError("GOAV audit design hash disagrees with its logged design")
            draw_value = event.get("draw")
            if strict:
                draw = _require_int(draw_value, "GOAV audit draw")
                if draw < 0 or draw in audit_draws[identity]:
                    raise RunIntegrityError("GOAV audit draw must be unique and non-negative")
                audit_draws[identity].add(draw)
            else:
                draw = _require_int(draw_value, "GOAV audit draw") if draw_value is not None else None
                audit_draws[identity].add(
                    draw if draw is not None else len(audit_draws[identity])
                )
            subset_index = _require_int(event.get("subset_index"), "GOAV subset_index")
            if subset_index < 0 or subset_index >= len(probabilities) or probabilities[subset_index] <= 0.0:
                raise RunIntegrityError("GOAV audit references an impossible subset")
            if strict:
                assert observed_seed is not None and draw is not None
                expected_index = int(
                    named_rng(observed_seed, f"goav_audit:{task}:{arm}:{draw}").choice(
                        len(probabilities), p=probabilities
                    )
                )
                if subset_index != expected_index:
                    raise RunIntegrityError("GOAV subset_index fails deterministic replay")
            selected = [index for index in range(len(pi)) if (subset_index >> index) & 1]
            selected_indices = event.get("selected_indices")
            if (
                not isinstance(selected_indices, list)
                or any(isinstance(index, bool) or not isinstance(index, int) for index in selected_indices)
                or selected_indices != selected
            ):
                raise RunIntegrityError("GOAV selected indices do not match subset_index")
            stored_pi = _strict_float_array(
                event.get("inclusion_probabilities"), "inclusion_probabilities", 1
            )
            if not np.array_equal(stored_pi, pi):
                raise RunIntegrityError("GOAV audit inclusion probabilities do not match its design")
            labels = event.get("selected_labels")
            if not isinstance(labels, Mapping) or set(labels) != {str(index) for index in selected}:
                raise RunIntegrityError("GOAV audit labels must include exactly the selected indices")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0, 1)
                for value in labels.values()
            ):
                raise RunIntegrityError("GOAV selected labels must be exact binary values")
            for index in selected:
                label = int(labels[str(index)])
                label_identity = (task, index)
                previous = trusted_labels.get(label_identity)
                if previous is not None and previous != label:
                    raise RunIntegrityError(
                        "GOAV audit contains a conflicting trusted label for the same "
                        "task and candidate"
                    )
                trusted_labels[label_identity] = label
            continue

        raise RunIntegrityError(f"unknown GOAV event type at position {position}")

    if not designs:
        if strict and expected_designs == set() and not records:
            return
        raise RunIntegrityError("GOAV events must contain designs followed by audit requests")
    if not any(audit_draws.values()):
        raise RunIntegrityError("GOAV events must contain designs followed by audit requests")
    if strict:
        assert expected_designs is not None and draws_per_design is not None
        if set(designs) != expected_designs:
            raise RunIntegrityError("GOAV logged designs do not cover every configured task/design arm")
        for identity in expected_designs:
            expected_count = 1 if identity[1] == "full_audit" else draws_per_design
            if audit_draws[identity] != set(range(expected_count)):
                raise RunIntegrityError("GOAV audit draws are missing, duplicated, or unconsumed")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunIntegrityError(f"{name} must contain a JSON object")
    return value


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunIntegrityError(f"{name} must be an integer")
    return value


def _config_hash(config: Any) -> str:
    try:
        return sha256(_canonical_json(config).encode("utf-8")).hexdigest()
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RunIntegrityError("resolved configuration is not canonicalizable") from exc


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unexpected = sorted(set(value) - expected)
        raise RunIntegrityError(f"{name} has missing={missing}, unexpected={unexpected}")


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RunIntegrityError(f"{name} must be a non-empty string")
    return value


def _validate_manifest_provenance(manifest: Mapping[str, Any]) -> None:
    source_configs = manifest.get("source_configs")
    if not isinstance(source_configs, list) or not source_configs:
        raise RunIntegrityError("manifest source_configs must be a non-empty list")
    seen_sources: set[str] = set()
    for index, value in enumerate(source_configs):
        source = _require_mapping(value, f"manifest source_configs[{index}]")
        _exact_keys(source, {"path", "sha256"}, f"manifest source_configs[{index}]")
        path = _nonempty_string(source["path"], f"manifest source_configs[{index}].path")
        digest = source["sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise RunIntegrityError(f"manifest source_configs[{index}].sha256 is invalid")
        if path in seen_sources:
            raise RunIntegrityError("manifest source_configs contains a duplicate path")
        seen_sources.add(path)

    git = _require_mapping(manifest.get("git"), "manifest git")
    _exact_keys(git, {"commit", "dirty"}, "manifest git")
    commit = git["commit"]
    if commit is not None and (
        not isinstance(commit, str)
        or len(commit) not in {40, 64}
        or re.fullmatch(r"[0-9a-f]+", commit) is None
    ):
        raise RunIntegrityError("manifest git.commit must be a 40/64-hex digest or null")
    if git["dirty"] is not None and not isinstance(git["dirty"], bool):
        raise RunIntegrityError("manifest git.dirty must be boolean or null")

    environment = _require_mapping(manifest.get("environment"), "manifest environment")
    _exact_keys(environment, _ENVIRONMENT_FIELDS, "manifest environment")
    if environment["kind"] != "cpu_phase0":
        raise RunIntegrityError("manifest environment.kind must be cpu_phase0")
    for field in ("python", "numpy", "scipy", "pyyaml"):
        _nonempty_string(environment[field], f"manifest environment.{field}")
    if environment["model_backend"] is not None or environment["model_checkpoint"] is not None:
        raise RunIntegrityError("Phase-0 manifest must not fabricate a model backend/checkpoint")
    if environment["container_digest"] != "not_applicable":
        raise RunIntegrityError("Phase-0 manifest container_digest must be not_applicable")


def _validate_initial_contract(
    run_dir: Path, config: Any, manifest: Any
) -> dict[str, Any]:
    config_map = _require_mapping(config, "resolved_config.json")
    manifest_map = _require_mapping(manifest, "manifest.json")
    _exact_keys(manifest_map, _MANIFEST_FIELDS, "manifest.json")
    if manifest_map.get("schema_version") != "phase0.run.v1":
        raise RunIntegrityError("manifest.json schema_version must be phase0.run.v1")

    direction = config_map.get("direction")
    if direction not in _DIRECTION_ALIASES:
        raise RunIntegrityError("resolved config has an unknown canonical direction")
    if config_map.get("schema_version") != _CONFIG_SCHEMA_BY_DIRECTION[direction]:
        raise RunIntegrityError("resolved config has the wrong direction schema_version")
    phase0 = _require_mapping(config_map.get("phase0"), "resolved config phase0")
    if not phase0:
        raise RunIntegrityError("resolved config phase0 registration must not be empty")
    alias = _DIRECTION_ALIASES[direction]
    runtime = _require_mapping(config_map.get("runtime"), "resolved config runtime")
    profile = runtime.get("profile")
    if profile not in {"smoke", "formal"}:
        raise RunIntegrityError("resolved config runtime.profile must be smoke or formal")
    if profile == "formal":
        # The frozen scientific projection is owned by the CLI configuration
        # boundary.  Import locally to avoid a module cycle at import time.
        from .cli import _validate_formal_projection
        from .config import ConfigError

        try:
            _validate_formal_projection(config_map, direction)
        except ConfigError as exc:
            raise RunIntegrityError(
                f"formal resolved config is not the frozen registration: {exc}"
            ) from exc
    output = _require_mapping(config_map.get("output"), "resolved config output")
    output_identity = _safe_identity(output.get("identity"), "resolved config output.identity")
    seed = _require_int(manifest_map.get("seed"), "manifest seed")
    if seed < 0:
        raise RunIntegrityError("manifest seed must be non-negative")
    configured_seeds = config_map.get("seeds")
    if (
        not isinstance(configured_seeds, list)
        or not configured_seeds
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in configured_seeds)
        or len(set(configured_seeds)) != len(configured_seeds)
        or seed not in configured_seeds
    ):
        raise RunIntegrityError("manifest seed must be a unique declared integer config seed")

    digest = _config_hash(config_map)
    expected_run_id = f"{output_identity}-seed{seed}-{digest[:8]}"
    if len(expected_run_id) > 192:
        raise RunIntegrityError("deterministic run_id exceeds 192 characters")
    run_id = _safe_identity(manifest_map.get("run_id"), "manifest run_id")
    expected_manifest = {
        "run_id": expected_run_id,
        "direction": direction,
        "direction_alias": alias,
        "output_identity": output_identity,
        "profile": profile,
        "config_sha256": digest,
    }
    for field, expected in expected_manifest.items():
        if manifest_map.get(field) != expected:
            raise RunIntegrityError(f"manifest {field} disagrees with the resolved configuration")
    if run_id != run_dir.name:
        raise RunIntegrityError("manifest run_id disagrees with the target directory basename")

    status = manifest_map.get("status")
    if status not in {"PASS", "FAIL", "INCOMPLETE", "INVALID"}:
        raise RunIntegrityError("manifest status is not a recognized scientific state")
    row_count = _require_int(manifest_map.get("row_count"), "manifest row_count")
    event_count = _require_int(manifest_map.get("event_count"), "manifest event_count")
    if row_count < 1 or event_count < 0:
        raise RunIntegrityError("manifest row/event counts are invalid")
    array_keys = manifest_map.get("array_keys")
    if (
        not isinstance(array_keys, list)
        or not array_keys
        or any(
            not isinstance(key, str)
            or not _SAFE_ARRAY_KEY.fullmatch(key)
            or key.endswith(".npy")
            for key in array_keys
        )
        or array_keys != sorted(set(array_keys))
    ):
        raise RunIntegrityError("manifest array_keys must be a sorted non-empty safe key list")
    _validate_manifest_provenance(manifest_map)
    return {
        "run_id": run_id,
        "direction": direction,
        "direction_alias": alias,
        "seed": seed,
        "profile": profile,
        "status": status,
        "config_sha256": digest,
        "row_count": row_count,
        "event_count": event_count,
        "array_keys": list(array_keys),
    }


def _validate_record_envelopes(
    rows: list[Any], events: list[Any], identity: Mapping[str, Any]
) -> None:
    for index, value in enumerate(rows):
        row = _require_mapping(value, f"rows.jsonl:{index + 1}")
        if not _ROW_ENVELOPE_FIELDS <= set(row) or not (set(row) - _ROW_ENVELOPE_FIELDS):
            raise RunIntegrityError(f"row {index} has an incomplete envelope or empty payload")
        if row["schema_version"] != "phase0.row.v1":
            raise RunIntegrityError(f"row {index} has an unknown schema_version")
        row_run_id = _safe_identity(row["run_id"], f"row {index} run_id")
        row_seed = _require_int(row["seed"], f"row {index} seed")
        if (
            row_run_id != identity["run_id"]
            or row["direction"] != identity["direction"]
            or row_seed != identity["seed"]
            or _require_int(row["row_index"], f"row {index} row_index") != index
            or not isinstance(row["record_type"], str)
            or not row["record_type"]
        ):
            raise RunIntegrityError(f"row {index} envelope disagrees with the manifest")
    for index, value in enumerate(events):
        event = _require_mapping(value, f"events.jsonl:{index + 1}")
        if not _EVENT_ENVELOPE_FIELDS <= set(event) or not (set(event) - _EVENT_ENVELOPE_FIELDS):
            raise RunIntegrityError(f"event {index} has an incomplete envelope or empty payload")
        _validate_common_event_envelope(
            event,
            index,
            identity["run_id"],
            identity["direction"],
            identity["seed"],
        )


def _require_number(value: Any, name: str, *, nullable: bool = False) -> float | int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunIntegrityError(f"{name} must be a finite JSON number")
    try:
        finite = math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RunIntegrityError(f"{name} must be representable as binary64") from exc
    if not finite:
        raise RunIntegrityError(f"{name} must be finite")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RunIntegrityError(f"{name} must be boolean")
    return value


def _require_string_list(value: Any, name: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise RunIntegrityError(f"{name} must be a{' non-empty' if nonempty else ''} string list")
    if any(not isinstance(item, str) or not item for item in value):
        raise RunIntegrityError(f"{name} must contain only non-empty strings")
    if len(set(value)) != len(value):
        raise RunIntegrityError(f"{name} must not contain duplicates")
    return list(value)


def _require_integer_list(
    value: Any, name: str, *, minimum: int = 0, nonempty: bool = True
) -> list[int]:
    if not isinstance(value, list) or (nonempty and not value):
        raise RunIntegrityError(f"{name} must be a{' non-empty' if nonempty else ''} integer list")
    result = [_require_int(item, f"{name}[{index}]") for index, item in enumerate(value)]
    if any(item < minimum for item in result) or len(set(result)) != len(result):
        raise RunIntegrityError(f"{name} contains an out-of-range or duplicate integer")
    return result


def _require_npz_array(
    arrays: Mapping[str, np.ndarray], key: str, *, ndim: int, boolean: bool = False
) -> np.ndarray:
    if key not in arrays:
        raise RunIntegrityError(f"arrays.npz is missing direction array {key!r}")
    array = np.asarray(arrays[key])
    expected_dtype = np.dtype(np.bool_) if boolean else np.dtype("<f8")
    if array.ndim != ndim or array.dtype != expected_dtype:
        raise RunIntegrityError(
            f"arrays.npz member {key!r} must be {expected_dtype} with {ndim} dimensions"
        )
    return array


def _runtime_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    if not slug:
        raise RunIntegrityError("scientific identity cannot be represented as an array key")
    return slug


def _validate_pbpf_contract(
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    gates: Mapping[str, Any],
    rows: list[Any],
    events: list[Any],
    arrays: Mapping[str, np.ndarray],
    identity: Mapping[str, Any],
) -> None:
    if events:
        raise RunIntegrityError("PBPF events.jsonl must be empty")
    phase0 = _require_mapping(config.get("phase0"), "PBPF phase0")
    belief = _require_mapping(config.get("belief"), "PBPF belief")
    episodes = _require_mapping(phase0.get("episodes"), "PBPF phase0.episodes")
    test_episodes = _require_int(episodes.get("test"), "PBPF phase0.episodes.test")
    prefixes = _require_integer_list(phase0.get("prefixes"), "PBPF phase0.prefixes")
    horizons_raw = phase0.get("forecast_horizons")
    if not isinstance(horizons_raw, list) or not horizons_raw:
        raise RunIntegrityError("PBPF phase0.forecast_horizons must be non-empty")
    horizons: list[int | str] = []
    for index, value in enumerate(horizons_raw):
        if value == "all_remaining":
            horizon: int | str = value
        else:
            horizon = _require_int(value, f"PBPF phase0.forecast_horizons[{index}]")
            if horizon < 1:
                raise RunIntegrityError("PBPF forecast horizons must be positive")
        if horizon in horizons:
            raise RunIntegrityError("PBPF forecast horizons must not contain duplicates")
        horizons.append(horizon)
    particle_sweep = _require_integer_list(
        phase0.get("particle_sweep", belief.get("particle_sweep")),
        "PBPF particle_sweep",
        minimum=1,
    )
    configured_arms = _require_string_list(phase0.get("arms"), "PBPF phase0.arms")
    implemented = {"exact_bayes", "pbpf", "map", "prior"}
    expected_rejected = [arm for arm in configured_arms if arm not in implemented]
    expected_arms = ["prior"]
    if "exact_bayes" in configured_arms:
        expected_arms.insert(0, "exact_bayes")
    if "map" in configured_arms:
        expected_arms.append("map")
    if "pbpf" in configured_arms:
        expected_arms.extend(f"particle_{count}" for count in particle_sweep)
    if len(set(expected_arms)) != len(expected_arms):
        raise RunIntegrityError("PBPF expanded arms are not unique")
    if test_episodes < 1:
        raise RunIntegrityError("PBPF test episode count must be positive")

    expected_row_required = _ROW_ENVELOPE_FIELDS | _PBPF_ROW_REQUIRED
    expected_identities = {
        (episode, prefix, horizon, arm)
        for episode in range(test_episodes)
        for prefix in prefixes
        for horizon in horizons
        for arm in expected_arms
    }
    identities: set[tuple[int, int, int | str, str]] = set()
    arm_counts: dict[str, int] = {}
    collapse_counts: dict[str, int] = {}
    numeric_expected = np.zeros((len(rows), len(_PBPF_COLUMNS)), dtype="<f8")
    present_expected = np.zeros(numeric_expected.shape, dtype=np.bool_)
    for index, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"PBPF row {index}")
        keys = set(row)
        if not expected_row_required <= keys or keys - expected_row_required - _PBPF_ROW_OPTIONAL:
            raise RunIntegrityError(f"PBPF row {index} has missing or unexpected fields")
        if row["record_type"] != "forecast_diagnostic":
            raise RunIntegrityError(f"PBPF row {index} has the wrong record_type")
        episode = _require_int(row["episode"], f"PBPF row {index} episode")
        prefix = _require_int(row["prefix"], f"PBPF row {index} prefix")
        horizon_value = row["horizon"]
        if horizon_value != "all_remaining":
            horizon_value = _require_int(horizon_value, f"PBPF row {index} horizon")
        arm = row["arm"]
        if not isinstance(arm, str) or not arm:
            raise RunIntegrityError(f"PBPF row {index} arm must be non-empty")
        row_identity = (episode, prefix, horizon_value, arm)
        if row_identity in identities:
            raise RunIntegrityError("PBPF (episode,prefix,horizon,arm) rows must be unique")
        identities.add(row_identity)
        arm_counts[arm] = arm_counts.get(arm, 0) + 1
        for column_index, column in enumerate(_PBPF_COLUMNS):
            nullable = column != "resampling"
            number = _require_number(row[column], f"PBPF row {index} {column}", nullable=nullable)
            if column == "resampling":
                integer = _require_int(row[column], f"PBPF row {index} resampling")
                if integer < 0:
                    raise RunIntegrityError("PBPF resampling count must be non-negative")
                number = integer
            if number is not None:
                numeric_expected[index, column_index] = float(number)
                present_expected[index, column_index] = True
        diagnostic = row.get("diagnostic")
        if diagnostic is not None:
            if diagnostic != "PARTICLE_COLLAPSE":
                raise RunIntegrityError(f"PBPF row {index} has an unknown diagnostic")
            collapse_counts[arm] = collapse_counts.get(arm, 0) + 1
        for field in (
            "pre_resampling_ess",
            "resampling_rate",
            "unique_ancestor_ratio",
            "post_rejuvenation_unique_state_ratio",
        ):
            if field in row:
                checked = _require_number(
                    row[field], f"PBPF row {index} {field}", nullable=True
                )
                if checked is not None and float(checked) < 0.0:
                    raise RunIntegrityError(f"PBPF row {index} {field} must be non-negative")
        if "particles_per_correct_equivalence_class" in row:
            if _require_int(
                row["particles_per_correct_equivalence_class"],
                f"PBPF row {index} particles_per_correct_equivalence_class",
            ) < 0:
                raise RunIntegrityError("PBPF particle equivalence count must be non-negative")
        if "pre_resampling_ess_history" in row:
            history = row["pre_resampling_ess_history"]
            if not isinstance(history, list):
                raise RunIntegrityError("PBPF pre_resampling_ess_history must be a list")
            for history_index, item in enumerate(history):
                checked = _require_number(
                    item, f"PBPF row {index} ESS history {history_index}"
                )
                if float(checked) < 0.0:
                    raise RunIntegrityError("PBPF ESS history must be non-negative")
        if "resampling_events" in row:
            resampling_events = row["resampling_events"]
            if not isinstance(resampling_events, list):
                raise RunIntegrityError("PBPF resampling_events must be a list")
            event_fields = {
                "position",
                "pre_resampling_ess",
                "unique_ancestor_ratio",
                "post_rejuvenation_unique_state_ratio",
            }
            for event_index, raw_event in enumerate(resampling_events):
                event = _require_mapping(
                    raw_event, f"PBPF row {index} resampling event {event_index}"
                )
                _exact_keys(
                    event, event_fields, f"PBPF row {index} resampling event {event_index}"
                )
                if _require_int(event["position"], "PBPF resampling position") < 0:
                    raise RunIntegrityError("PBPF resampling position must be non-negative")
                for field in event_fields - {"position"}:
                    checked = _require_number(
                        event[field], f"PBPF resampling event {field}", nullable=True
                    )
                    if checked is not None and float(checked) < 0.0:
                        raise RunIntegrityError("PBPF resampling diagnostics must be non-negative")
    if identities != expected_identities:
        raise RunIntegrityError("PBPF rows do not cover the registered episode/prefix/horizon/arm grid")

    schema = _require_mapping(metrics.get("diagnostic_column_schema"), "PBPF diagnostic schema")
    expected_schema = {
        "matrix_key": "pbpf_row_numeric",
        "present_mask_key": "pbpf_row_numeric_present",
        "columns": list(_PBPF_COLUMNS),
        "missing_value": "zero_with_present_mask",
    }
    if dict(schema) != expected_schema:
        raise RunIntegrityError("PBPF diagnostic column schema is not canonical")
    if metrics.get("arm_counts") != dict(sorted(arm_counts.items())):
        raise RunIntegrityError("PBPF metrics arm_counts disagree with rows")
    expected_collapse = {
        "total": sum(collapse_counts.values()),
        "by_arm": dict(sorted(collapse_counts.items())),
    }
    if metrics.get("collapse_counts") != expected_collapse:
        raise RunIntegrityError("PBPF metrics collapse_counts disagree with rows")
    if metrics.get("rejected_arms") != expected_rejected:
        raise RunIntegrityError("PBPF rejected_arms disagree with the registered arms")

    gate_inputs = _require_mapping(gates.get("inputs"), "PBPF gates inputs")
    _exact_keys(gate_inputs, _PBPF_GATE_INPUT_FIELDS, "PBPF gates inputs")
    nominal = _require_number(gate_inputs["hpd_nominal"], "PBPF hpd_nominal")
    if not 0.0 < float(nominal) <= 1.0:
        raise RunIntegrityError("PBPF hpd_nominal must lie in (0,1]")
    by_prefix = _require_mapping(
        gate_inputs["by_prefix_horizon"], "PBPF gates inputs.by_prefix_horizon"
    )
    if set(by_prefix) != {str(prefix) for prefix in prefixes}:
        raise RunIntegrityError("PBPF gate prefix keys disagree with config")
    for prefix in prefixes:
        by_horizon = _require_mapping(
            by_prefix[str(prefix)], f"PBPF gate prefix {prefix}"
        )
        expected_horizon_keys = [str(horizon) for horizon in horizons]
        if set(by_horizon) != set(expected_horizon_keys):
            raise RunIntegrityError("PBPF gate horizon keys disagree with config")
        for horizon in horizons:
            cell = _require_mapping(
                by_horizon[str(horizon)], f"PBPF gate cell {prefix}/{horizon}"
            )
            _exact_keys(cell, _PBPF_GATE_CELL_FIELDS, f"PBPF gate cell {prefix}/{horizon}")
            expected_ids = _require_integer_list(
                cell["expected_episode_ids"],
                f"PBPF gate cell {prefix}/{horizon} expected ids",
                nonempty=test_episodes > 0,
            )
            if expected_ids != list(range(test_episodes)):
                raise RunIntegrityError("PBPF gate expected episode ids disagree with config")
            for field in (
                "p16_collapse_episode_ids",
                "p16_missing_episode_ids",
            ):
                values = _require_integer_list(
                    cell[field], f"PBPF gate cell {prefix}/{horizon} {field}", nonempty=False
                )
                if any(value not in expected_ids for value in values):
                    raise RunIntegrityError("PBPF gate episode diagnostics reference unknown episodes")
            _require_bool(cell["p16_complete_coverage"], "PBPF p16_complete_coverage")
            if _require_int(cell["particle_collapse_count"], "PBPF particle_collapse_count") < 0:
                raise RunIntegrityError("PBPF particle collapse count must be non-negative")
            for field in (
                "paired_exact_vs_map_inputs",
                "paired_exact_vs_prior_inputs",
                "paired_p16_to_exact_inputs",
                "hpd_coverage_inputs",
            ):
                _strict_float_array(cell[field], f"PBPF gate {field}", 1)
            for field in (
                "exact_mixture_vs_map_nll_delta",
                "exact_vs_prior_nll_delta",
                "map_gap_fraction_closed",
                "p16_to_exact_nll_gap",
                "p16_to_exact_one_sided_upper_95",
            ):
                _require_number(cell[field], f"PBPF gate cell {field}", nullable=True)
            for field in ("exact_vs_map_paired_ci", "hpd_coverage_ci"):
                interval = cell[field]
                if interval is not None:
                    values = _strict_float_array(interval, f"PBPF gate {field}", 1)
                    if values.shape != (2,):
                        raise RunIntegrityError("PBPF confidence intervals must contain two values")
    prefix_4 = _require_mapping(gate_inputs["prefix_4"], "PBPF gates inputs.prefix_4")
    registered_prefix_4 = (
        by_prefix.get("4", {}).get("all_remaining", {})
        if isinstance(by_prefix.get("4", {}), Mapping)
        else {}
    )
    if dict(prefix_4) != dict(registered_prefix_4):
        raise RunIntegrityError("PBPF gates inputs.prefix_4 does not mirror its registered cell")
    for field in (
        "exact_vs_prior_nll_delta",
        "exact_mixture_vs_map_nll_delta",
        "p16_to_exact_nll_gap",
        "map_gap_fraction_closed",
    ):
        if gate_inputs[field] != prefix_4.get(field):
            raise RunIntegrityError(f"PBPF top-level gate input {field} does not mirror prefix_4")
    present_gate_sources = {
        source_key for source_key in _PBPF_GATE_ARRAYS if source_key in prefix_4
    }
    if gates.get("formal_evidence") is True and present_gate_sources != set(_PBPF_GATE_ARRAYS):
        raise RunIntegrityError("formal PBPF evidence requires all four registered gate arrays")
    expected_array_keys = {"pbpf_row_numeric", "pbpf_row_numeric_present"} | {
        _PBPF_GATE_ARRAYS[source_key] for source_key in present_gate_sources
    }
    if set(arrays) != expected_array_keys:
        raise RunIntegrityError("PBPF arrays.npz has a missing or orphaned direction array")
    numeric = _require_npz_array(arrays, "pbpf_row_numeric", ndim=2)
    present = _require_npz_array(arrays, "pbpf_row_numeric_present", ndim=2, boolean=True)
    if not np.array_equal(numeric, numeric_expected) or not np.array_equal(present, present_expected):
        raise RunIntegrityError("PBPF numeric matrix/present mask disagree with row null semantics")
    for source_key, array_key in _PBPF_GATE_ARRAYS.items():
        if source_key not in prefix_4:
            continue
        expected = _strict_float_array(prefix_4[source_key], source_key, 1)
        observed = _require_npz_array(arrays, array_key, ndim=1)
        if not np.array_equal(observed, expected):
            raise RunIntegrityError(f"PBPF gate array {array_key} disagrees with gates inputs.prefix_4")

    can_reconstruct = (
        {"exact_bayes", "map", "pbpf"}.issubset(configured_arms)
        and 16 in particle_sweep
    )
    profile = _require_mapping(config.get("runtime"), "PBPF runtime").get("profile")
    expected_source_status = "INCOMPLETE"
    if can_reconstruct:
        from .pbpf.experiment import _formal_status, _group_gate_inputs

        grouped: dict[tuple[int, int | str], dict[str, Any]] = {}
        for prefix in prefixes:
            for horizon in horizons:
                grouped[(prefix, horizon)] = {
                    "exact": {},
                    "prior": {},
                    "map": {},
                    "particle_16": {},
                    "hpd": {},
                    "collapses": 0,
                    "p16_collapse_ids": set(),
                    "expected_episode_ids": set(),
                }
        for raw_row in rows:
            row = _require_mapping(raw_row, "PBPF reconstruct row")
            cell = grouped[(int(row["prefix"]), row["horizon"])]
            episode = int(row["episode"])
            arm = row["arm"]
            if arm == "exact_bayes":
                cell["exact"][episode] = float(row["nll"])
                cell["hpd"][episode] = float(row["hpd_true_inclusion"])
                cell["expected_episode_ids"].add(episode)
            elif arm == "prior":
                cell["prior"][episode] = float(row["nll"])
            elif arm == "map":
                cell["map"][episode] = float(row["nll"])
            elif arm == "particle_16":
                if row.get("diagnostic") == "PARTICLE_COLLAPSE":
                    cell["p16_collapse_ids"].add(episode)
                elif row["nll"] is not None:
                    cell["particle_16"][episode] = float(row["nll"])
            if row.get("diagnostic") == "PARTICLE_COLLAPSE":
                cell["collapses"] += 1
        expected_by_prefix: dict[str, dict[str, Any]] = {}
        bootstrap_resamples = (
            int(_require_mapping(config.get("statistics", {}), "PBPF statistics").get("bootstrap_resamples", 10_000))
            if profile == "formal"
            else 0
        )
        for (prefix, horizon), values in grouped.items():
            resamples = bootstrap_resamples if (prefix, horizon) == (4, "all_remaining") else 0
            expected_cell = _group_gate_inputs(
                values,
                outer_seed=int(rows[0]["seed"]),
                label=f"prefix={prefix}/horizon={horizon}",
                bootstrap_resamples=resamples,
            )
            expected_by_prefix.setdefault(str(prefix), {})[str(horizon)] = expected_cell
        if dict(by_prefix) != expected_by_prefix:
            raise RunIntegrityError("PBPF gate inputs disagree with statistics reconstructed from rows")
        expected_prefix_4 = expected_by_prefix.get("4", {}).get("all_remaining", {})
        expected_source_status = (
            "INCOMPLETE"
            if profile == "smoke"
            else _formal_status(phase0, expected_prefix_4, expected_rejected)
        )
    if gates.get("source_status") != expected_source_status:
        raise RunIntegrityError("PBPF gates source_status disagrees with reconstructed evidence")

    # Phase-0 is a finite deterministic mechanism experiment.  Replaying its
    # registered config/seed is the only closed check of the particle path:
    # local ESS/event invariants alone cannot rule out a coordinated rewrite of
    # the particle trajectory, forecasts, gate inputs, and mirrored NPZ arrays.
    from .pbpf.experiment import run_pbpf_phase0
    from .results import normalize_result

    try:
        expected_raw = run_pbpf_phase0(config, int(identity["seed"]))
    except Exception as exc:
        raise RunIntegrityError(
            "PBPF deterministic config/seed replay could not be completed"
        ) from exc
    expected_raw_keys = {
        "seed",
        "status",
        "rows",
        "gate_status",
        "gate_inputs",
        "rejected_arms",
    }
    if not isinstance(expected_raw, Mapping) or set(expected_raw) != expected_raw_keys:
        raise RunIntegrityError("PBPF deterministic replay returned a non-canonical result")
    if expected_raw["seed"] != identity["seed"]:
        raise RunIntegrityError("PBPF deterministic replay returned the wrong seed")
    if expected_raw["status"] != expected_raw["gate_status"]:
        raise RunIntegrityError("PBPF deterministic replay returned inconsistent gate status")

    observed_native_rows = [
        {key: value for key, value in _require_mapping(row, "PBPF replay row").items()
         if key not in _ROW_ENVELOPE_FIELDS}
        for row in rows
    ]
    if _canonical_json(observed_native_rows) != _canonical_json(expected_raw["rows"]):
        raise RunIntegrityError(
            "PBPF rows disagree with deterministic config/seed replay"
        )
    if _canonical_json(gate_inputs) != _canonical_json(expected_raw["gate_inputs"]):
        raise RunIntegrityError(
            "PBPF gate inputs disagree with deterministic config/seed replay"
        )
    if metrics.get("rejected_arms") != expected_raw["rejected_arms"]:
        raise RunIntegrityError(
            "PBPF rejected arms disagree with deterministic config/seed replay"
        )
    try:
        replay = normalize_result(
            "pbpf",
            identity["direction"],
            config,
            identity["seed"],
            identity["run_id"],
            expected_raw,
        )
    except Exception as exc:
        raise RunIntegrityError("PBPF deterministic replay normalization failed") from exc
    if _deterministic_npz_bytes(arrays) != _deterministic_npz_bytes(replay.arrays):
        raise RunIntegrityError(
            "PBPF arrays disagree byte-for-byte with deterministic config/seed replay"
        )

    source_status = expected_raw["status"]
    if source_status not in {"PASS", "FAIL", "INCOMPLETE", "INVALID"}:
        raise RunIntegrityError("PBPF deterministic replay returned an invalid status")
    profile = identity["profile"]
    if source_status == "INVALID":
        final_status = "INVALID"
        formal_evidence = False
        reason = "source_result_invalid"
    elif profile == "smoke":
        final_status = "INCOMPLETE"
        formal_evidence = False
        reason = "smoke_profile_below_formal_registration"
    else:
        final_status = source_status
        formal_evidence = source_status in {"PASS", "FAIL"}
        reason = {
            "PASS": "source_formal_gate_passed",
            "FAIL": "source_formal_gate_failed",
            "INCOMPLETE": "source_formal_evidence_incomplete",
        }[source_status]
    expected_gates = {
        "schema_version": "phase0.gates.v1",
        "direction": identity["direction"],
        "seed": identity["seed"],
        "profile": profile,
        "status": final_status,
        "formal_evidence": formal_evidence,
        "source_status": source_status,
        "reason": reason,
        "inputs": expected_raw["gate_inputs"],
    }
    if _canonical_json(dict(gates)) != _canonical_json(expected_gates):
        raise RunIntegrityError(
            "PBPF gates disagree with the canonical deterministic replay state"
        )


def _goav_design_array_keys(task: int, arm_index: int, arm: str) -> dict[str, str]:
    base = f"goav_design_t{task:06d}_a{arm_index:03d}_{_runtime_slug(arm)}"
    return {suffix: f"{base}_{suffix}" for suffix in ("probabilities", "pi", "pi2")}


def _validate_goav_contract(
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    gates: Mapping[str, Any],
    rows: list[Any],
    events: list[Any],
    arrays: Mapping[str, np.ndarray],
    identity: Mapping[str, Any],
) -> None:
    gate_inputs = _require_mapping(gates.get("inputs"), "GOAV gates inputs")
    if gate_inputs:
        raise RunIntegrityError("GOAV gates inputs must be the exact empty Phase-0 mapping")
    if gates.get("source_status") != "INCOMPLETE" or gates.get("formal_evidence") is not False:
        raise RunIntegrityError("single-seed GOAV Phase-0 cannot claim formal PASS/FAIL evidence")
    phase0 = _require_mapping(config.get("phase0"), "GOAV phase0")
    acquisition = _require_mapping(config.get("acquisition"), "GOAV acquisition")
    arms = _require_string_list(phase0.get("arms"), "GOAV phase0.arms")
    unknown_arms = set(arms) - _GOAV_ALLOWED_ARMS
    if unknown_arms:
        raise RunIntegrityError(f"GOAV config contains unknown arms: {sorted(unknown_arms)}")
    tasks = _require_int(phase0.get("tasks_min"), "GOAV phase0.tasks_min")
    candidates = _require_int(phase0.get("group_size"), "GOAV phase0.group_size")
    tests = _require_int(phase0.get("tests_per_task_min"), "GOAV phase0.tests_per_task_min")
    if tasks < 1 or candidates < 1 or tests < 1:
        raise RunIntegrityError("GOAV registered task/candidate/test counts must be positive")
    expected_row_fields = _ROW_ENVELOPE_FIELDS | {"arm", *_GOAV_ROW_COLUMNS}
    numeric_expected = np.zeros((len(rows), len(_GOAV_ROW_COLUMNS)), dtype="<f8")
    support_violations = 0
    observed_arms: list[str] = []
    for index, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"GOAV row {index}")
        _exact_keys(row, expected_row_fields, f"GOAV row {index}")
        if row["record_type"] != "arm_summary":
            raise RunIntegrityError(f"GOAV row {index} has the wrong record_type")
        arm = row["arm"]
        if not isinstance(arm, str) or not arm:
            raise RunIntegrityError(f"GOAV row {index} arm must be non-empty")
        observed_arms.append(arm)
        for column_index, column in enumerate(_GOAV_ROW_COLUMNS):
            if column == "support_violations":
                number = _require_int(row[column], f"GOAV row {index} support_violations")
                if number < 0:
                    raise RunIntegrityError("GOAV support violations must be non-negative")
                support_violations += number
            else:
                checked = _require_number(row[column], f"GOAV row {index} {column}")
                assert checked is not None
                number = checked
            numeric_expected[index, column_index] = float(number)
    if observed_arms != arms:
        raise RunIntegrityError("GOAV rows must contain each configured arm once in configured order")

    expected_numeric_schema = {
        "matrix_key": "goav_row_numeric",
        "present_mask_key": "goav_row_numeric_present",
        "columns": list(_GOAV_ROW_COLUMNS),
        "missing_value": "zero_with_present_mask",
    }
    if metrics.get("numeric_row_schema") != expected_numeric_schema:
        raise RunIntegrityError("GOAV numeric row schema is not canonical")
    design_events = [event for event in events if isinstance(event, Mapping) and event.get("event") == "design_logged"]
    audit_events = [event for event in events if isinstance(event, Mapping) and event.get("event") == "audit_request"]
    if _require_int(metrics.get("design_event_count"), "GOAV design_event_count") != len(design_events):
        raise RunIntegrityError("GOAV design_event_count disagrees with events")
    if _require_int(metrics.get("audit_event_count"), "GOAV audit_event_count") != len(audit_events):
        raise RunIntegrityError("GOAV audit_event_count disagrees with events")
    if _require_int(metrics.get("support_violation_count"), "GOAV support_violation_count") != support_violations:
        raise RunIntegrityError("GOAV support_violation_count disagrees with rows")

    epsilon = _require_number(metrics.get("epsilon_G"), "GOAV epsilon_G")
    assert epsilon is not None
    if float(epsilon) <= 0.0:
        raise RunIntegrityError("GOAV epsilon_G must be positive")
    provenance = _require_mapping(metrics.get("epsilon_G_provenance"), "GOAV epsilon provenance")
    explicit = phase0.get("epsilon_G")
    if explicit is not None and float(_require_number(explicit, "GOAV configured epsilon_G")) != float(epsilon):
        raise RunIntegrityError("GOAV epsilon_G disagrees with its explicit registration")
    if provenance.get("source") == "explicit":
        _exact_keys(provenance, {"source", "input_hash"}, "GOAV epsilon provenance")
        if provenance["input_hash"] != _array_hash(np.asarray([epsilon], dtype="<f8")):
            raise RunIntegrityError("GOAV explicit epsilon provenance hash is invalid")
    elif provenance.get("source") == "synthetic_dev_panel":
        fields = {
            "source", "seed", "tasks", "candidates", "tests", "panel_rng_label",
            "score_rng_label", "input_hash", "formula",
        }
        _exact_keys(provenance, fields, "GOAV epsilon provenance")
        expected_dev_tasks = _require_int(
            phase0.get("epsilon_g_dev_tasks", max(4, tasks)), "GOAV epsilon dev tasks"
        )
        expected_dev_seed = _require_int(
            phase0.get("epsilon_g_dev_seed", 730_241), "GOAV epsilon dev seed"
        )
        expected_values = {
            "seed": expected_dev_seed,
            "tasks": expected_dev_tasks,
            "candidates": candidates,
            "tests": tests,
            "panel_rng_label": "goav_synthetic_oracle",
            "score_rng_label": "goav_epsilon_dev_scores",
            "formula": "max(0.01*median_squared_gradient_norm,float64_epsilon)",
        }
        for field, expected in expected_values.items():
            if provenance.get(field) != expected:
                raise RunIntegrityError(f"GOAV epsilon provenance {field} disagrees with config")
        if not isinstance(provenance.get("input_hash"), str) or not _SHA256.fullmatch(provenance["input_hash"]):
            raise RunIntegrityError("GOAV epsilon provenance input_hash is invalid")
        noise_panel = _require_mapping(
            _require_mapping(config.get("oracle_noise_model", {}), "GOAV oracle_noise_model").get(
                "medium_noise", {}
            ),
            "GOAV oracle_noise_model.medium_noise",
        )
        cluster_size = _require_int(noise_panel.get("cluster_size", 2), "GOAV cluster_size")
        if cluster_size < 1:
            raise RunIntegrityError("GOAV cluster_size must be positive")
        false_negative = float(
            _require_number(noise_panel.get("false_negative", 0.1), "GOAV false_negative")
        )
        false_positive = float(
            _require_number(noise_panel.get("false_positive", 0.2), "GOAV false_positive")
        )
        rho = float(_require_number(noise_panel.get("flip_icc", 0.6), "GOAV flip_icc"))
        cluster_ids = np.arange(tests) // cluster_size
        from .goav.estimator import loo_influence
        from .goav.noise import simulate_synthetic_oracle

        dev_labels, _ = simulate_synthetic_oracle(
            expected_dev_tasks,
            candidates,
            cluster_ids,
            false_negative,
            false_positive,
            rho,
            expected_dev_seed,
        )
        dev_scores = named_rng(expected_dev_seed, "goav_epsilon_dev_scores").normal(
            size=(expected_dev_tasks, candidates, max(2, candidates // 2))
        )
        dev_targets = np.asarray(
            [
                loo_influence(dev_scores[index]) @ dev_labels[index].astype(np.float64)
                for index in range(expected_dev_tasks)
            ]
        )
        squared_norms = np.einsum("tp,tp->t", dev_targets, dev_targets)
        recomputed_epsilon = max(
            0.01 * float(np.median(squared_norms)), float(np.finfo(np.float64).eps)
        )
        if float(epsilon) != recomputed_epsilon or provenance["input_hash"] != _array_hash(dev_targets):
            raise RunIntegrityError("GOAV epsilon value/provenance does not match the frozen dev panel")
    else:
        raise RunIntegrityError("GOAV epsilon provenance has an unknown source")

    solver = _require_mapping(metrics.get("solver"), "GOAV solver metrics")
    _exact_keys(solver, {"steps", "learning_rate", "restarts"}, "GOAV solver metrics")
    registered_solver = _require_mapping(acquisition.get("solver"), "GOAV acquisition.solver")
    default_initializations = [
        "uniform", "poisson_neyman", "bayes_voi", "random_1", "random_2",
        "random_3", "random_4", "random_5",
    ]
    initializations = _require_string_list(
        registered_solver.get("initialization", default_initializations),
        "GOAV solver initialization",
        nonempty=False,
    )
    expected_solver = {
        "steps": _require_int(
            phase0.get("solver_steps", registered_solver.get("steps", 200)),
            "GOAV solver steps",
        ),
        "learning_rate": phase0.get(
            "solver_learning_rate", registered_solver.get("learning_rate", 0.05)
        ),
        "restarts": _require_int(
            phase0.get("solver_restarts", len(initializations) if initializations else 8),
            "GOAV solver restarts",
        ),
    }
    _require_number(expected_solver["learning_rate"], "GOAV solver learning_rate")
    if dict(solver) != expected_solver:
        raise RunIntegrityError("GOAV solver metrics disagree with the registered solver")

    validate_goav_events(
        events,
        run_id=identity["run_id"],
        seed=identity["seed"],
        require_envelope=True,
        config=config,
        row_arms=tuple(observed_arms),
    )
    expected_array_keys = {"goav_row_numeric", "goav_row_numeric_present"}
    for event in design_events:
        task = _require_int(event.get("task"), "GOAV design task")
        arm = str(event.get("arm"))
        arm_index = arms.index(arm)
        keys = _goav_design_array_keys(task, arm_index, arm)
        for source, key in keys.items():
            expected_array_keys.add(key)
            expected = _strict_float_array(event[source], f"GOAV design {source}", 2 if source == "pi2" else 1)
            observed = _require_npz_array(arrays, key, ndim=expected.ndim)
            if not np.array_equal(observed, expected):
                raise RunIntegrityError(f"GOAV design array {key} disagrees with its logged event")
    if set(arrays) != expected_array_keys:
        raise RunIntegrityError("GOAV arrays.npz has a missing or orphaned direction array")
    row_numeric = _require_npz_array(arrays, "goav_row_numeric", ndim=2)
    row_present = _require_npz_array(arrays, "goav_row_numeric_present", ndim=2, boolean=True)
    if not np.array_equal(row_numeric, numeric_expected):
        raise RunIntegrityError("GOAV row matrix disagrees with rows.jsonl")
    if not np.array_equal(row_present, np.ones(numeric_expected.shape, dtype=np.bool_)):
        raise RunIntegrityError("GOAV present mask must be true for every exact row value")

    # GOAV's aggregate rows cannot be reconstructed from the audit log alone:
    # the control arms deliberately have no audit events, and the gradient
    # geometry is generated from a named seed stream.  Replay the finite
    # experiment from the trusted resolved config and seed instead of keeping
    # a second, potentially drifting implementation of the twelve statistics.
    from .goav.experiment import run_goav_phase0
    from .results import normalize_result

    try:
        expected_raw = run_goav_phase0(config, identity["seed"])
    except (TypeError, ValueError, RuntimeError, OverflowError, FloatingPointError) as exc:
        raise RunIntegrityError("GOAV trusted deterministic replay failed") from exc
    if expected_raw.get("gates") != {"status": "INCOMPLETE", "formal_evidence": False}:
        raise RunIntegrityError("GOAV runner violated its Phase-0 gate contract")
    observed_native_rows = [
        {key: value for key, value in dict(row).items() if key not in _ROW_ENVELOPE_FIELDS}
        for row in rows
    ]
    observed_native_events = [
        {key: value for key, value in dict(event).items() if key not in _EVENT_ENVELOPE_FIELDS}
        for event in events
    ]
    if _canonical_json(observed_native_rows) != _canonical_json(expected_raw["rows"]):
        raise RunIntegrityError("GOAV rows disagree with deterministic config/seed replay")
    if _canonical_json(observed_native_events) != _canonical_json(expected_raw["events"]):
        raise RunIntegrityError("GOAV events disagree with deterministic config/seed replay")
    expected_science = {
        key: expected_raw[key]
        for key in ("epsilon_G", "epsilon_G_provenance", "solver")
    }
    observed_science = {key: metrics[key] for key in expected_science}
    if _canonical_json(observed_science) != _canonical_json(expected_science):
        raise RunIntegrityError(
            "GOAV scalar scientific metadata disagrees with deterministic replay"
        )
    try:
        replay = normalize_result(
            "goav",
            identity["direction"],
            config,
            identity["seed"],
            identity["run_id"],
            expected_raw,
        )
    except Exception as exc:
        raise RunIntegrityError("GOAV deterministic replay normalization failed") from exc
    if _deterministic_npz_bytes(arrays) != _deterministic_npz_bytes(replay.arrays):
        raise RunIntegrityError(
            "GOAV arrays disagree byte-for-byte with deterministic config/seed replay"
        )
    expected_reason = (
        "smoke_profile_below_formal_registration"
        if identity["profile"] == "smoke"
        else "source_formal_evidence_incomplete"
    )
    expected_gates = {
        "schema_version": "phase0.gates.v1",
        "direction": identity["direction"],
        "seed": identity["seed"],
        "profile": identity["profile"],
        "status": "INCOMPLETE",
        "formal_evidence": False,
        "source_status": "INCOMPLETE",
        "reason": expected_reason,
        "inputs": {},
    }
    if _canonical_json(dict(gates)) != _canonical_json(expected_gates):
        raise RunIntegrityError("GOAV gates are not the canonical Phase-0 INCOMPLETE gate")


def _jag_covariance_array_keys(family_index: int, family: str) -> tuple[str, str]:
    base = f"jag_covariance_f{family_index:03d}_{_runtime_slug(family)}"
    return f"{base}_predicted", f"{base}_realized"


def _validate_jag_row_nested(
    row: Mapping[str, Any],
    *,
    family: str,
    arm: str,
    budget: int,
    sample_count: int,
    seed: int,
    baseline: str,
    max_branching: int,
) -> None:
    allocation = _require_mapping(row["allocation"], "JAG row allocation")
    _exact_keys(allocation, _JAG_ALLOCATION_FIELDS, "JAG row allocation")
    if _require_int(allocation["replication_count"], "JAG allocation replication_count") != sample_count:
        raise RunIntegrityError("JAG allocation replication_count disagrees with sample_count")
    if _require_int(allocation["total_edge_samples"], "JAG allocation total_edge_samples") != sample_count * budget:
        raise RunIntegrityError("JAG allocation total_edge_samples disagrees with sample_count*budget")
    histogram = _require_mapping(allocation["plan_histogram"], "JAG allocation plan_histogram")
    if _require_int(allocation["unique_plan_count"], "JAG allocation unique_plan_count") != len(histogram):
        raise RunIntegrityError("JAG allocation unique_plan_count disagrees with histogram")
    histogram_total = 0
    for digest, raw_entry in histogram.items():
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise RunIntegrityError("JAG allocation plan hash is invalid")
        entry = _require_mapping(raw_entry, f"JAG allocation plan {digest}")
        _exact_keys(entry, {"count", "edge_count", "canonical_plan"}, f"JAG plan {digest}")
        count = _require_int(entry["count"], "JAG plan count")
        edge_count = _require_int(entry["edge_count"], "JAG plan edge_count")
        plan = entry["canonical_plan"]
        if count < 1 or edge_count != budget or not isinstance(plan, list) or not plan:
            raise RunIntegrityError("JAG allocation plan count/edge_count/list is invalid")
        nodes: set[str] = set()
        edge_sum = 0
        previous_node: str | None = None
        for plan_index, item in enumerate(plan):
            if not isinstance(item, list) or len(item) != 2:
                raise RunIntegrityError("JAG canonical plan items must be [node_id, branching]")
            node = _nonempty_string(item[0], f"JAG plan node {plan_index}")
            branching = _require_int(item[1], f"JAG plan branching {plan_index}")
            if node in nodes or (previous_node is not None and node <= previous_node):
                raise RunIntegrityError("JAG canonical plan nodes must be unique and sorted")
            if branching < 1 or branching > max_branching:
                raise RunIntegrityError("JAG canonical plan branching violates estimator.max_branching")
            nodes.add(node)
            previous_node = node
            edge_sum += branching
        if edge_sum != edge_count:
            raise RunIntegrityError("JAG canonical plan branching does not sum to edge_count")
        encoded = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
        if sha256(encoded.encode("utf-8")).hexdigest() != digest:
            raise RunIntegrityError("JAG canonical plan hash disagrees with its payload")
        histogram_total += count
    if histogram_total != sample_count:
        raise RunIntegrityError("JAG allocation plan counts do not sum to sample_count")

    evidence = _require_mapping(row["allocation_evidence"], "JAG row allocation_evidence")
    _exact_keys(evidence, _JAG_ALLOCATION_EVIDENCE_FIELDS, "JAG row allocation_evidence")
    _nonempty_string(evidence["planner"], "JAG allocation planner")
    _nonempty_string(evidence["score_formula"], "JAG allocation score_formula")
    for field in (
        "branch_counts_frozen_before_child_draws",
        "exact_edge_budget_every_replication",
    ):
        _require_bool(evidence[field], f"JAG allocation {field}")
    if evidence["branch_counts_frozen_before_child_draws"] is not True:
        raise RunIntegrityError("JAG branch counts must be frozen before child draws")
    for field in (
        "decision_event_count", "transport_conditioned_event_count", "oracle_frontier_solve_count"
    ):
        if _require_int(evidence[field], f"JAG allocation {field}") < 0:
            raise RunIntegrityError("JAG allocation event counts must be non-negative")
    for field in ("max_score_identity_error", "oracle_max_optimality_gap"):
        value = _require_number(evidence[field], f"JAG allocation {field}")
        if float(value) < 0.0:
            raise RunIntegrityError("JAG allocation errors/gaps must be non-negative")
    trace_histogram = _require_mapping(
        evidence["trace_hash_histogram"], "JAG allocation trace_hash_histogram"
    )
    trace_total = 0
    for digest, count_value in trace_histogram.items():
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise RunIntegrityError("JAG trace hash is invalid")
        count = _require_int(count_value, f"JAG trace count {digest}")
        if count < 1:
            raise RunIntegrityError("JAG trace count must be positive")
        trace_total += count
    if trace_total != sample_count:
        raise RunIntegrityError("JAG trace hash counts do not sum to sample_count")
    unavailable = row["status"] == "UNAVAILABLE_ORACLE_SCOPE"
    if evidence["exact_edge_budget_every_replication"] is not (not unavailable):
        raise RunIntegrityError("JAG exact-edge-budget evidence disagrees with availability")

    replicates = row["paired_replicates"]
    if not isinstance(replicates, list) or len(replicates) != sample_count:
        raise RunIntegrityError("JAG paired_replicates must contain sample_count records")
    for replication, raw_replicate in enumerate(replicates):
        item = _require_mapping(raw_replicate, f"JAG paired replicate {replication}")
        _exact_keys(item, _JAG_PAIRED_REPLICATE_FIELDS, f"JAG paired replicate {replication}")
        if _require_int(item["replication"], "JAG paired replication") != replication:
            raise RunIntegrityError("JAG paired replication index is non-canonical")
        expected_stream = sha256(
            f"{seed}:jag-crn:{family}:{budget}:{replication}".encode("utf-8")
        ).hexdigest()
        if item["crn_stream_id"] != expected_stream:
            raise RunIntegrityError("JAG CRN stream id fails deterministic replay")
        for field in ("gradient_hash", "gradient_delta_hash"):
            if not isinstance(item[field], str) or not _SHA256.fullmatch(item[field]):
                raise RunIntegrityError("JAG paired gradient hash is invalid")
        squared_error = _require_number(item["squared_error"], "JAG paired squared_error")
        if float(squared_error) < 0.0:
            raise RunIntegrityError("JAG paired squared_error must be non-negative")

    metadata = _require_mapping(row["arm_metadata"], "JAG row arm_metadata")
    expected_metadata_fields = _JAG_ARM_METADATA_FIELDS | (
        _JAG_LEAF_METADATA_FIELDS if arm == "leaf_equal_naive" else set()
    )
    _exact_keys(metadata, expected_metadata_fields, "JAG row arm_metadata")
    if metadata["baseline"] != baseline:
        raise RunIntegrityError("JAG row baseline disagrees with estimator_policy")
    for field in (
        "jag_status", "estimator", "allocation_rule", "moment_source", "moment_estimator",
        "baseline", "baseline_evaluation_frequency",
    ):
        _nonempty_string(metadata[field], f"JAG arm_metadata {field}")
    continuation = metadata["moment_continuation_branching"]
    if continuation is not None and _require_int(continuation, "JAG moment continuation branching") < 1:
        raise RunIntegrityError("JAG moment continuation branching must be positive or null")
    for field in (
        "oracle_ablation", "deployable", "baseline_frozen_before_evaluation",
        "baseline_reads_evaluation_outcomes", "unbiased_claim",
    ):
        _require_bool(metadata[field], f"JAG arm_metadata {field}")
    if arm == "leaf_equal_naive":
        _require_bool(metadata["uses_same_tree_outcomes_for_allocation"], "JAG leaf outcome use")
        _nonempty_string(metadata["negative_control_mechanism"], "JAG leaf mechanism")
        _nonempty_string(metadata["negative_control_scope"], "JAG leaf scope")
    # The runner owns the arm semantics.  Rebuild this small deterministic
    # payload so resealing cannot invert any causal-safety flag.
    from .jag.experiment import _arm_metadata

    if dict(metadata) != _arm_metadata(arm, baseline):
        raise RunIntegrityError("JAG arm_metadata disagrees with the registered arm")

    if row["negative_control"] is not (arm == "leaf_equal_naive"):
        raise RunIntegrityError("JAG negative_control disagrees with arm")
    expected_status = (
        "NON_JAG_NEGATIVE_CONTROL" if arm == "leaf_equal_naive"
        else "LAGGED_RIDGE" if arm == "jag_learned"
        else "EXACT_FRONTIER_ORACLE" if arm == "jag_oracle" and not unavailable
        else "UNAVAILABLE_ORACLE_SCOPE" if arm == "jag_oracle" and unavailable
        else "OK"
    )
    if row["status"] != expected_status:
        raise RunIntegrityError("JAG row status disagrees with its registered arm")
    for field in (
        "bias", "absolute_bias", "relative_bias", "bias_z_score", "mse",
        "budget_times_mse", "cosine",
    ):
        value = _require_number(row[field], f"JAG row {field}", nullable=True)
        if value is not None and field in {
            "bias", "absolute_bias", "relative_bias", "mse", "budget_times_mse"
        } and float(value) < 0.0:
            raise RunIntegrityError(f"JAG row {field} must be non-negative")
        if value is not None and field == "cosine" and not -1.0 <= float(value) <= 1.0:
            raise RunIntegrityError("JAG row cosine must lie in [-1,1]")
    _nonempty_string(row["bias_z_score_status"], "JAG bias_z_score_status")
    if unavailable:
        if sample_count != 0 or row["status_reason"] is None:
            raise RunIntegrityError("unavailable JAG oracle rows need zero samples and a reason")
    elif sample_count < 1 or row["status_reason"] is not None:
        raise RunIntegrityError("available JAG rows need samples and null status_reason")
    if sample_count:
        mse = float(row["mse"])
        paired_mse = float(np.mean([item["squared_error"] for item in replicates]))
        if not math.isclose(mse, paired_mse, rel_tol=1e-15, abs_tol=1e-15) or not math.isclose(
            float(row["budget_times_mse"]), budget * mse, rel_tol=1e-15, abs_tol=1e-15
        ):
            raise RunIntegrityError("JAG MSE/budget_times_mse disagree with paired replicates")
        if row["bias"] != row["absolute_bias"]:
            raise RunIntegrityError("JAG scalar bias must equal absolute_bias")


def _jag_paired_delta(
    first: Mapping[str, Any] | None, second: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    if first is None or second is None:
        return None
    first_by_stream = {item["crn_stream_id"]: item for item in first["paired_replicates"]}
    second_by_stream = {item["crn_stream_id"]: item for item in second["paired_replicates"]}
    shared = sorted(set(first_by_stream) & set(second_by_stream))
    if not shared:
        return None
    deltas = [
        float(first_by_stream[stream]["squared_error"])
        - float(second_by_stream[stream]["squared_error"])
        for stream in shared
    ]
    payload = json.dumps(deltas, separators=(",", ":"), allow_nan=False)
    return {
        "definition": "first_squared_error-minus-second_squared_error_on_shared_crn_stream",
        "paired_count": len(shared),
        "mean_delta": float(np.mean(deltas)),
        "delta_hash": sha256(payload.encode("utf-8")).hexdigest(),
    }


def _jag_registered_comparisons(
    rows: list[Any], families: list[str], arms: list[str], budgets: list[int]
) -> list[dict[str, Any]]:
    by_cell = {
        (row["family"], row["arm"], row["budget"]): row
        for row in rows
        if isinstance(row, Mapping)
    }
    baseline_arms = {
        "flat_iid", "uniform_tree", "entropy_tree", "value_variance_tree",
        "gradient_only", "joint_no_cross",
    }
    comparisons: list[dict[str, Any]] = []
    for family in families:
        for budget in budgets:
            baseline_mse = [
                float(by_cell[(family, arm, budget)]["mse"])
                for arm in arms
                if arm in baseline_arms
                and (family, arm, budget) in by_cell
                and by_cell[(family, arm, budget)]["mse"] is not None
            ]
            oracle = by_cell.get((family, "jag_oracle", budget))
            uniform = by_cell.get((family, "uniform_tree", budget))
            learned = by_cell.get((family, "jag_learned", budget))
            oracle_mse = None if oracle is None else oracle["mse"]
            uniform_mse = None if uniform is None else uniform["mse"]
            learned_mse = None if learned is None else learned["mse"]
            best = min(baseline_mse) if baseline_mse else None
            oracle_reduction = None
            if best not in (None, 0.0) and oracle_mse is not None:
                oracle_reduction = (float(best) - float(oracle_mse)) / float(best)
            gap_closed = None
            if oracle_mse is not None and learned_mse is not None and uniform_mse is not None:
                denominator = float(uniform_mse) - float(oracle_mse)
                gap_closed = None if denominator == 0.0 else (
                    float(uniform_mse) - float(learned_mse)
                ) / denominator
            comparisons.append(
                {
                    "family": family,
                    "budget": budget,
                    "oracle_vs_best_unbiased_relative_mse_reduction": oracle_reduction,
                    "learned_oracle_uniform_gap_closed": gap_closed,
                    "learned_minus_uniform_paired_squared_error": _jag_paired_delta(
                        learned, uniform
                    ),
                }
            )
    for budget in budgets:
        full = by_cell.get(("covariance_reversal", "jag_oracle", budget))
        no_cross = by_cell.get(("covariance_reversal", "joint_no_cross", budget))
        full_mse = None if full is None else full["mse"]
        no_cross_mse = None if no_cross is None else no_cross["mse"]
        gain = None
        if full_mse is not None and no_cross_mse not in (None, 0.0):
            gain = (float(no_cross_mse) - float(full_mse)) / float(no_cross_mse)
        comparisons.append(
            {
                "family": "covariance_reversal",
                "budget": budget,
                "full_joint_vs_no_cross_relative_mse_gain": gain,
                "full_joint_minus_no_cross_paired_squared_error": _jag_paired_delta(
                    full, no_cross
                ),
            }
        )
    return comparisons


def _validate_jag_smoke_replay(
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    gates: Mapping[str, Any],
    rows: list[Any],
    events: list[Any],
    arrays: Mapping[str, np.ndarray],
    identity: Mapping[str, Any],
) -> None:
    """Bind every executable JAG artifact to one trusted finite replay."""

    from .jag import run_jag_phase0
    from .results import normalize_result

    try:
        replay = normalize_result(
            "jag",
            identity["direction"],
            config,
            identity["seed"],
            identity["run_id"],
            run_jag_phase0(config, identity["seed"]),
        )
    except Exception as exc:
        raise RunIntegrityError(
            "JAG deterministic config/seed replay could not be completed"
        ) from exc
    for name, observed, expected in (
        ("metrics/covariance evidence", metrics, replay.metrics),
        ("gates", gates, replay.gates),
        ("rows", rows, list(replay.rows)),
        ("events", events, list(replay.events)),
    ):
        if _canonical_json(observed) != _canonical_json(expected):
            raise RunIntegrityError(
                f"JAG {name} disagree with deterministic config/seed replay"
            )
    if (
        set(arrays) != set(replay.arrays)
        or _deterministic_npz_bytes(arrays)
        != _deterministic_npz_bytes(replay.arrays)
    ):
        raise RunIntegrityError(
            "JAG arrays/covariance evidence disagree with deterministic config/seed replay"
        )


def _validate_jag_contract(
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    gates: Mapping[str, Any],
    rows: list[Any],
    events: list[Any],
    arrays: Mapping[str, np.ndarray],
    identity: Mapping[str, Any],
) -> None:
    if events:
        raise RunIntegrityError("JAG events.jsonl must be empty")
    # The checked-in executable profile is deliberately small and finite.  A
    # complete replay is both stronger and cheaper than replaying covariance
    # and rollout sub-contracts separately.  Formal registrations are not
    # directly executable by this CPU reference runner and use the explicit
    # mathematical checks below instead.
    if identity["profile"] == "smoke":
        _validate_jag_smoke_replay(
            config, metrics, gates, rows, events, arrays, identity
        )
        return
    phase0 = _require_mapping(config.get("phase0"), "JAG phase0")
    estimator = _require_mapping(config.get("estimator", {}), "JAG estimator")
    families = _require_string_list(phase0.get("reward_families"), "JAG reward_families")
    arms = _require_string_list(phase0.get("arms"), "JAG phase0.arms")
    budgets = _require_integer_list(
        phase0.get("budgets", [64, 128, 256]), "JAG phase0.budgets", minimum=1
    )
    if metrics.get("family_order") != families:
        raise RunIntegrityError("JAG family_order disagrees with config reward_families")
    policy = _require_mapping(metrics.get("estimator_policy"), "JAG estimator_policy")
    _exact_keys(
        policy,
        {"primary_baseline", "max_branching", "branchable_depth_count", "branchable_depths"},
        "JAG estimator_policy",
    )
    horizon = _require_int(phase0.get("horizon"), "JAG phase0.horizon")
    actions = _require_int(phase0.get("actions"), "JAG phase0.actions")
    if horizon < 1 or actions < 1:
        raise RunIntegrityError("JAG horizon and action count must be positive")
    depth_count = _require_int(policy["branchable_depth_count"], "JAG branchable_depth_count")
    expected_policy = {
        "primary_baseline": estimator.get("primary_baseline", "zero"),
        "max_branching": estimator.get("max_branching", max(budgets)),
        "branchable_depth_count": estimator.get("branchable_depth_count", horizon),
        "branchable_depths": list(range(depth_count)),
    }
    if dict(policy) != expected_policy:
        raise RunIntegrityError("JAG estimator_policy disagrees with config")
    oracle_scope = _require_mapping(metrics.get("oracle_scope"), "JAG oracle_scope")
    _exact_keys(
        oracle_scope,
        {"target", "global_tree_oracle_claimed", "max_horizon", "max_budget", "max_frontier_nodes", "max_states"},
        "JAG oracle_scope",
    )
    registered_scope_raw = phase0.get("oracle_scope", {})
    registered_scope = _require_mapping(registered_scope_raw, "JAG phase0.oracle_scope")
    default_scope = {
        "max_horizon": 3,
        "max_budget": 12,
        "max_frontier_nodes": 8,
        "max_states": 100_000,
    }
    for field in ("max_horizon", "max_budget", "max_frontier_nodes", "max_states"):
        if oracle_scope.get(field) != registered_scope.get(field, default_scope[field]):
            raise RunIntegrityError(f"JAG oracle_scope {field} disagrees with config")
    if oracle_scope.get("target") != "exact_each_observed_unopened_level_frontier" or oracle_scope.get("global_tree_oracle_claimed") is not False:
        raise RunIntegrityError("JAG oracle_scope overclaims its registered target")

    expected_row_fields = _ROW_ENVELOPE_FIELDS | {"family"} | _JAG_SUMMARY_FIELDS
    observed_cells: set[tuple[str, str, int]] = set()
    observed_cell_order: list[tuple[str, str, int]] = []
    rows_by_family: dict[str, list[Mapping[str, Any]]] = {family: [] for family in families}
    replications = _require_int(
        phase0.get("rollout_replications"), "JAG phase0.rollout_replications"
    )
    if replications < 1:
        raise RunIntegrityError("JAG rollout_replications must be positive")
    for index, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"JAG row {index}")
        _exact_keys(row, expected_row_fields, f"JAG row {index}")
        if row["record_type"] != "arm_budget_summary":
            raise RunIntegrityError(f"JAG row {index} has the wrong record_type")
        family = row["family"]
        arm = row["arm"]
        if not isinstance(family, str) or not isinstance(arm, str):
            raise RunIntegrityError("JAG family and arm must be strings")
        budget = _require_int(row["budget"], f"JAG row {index} budget")
        if family not in families or arm not in arms or budget not in budgets:
            raise RunIntegrityError(f"JAG row {index} identifies an unregistered cell")
        cell = (family, arm, budget)
        if cell in observed_cells:
            raise RunIntegrityError("JAG (family,arm,budget) rows must be unique")
        observed_cells.add(cell)
        observed_cell_order.append(cell)
        rows_by_family[family].append(row)
        sample_count = _require_int(row["sample_count"], f"JAG row {index} sample_count")
        static_oracle_reason: str | None = None
        if arm == "jag_oracle":
            if horizon > int(oracle_scope["max_horizon"]):
                static_oracle_reason = (
                    f"horizon {horizon} exceeds exact oracle "
                    f"max_horizon={oracle_scope['max_horizon']}"
                )
            elif budget > int(oracle_scope["max_budget"]):
                static_oracle_reason = (
                    f"budget {budget} exceeds exact oracle "
                    f"max_budget={oracle_scope['max_budget']}"
                )
        observed_oracle_unavailable = (
            arm == "jag_oracle" and row["status"] == "UNAVAILABLE_ORACLE_SCOPE"
        )
        if static_oracle_reason is not None:
            if not observed_oracle_unavailable or row["status_reason"] != static_oracle_reason:
                raise RunIntegrityError(
                    "JAG oracle availability disagrees with its static registered scope"
                )
        elif observed_oracle_unavailable:
            reason = row["status_reason"]
            if not isinstance(reason, str):
                raise RunIntegrityError("JAG dynamic oracle unavailability needs a reason")
            frontier_match = re.fullmatch(
                r"frontier node count ([0-9]+) exceeds max_frontier_nodes=([0-9]+)",
                reason,
            )
            state_match = re.fullmatch(
                r"frontier enumeration needs more than max_states=([0-9]+)",
                reason,
            )
            frontier_valid = (
                frontier_match is not None
                and int(frontier_match.group(2)) == int(oracle_scope["max_frontier_nodes"])
                and int(frontier_match.group(1)) > int(frontier_match.group(2))
            )
            state_valid = (
                state_match is not None
                and int(state_match.group(1)) == int(oracle_scope["max_states"])
            )
            if not (frontier_valid or state_valid):
                raise RunIntegrityError(
                    "JAG dynamic oracle reason disagrees with its registered scope"
                )
        oracle_unavailable = static_oracle_reason is not None or observed_oracle_unavailable
        expected_samples = 0 if oracle_unavailable else replications
        if sample_count != expected_samples:
            raise RunIntegrityError("JAG row sample_count disagrees with registration/oracle scope")
        _validate_jag_row_nested(
            row,
            family=family,
            arm=arm,
            budget=budget,
            sample_count=sample_count,
            seed=identity["seed"],
            baseline=str(policy["primary_baseline"]),
            max_branching=int(policy["max_branching"]),
        )
    expected_cells = {(family, arm, budget) for family in families for arm in arms for budget in budgets}
    if observed_cells != expected_cells:
        raise RunIntegrityError("JAG rows do not cover the registered family/arm/budget grid")
    expected_cell_order = [
        (family, arm, budget) for family in families for arm in arms for budget in budgets
    ]
    if observed_cell_order != expected_cell_order:
        raise RunIntegrityError("JAG rows are not in canonical family/arm/budget order")

    allocation = metrics.get("allocation_evidence")
    if not isinstance(allocation, list) or len(allocation) != len(rows):
        raise RunIntegrityError("JAG allocation_evidence must contain one entry per row")
    allocation_by_cell: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    allocation_fields = {"family", "arm", "budget", "allocation", "allocation_evidence", "arm_status"}
    for index, raw_entry in enumerate(allocation):
        entry = _require_mapping(raw_entry, f"JAG allocation_evidence {index}")
        _exact_keys(entry, allocation_fields, f"JAG allocation_evidence {index}")
        cell = (entry.get("family"), entry.get("arm"), _require_int(entry.get("budget"), "JAG allocation budget"))
        if cell in allocation_by_cell:
            raise RunIntegrityError("JAG allocation_evidence contains a duplicate cell")
        allocation_by_cell[cell] = entry
    for row in rows:
        cell = (row["family"], row["arm"], row["budget"])
        entry = allocation_by_cell.get(cell)
        if entry is None or entry["allocation"] != row["allocation"] or entry["allocation_evidence"] != row["allocation_evidence"] or entry["arm_status"] != row["status"]:
            raise RunIntegrityError("JAG allocation_evidence disagrees with its row")

    family_metadata = _require_mapping(metrics.get("family_metadata"), "JAG family_metadata")
    if set(family_metadata) != set(families):
        raise RunIntegrityError("JAG family_metadata keys disagree with family_order")
    expected_array_keys: set[str] = set()
    covariance_by_family: dict[str, Mapping[str, Any]] = {}
    metadata_fields = {
        "probability_sum", "expected_reward", "structural_audit", "covariance_audit",
        "predictor_provenance", "summary_count", "exact_gradient_array_key",
    }
    covariance_base_fields = {
        "status", "prefix", "sample_count", "held_out_from_rollout_and_calibration", "object",
        "numerator", "numerator_definition", "denominator", "denominator_definition",
        "relative_frobenius_error", "predicted_covariance_hash", "realized_covariance_hash",
        "joint_dimension", "covariance_dtype", "covariance_normalization",
        "realized_joint_mean", "predicted_covariance_array_key", "realized_covariance_array_key",
    }
    for family_index, family in enumerate(families):
        metadata = _require_mapping(family_metadata[family], f"JAG family_metadata.{family}")
        _exact_keys(metadata, metadata_fields, f"JAG family_metadata.{family}")
        if _require_int(metadata["summary_count"], f"JAG {family} summary_count") != len(rows_by_family[family]):
            raise RunIntegrityError("JAG family summary_count disagrees with rows")
        gradient_key = metadata["exact_gradient_array_key"]
        if not isinstance(gradient_key, str):
            raise RunIntegrityError("JAG exact gradient array key must be a string")
        expected_gradient_key = f"jag_exact_gradient_f{family_index:03d}_{_runtime_slug(family)}"
        if gradient_key != expected_gradient_key:
            raise RunIntegrityError("JAG exact gradient array key is not canonical")
        gradient = _require_npz_array(arrays, gradient_key, ndim=1)
        if gradient.size < 1:
            raise RunIntegrityError("JAG exact gradient arrays must not be empty")
        # Keep this import local: JAG's experiment module imports ``named_rng``
        # from this module, so a top-level import would create a cycle.
        from .jag.mdp import exact_target, make_phase0_mdp, structural_audit

        mdp = make_phase0_mdp(horizon, actions, family, identity["seed"])
        exact = exact_target(mdp)
        if not np.array_equal(gradient, np.asarray(exact.gradient, dtype="<f8")):
            raise RunIntegrityError("JAG exact gradient array disagrees with exact finite-MDP enumeration")
        probability_sum = _require_number(
            metadata["probability_sum"], f"JAG {family} probability_sum"
        )
        expected_reward = _require_number(
            metadata["expected_reward"], f"JAG {family} expected_reward"
        )
        if float(probability_sum) != float(exact.probability_sum) or float(expected_reward) != float(exact.expected_reward):
            raise RunIntegrityError("JAG exact target scalars disagree with finite-MDP enumeration")
        if metadata["structural_audit"] != structural_audit(mdp, exact):
            raise RunIntegrityError("JAG structural_audit disagrees with exact finite-MDP recomputation")
        predictor = metadata["predictor_provenance"]
        predictor_required = "jag_learned" in arms or policy["primary_baseline"] == "lagged_cross_fitted_value"
        if (predictor is not None) is not predictor_required:
            raise RunIntegrityError("JAG predictor provenance presence disagrees with estimator/arms")
        calibration_record = None
        if predictor is not None:
            provenance = _require_mapping(predictor, f"JAG {family} predictor_provenance")
            _exact_keys(provenance, _JAG_PREDICTOR_FIELDS, f"JAG {family} predictor_provenance")
            expected_calibration_seeds = _require_integer_list(
                phase0.get(
                    "learned_calibration_seeds",
                    [identity["seed"] - 2, identity["seed"] - 1],
                ),
                "JAG learned_calibration_seeds",
            )
            calibration_horizon = _require_int(
                phase0.get("learned_calibration_horizon", horizon), "JAG calibration horizon"
            )
            from .jag.experiment import build_calibration_record

            calibration_record = build_calibration_record(
                family,
                calibration_horizon,
                actions,
                tuple(expected_calibration_seeds),
                risk_label_baseline=str(policy["primary_baseline"]),
            )
            expected_provenance = {
                "schema_version": calibration_record.schema_version,
                "completed": calibration_record.completed,
                "calibration_seeds": list(calibration_record.environment_seeds),
                "environment_ids": list(calibration_record.environment_ids),
                "environment_hashes": list(calibration_record.environment_hashes),
                "row_count": len(calibration_record.targets),
                "data_hash": calibration_record.data_hash,
                "provenance_hash": calibration_record.provenance_hash,
                "risk_label_baseline": calibration_record.risk_label_baseline,
                "evaluation_seed": identity["seed"],
                "oracle_labels_from_evaluation_environment": False,
                "evaluation_baseline_fit_from_completed_record_only": (
                    policy["primary_baseline"] == "lagged_cross_fitted_value"
                ),
            }
            if dict(provenance) != expected_provenance:
                raise RunIntegrityError(
                    "JAG predictor provenance disagrees with the trusted calibration replay"
                )
        expected_array_keys.add(gradient_key)

        covariance = _require_mapping(metadata["covariance_audit"], f"JAG {family} covariance_audit")
        _exact_keys(covariance, covariance_base_fields, f"JAG {family} covariance_audit")
        predicted_key, realized_key = _jag_covariance_array_keys(family_index, family)
        if covariance["predicted_covariance_array_key"] != predicted_key or covariance["realized_covariance_array_key"] != realized_key:
            raise RunIntegrityError("JAG covariance array keys are not canonical")
        predicted = _require_npz_array(arrays, predicted_key, ndim=2)
        realized = _require_npz_array(arrays, realized_key, ndim=2)
        dimension = _require_int(covariance["joint_dimension"], f"JAG {family} covariance joint_dimension")
        expected_covariance_samples = _require_int(
            phase0.get("covariance_audit_replications", 32),
            "JAG covariance_audit_replications",
        )
        if covariance["sample_count"] != expected_covariance_samples:
            raise RunIntegrityError("JAG covariance sample_count disagrees with config")
        if dimension < 1 or predicted.shape != (dimension, dimension) or realized.shape != predicted.shape or dimension != gradient.size + 1:
            raise RunIntegrityError("JAG covariance matrix dimensions disagree with metadata")
        if covariance["covariance_dtype"] != "<f8" or covariance["covariance_normalization"] != "population_1_over_n":
            raise RunIntegrityError("JAG covariance dtype/normalization metadata is invalid")
        if covariance["held_out_from_rollout_and_calibration"] is not True:
            raise RunIntegrityError("JAG covariance audit must be held out")
        if covariance["object"] != "joint_node_contribution_[value,gradient]_covariance":
            raise RunIntegrityError("JAG covariance audit object is not registered")
        if covariance["numerator_definition"] != "squared_frobenius(realized_covariance-predicted_covariance)" or covariance["denominator_definition"] != "squared_frobenius(predicted_covariance)":
            raise RunIntegrityError("JAG covariance audit definitions are not canonical")
        for matrix in (predicted, realized):
            symmetry_tolerance = (
                64.0
                * np.finfo(np.float64).eps
                * max(1.0, float(np.linalg.norm(matrix, ord=np.inf)))
            )
            if float(np.max(np.abs(matrix - matrix.T))) > symmetry_tolerance:
                raise RunIntegrityError("JAG covariance matrices must be symmetric")
        for matrix in (predicted, realized):
            symmetric = (matrix + matrix.T) / 2.0
            tolerance = 1e-10 * max(1.0, float(np.linalg.norm(symmetric, ord=2)))
            if float(np.min(np.linalg.eigvalsh(symmetric))) < -tolerance:
                raise RunIntegrityError("JAG covariance matrices must be positive semidefinite")
        realized_joint_mean = _strict_float_array(
            covariance["realized_joint_mean"], f"JAG {family} realized_joint_mean", 1
        )
        if realized_joint_mean.shape != (dimension,):
            raise RunIntegrityError("JAG realized_joint_mean dimension disagrees with covariance")
        if covariance["predicted_covariance_hash"] != _array_hash(predicted) or covariance["realized_covariance_hash"] != _array_hash(realized):
            raise RunIntegrityError("JAG covariance matrix hash disagrees with arrays.npz")
        difference = realized - predicted
        numerator = float(np.sum(difference * difference))
        denominator = float(np.sum(predicted * predicted))
        relative = None if denominator == 0.0 else float(np.sqrt(numerator / denominator))
        expected_covariance_status = "OK" if denominator > 0.0 else "ZERO_PREDICTED_COVARIANCE"
        if covariance["status"] != expected_covariance_status:
            raise RunIntegrityError("JAG covariance status disagrees with its denominator")
        for field, expected in (("numerator", numerator), ("denominator", denominator)):
            observed = _require_number(covariance[field], f"JAG {family} covariance {field}")
            if float(observed) != expected:
                raise RunIntegrityError(f"JAG covariance {field} disagrees with sealed matrices")
        observed_relative = _require_number(
            covariance["relative_frobenius_error"],
            f"JAG {family} covariance relative error",
            nullable=True,
        )
        if observed_relative != relative:
            raise RunIntegrityError("JAG covariance relative error disagrees with sealed matrices")
        from .jag.experiment import _held_out_covariance_audit, fit_value_baseline

        if policy["primary_baseline"] == "lagged_cross_fitted_value":
            if calibration_record is None:
                raise RunIntegrityError("JAG lagged baseline is missing its calibration record")
            value_baseline = fit_value_baseline(calibration_record, identity["seed"])
            baseline_prefix: Any = lambda prefix, model=mdp, baseline=value_baseline: baseline.predict(
                model, prefix
            )
            baseline_node: Any = lambda node, model=mdp, baseline=value_baseline: baseline.predict(
                model, node.prefix
            )
        else:
            baseline_prefix = 0.0
            baseline_node = 0.0
        trusted_covariance = _held_out_covariance_audit(
            mdp,
            identity["seed"],
            expected_covariance_samples,
            baseline_prefix,
            baseline_node,
        )
        sealed_covariance = dict(covariance)
        del sealed_covariance["predicted_covariance_array_key"]
        del sealed_covariance["realized_covariance_array_key"]
        sealed_covariance["predicted_covariance"] = predicted.tolist()
        sealed_covariance["realized_covariance"] = realized.tolist()
        if sealed_covariance != trusted_covariance:
            raise RunIntegrityError(
                "JAG covariance audit disagrees with deterministic held-out replay"
            )
        expected_array_keys.update({predicted_key, realized_key})
        covariance_by_family[family] = covariance
    if set(arrays) != expected_array_keys:
        raise RunIntegrityError("JAG arrays.npz has a missing or orphaned family array")

    gate_inputs = _require_mapping(gates.get("inputs"), "JAG gates inputs")
    _exact_keys(gate_inputs, _JAG_GATE_INPUT_FIELDS, "JAG gates inputs")
    if gate_inputs["thresholds"] != phase0.get("gate", {}):
        raise RunIntegrityError("JAG gate thresholds disagree with config")
    expected_relative_bias = [
        {
            "family": row["family"],
            "arm": row["arm"],
            "budget": row["budget"],
            "value": row["relative_bias"],
        }
        for row in rows
    ]
    expected_bias_z = [
        {
            "family": row["family"],
            "arm": row["arm"],
            "budget": row["budget"],
            "value": row["bias_z_score"],
            "status": row["bias_z_score_status"],
        }
        for row in rows
    ]
    if gate_inputs["relative_bias"] != expected_relative_bias or gate_inputs["bias_z_score"] != expected_bias_z:
        raise RunIntegrityError("JAG gate row diagnostics do not exactly mirror rows.jsonl")
    comparisons = gate_inputs["registered_comparisons"]
    if not isinstance(comparisons, list):
        raise RunIntegrityError("JAG registered_comparisons must be a list")
    general_comparison_fields = {
        "family", "budget", "oracle_vs_best_unbiased_relative_mse_reduction",
        "learned_oracle_uniform_gap_closed", "learned_minus_uniform_paired_squared_error",
    }
    covariance_comparison_fields = {
        "family", "budget", "full_joint_vs_no_cross_relative_mse_gain",
        "full_joint_minus_no_cross_paired_squared_error",
    }
    paired_delta_fields = {"definition", "paired_count", "mean_delta", "delta_hash"}
    for index, raw_comparison in enumerate(comparisons):
        comparison = _require_mapping(raw_comparison, f"JAG comparison {index}")
        keys = set(comparison)
        if keys == general_comparison_fields:
            nullable_fields = (
                "oracle_vs_best_unbiased_relative_mse_reduction",
                "learned_oracle_uniform_gap_closed",
            )
            paired_field = "learned_minus_uniform_paired_squared_error"
        elif keys == covariance_comparison_fields:
            nullable_fields = ("full_joint_vs_no_cross_relative_mse_gain",)
            paired_field = "full_joint_minus_no_cross_paired_squared_error"
        else:
            raise RunIntegrityError("JAG registered comparison has an unknown schema")
        _nonempty_string(comparison["family"], "JAG comparison family")
        if _require_int(comparison["budget"], "JAG comparison budget") < 1:
            raise RunIntegrityError("JAG comparison budget must be positive")
        for field in nullable_fields:
            _require_number(comparison[field], f"JAG comparison {field}", nullable=True)
        paired = comparison[paired_field]
        if paired is not None:
            paired_map = _require_mapping(paired, f"JAG comparison {paired_field}")
            _exact_keys(paired_map, paired_delta_fields, f"JAG comparison {paired_field}")
            _nonempty_string(paired_map["definition"], "JAG paired delta definition")
            if _require_int(paired_map["paired_count"], "JAG paired delta count") < 1:
                raise RunIntegrityError("JAG paired delta count must be positive")
            _require_number(paired_map["mean_delta"], "JAG paired delta mean")
            if not isinstance(paired_map["delta_hash"], str) or not _SHA256.fullmatch(paired_map["delta_hash"]):
                raise RunIntegrityError("JAG paired delta hash is invalid")
    if comparisons != _jag_registered_comparisons(rows, families, arms, budgets):
        raise RunIntegrityError("JAG registered comparisons disagree with rows")
    held_out = gate_inputs.get("held_out_covariance_audit")
    if held_out is not None:
        if not isinstance(held_out, list) or len(held_out) != len(families):
            raise RunIntegrityError("JAG held_out_covariance_audit must contain one entry per family")
        seen: set[str] = set()
        for index, raw_audit in enumerate(held_out):
            audit = _require_mapping(raw_audit, f"JAG held-out covariance audit {index}")
            gate_fields = (
                covariance_base_fields
                - {"predicted_covariance_array_key", "realized_covariance_array_key"}
                | {"family", "predicted_covariance", "realized_covariance"}
            )
            _exact_keys(audit, gate_fields, f"JAG held-out covariance audit {index}")
            family = audit.get("family")
            if family not in covariance_by_family or family in seen:
                raise RunIntegrityError("JAG held-out covariance audit family is missing or duplicated")
            seen.add(family)
            metadata_audit = covariance_by_family[family]
            predicted_key = metadata_audit["predicted_covariance_array_key"]
            realized_key = metadata_audit["realized_covariance_array_key"]
            expected_gate_audit = dict(metadata_audit)
            del expected_gate_audit["predicted_covariance_array_key"]
            del expected_gate_audit["realized_covariance_array_key"]
            expected_gate_audit["family"] = family
            expected_gate_audit["predicted_covariance"] = arrays[predicted_key].tolist()
            expected_gate_audit["realized_covariance"] = arrays[realized_key].tolist()
            if dict(audit) != expected_gate_audit:
                raise RunIntegrityError("JAG gate covariance audit disagrees with family metadata/arrays")
            gate_predicted = _strict_float_array(
                audit["predicted_covariance"],
                f"JAG held-out covariance {family} predicted",
                2,
            )
            gate_realized = _strict_float_array(
                audit["realized_covariance"],
                f"JAG held-out covariance {family} realized",
                2,
            )
            if not np.array_equal(gate_predicted, arrays[predicted_key]) or not np.array_equal(
                gate_realized, arrays[realized_key]
            ):
                raise RunIntegrityError("JAG gate covariance matrices disagree with arrays.npz")

    evidence = _require_mapping(gates.get("evidence"), "JAG gates evidence")
    _exact_keys(evidence, _JAG_GATE_EVIDENCE_FIELDS, "JAG gates evidence")
    if evidence["active_seed"] != identity["seed"] or evidence["configured_seed_ids"] != config["seeds"]:
        raise RunIntegrityError("JAG gate evidence seed identities disagree with config")
    if evidence["active_seed_declared"] is not True:
        raise RunIntegrityError("JAG gate evidence must acknowledge the declared active seed")
    if evidence["observed_seed_count"] != 1 or evidence["required_seed_count"] != 50:
        raise RunIntegrityError("JAG gate evidence seed counts are not the single-run registration")
    if evidence["rollout_replications_per_cell"] != replications:
        raise RunIntegrityError("JAG gate evidence replication count disagrees with config")
    available_cells = sum(_require_int(row["sample_count"], "JAG sample_count") > 0 for row in rows)
    if evidence["available_cell_count"] != available_cells or evidence["required_cell_count"] != 108:
        raise RunIntegrityError("JAG gate evidence cell counts disagree with rows/registration")
    checks = _require_mapping(evidence["checks"], "JAG gates evidence checks")
    _exact_keys(checks, _JAG_GATE_CHECK_FIELDS, "JAG gates evidence checks")
    registered_families = {"root_only", "suffix_only", "entropy_distractor", "covariance_reversal"}
    registered_arms = {
        "flat_iid", "uniform_tree", "leaf_equal_naive", "entropy_tree",
        "value_variance_tree", "gradient_only", "joint_no_cross", "jag_oracle", "jag_learned",
    }
    required_cells = {
        (family, arm, budget)
        for family in registered_families
        for arm in registered_arms
        for budget in (64, 128, 256)
    }
    diagnostics_pass = all(
        family not in families
        or _require_mapping(
            _require_mapping(family_metadata[family], f"JAG {family} metadata")["structural_audit"],
            f"JAG {family} structural audit",
        ).get("status") == "PASS"
        for family in ("entropy_distractor", "covariance_reversal")
    ) and {"entropy_distractor", "covariance_reversal"}.issubset(families)
    expected_checks = {
        "registered_families_present": registered_families.issubset(families),
        "registered_arms_present": registered_arms.issubset(arms),
        "registered_budgets_present": {64, 128, 256}.issubset(budgets),
        "registered_replications_present": replications >= 100_000,
        "all_registered_cells_have_samples": required_cells.issubset(
            {cell for cell in observed_cells if next(row for row in rows if (row["family"], row["arm"], row["budget"]) == cell)["sample_count"] == replications}
        ),
        "fifty_environment_seed_results_present": False,
        "diagnostic_structures_pass": diagnostics_pass,
        "registered_covariance_coverage_gate_implemented": False,
    }
    if dict(checks) != expected_checks:
        raise RunIntegrityError("JAG gate evidence checks disagree with config/rows")
    expected_missing = [field for field, passed in expected_checks.items() if not passed]
    if evidence["missing"] != expected_missing:
        raise RunIntegrityError("JAG gate evidence missing list disagrees with failed checks")
    failed_requested_diagnostics = [
        family
        for family in ("entropy_distractor", "covariance_reversal")
        if family in families
        and _require_mapping(family_metadata[family]["structural_audit"], "JAG structural audit").get("status") != "PASS"
    ]
    expected_source_status = "INVALID" if failed_requested_diagnostics else "INCOMPLETE"
    if gates.get("source_status") != expected_source_status or gates.get("formal_evidence") is not False:
        raise RunIntegrityError("JAG gates source_status disagrees with recomputed diagnostics")
    if expected_source_status == "INVALID":
        expected_reason = "source_result_invalid"
    elif identity["profile"] == "smoke":
        expected_reason = "smoke_profile_below_formal_registration"
    elif expected_missing:
        expected_reason = "missing registered evidence: " + ", ".join(expected_missing)
    else:
        expected_reason = (
            "formal gate aggregation across 50 seeds is outside a single-seed runner"
        )
    if gates.get("reason") != expected_reason:
        raise RunIntegrityError("JAG gates reason disagrees with recomputed evidence")

def _expected_report(identity: Mapping[str, Any], gates: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    reason = _nonempty_string(gates.get("reason"), "gates reason")
    return (
        "# Phase-0 run report\n\n"
        f"- Run ID: {identity['run_id']}\n"
        f"- Direction: {identity['direction']} ({identity['direction_alias']})\n"
        f"- Seed: {identity['seed']}\n"
        f"- Profile: {identity['profile']}\n"
        f"- Scientific status: {identity['status']}\n"
        f"- Status reason: {reason}\n"
        f"- Rows/events/arrays: {identity['row_count']} / "
        f"{identity['event_count']} / {len(arrays)}\n\n"
        "This is a deterministic finite CPU Phase-0 mechanism experiment. "
        "It is not a 7B-model result or a coding-benchmark score. COMPLETE "
        "means that the artifacts are sealed; gates.json is authoritative "
        "for the scientific status.\n"
    )


def _validate_direction_contract(
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    gates: Mapping[str, Any],
    rows: list[Any],
    events: list[Any],
    arrays: Mapping[str, np.ndarray],
    identity: Mapping[str, Any],
) -> None:
    expected_metrics = _METRICS_FIELDS | _DIRECTION_METRICS_FIELDS[identity["direction"]]
    _exact_keys(metrics, expected_metrics, "metrics.json")
    if identity["direction"] == "predictive_belief_particle_filter":
        _validate_pbpf_contract(config, metrics, gates, rows, events, arrays, identity)
    elif identity["direction"] == "gradient_optimal_active_verification":
        _validate_goav_contract(config, metrics, gates, rows, events, arrays, identity)
    else:
        _validate_jag_contract(config, metrics, gates, rows, events, arrays, identity)


def _validate_cross_artifact_contract(
    run_dir: Path,
    config: Any,
    manifest: Any,
    metrics: Any,
    gates: Any,
    events: list[Any],
    rows: list[Any],
    arrays: Mapping[str, np.ndarray],
    report: str,
) -> dict[str, Any]:
    identity = _validate_initial_contract(run_dir, config, manifest)
    metrics_map = _require_mapping(metrics, "metrics.json")
    gates_map = _require_mapping(gates, "gates.json")
    if not _METRICS_FIELDS <= set(metrics_map):
        raise RunIntegrityError("metrics.json lacks its exact common fields or scientific payload")
    if metrics_map["schema_version"] != "phase0.metrics.v1":
        raise RunIntegrityError("metrics.json schema_version must be phase0.metrics.v1")
    expected_gate_fields = _GATES_FIELDS | ({"evidence"} if identity["direction"] == "jag_tree" else set())
    _exact_keys(gates_map, expected_gate_fields, "gates.json")
    if gates_map["schema_version"] != "phase0.gates.v1":
        raise RunIntegrityError("gates.json schema_version must be phase0.gates.v1")
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        raise RunIntegrityError("rows.jsonl must contain at least one object")
    if not arrays:
        raise RunIntegrityError("arrays.npz must contain at least one array")
    if not report.strip():
        raise RunIntegrityError("report.md must contain non-whitespace text")

    for name, document in (("metrics.json", metrics_map), ("gates.json", gates_map)):
        _require_int(document.get("seed"), f"{name} seed")
        expected_identity = {
            "direction": identity["direction"],
            "profile": identity["profile"],
            "seed": identity["seed"],
            "status": identity["status"],
        }
        for field, expected in expected_identity.items():
            if document.get(field) != expected:
                raise RunIntegrityError(f"{name} {field} disagrees with the manifest")
    if (
        _require_int(metrics_map["row_count"], "metrics row_count") != identity["row_count"]
        or _require_int(metrics_map["event_count"], "metrics event_count") != identity["event_count"]
        or identity["row_count"] != len(rows)
        or identity["event_count"] != len(events)
    ):
        raise RunIntegrityError("manifest row/event counts disagree with JSONL artifacts")
    if metrics_map["array_keys"] != identity["array_keys"] or identity["array_keys"] != sorted(arrays):
        raise RunIntegrityError("manifest array_keys disagree with arrays.npz")

    formal_evidence = gates_map["formal_evidence"]
    if not isinstance(formal_evidence, bool):
        raise RunIntegrityError("gates formal_evidence must be boolean")
    status = identity["status"]
    profile = identity["profile"]
    source_status = gates_map["source_status"]
    if source_status not in {"PASS", "FAIL", "INCOMPLETE", "INVALID"}:
        raise RunIntegrityError("gates source_status is invalid")
    if profile == "smoke":
        expected_status = "INVALID" if source_status == "INVALID" else "INCOMPLETE"
        expected_evidence = False
    elif source_status in {"PASS", "FAIL"} and formal_evidence:
        expected_status = source_status
        expected_evidence = True
    elif source_status == "INVALID":
        expected_status = "INVALID"
        expected_evidence = False
    else:
        expected_status = "INCOMPLETE"
        expected_evidence = False
    if status != expected_status or formal_evidence is not expected_evidence:
        raise RunIntegrityError(
            "gates source_status/status/formal_evidence/profile matrix is invalid"
        )
    _nonempty_string(gates_map["reason"], "gates reason")
    _require_mapping(gates_map["inputs"], "gates inputs")
    if identity["direction"] == "jag_tree":
        _require_mapping(gates_map["evidence"], "gates evidence")

    _validate_record_envelopes(rows, events, identity)
    config_map = _require_mapping(config, "resolved_config.json")
    _validate_direction_contract(
        config_map, metrics_map, gates_map, rows, events, arrays, identity
    )
    if report != _expected_report(identity, gates_map, arrays):
        raise RunIntegrityError("report.md does not match the canonical Phase-0 report template")

    return dict(identity)


def _read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RunIntegrityError(
                f"required artifact is not a regular unsymlinked file: {path.name}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        signature_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        signature_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        raw = b"".join(chunks)
        if signature_before != signature_after or len(raw) != before.st_size:
            raise RunIntegrityError(f"artifact changed while being read: {path.name}")
        return raw
    except OSError as exc:
        raise RunIntegrityError(f"could not read artifact: {path.name}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _parse_scientific_artifacts(
    run_dir: Path, retained: Mapping[str, bytes] | None = None
) -> tuple[dict[str, Any], dict[str, bytes]]:
    if retained is None:
        raw = {name: _read_regular(run_dir / name) for name in sorted(_REQUIRED_ARTIFACTS)}
    else:
        missing = _REQUIRED_ARTIFACTS - set(retained)
        if missing:
            raise RunIntegrityError(f"retained snapshot lacks required artifacts: {sorted(missing)}")
        raw = {name: retained[name] for name in sorted(_REQUIRED_ARTIFACTS)}
    config = _parse_canonical_json(raw["resolved_config.json"], "resolved_config.json")
    manifest = _parse_canonical_json(raw["manifest.json"], "manifest.json")
    metrics = _parse_canonical_json(raw["metrics.json"], "metrics.json")
    gates = _parse_canonical_json(raw["gates.json"], "gates.json")
    events = _parse_canonical_jsonl(raw["events.jsonl"], "events.jsonl")
    rows = _parse_canonical_jsonl(raw["rows.jsonl"], "rows.jsonl")
    arrays = _parse_deterministic_npz(raw["arrays.npz"])
    try:
        report = raw["report.md"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RunIntegrityError("report.md is not valid UTF-8") from exc
    summary = _validate_cross_artifact_contract(
        run_dir, config, manifest, metrics, gates, events, rows, arrays, report
    )
    return summary, raw


def _json_snapshot(value: Any, name: str) -> tuple[Any, bytes]:
    """Return a defensive JSON snapshot and the exact canonical bytes to persist."""

    try:
        encoded = (_canonical_json(value) + "\n").encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        raise RunIntegrityError(f"{name} is not finite JSON-compatible data") from exc
    return _parse_canonical_json(encoded, name), encoded


def _preflight_run_path(run_dir: Path) -> None:
    """Reject lexical traversal, existing targets, and symlink/non-directory parents."""

    if ".." in run_dir.parts:
        raise RunIntegrityError("run path must not contain parent traversal")
    absolute = run_dir if run_dir.is_absolute() else Path.cwd() / run_dir
    current = Path(absolute.anchor)
    for component in absolute.parts[1:-1]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RunIntegrityError(f"could not inspect run parent: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RunIntegrityError(f"run parent is not a real directory: {current}")
    try:
        run_dir.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RunIntegrityError(f"could not inspect run target: {run_dir}") from exc
    raise FileExistsError(f"run directory already exists and cannot be reused: {run_dir}")


class RunWriter:
    """Exclusively own a new run directory and atomically create its artifacts.

    There is deliberately no resume mode. A failed run retains ``RUNNING``;
    a valid run is sealed once with checksums and a final ``COMPLETE`` anchor.
    """

    def __init__(
        self,
        run_dir: str | Path,
        resolved_config: Mapping[str, Any],
        manifest: Mapping[str, Any] | None = None,
    ):
        self.run_dir = Path(run_dir)
        config_snapshot, config_bytes = _json_snapshot(resolved_config, "resolved_config.json")
        manifest_snapshot, manifest_bytes = _json_snapshot(manifest, "manifest.json")
        _validate_initial_contract(self.run_dir, config_snapshot, manifest_snapshot)
        _preflight_run_path(self.run_dir)
        self.run_dir.parent.mkdir(parents=True, exist_ok=True)
        _preflight_run_path(self.run_dir)
        try:
            self.run_dir.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise FileExistsError(f"run directory already exists and cannot be reused: {self.run_dir}") from exc
        metadata = self.run_dir.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RunIntegrityError(f"run path is not a secure directory: {self.run_dir}")
        _fsync_directory(self.run_dir.parent)
        self._create_exclusive("RUNNING", b"running\n")
        self._create_exclusive("resolved_config.json", config_bytes)
        self._create_exclusive("manifest.json", manifest_bytes)

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False

    def _path(self, name: str) -> Path:
        return self.run_dir / _safe_artifact_name(name)

    def _ensure_active(self) -> None:
        if self.run_dir.is_symlink() or not self.run_dir.is_dir():
            raise RunIntegrityError(f"run directory is not secure: {self.run_dir}")
        marker = self.run_dir / "RUNNING"
        try:
            (self.run_dir / "COMPLETE").lstat()
            complete_present = True
        except FileNotFoundError:
            complete_present = False
        except OSError as exc:
            raise RunIntegrityError("could not inspect COMPLETE marker") from exc
        if complete_present or not _is_regular_unsymlinked(marker):
            raise FileExistsError(f"run is not writable: {self.run_dir}")

    def _create_exclusive(self, name: str, contents: bytes) -> Path:
        target = self._path(name)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=self.run_dir)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_name, target)
            except FileExistsError:
                raise FileExistsError(f"artifact already exists and cannot be replaced: {target}") from None
            _fsync_directory(self.run_dir)
        finally:
            try:
                os.unlink(temporary_name)
                _fsync_directory(self.run_dir)
            except FileNotFoundError:
                pass
        return target

    def write_text(self, name: str, contents: str) -> Path:
        """Atomically create a UTF-8 artifact without replacing existing data."""

        if not isinstance(contents, str):
            raise TypeError("text artifact contents must be a string")
        self._ensure_active()
        return self._create_exclusive(name, contents.encode("utf-8"))

    def write_json(self, name: str, data: Any) -> Path:
        self._ensure_active()
        try:
            encoded = (_canonical_json(data) + "\n").encode("utf-8")
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ValueError(f"{name} must contain finite JSON-compatible data") from exc
        return self._create_exclusive(name, encoded)

    def write_jsonl(self, name: str, records: Iterable[Mapping[str, Any]]) -> Path:
        self._ensure_active()
        if isinstance(records, (str, bytes, Mapping)):
            raise TypeError("JSONL records must be an iterable of mappings")
        encoded_lines: list[str] = []
        try:
            for record in records:
                if not isinstance(record, Mapping):
                    raise TypeError("each JSONL record must be a mapping")
                encoded_lines.append(_canonical_json(dict(record)) + "\n")
            encoded = "".join(encoded_lines).encode("utf-8")
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ValueError(f"{name} must contain finite JSON-compatible records") from exc
        return self._create_exclusive(name, encoded)

    def write_resolved_config(self, data: Mapping[str, Any]) -> Path:
        return self.write_json("resolved_config.json", dict(data))

    def write_manifest(self, data: Mapping[str, Any]) -> Path:
        return self.write_json("manifest.json", dict(data))

    def write_metrics(self, data: Mapping[str, Any]) -> Path:
        return self.write_json("metrics.json", dict(data))

    def write_gates(self, data: Mapping[str, Any]) -> Path:
        return self.write_json("gates.json", dict(data))

    def write_report(self, markdown: str) -> Path:
        return self.write_text("report.md", markdown)

    def write_rows(self, rows: Iterable[Mapping[str, Any]]) -> Path:
        return self.write_jsonl("rows.jsonl", rows)

    def write_events(self, events: Iterable[Mapping[str, Any]]) -> Path:
        return self.write_jsonl("events.jsonl", events)

    def write_event(self, event: Mapping[str, Any]) -> Path:
        """Compatibility convenience for a one-record, one-shot event log."""

        return self.write_events([event])

    def write_npz(self, arrays: Mapping[str, Any]) -> Path:
        self._ensure_active()
        encoded = _deterministic_npz_bytes(arrays)
        return self._create_exclusive("arrays.npz", encoded)

    def complete(self) -> Path:
        """Validate, checksum, and irreversibly seal the complete artifact set."""

        self._ensure_active()
        names = {entry.name for entry in self.run_dir.iterdir()}
        expected = _REQUIRED_ARTIFACTS | {"RUNNING"}
        missing = sorted(expected - names)
        unexpected = sorted(names - expected)
        if missing or unexpected:
            raise RunIntegrityError(f"cannot complete run; missing={missing}, unexpected={unexpected}")
        _, raw = _parse_scientific_artifacts(self.run_dir)
        checksums = {name: sha256(raw[name]).hexdigest() for name in sorted(_REQUIRED_ARTIFACTS)}
        checksums_bytes = (_canonical_json(checksums) + "\n").encode("utf-8")
        self._create_exclusive("checksums.json", checksums_bytes)
        _fsync_directory(self.run_dir)
        anchor = sha256(checksums_bytes).hexdigest().encode("ascii") + b"\n"
        complete = self._create_exclusive("COMPLETE", anchor)
        try:
            (self.run_dir / "RUNNING").unlink()
        except OSError as exc:
            raise RunIntegrityError("could not remove RUNNING after preparing COMPLETE") from exc
        _fsync_directory(self.run_dir)
        return complete


def verify_run_dir(path: str | Path) -> dict[str, Any]:
    """Read and verify a sealed run directory without modifying it."""

    run_dir = Path(path)
    try:
        metadata = run_dir.lstat()
    except OSError as exc:
        raise RunIntegrityError(f"run directory does not exist: {run_dir}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RunIntegrityError("run directory must be a real directory, not a symlink")
    try:
        names = {entry.name for entry in run_dir.iterdir()}
    except OSError as exc:
        raise RunIntegrityError("could not list run directory") from exc
    if "RUNNING" in names:
        raise RunIntegrityError("run is still marked RUNNING")
    missing = sorted(_FINAL_ARTIFACTS - names)
    unexpected = sorted(names - _FINAL_ARTIFACTS)
    if missing or unexpected:
        raise RunIntegrityError(f"invalid sealed artifact set; missing={missing}, unexpected={unexpected}")

    raw = {name: _read_regular(run_dir / name) for name in sorted(_FINAL_ARTIFACTS)}
    checksums_raw = raw["checksums.json"]
    complete_raw = raw["COMPLETE"]
    expected_anchor = sha256(checksums_raw).hexdigest().encode("ascii") + b"\n"
    if complete_raw != expected_anchor:
        raise RunIntegrityError("COMPLETE does not anchor checksums.json")
    checksums = _parse_canonical_json(checksums_raw, "checksums.json")
    if not isinstance(checksums, Mapping) or set(checksums) != _REQUIRED_ARTIFACTS:
        raise RunIntegrityError("checksums.json has the wrong artifact key set")
    for name, digest in checksums.items():
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise RunIntegrityError(f"checksums.json has an invalid digest for {name}")
        if sha256(raw[name]).hexdigest() != digest:
            raise RunIntegrityError(f"artifact checksum mismatch: {name}")
    summary, _ = _parse_scientific_artifacts(run_dir, raw)
    return summary
