"""Integrity and determinism tests for sealed Phase-0 run artifacts."""

from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
from functools import lru_cache
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

from coding_opsd.config import config_hash
from coding_opsd.runtime import RunIntegrityError, RunWriter, named_rng, validate_goav_events, verify_run_dir


REQUIRED_PRECOMPLETION = {
    "resolved_config.json",
    "manifest.json",
    "metrics.json",
    "gates.json",
    "events.jsonl",
    "rows.jsonl",
    "arrays.npz",
    "report.md",
}


def _identity(direction: str, alias: str, seed: int = 101) -> tuple[dict[str, object], dict[str, object], str]:
    config: dict[str, object] = {
        "schema_version": "0.2" if alias == "goav" else "0.1",
        "direction": direction,
        "runtime": {"profile": "smoke"},
        "output": {"identity": f"{alias}_phase0_smoke"},
        "seeds": [seed, 202],
    }
    if alias == "jag":
        config["estimator"] = {
            "primary_baseline": "zero",
            "max_branching": 1,
            "branchable_depth_count": 1,
        }
        config["phase0"] = {
            "horizon": 1,
            "actions": 1,
            "reward_families": ["root_only"],
            "arms": ["flat_iid"],
            "budgets": [1],
            "rollout_replications": 1,
            "covariance_audit_replications": 2,
            "gate": {},
            "oracle_scope": {
                "max_horizon": 1,
                "max_budget": 1,
                "max_frontier_nodes": 2,
                "max_states": 100,
            },
        }
    elif alias == "goav":
        config["phase0"] = {
            "tasks_min": 1,
            "group_size": 2,
            "tests_per_task_min": 2,
            "subset_draws_per_group_design": 1,
            "primary_budget_fraction": 0.65,
            "epsilon_G": 1.0,
            "arms": ["goav_exact_subset_aipw"],
        }
        config["acquisition"] = {
            "primary_inclusion_floor": 0.1,
            "solver": {"steps": 2, "learning_rate": 0.1, "initialization": ["uniform"]},
        }
    else:
        config["belief"] = {"particle_sweep": [16]}
        config["phase0"] = {
            "episodes": {"test": 1},
            "prefixes": [4],
            "forecast_horizons": ["all_remaining"],
            "arms": ["exact_bayes", "pbpf", "map"],
        }
    digest = config_hash(config)
    run_id = f"{alias}_phase0_smoke-seed{seed}-{digest[:8]}"
    manifest: dict[str, object] = {
        "schema_version": "phase0.run.v1",
        "run_id": run_id,
        "direction": direction,
        "direction_alias": alias,
        "output_identity": f"{alias}_phase0_smoke",
        "seed": seed,
        "profile": "smoke",
        "status": "INCOMPLETE",
        "config_sha256": digest,
        "source_configs": [{"path": "/registered/config.yaml", "sha256": "a" * 64}],
        "git": {"commit": "b" * 40, "dirty": False},
        "environment": {
            "kind": "cpu_phase0",
            "python": "3.12.0",
            "numpy": "2.0.0",
            "scipy": "1.13.0",
            "pyyaml": "6.0.0",
            "model_backend": None,
            "model_checkpoint": None,
            "container_digest": "not_applicable",
        },
        "row_count": 1,
        "event_count": 0,
        "array_keys": (
            [
                "jag_covariance_f000_root_only_predicted",
                "jag_covariance_f000_root_only_realized",
                "jag_exact_gradient_f000_root_only",
            ]
            if alias == "jag"
            else ["values"]
        ),
    }
    return config, manifest, run_id


def _replace_output_identity(
    config: dict[str, object], manifest: dict[str, object], output_identity: str
) -> tuple[dict[str, object], dict[str, object], str]:
    config = json.loads(json.dumps(config))
    manifest = json.loads(json.dumps(manifest))
    config["output"]["identity"] = output_identity
    digest = config_hash(config)
    seed = manifest["seed"]
    run_id = f"{output_identity}-seed{seed}-{digest[:8]}"
    manifest["output_identity"] = output_identity
    manifest["config_sha256"] = digest
    manifest["run_id"] = run_id
    return config, manifest, run_id


def _scientific_identity(direction: str, seed: int = 101) -> dict[str, object]:
    return {
        "direction": direction,
        "profile": "smoke",
        "seed": seed,
        "status": "INCOMPLETE",
    }


def _metrics(
    direction: str,
    *,
    row_count: int = 1,
    event_count: int = 0,
    array_keys: list[str] | None = None,
    status: str = "INCOMPLETE",
) -> dict[str, object]:
    default_array_keys = (
        sorted(_jag_arrays()) if direction == "jag_tree" else ["values"]
    )
    result: dict[str, object] = {
        "schema_version": "phase0.metrics.v1",
        **_scientific_identity(direction),
        "status": status,
        "row_count": row_count,
        "event_count": event_count,
        "array_keys": default_array_keys if array_keys is None else array_keys,
    }
    if direction == "jag_tree":
        result.update(_jag_metrics_specific())
    return result


def _gates(direction: str, *, status: str = "INCOMPLETE") -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": "phase0.gates.v1",
        **_scientific_identity(direction),
        "status": status,
        "formal_evidence": False,
        "source_status": status,
        "reason": "smoke_profile_below_formal_registration",
        "inputs": {} if direction == "gradient_optimal_active_verification" else {"registered": True},
    }
    if direction == "jag_tree":
        native_gate = _jag_native()["gate"]
        result["inputs"] = json.loads(json.dumps(native_gate["inputs"]))
        result["inputs"]["held_out_covariance_audit"] = [
            {"family": "root_only", **_jag_covariance(raw_matrices=True)}
        ]
        result["evidence"] = json.loads(json.dumps(native_gate["evidence"]))
    return result


@lru_cache(maxsize=1)
def _jag_native() -> dict[str, object]:
    from coding_opsd.jag import run_jag_phase0

    return run_jag_phase0(
        {
            "runtime": {"profile": "smoke"},
            "seeds": [101, 202],
            "estimator": {
                "primary_baseline": "zero",
                "max_branching": 1,
                "branchable_depth_count": 1,
            },
            "phase0": {
                "horizon": 1,
                "actions": 1,
                "reward_families": ["root_only"],
                "arms": ["flat_iid"],
                "budgets": [1],
                "rollout_replications": 1,
                "covariance_audit_replications": 2,
                "gate": {},
                "oracle_scope": {
                    "max_horizon": 1,
                    "max_budget": 1,
                    "max_frontier_nodes": 2,
                    "max_states": 100,
                },
            },
        },
        101,
    )


def _jag_exact(seed: int = 101) -> tuple[float, float, np.ndarray]:
    from coding_opsd.jag.mdp import exact_target, make_phase0_mdp

    target = exact_target(make_phase0_mdp(1, 1, "root_only", seed))
    return (
        float(target.probability_sum),
        float(target.expected_reward),
        np.asarray(target.gradient, dtype="<f8"),
    )


def _jag_covariance(*, raw_matrices: bool = False) -> dict[str, object]:
    payload = json.loads(
        json.dumps(_jag_native()["families"]["root_only"]["covariance_audit"])
    )
    if raw_matrices:
        return payload
    del payload["predicted_covariance"]
    del payload["realized_covariance"]
    payload["predicted_covariance_array_key"] = "jag_covariance_f000_root_only_predicted"
    payload["realized_covariance_array_key"] = "jag_covariance_f000_root_only_realized"
    return payload


def _jag_row_native(arm: str = "flat_iid") -> dict[str, object]:
    row = json.loads(
        json.dumps(_jag_native()["families"]["root_only"]["summaries"][0])
    )
    if arm != "flat_iid":
        row["arm"] = arm
    return {"family": "root_only", **row}


def _jag_arrays() -> dict[str, np.ndarray]:
    family = _jag_native()["families"]["root_only"]
    covariance = family["covariance_audit"]
    return {
        "jag_exact_gradient_f000_root_only": np.asarray(family["exact"]["gradient"], dtype="<f8"),
        "jag_covariance_f000_root_only_predicted": np.asarray(
            covariance["predicted_covariance"], dtype="<f8"
        ),
        "jag_covariance_f000_root_only_realized": np.asarray(
            covariance["realized_covariance"], dtype="<f8"
        ),
    }


def _jag_metrics_specific() -> dict[str, object]:
    family = _jag_native()["families"]["root_only"]
    row = _jag_row_native()
    return {
        "family_order": ["root_only"],
        "family_metadata": {
            "root_only": {
                "probability_sum": family["exact"]["probability_sum"],
                "expected_reward": family["exact"]["expected_reward"],
                "structural_audit": json.loads(json.dumps(family["structural_audit"])),
                "covariance_audit": _jag_covariance(),
                "predictor_provenance": family["predictor_provenance"],
                "summary_count": 1,
                "exact_gradient_array_key": "jag_exact_gradient_f000_root_only",
            }
        },
        "allocation_evidence": [
            {
                "family": "root_only",
                "arm": "flat_iid",
                "budget": 1,
                "allocation": row["allocation"],
                "allocation_evidence": row["allocation_evidence"],
                "arm_status": "OK",
            }
        ],
        "oracle_scope": json.loads(json.dumps(_jag_native()["oracle_scope"])),
        "estimator_policy": json.loads(json.dumps(_jag_native()["estimator_policy"])),
    }


@lru_cache(maxsize=1)
def _goav_normalized():
    """Return one real, deterministic GOAV runner/adapter fixture."""

    from coding_opsd.goav.experiment import run_goav_phase0
    from coding_opsd.results import normalize_result

    config, _, run_id = _identity("gradient_optimal_active_verification", "goav")
    return normalize_result(
        "goav",
        "gradient_optimal_active_verification",
        config,
        101,
        run_id,
        run_goav_phase0(config, 101),
    )


def _canonical_report(
    manifest: dict[str, object], gates: dict[str, object], *, array_count: int | None = None
) -> str:
    if array_count is None:
        array_count = len(manifest["array_keys"])
    return (
        "# Phase-0 run report\n\n"
        f"- Run ID: {manifest['run_id']}\n"
        f"- Direction: {manifest['direction']} ({manifest['direction_alias']})\n"
        f"- Seed: {manifest['seed']}\n"
        f"- Profile: {manifest['profile']}\n"
        f"- Scientific status: {manifest['status']}\n"
        f"- Status reason: {gates['reason']}\n"
        f"- Rows/events/arrays: {manifest['row_count']} / {manifest['event_count']} / {array_count}\n\n"
        "This is a deterministic finite CPU Phase-0 mechanism experiment. "
        "It is not a 7B-model result or a coding-benchmark score. COMPLETE "
        "means that the artifacts are sealed; gates.json is authoritative "
        "for the scientific status.\n"
    )


def _create_run(
    root: Path,
    *,
    direction: str = "jag_tree",
    alias: str = "jag",
    events: list[dict[str, object]] | None = None,
    arrays: dict[str, np.ndarray] | None = None,
) -> Path:
    config, manifest, run_id = _identity(direction, alias)
    event_rows = [] if events is None else events
    event_rows = _envelope_events(event_rows, run_id, direction)
    run = root / run_id
    gates = _gates(direction)
    if alias == "jag":
        array_values = _jag_arrays() if arrays is None else arrays
        rows = _envelope_rows([_jag_row_native()], run_id, direction)
        metrics = _metrics(
            direction,
            row_count=len(rows),
            event_count=len(event_rows),
            array_keys=sorted(array_values),
        )
    elif alias == "goav":
        # Full sealed-contract tests must start from a result the registered
        # runner can actually produce.  Hand-built events remain below for the
        # focused validate_goav_events unit tests only.
        normalized = _goav_normalized()
        metrics = json.loads(json.dumps(normalized.metrics))
        gates = json.loads(json.dumps(normalized.gates))
        rows = [json.loads(json.dumps(row)) for row in normalized.rows]
        event_rows = [json.loads(json.dumps(event)) for event in normalized.events]
        array_values = {key: value.copy() for key, value in normalized.arrays.items()}
    else:
        raise AssertionError("PBPF fixture is constructed by dedicated tests")
    manifest["event_count"] = len(event_rows)
    manifest["row_count"] = len(rows)
    manifest["array_keys"] = sorted(array_values)
    with RunWriter(run, config, manifest) as writer:
        writer.write_metrics(metrics)
        writer.write_gates(gates)
        writer.write_events(event_rows)
        writer.write_rows(rows)
        writer.write_npz(array_values)
        writer.write_report(_canonical_report(manifest, gates))
        writer.complete()
    return run


def _create_native_jag_run(root: Path, config: dict[str, object]) -> Path:
    """Seal the real runner/adapter output for a small JAG registration."""

    from coding_opsd.jag import run_jag_phase0
    from coding_opsd.results import normalize_result

    seed = int(config["seeds"][0])
    digest = config_hash(config)
    output_identity = str(config["output"]["identity"])
    run_id = f"{output_identity}-seed{seed}-{digest[:8]}"
    normalized = normalize_result(
        "jag",
        "jag_tree",
        config,
        seed,
        run_id,
        run_jag_phase0(config, seed),
    )
    _, template, _ = _identity("jag_tree", "jag", seed=seed)
    manifest = {
        **template,
        "run_id": run_id,
        "output_identity": output_identity,
        "config_sha256": digest,
        "status": normalized.metrics["status"],
        "row_count": len(normalized.rows),
        "event_count": len(normalized.events),
        "array_keys": sorted(normalized.arrays),
    }
    run = root / run_id
    with RunWriter(run, config, manifest) as writer:
        writer.write_metrics(normalized.metrics)
        writer.write_gates(normalized.gates)
        writer.write_events(normalized.events)
        writer.write_rows(normalized.rows)
        writer.write_npz(normalized.arrays)
        writer.write_report(_canonical_report(manifest, normalized.gates))
        writer.complete()
    return run


def _envelope_goav_events(
    events: list[dict[str, object]], run_id: str, seed: int = 101
) -> list[dict[str, object]]:
    return _envelope_events(events, run_id, "gradient_optimal_active_verification", seed)


def _envelope_events(
    events: list[dict[str, object]], run_id: str, direction: str, seed: int = 101
) -> list[dict[str, object]]:
    return [
        {
            "schema_version": "phase0.event.v1",
            "run_id": run_id,
            "direction": direction,
            "seed": seed,
            "sequence": sequence,
            **event,
        }
        for sequence, event in enumerate(events)
    ]


def _envelope_rows(
    rows: list[dict[str, object]], run_id: str, direction: str, seed: int = 101
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for index, input_row in enumerate(rows):
        row = input_row
        if direction == "jag_tree" and "family" not in row:
            row = _jag_row_native()
        normalized.append(
            {
            "schema_version": "phase0.row.v1",
            "run_id": run_id,
            "direction": direction,
            "seed": seed,
            "row_index": index,
            "record_type": (
                "arm_budget_summary"
                if direction == "jag_tree"
                else "forecast_diagnostic"
                if direction == "predictive_belief_particle_filter"
                else "arm_summary"
            ),
            **row,
        }
        )
    return normalized


def _reseal(run: Path) -> None:
    """Test-only helper: reseal a deliberately tampered fixture."""

    checksums = {
        name: sha256((run / name).read_bytes()).hexdigest()
        for name in sorted(REQUIRED_PRECOMPLETION)
    }
    encoded = json.dumps(checksums, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    (run / "checksums.json").write_text(encoded, encoding="utf-8", newline="")
    anchor = sha256((run / "checksums.json").read_bytes()).hexdigest() + "\n"
    (run / "COMPLETE").write_text(anchor, encoding="utf-8", newline="")


def _goav_design(arm: str, probabilities: list[float]) -> dict[str, object]:
    probability_array = np.asarray(probabilities, dtype="<f8")
    candidates = len(probabilities).bit_length() - 1
    masks = (
        (np.arange(len(probabilities), dtype=np.uint64)[:, None] >> np.arange(candidates, dtype=np.uint64)[None, :])
        & 1
    ).astype(bool)
    pi = masks.T @ probability_array
    pi2 = masks.T @ (probability_array[:, None] * masks)
    digest = sha256(probability_array.tobytes(order="C")).hexdigest()
    return {
        "event": "design_logged",
        "task": 0,
        "arm": arm,
        "design_hash": digest,
        "probabilities": probability_array.tolist(),
        "pi": pi.tolist(),
        "pi2": pi2.tolist(),
        "pi_hash": sha256(np.asarray(pi, dtype="<f8").tobytes(order="C")).hexdigest(),
        "pi2_hash": sha256(np.asarray(pi2, dtype="<f8").tobytes(order="C")).hexdigest(),
    }


def _goav_audit(design: dict[str, object], *, seed: int = 101, draw: int = 0) -> dict[str, object]:
    probabilities = np.asarray(design["probabilities"], dtype=np.float64)
    pi = list(design["pi"])
    arm = str(design["arm"])
    task = int(design["task"])
    subset_index = int(named_rng(seed, f"goav_audit:{task}:{arm}:{draw}").choice(len(probabilities), p=probabilities))
    selected_indices = [index for index in range(len(pi)) if (subset_index >> index) & 1]
    labels = [1.0, 0.0]
    return {
        "event": "audit_request",
        "task": task,
        "arm": arm,
        "draw": draw,
        "subset_index": subset_index,
        "selected_indices": selected_indices,
        "design_hash": design["design_hash"],
        "inclusion_probabilities": pi,
        "selected_labels": {str(index): labels[index] for index in selected_indices},
    }


def _goav_events(seed: int = 101) -> list[dict[str, object]]:
    design = _goav_design("goav_exact_subset_aipw", [0.1, 0.2, 0.3, 0.4])
    return [design, _goav_audit(design, seed=seed)]


class RunArtifactTests(unittest.TestCase):
    def test_jag_accepts_the_runner_default_when_estimator_mapping_is_absent(self) -> None:
        config, _, _ = _identity("jag_tree", "jag")
        del config["estimator"]
        config["output"]["identity"] = "jag_without_estimator"
        with tempfile.TemporaryDirectory() as directory:
            run = _create_native_jag_run(Path(directory), config)
            self.assertEqual(verify_run_dir(run)["status"], "INCOMPLETE")

    def test_jag_accepts_single_environment_lagged_calibration_fallback(self) -> None:
        config, _, _ = _identity("jag_tree", "jag")
        config["output"]["identity"] = "jag_lagged_single_environment"
        config["estimator"]["primary_baseline"] = "lagged_cross_fitted_value"
        config["phase0"]["arms"] = ["jag_learned"]
        config["phase0"]["learned_calibration_seeds"] = [100]
        with tempfile.TemporaryDirectory() as directory:
            run = _create_native_jag_run(Path(directory), config)
            self.assertEqual(verify_run_dir(run)["status"], "INCOMPLETE")

    def test_jag_rejects_self_consistent_but_unreplayed_covariance_after_reseal(self) -> None:
        from coding_opsd.runtime import _deterministic_npz_bytes

        with tempfile.TemporaryDirectory() as directory:
            run = _create_run(Path(directory))
            with np.load(run / "arrays.npz", allow_pickle=False) as loaded:
                arrays = {key: loaded[key].copy() for key in loaded.files}
            predicted_key = "jag_covariance_f000_root_only_predicted"
            realized_key = "jag_covariance_f000_root_only_realized"
            dimension = arrays[predicted_key].shape[0]
            predicted = np.eye(dimension, dtype="<f8")
            realized = np.eye(dimension, dtype="<f8") * 2.0
            arrays[predicted_key] = predicted
            arrays[realized_key] = realized
            (run / "arrays.npz").write_bytes(_deterministic_npz_bytes(arrays))

            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            covariance = metrics["family_metadata"]["root_only"]["covariance_audit"]
            covariance.update(
                {
                    "status": "OK",
                    "realized_joint_mean": [0.0] * dimension,
                    "numerator": float(dimension),
                    "denominator": float(dimension),
                    "relative_frobenius_error": 1.0,
                    "predicted_covariance_hash": sha256(predicted.tobytes(order="C")).hexdigest(),
                    "realized_covariance_hash": sha256(realized.tobytes(order="C")).hexdigest(),
                }
            )
            gates = json.loads((run / "gates.json").read_text(encoding="utf-8"))
            gate_covariance = dict(covariance)
            del gate_covariance["predicted_covariance_array_key"]
            del gate_covariance["realized_covariance_array_key"]
            gate_covariance.update(
                {
                    "family": "root_only",
                    "predicted_covariance": predicted.tolist(),
                    "realized_covariance": realized.tolist(),
                }
            )
            gates["inputs"]["held_out_covariance_audit"] = [gate_covariance]
            for filename, document in (("metrics.json", metrics), ("gates.json", gates)):
                (run / filename).write_text(
                    json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                    newline="",
                )
            _reseal(run)
            with self.assertRaisesRegex(RunIntegrityError, "covariance"):
                verify_run_dir(run)

    def test_jag_rejects_false_frozen_evidence_flags_after_reseal(self) -> None:
        for field in ("arm_metadata", "allocation_evidence"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                run = _create_run(Path(directory))
                row = json.loads((run / "rows.jsonl").read_text(encoding="utf-8"))
                metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
                if field == "arm_metadata":
                    row[field]["baseline_frozen_before_evaluation"] = False
                else:
                    row[field]["branch_counts_frozen_before_child_draws"] = False
                    metrics["allocation_evidence"][0][field] = row[field]
                (run / "rows.jsonl").write_text(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                    newline="",
                )
                (run / "metrics.json").write_text(
                    json.dumps(metrics, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                    newline="",
                )
                _reseal(run)
                with self.assertRaises(RunIntegrityError):
                    verify_run_dir(run)

    def test_resolved_config_requires_nonempty_direction_phase0_registration(self) -> None:
        config, manifest, run_id = _identity("jag_tree", "jag")
        del config["phase0"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RunIntegrityError, "phase0"):
                RunWriter(Path(directory) / run_id, config, manifest)

    def test_report_is_reconstructed_from_sealed_identity_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = _create_run(Path(directory))
            (run / "report.md").write_text("# plausible but unregistered report\n", encoding="utf-8")
            _reseal(run)
            with self.assertRaisesRegex(RunIntegrityError, "report"):
                verify_run_dir(run)

    def test_named_rng_rejects_boolean_and_float_seeds(self) -> None:
        for seed in (True, False, 1.0):
            with self.subTest(seed=seed), self.assertRaises(TypeError):
                named_rng(seed, "stream")

    def test_constructor_validates_identity_config_manifest_and_path_before_mkdir(self) -> None:
        invalid_cases: list[tuple[str, object]] = []
        config, manifest, run_id = _identity("jag_tree", "jag")
        float_seed = json.loads(json.dumps(manifest))
        float_seed["seed"] = 101.0
        invalid_cases.append((run_id, (config, float_seed)))

        boolean_seed = json.loads(json.dumps(manifest))
        boolean_seed["seed"] = True
        invalid_cases.append((run_id, (config, boolean_seed)))

        unsafe_config = json.loads(json.dumps(config))
        unsafe_manifest = json.loads(json.dumps(manifest))
        unsafe_config["output"]["identity"] = "../escape"
        unsafe_manifest["output_identity"] = "../escape"
        invalid_cases.append((run_id, (unsafe_config, unsafe_manifest)))

        missing_provenance = json.loads(json.dumps(manifest))
        del missing_provenance["environment"]
        invalid_cases.append((run_id, (config, missing_provenance)))

        extra_manifest = json.loads(json.dumps(manifest))
        extra_manifest["unexpected"] = True
        invalid_cases.append((run_id, (config, extra_manifest)))

        duplicate_source = json.loads(json.dumps(manifest))
        duplicate_source["source_configs"].append(
            {"path": "/registered/config.yaml", "sha256": "c" * 64}
        )
        invalid_cases.append((run_id, (config, duplicate_source)))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (candidate_name, payload) in enumerate(invalid_cases):
                with self.subTest(index=index):
                    candidate_parent = root / f"not-created-{index}"
                    candidate_config, candidate_manifest = payload
                    with self.assertRaises((RunIntegrityError, TypeError, ValueError)):
                        RunWriter(candidate_parent / candidate_name, candidate_config, candidate_manifest)
                    self.assertFalse(candidate_parent.exists())

            exact_config, exact_manifest, _ = _identity("jag_tree", "jag")
            exact_config, exact_manifest, exact_id = _replace_output_identity(
                exact_config, exact_manifest, "a" * 175
            )
            self.assertEqual(len(exact_id), 192)
            exact_writer = RunWriter(root / exact_id, exact_config, exact_manifest)
            self.assertTrue((exact_writer.run_dir / "RUNNING").is_file())

            long_config, long_manifest, _ = _identity("jag_tree", "jag")
            long_config, long_manifest, long_id = _replace_output_identity(
                long_config, long_manifest, "a" * 176
            )
            self.assertEqual(len(long_id), 193)
            long_parent = root / "long-not-created"
            with self.assertRaises(RunIntegrityError):
                RunWriter(long_parent / long_id, long_config, long_manifest)
            self.assertFalse(long_parent.exists())

            traversal_parent = root / "traversal-not-created"
            with self.assertRaises(RunIntegrityError):
                RunWriter(
                    traversal_parent / "nested" / ".." / run_id,
                    config,
                    manifest,
                )
            self.assertFalse(traversal_parent.exists())

            if hasattr(os, "symlink"):
                real_parent = root / "real-parent"
                real_parent.mkdir()
                linked_parent = root / "linked-parent"
                linked_parent.symlink_to(real_parent, target_is_directory=True)
                with self.assertRaises(RunIntegrityError):
                    RunWriter(linked_parent / run_id, config, manifest)
                self.assertFalse((real_parent / run_id).exists())

    def test_complete_seals_exact_contract_and_verify_returns_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = _create_run(Path(directory))
            self.assertEqual(
                {item.name for item in run.iterdir()},
                REQUIRED_PRECOMPLETION | {"checksums.json", "COMPLETE"},
            )
            self.assertFalse((run / "RUNNING").exists())
            summary = verify_run_dir(run)
            self.assertEqual(summary["run_id"], run.name)
            self.assertEqual(summary["direction"], "jag_tree")
            self.assertEqual(summary["status"], "INCOMPLETE")
            self.assertEqual(summary["row_count"], 1)
            self.assertEqual(summary["event_count"], 0)
            self.assertEqual(summary["array_keys"], sorted(_jag_arrays()))

    def test_every_row_and_event_envelope_and_metrics_counts_are_cross_checked(self) -> None:
        mutations = (
            "metric_count",
            "metric_float_seed",
            "gate_float_seed",
            "row_index",
            "row_run_id",
            "row_float_seed",
            "event_sequence",
            "event_float_seed",
            "gate_matrix",
            "jag_evidence",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                if mutation in {"event_sequence", "event_float_seed"}:
                    run = _create_run(
                        Path(directory),
                        direction="gradient_optimal_active_verification",
                        alias="goav",
                        events=_goav_events(),
                    )
                else:
                    run = _create_run(Path(directory))
                if mutation in {"metric_count", "metric_float_seed"}:
                    document = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
                    if mutation == "metric_count":
                        document["event_count"] = 9
                    else:
                        document["seed"] = 101.0
                    filename = "metrics.json"
                elif mutation in {"row_index", "row_run_id", "row_float_seed"}:
                    document = json.loads((run / "rows.jsonl").read_text(encoding="utf-8"))
                    if mutation == "row_index":
                        document["row_index"] = 9
                    elif mutation == "row_run_id":
                        document["run_id"] = "wrong"
                    else:
                        document["seed"] = 101.0
                    filename = "rows.jsonl"
                elif mutation in {"event_sequence", "event_float_seed"}:
                    documents = [
                        json.loads(line)
                        for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
                    ]
                    documents[0]["sequence" if mutation == "event_sequence" else "seed"] = (
                        4 if mutation == "event_sequence" else 101.0
                    )
                    document = documents
                    filename = "events.jsonl"
                else:
                    document = json.loads((run / "gates.json").read_text(encoding="utf-8"))
                    if mutation == "gate_matrix":
                        document["formal_evidence"] = True
                    elif mutation == "gate_float_seed":
                        document["seed"] = 101.0
                    else:
                        del document["evidence"]
                    filename = "gates.json"
                encoded = (
                    "".join(
                        json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                        for item in document
                    )
                    if isinstance(document, list)
                    else json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
                )
                (run / filename).write_text(
                    encoded,
                    encoding="utf-8",
                    newline="",
                )
                _reseal(run)
                with self.assertRaises(RunIntegrityError):
                    verify_run_dir(run)

    def test_complete_failure_while_creating_anchor_never_removes_running(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, manifest, run_id = _identity("jag_tree", "jag")
            writer = RunWriter(Path(directory) / run_id, config, manifest)
            gates = _gates("jag_tree")
            writer.write_metrics(_metrics("jag_tree"))
            writer.write_gates(gates)
            writer.write_events([])
            writer.write_rows(_envelope_rows([_jag_row_native()], run_id, "jag_tree"))
            writer.write_npz(_jag_arrays())
            writer.write_report(_canonical_report(manifest, gates))
            original = writer._create_exclusive

            def injected_failure(name: str, contents: bytes) -> Path:
                if name == "COMPLETE":
                    raise OSError("injected COMPLETE failure")
                return original(name, contents)

            with mock.patch.object(writer, "_create_exclusive", side_effect=injected_failure):
                with self.assertRaises(OSError):
                    writer.complete()
            self.assertTrue((writer.run_dir / "RUNNING").is_file())
            self.assertFalse((writer.run_dir / "COMPLETE").exists())

    def test_verify_uses_one_retained_nofollow_snapshot_per_file(self) -> None:
        import coding_opsd.runtime as runtime_module

        with tempfile.TemporaryDirectory() as directory:
            run = _create_run(Path(directory))
            reads: dict[str, int] = {}
            original = runtime_module._read_regular

            def counted_read(path: Path) -> bytes:
                reads[path.name] = reads.get(path.name, 0) + 1
                return original(path)

            with mock.patch.object(runtime_module, "_read_regular", side_effect=counted_read):
                self.assertEqual(verify_run_dir(run)["run_id"], run.name)
            self.assertEqual(reads, {name: 1 for name in REQUIRED_PRECOMPLETION | {"checksums.json", "COMPLETE"}})

    def test_completion_rejects_identity_only_science_and_non_incomplete_smoke(self) -> None:
        variants = ("identity_only", "smoke_pass")
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                config, manifest, run_id = _identity("jag_tree", "jag")
                metrics = _metrics("jag_tree")
                gates = _gates("jag_tree")
                if variant == "smoke_pass":
                    manifest["status"] = "PASS"
                    metrics["status"] = "PASS"
                    gates["status"] = "PASS"
                    gates["source_status"] = "PASS"
                writer = RunWriter(Path(directory) / run_id, config, manifest)
                if variant == "identity_only":
                    metrics = {key: metrics[key] for key in _scientific_identity("jag_tree")}
                    gates = {key: gates[key] for key in _scientific_identity("jag_tree")}
                writer.write_metrics(metrics)
                writer.write_gates(gates)
                writer.write_events([])
                writer.write_rows(_envelope_rows([{"arm": "reference"}], run_id, "jag_tree"))
                writer.write_npz({"values": np.array([1.0])})
                writer.write_report("# report\n")
                with self.assertRaises(RunIntegrityError):
                    writer.complete()

    def test_formal_status_requires_exact_formal_evidence_matrix(self) -> None:
        cases = (
            ("PASS", True, False),
            ("FAIL", True, False),
            ("INCOMPLETE", False, True),
            ("INVALID", False, False),
            ("PASS", False, False),
            ("INCOMPLETE", True, False),
        )
        for index, (status, formal_evidence, accepted) in enumerate(cases):
            with (
                self.subTest(status=status, formal_evidence=formal_evidence),
                tempfile.TemporaryDirectory() as directory,
            ):
                config, manifest, _ = _identity("jag_tree", "jag")
                config["runtime"]["profile"] = "formal"
                manifest["profile"] = "formal"
                manifest["status"] = status
                config, manifest, run_id = _replace_output_identity(
                    config, manifest, f"jag_formal_{index}"
                )
                metrics = _metrics("jag_tree", status=status)
                metrics["profile"] = "formal"
                gates = _gates("jag_tree", status=status)
                gates["profile"] = "formal"
                gates["formal_evidence"] = formal_evidence
                if accepted:
                    gates["reason"] = "missing registered evidence: " + ", ".join(
                        (
                            "registered_families_present",
                            "registered_arms_present",
                            "registered_budgets_present",
                            "registered_replications_present",
                            "all_registered_cells_have_samples",
                            "fifty_environment_seed_results_present",
                            "diagnostic_structures_pass",
                            "registered_covariance_coverage_gate_implemented",
                        )
                    )
                with mock.patch("coding_opsd.cli._validate_formal_projection"):
                    writer = RunWriter(Path(directory) / run_id, config, manifest)
                    writer.write_metrics(metrics)
                    writer.write_gates(gates)
                    writer.write_events([])
                    writer.write_rows(
                        _envelope_rows([_jag_row_native()], run_id, "jag_tree")
                    )
                    writer.write_npz(_jag_arrays())
                    writer.write_report(_canonical_report(manifest, gates))
                    if accepted:
                        writer.complete()
                        self.assertEqual(verify_run_dir(writer.run_dir)["status"], status)
                    else:
                        with self.assertRaises(RunIntegrityError):
                            writer.complete()
                        self.assertTrue((writer.run_dir / "RUNNING").is_file())

    def test_gate_source_status_cannot_invert_or_fabricate_final_status(self) -> None:
        cases = (
            ("formal", "PASS", True, "FAIL"),
            ("formal", "FAIL", True, "PASS"),
            ("formal", "PASS", True, "INCOMPLETE"),
            ("formal", "INCOMPLETE", False, "INVALID"),
            ("smoke", "INCOMPLETE", False, "INVALID"),
            ("smoke", "INVALID", False, "PASS"),
        )
        for index, (profile, final_status, evidence, source_status) in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                config, manifest, _ = _identity("jag_tree", "jag")
                config["runtime"]["profile"] = profile
                manifest["profile"] = profile
                manifest["status"] = final_status
                config, manifest, run_id = _replace_output_identity(
                    config, manifest, f"jag_status_inversion_{index}"
                )
                metrics = _metrics("jag_tree", status=final_status)
                metrics["profile"] = profile
                gates = _gates("jag_tree", status=final_status)
                gates["profile"] = profile
                gates["formal_evidence"] = evidence
                gates["source_status"] = source_status
                context = (
                    mock.patch("coding_opsd.cli._validate_formal_projection")
                    if profile == "formal"
                    else nullcontext()
                )
                with context:
                    writer = RunWriter(Path(directory) / run_id, config, manifest)
                    writer.write_metrics(metrics)
                    writer.write_gates(gates)
                    writer.write_events([])
                    writer.write_rows(
                        _envelope_rows([_jag_row_native()], run_id, "jag_tree")
                    )
                    writer.write_npz(_jag_arrays())
                    writer.write_report(_canonical_report(manifest, gates))
                    with self.assertRaisesRegex(RunIntegrityError, "source_status"):
                        writer.complete()

    def test_completion_rejects_each_missing_or_empty_scientific_artifact(self) -> None:
        artifact_names = {
            "metrics.json",
            "gates.json",
            "events.jsonl",
            "rows.jsonl",
            "arrays.npz",
            "report.md",
        }
        for omitted in artifact_names:
            with self.subTest(omitted=omitted), tempfile.TemporaryDirectory() as directory:
                config, manifest, run_id = _identity("jag_tree", "jag")
                run = Path(directory) / run_id
                writer = RunWriter(run, config, manifest)
                if omitted != "metrics.json":
                    writer.write_metrics(_metrics("jag_tree"))
                if omitted != "gates.json":
                    writer.write_gates(_gates("jag_tree"))
                if omitted != "events.jsonl":
                    writer.write_events([])
                if omitted != "rows.jsonl":
                    writer.write_rows(_envelope_rows([{"arm": "reference"}], run_id, "jag_tree"))
                if omitted != "arrays.npz":
                    writer.write_npz({"values": np.array([1.0])})
                if omitted != "report.md":
                    writer.write_report("# report\n")
                with self.assertRaises(RunIntegrityError):
                    writer.complete()
                self.assertTrue((run / "RUNNING").is_file())
                self.assertFalse((run / "COMPLETE").exists())

        invalid_payloads = (
            ("rows", lambda writer: writer.write_rows([])),
            ("arrays", lambda writer: writer.write_npz({})),
            ("report", lambda writer: writer.write_report(" \n")),
        )
        for label, invalid_write in invalid_payloads:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                config, manifest, run_id = _identity("jag_tree", "jag")
                run = Path(directory) / run_id
                writer = RunWriter(run, config, manifest)
                writer.write_metrics(_metrics("jag_tree"))
                writer.write_gates(_gates("jag_tree"))
                writer.write_events([])
                if label != "rows":
                    writer.write_rows(_envelope_rows([{"arm": "reference"}], run_id, "jag_tree"))
                if label != "arrays":
                    writer.write_npz({"values": np.array([1.0])})
                if label != "report":
                    writer.write_report("# report\n")
                invalid_write(writer)
                with self.assertRaises(RunIntegrityError):
                    writer.complete()
                self.assertFalse((run / "COMPLETE").exists())

    def test_writes_are_one_shot_path_safe_and_collision_preserves_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, manifest, run_id = _identity("jag_tree", "jag")
            run = Path(directory) / run_id
            writer = RunWriter(run, config, manifest)
            writer.write_events([{"event": "first"}])
            original = (run / "events.jsonl").read_bytes()
            with self.assertRaises(FileExistsError):
                writer.write_events([{"event": "second"}])
            self.assertEqual((run / "events.jsonl").read_bytes(), original)
            with self.assertRaises(ValueError):
                writer.write_text("../escape", "bad")
            with self.assertRaises(ValueError):
                writer.write_text("nested/name", "bad")
            self.assertFalse((Path(directory) / "escape").exists())

            if hasattr(os, "symlink"):
                (run / "COMPLETE").symlink_to(run / "missing-anchor")
                with self.assertRaises(FileExistsError):
                    writer.write_text("late.txt", "must not be written")
                self.assertFalse((run / "late.txt").exists())

    def test_completed_writer_rejects_every_write_api_before_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = _create_run(Path(directory))
            config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            writer = object.__new__(RunWriter)
            writer.run_dir = run
            attempts = (
                lambda: writer.write_text("late.txt", "late"),
                lambda: writer.write_json("late.json", {"bad": np.nan}),
                lambda: writer.write_jsonl("late.jsonl", [{"bad": np.nan}]),
                lambda: writer.write_npz({"bad": np.array([np.nan])}),
                lambda: writer.write_manifest(manifest),
                lambda: writer.write_resolved_config(config),
            )
            for attempt in attempts:
                with self.subTest(attempt=attempt), self.assertRaises(FileExistsError):
                    attempt()

    def test_npz_is_deterministic_safe_and_rejects_bad_arrays(self) -> None:
        arrays = {
            "z_float": np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=">f8")),
            "a_bool": np.array([True, False]),
            "m_int": np.array([1, 2], dtype=">i4"),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest, first_id = _identity("jag_tree", "jag", seed=101)
            first = RunWriter(root / first_id, config, manifest)
            first.write_npz(arrays)
            first_bytes = (first.run_dir / "arrays.npz").read_bytes()

            config2, manifest2, second_id = _identity("jag_tree", "jag", seed=102)
            second = RunWriter(root / second_id, config2, manifest2)
            second.write_npz(dict(reversed(list(arrays.items()))))
            second_bytes = (second.run_dir / "arrays.npz").read_bytes()
            self.assertEqual(first_bytes, second_bytes)
            with zipfile.ZipFile(io.BytesIO(first_bytes)) as archive:
                self.assertEqual(archive.namelist(), ["a_bool.npy", "m_int.npy", "z_float.npy"])
                self.assertTrue(all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()))
                self.assertTrue(all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist()))
            with np.load(io.BytesIO(first_bytes), allow_pickle=False) as loaded:
                self.assertEqual(loaded.files, ["a_bool", "m_int", "z_float"])
                np.testing.assert_array_equal(loaded["z_float"], arrays["z_float"])

            for bad_arrays in (
                {"object": np.array([object()], dtype=object)},
                {"nan": np.array([np.nan])},
                {"inf": np.array([np.inf])},
                {"../unsafe": np.array([1])},
                {"bad/name": np.array([1])},
                {"unicode_λ": np.array([1])},
            ):
                with self.subTest(keys=list(bad_arrays)):
                    config3, manifest3, third_id = _identity("jag_tree", "jag", seed=103 + len(list(root.iterdir())))
                    bad_writer = RunWriter(root / third_id, config3, manifest3)
                    with self.assertRaises((TypeError, ValueError)):
                        bad_writer.write_npz(bad_arrays)
                    self.assertFalse((bad_writer.run_dir / "arrays.npz").exists())

    def test_verify_detects_artifact_checksum_and_anchor_mutations(self) -> None:
        mutations = ("metrics", "checksums", "anchor")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                run = _create_run(Path(directory))
                if mutation == "metrics":
                    (run / "metrics.json").write_text('{"changed":true}\n', encoding="utf-8")
                elif mutation == "checksums":
                    payload = json.loads((run / "checksums.json").read_text(encoding="utf-8"))
                    payload["metrics.json"] = "0" * 64
                    (run / "checksums.json").write_text(
                        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
                    )
                else:
                    (run / "COMPLETE").write_text("0" * 64 + "\n", encoding="utf-8")
                with self.assertRaises(RunIntegrityError):
                    verify_run_dir(run)

    def test_artifact_specific_schema_versions_are_distinct_and_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = _create_run(Path(directory))
            self.assertEqual(verify_run_dir(run)["status"], "INCOMPLETE")
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            metrics["schema_version"] = "phase0.run.v1"
            (run / "metrics.json").write_text(
                json.dumps(metrics, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
                newline="",
            )
            _reseal(run)
            with self.assertRaises(RunIntegrityError):
                verify_run_dir(run)

    def test_verify_parses_every_format_and_rejects_resealed_malformed_data(self) -> None:
        mutations = {
            "json_duplicate": ("metrics.json", '{"seed":101,"seed":101}\n'),
            "json_noncanonical": (
                "metrics.json",
                '{ "direction": "jag_tree", "profile": "smoke", "seed": 101, "status": "INCOMPLETE" }\n',
            ),
            "jsonl_duplicate": ("rows.jsonl", '{"arm":"a","arm":"b"}\n'),
            "jsonl_blank": ("rows.jsonl", '{"arm":"a"}\n\n'),
        }
        for label, (filename, contents) in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                run = _create_run(Path(directory))
                (run / filename).write_text(contents, encoding="utf-8", newline="")
                _reseal(run)
                with self.assertRaises(RunIntegrityError):
                    verify_run_dir(run)

        with tempfile.TemporaryDirectory() as directory:
            run = _create_run(Path(directory))
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
                npy = io.BytesIO()
                np.lib.format.write_array(npy, np.array([object()], dtype=object), allow_pickle=True)
                archive.writestr("values.npy", npy.getvalue())
            (run / "arrays.npz").write_bytes(buffer.getvalue())
            _reseal(run)
            with self.assertRaises(RunIntegrityError):
                verify_run_dir(run)

    def test_verify_rejects_identity_counts_unexpected_files_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = _create_run(Path(directory))
            (run / "unexpected.txt").write_text("surprise", encoding="utf-8")
            with self.assertRaises(RunIntegrityError):
                verify_run_dir(run)

        for field, replacement in (("row_count", 9), ("event_count", 9), ("array_keys", ["other"])):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                run = _create_run(Path(directory))
                manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
                manifest[field] = replacement
                (run / "manifest.json").write_text(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
                )
                _reseal(run)
                with self.assertRaises(RunIntegrityError):
                    verify_run_dir(run)

        if hasattr(os, "symlink"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = _create_run(root)
                target = root / "saved-metrics.json"
                (run / "metrics.json").replace(target)
                (run / "metrics.json").symlink_to(target)
                _reseal(run)
                with self.assertRaises(RunIntegrityError):
                    verify_run_dir(run)

    def test_goav_semantics_detect_reorder_and_tampering_after_reseal(self) -> None:
        direction = "gradient_optimal_active_verification"
        original = _goav_events()
        mutations: list[tuple[str, list[dict[str, object]]]] = []
        mutations.append(("reordered", [original[1], original[0]]))
        wrong_subset = json.loads(json.dumps(original))
        wrong_subset[1]["subset_index"] = (wrong_subset[1]["subset_index"] + 1) % 4
        mutations.append(("wrong_subset", wrong_subset))
        leaked_label = json.loads(json.dumps(original))
        leaked_label[1]["selected_labels"]["2"] = 1.0
        mutations.append(("unselected_label", leaked_label))
        extra_outcome = json.loads(json.dumps(original))
        extra_outcome[1]["full_labels"] = [1.0, 0.0]
        mutations.append(("extra_outcome", extra_outcome))
        huge_label = json.loads(json.dumps(original))
        huge_label[1]["selected_labels"]["0"] = 10**400
        mutations.append(("nonfinite_cast_label", huge_label))
        wrong_pi = json.loads(json.dumps(original))
        wrong_pi[0]["pi"][0] = 0.61
        mutations.append(("wrong_pi", wrong_pi))
        wrong_hash = json.loads(json.dumps(original))
        wrong_hash[0]["pi_hash"] = "0" * 64
        mutations.append(("wrong_hash", wrong_hash))

        for label, events in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                run = _create_run(Path(directory), direction=direction, alias="goav", events=original)
                event_rows = _envelope_goav_events(events, run.name)
                encoded = "".join(
                    json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
                    for event in event_rows
                )
                (run / "events.jsonl").write_text(encoded, encoding="utf-8", newline="")
                _reseal(run)
                with self.assertRaises(RunIntegrityError):
                    verify_run_dir(run)

        envelope_mutations = (("sequence", 9), ("run_id", "wrong-run"), ("seed", 202))
        for field, value in envelope_mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                run = _create_run(Path(directory), direction=direction, alias="goav", events=original)
                event_rows = [
                    json.loads(line)
                    for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                event_rows[1][field] = value
                encoded = "".join(
                    json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
                    for event in event_rows
                )
                (run / "events.jsonl").write_text(encoded, encoding="utf-8", newline="")
                _reseal(run)
                with self.assertRaises(RunIntegrityError):
                    verify_run_dir(run)

        with tempfile.TemporaryDirectory() as directory:
            run = _create_run(Path(directory), direction=direction, alias="goav", events=original)
            self.assertEqual(verify_run_dir(run)["event_count"], 2)

    def test_goav_replay_rejects_coordinated_scientific_rewrites(self) -> None:
        """Checksums cannot legitimize rows, labels, or gates that replay disproves."""

        from coding_opsd.runtime import _deterministic_npz_bytes

        direction = "gradient_optimal_active_verification"
        with tempfile.TemporaryDirectory() as directory:
            run = _create_run(Path(directory), direction=direction, alias="goav")
            rows = [
                json.loads(line)
                for line in (run / "rows.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["nMSE"] += 1.0
            (run / "rows.jsonl").write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
                newline="",
            )
            with np.load(run / "arrays.npz", allow_pickle=False) as loaded:
                arrays = {key: loaded[key].copy() for key in loaded.files}
            arrays["goav_row_numeric"][0, 0] = rows[0]["nMSE"]
            (run / "arrays.npz").write_bytes(_deterministic_npz_bytes(arrays))
            _reseal(run)
            with self.assertRaisesRegex(RunIntegrityError, "deterministic config/seed replay"):
                verify_run_dir(run)

        with tempfile.TemporaryDirectory() as directory:
            run = _create_run(Path(directory), direction=direction, alias="goav")
            events = [
                json.loads(line)
                for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            events[1]["selected_labels"]["0"] = 1.0
            (run / "events.jsonl").write_text(
                "".join(
                    json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
                    for event in events
                ),
                encoding="utf-8",
                newline="",
            )
            _reseal(run)
            with self.assertRaisesRegex(RunIntegrityError, "deterministic config/seed replay"):
                verify_run_dir(run)

        with tempfile.TemporaryDirectory() as directory:
            run = _create_run(Path(directory), direction=direction, alias="goav")
            gates = json.loads((run / "gates.json").read_text(encoding="utf-8"))
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            gates["reason"] = "forged-but-resealed"
            (run / "gates.json").write_text(
                json.dumps(gates, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
                newline="",
            )
            (run / "report.md").write_text(
                _canonical_report(manifest, gates), encoding="utf-8", newline=""
            )
            _reseal(run)
            with self.assertRaisesRegex(RunIntegrityError, "canonical Phase-0"):
                verify_run_dir(run)

        with tempfile.TemporaryDirectory() as directory:
            run = _create_run(Path(directory), direction=direction, alias="goav")
            with np.load(run / "arrays.npz", allow_pickle=False) as loaded:
                arrays = {key: loaded[key].copy() for key in loaded.files}
            matrix = arrays["goav_row_numeric"]
            zero_index = tuple(np.argwhere(matrix == 0.0)[0])
            matrix[zero_index] = -0.0
            (run / "arrays.npz").write_bytes(_deterministic_npz_bytes(arrays))
            _reseal(run)
            with self.assertRaisesRegex(RunIntegrityError, "byte-for-byte"):
                verify_run_dir(run)

    def test_goav_config_support_counts_replay_and_strict_numeric_semantics(self) -> None:
        direction = "gradient_optimal_active_verification"
        config, _, run_id = _identity(direction, "goav")
        row_arms = {"goav_exact_subset_aipw"}

        def check(events: list[dict[str, object]], candidate_config: dict[str, object] | None = None) -> None:
            validate_goav_events(
                _envelope_goav_events(events, run_id),
                run_id=run_id,
                seed=101,
                require_envelope=True,
                config=config if candidate_config is None else candidate_config,
                row_arms=row_arms,
            )

        check(_goav_events())

        numeric_string = json.loads(json.dumps(_goav_events()))
        numeric_string[0]["probabilities"][0] = "0.1"
        with self.assertRaises(RunIntegrityError):
            check(numeric_string)

        boolean_probability = json.loads(json.dumps(_goav_events()))
        boolean_probability[0]["probabilities"][0] = True
        with self.assertRaises(RunIntegrityError):
            check(boolean_probability)

        near_pi = json.loads(json.dumps(_goav_events()))
        near_pi[0]["pi"][0] += 1e-15
        near_pi[0]["pi_hash"] = sha256(
            np.asarray(near_pi[0]["pi"], dtype="<f8").tobytes(order="C")
        ).hexdigest()
        near_pi[1]["inclusion_probabilities"] = near_pi[0]["pi"]
        with self.assertRaises(RunIntegrityError):
            check(near_pi)

        nonbinary_label = json.loads(json.dumps(_goav_events()))
        nonbinary_label[1]["selected_labels"]["0"] = 0.5
        with self.assertRaises(RunIntegrityError):
            check(nonbinary_label)

        no_full_support = _goav_design("goav_exact_subset_aipw", [0.0, 0.2, 0.3, 0.5])
        with self.assertRaises(RunIntegrityError):
            check([no_full_support, _goav_audit(no_full_support)])

        wrong_budget = json.loads(json.dumps(config))
        wrong_budget["phase0"]["primary_budget_fraction"] = 0.5
        with self.assertRaises(RunIntegrityError):
            check(_goav_events(), wrong_budget)

        impossible_floor = json.loads(json.dumps(config))
        impossible_floor["acquisition"]["primary_inclusion_floor"] = 0.7
        with self.assertRaises(RunIntegrityError):
            check(_goav_events(), impossible_floor)

        full_config = json.loads(json.dumps(config))
        full_config["phase0"]["arms"] = ["full_audit"]
        full_design = _goav_design("full_audit", [0.0, 0.0, 0.0, 1.0])
        validate_goav_events(
            _envelope_goav_events([full_design, _goav_audit(full_design)], run_id),
            run_id=run_id,
            seed=101,
            require_envelope=True,
            config=full_config,
            row_arms={"full_audit"},
        )
        bad_full = _goav_design("full_audit", [0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(RunIntegrityError):
            validate_goav_events(
                _envelope_goav_events([bad_full, _goav_audit(bad_full)], run_id),
                run_id=run_id,
                seed=101,
                require_envelope=True,
                config=full_config,
                row_arms={"full_audit"},
            )

        two_draw_config = json.loads(json.dumps(config))
        two_draw_config["phase0"]["subset_draws_per_group_design"] = 2
        with self.assertRaises(RunIntegrityError):
            check(_goav_events(), two_draw_config)

        wrong_replay = json.loads(json.dumps(_goav_events()))
        wrong_replay[1]["subset_index"] = (wrong_replay[1]["subset_index"] + 1) % 4
        wrong_replay[1]["selected_indices"] = [
            index for index in range(2) if (wrong_replay[1]["subset_index"] >> index) & 1
        ]
        wrong_replay[1]["selected_labels"] = {
            str(index): float(index == 0) for index in wrong_replay[1]["selected_indices"]
        }
        with self.assertRaises(RunIntegrityError):
            check(wrong_replay)

    def test_goav_rejects_missing_or_late_designs_and_unconfigured_event_arms(self) -> None:
        direction = "gradient_optimal_active_verification"
        config, _, run_id = _identity(direction, "goav")
        config["phase0"]["arms"] = ["goav_exact_subset_aipw", "uniform_subset_aipw"]
        first = _goav_design("goav_exact_subset_aipw", [0.1, 0.2, 0.3, 0.4])
        second = _goav_design("uniform_subset_aipw", [0.1, 0.2, 0.3, 0.4])

        for label, events, row_arms in (
            ("missing", [first, _goav_audit(first)], {"goav_exact_subset_aipw", "uniform_subset_aipw"}),
            (
                "late",
                [first, _goav_audit(first), second, _goav_audit(second)],
                {"goav_exact_subset_aipw", "uniform_subset_aipw"},
            ),
            ("row_arm", [first, second, _goav_audit(first), _goav_audit(second)], {"goav_exact_subset_aipw"}),
        ):
            with self.subTest(label=label), self.assertRaises(RunIntegrityError):
                validate_goav_events(
                    _envelope_goav_events(events, run_id),
                    run_id=run_id,
                    seed=101,
                    require_envelope=True,
                    config=config,
                    row_arms=row_arms,
                )

        control_config = json.loads(json.dumps(config))
        control_config["phase0"]["arms"] = ["goav_exact_subset_aipw", "deterministic_topk_invalid"]
        validate_goav_events(
            _envelope_goav_events([first, _goav_audit(first)], run_id),
            run_id=run_id,
            seed=101,
            require_envelope=True,
            config=control_config,
            row_arms={"goav_exact_subset_aipw", "deterministic_topk_invalid"},
        )

    def test_goav_rejects_conflicting_trusted_labels_across_arms_and_draws(self) -> None:
        direction = "gradient_optimal_active_verification"
        config, _, run_id = _identity(direction, "goav", seed=0)
        config["phase0"]["arms"] = [
            "goav_exact_subset_aipw",
            "uniform_subset_aipw",
        ]
        first = _goav_design("goav_exact_subset_aipw", [0.1, 0.2, 0.3, 0.4])
        second = _goav_design("uniform_subset_aipw", [0.1, 0.2, 0.3, 0.4])
        first_audit = _goav_audit(first, seed=0)
        second_audit = _goav_audit(second, seed=0)
        self.assertIn("1", first_audit["selected_labels"])
        self.assertIn("1", second_audit["selected_labels"])
        second_audit["selected_labels"]["1"] = 1.0

        with self.assertRaisesRegex(RunIntegrityError, "conflicting trusted label"):
            validate_goav_events(
                _envelope_goav_events(
                    [first, second, first_audit, second_audit], run_id, seed=0
                ),
                run_id=run_id,
                seed=0,
                require_envelope=True,
                config=config,
                row_arms={"goav_exact_subset_aipw", "uniform_subset_aipw"},
            )


if __name__ == "__main__":
    unittest.main()
