"""Regression tests for trusted replay of sealed JAG Phase-0 artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from coding_opsd.config import config_hash
from coding_opsd.jag import run_jag_phase0
from coding_opsd.results import NormalizedRun, normalize_result
from coding_opsd.runtime import (
    RunIntegrityError,
    RunWriter,
    _deterministic_npz_bytes,
    verify_run_dir,
)


_PRECOMPLETION_ARTIFACTS = {
    "resolved_config.json",
    "manifest.json",
    "metrics.json",
    "gates.json",
    "events.jsonl",
    "rows.jsonl",
    "arrays.npz",
    "report.md",
}


def _config(identity: str, *, dynamic_oracle_unavailable: bool = False) -> dict[str, object]:
    if dynamic_oracle_unavailable:
        horizon = 3
        max_branching = 4
        branchable_depth_count = 3
        arms = ["jag_oracle"]
        budget = 6
        replications = 1
        max_states = 1
    else:
        horizon = 2
        max_branching = 2
        branchable_depth_count = 2
        arms = ["uniform_tree", "jag_oracle", "jag_learned"]
        budget = 2
        replications = 2
        max_states = 100_000
    return {
        "schema_version": "0.1",
        "direction": "jag_tree",
        "runtime": {"profile": "smoke"},
        "output": {"identity": identity},
        "seeds": [101],
        "estimator": {
            "primary_baseline": "zero",
            "max_branching": max_branching,
            "branchable_depth_count": branchable_depth_count,
        },
        "phase0": {
            "horizon": horizon,
            "actions": 2,
            "reward_families": ["root_only"],
            "arms": arms,
            "budgets": [budget],
            "rollout_replications": replications,
            "covariance_audit_replications": 2,
            "learned_calibration_seeds": [71],
            "learned_calibration_horizon": horizon,
            "gate": {},
            "oracle_scope": {
                "max_horizon": 3,
                "max_budget": 12,
                "max_frontier_nodes": 8,
                "max_states": max_states,
            },
        },
    }


def _manifest(config: dict[str, object], normalized: NormalizedRun, run_id: str) -> dict[str, object]:
    return {
        "schema_version": "phase0.run.v1",
        "run_id": run_id,
        "direction": "jag_tree",
        "direction_alias": "jag",
        "output_identity": config["output"]["identity"],
        "seed": 101,
        "profile": "smoke",
        "status": normalized.metrics["status"],
        "config_sha256": config_hash(config),
        "source_configs": [{"path": "/registered/jag.yaml", "sha256": "a" * 64}],
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
        "row_count": len(normalized.rows),
        "event_count": len(normalized.events),
        "array_keys": sorted(normalized.arrays),
    }


def _report(manifest: dict[str, object], normalized: NormalizedRun) -> str:
    return (
        "# Phase-0 run report\n\n"
        f"- Run ID: {manifest['run_id']}\n"
        "- Direction: jag_tree (jag)\n"
        "- Seed: 101\n"
        "- Profile: smoke\n"
        f"- Scientific status: {manifest['status']}\n"
        f"- Status reason: {normalized.gates['reason']}\n"
        f"- Rows/events/arrays: {len(normalized.rows)} / "
        f"{len(normalized.events)} / {len(normalized.arrays)}\n\n"
        "This is a deterministic finite CPU Phase-0 mechanism experiment. "
        "It is not a 7B-model result or a coding-benchmark score. COMPLETE "
        "means that the artifacts are sealed; gates.json is authoritative "
        "for the scientific status.\n"
    )


def _seal(root: Path, config: dict[str, object]) -> tuple[Path, NormalizedRun]:
    digest = config_hash(config)
    identity = config["output"]["identity"]
    run_id = f"{identity}-seed101-{digest[:8]}"
    normalized = normalize_result(
        "jag",
        "jag_tree",
        config,
        101,
        run_id,
        run_jag_phase0(config, 101),
    )
    manifest = _manifest(config, normalized, run_id)
    run = root / run_id
    with RunWriter(run, config, manifest) as writer:
        writer.write_metrics(normalized.metrics)
        writer.write_gates(normalized.gates)
        writer.write_events(normalized.events)
        writer.write_rows(normalized.rows)
        writer.write_npz(normalized.arrays)
        writer.write_report(_report(manifest, normalized))
        writer.complete()
    return run, normalized


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    path.write_text(encoded, encoding="utf-8", newline="")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    encoded = "".join(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for value in values
    )
    path.write_text(encoded, encoding="utf-8", newline="")


def _reseal(run: Path) -> None:
    checksums = {
        name: sha256((run / name).read_bytes()).hexdigest()
        for name in sorted(_PRECOMPLETION_ARTIFACTS)
    }
    _write_json(run / "checksums.json", checksums)
    anchor = sha256((run / "checksums.json").read_bytes()).hexdigest() + "\n"
    (run / "COMPLETE").write_text(anchor, encoding="utf-8", newline="")


class JagArtifactReplayTests(unittest.TestCase):
    def test_rejects_resealed_array_dtype_and_signed_zero_rewrites(self) -> None:
        config = _config("jag_array_bytes_replay")
        phase = config["phase0"]
        estimator = config["estimator"]
        phase.update(
            {
                "horizon": 1,
                "actions": 1,
                "arms": ["flat_iid"],
                "budgets": [1],
                "rollout_replications": 1,
                "learned_calibration_horizon": 1,
            }
        )
        estimator.update(
            {"max_branching": 1, "branchable_depth_count": 1}
        )
        for mutation in ("dtype", "signed_zero"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                run, _ = _seal(Path(directory), config)
                with np.load(run / "arrays.npz", allow_pickle=False) as archive:
                    arrays = {
                        key: np.asarray(archive[key]).copy() for key in archive.files
                    }
                if mutation == "dtype":
                    arrays = {
                        key: value.astype("<i8") for key, value in arrays.items()
                    }
                else:
                    key = next(
                        name
                        for name, value in arrays.items()
                        if np.any(value == 0.0)
                    )
                    flat = arrays[key].reshape(-1)
                    zero_index = int(np.flatnonzero(flat == 0.0)[0])
                    flat[zero_index] = -0.0
                (run / "arrays.npz").write_bytes(
                    _deterministic_npz_bytes(arrays)
                )
                _reseal(run)

                with self.assertRaisesRegex(
                    RunIntegrityError,
                    "JAG arrays/covariance evidence disagree with deterministic config/seed replay",
                ):
                    verify_run_dir(run)

    def test_rejects_resealed_gradient_hash_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run, _ = _seal(Path(directory), _config("jag_gradient_hash_replay"))
            rows = _read_jsonl(run / "rows.jsonl")
            rows[0]["paired_replicates"][0]["gradient_hash"] = "f" * 64
            _write_jsonl(run / "rows.jsonl", rows)
            _reseal(run)

            with self.assertRaisesRegex(
                RunIntegrityError, "JAG rows disagree with deterministic config/seed replay"
            ):
                verify_run_dir(run)

    def test_rejects_resealed_coordinated_mse_and_comparison_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run, _ = _seal(Path(directory), _config("jag_mse_replay"))
            rows = _read_jsonl(run / "rows.jsonl")
            for row in rows:
                for replicate in row["paired_replicates"]:
                    replicate["squared_error"] = 0.0
                row["mse"] = 0.0
                row["budget_times_mse"] = 0.0
            _write_jsonl(run / "rows.jsonl", rows)

            gates = json.loads((run / "gates.json").read_text(encoding="utf-8"))
            paired_zeros = {
                "definition": "first_squared_error-minus-second_squared_error_on_shared_crn_stream",
                "paired_count": 2,
                "mean_delta": 0.0,
                "delta_hash": sha256(b"[0.0,0.0]").hexdigest(),
            }
            gates["inputs"]["registered_comparisons"] = [
                {
                    "family": "root_only",
                    "budget": 2,
                    "oracle_vs_best_unbiased_relative_mse_reduction": None,
                    "learned_oracle_uniform_gap_closed": None,
                    "learned_minus_uniform_paired_squared_error": paired_zeros,
                },
                {
                    "family": "covariance_reversal",
                    "budget": 2,
                    "full_joint_vs_no_cross_relative_mse_gain": None,
                    "full_joint_minus_no_cross_paired_squared_error": None,
                },
            ]
            _write_json(run / "gates.json", gates)
            _reseal(run)

            with self.assertRaisesRegex(
                RunIntegrityError, "JAG (rows|gates) disagree with deterministic config/seed replay"
            ):
                verify_run_dir(run)

    def test_accepts_runner_artifact_with_dynamic_max_states_unavailability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run, normalized = _seal(
                Path(directory),
                _config("jag_dynamic_oracle_scope", dynamic_oracle_unavailable=True),
            )
            self.assertEqual(normalized.rows[0]["status"], "UNAVAILABLE_ORACLE_SCOPE")
            self.assertEqual(
                normalized.rows[0]["status_reason"],
                "frontier enumeration needs more than max_states=1",
            )

            self.assertEqual(verify_run_dir(run)["status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
