"""Adversarial sealed-artifact contract tests for the PBPF Phase-0 runner."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

from coding_opsd.config import config_hash
from coding_opsd.pbpf.experiment import _group_gate_inputs, run_pbpf_phase0
from coding_opsd.results import normalize_result
from coding_opsd.runtime import (
    RunIntegrityError,
    RunWriter,
    _deterministic_npz_bytes,
    verify_run_dir,
)


_DIRECTION = "predictive_belief_particle_filter"
_SEED = 101
_REQUIRED_PRECOMPLETION = {
    "resolved_config.json",
    "manifest.json",
    "metrics.json",
    "gates.json",
    "events.jsonl",
    "rows.jsonl",
    "arrays.npz",
    "report.md",
}
_GATE_ARRAYS = {
    "paired_exact_vs_prior_inputs": "pbpf_gate_prefix4_paired_exact_vs_prior",
    "paired_exact_vs_map_inputs": "pbpf_gate_prefix4_paired_exact_vs_map",
    "paired_p16_to_exact_inputs": "pbpf_gate_prefix4_paired_p16_to_exact",
    "hpd_coverage_inputs": "pbpf_gate_prefix4_hpd_coverage",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _tiny_config() -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "direction": _DIRECTION,
        "runtime": {"profile": "smoke"},
        "output": {"identity": "pbpf_artifact_contract_smoke"},
        "seeds": [_SEED],
        "belief": {"particle_sweep": [2, 16]},
        "phase0": {
            "episodes": {"test": 1},
            "candidates": 4,
            "prefixes": [0, 1, 4],
            "forecast_horizons": [1, "all_remaining"],
            "particle_sweep": [2, 16],
            "test_order_seed": 61030,
            "arms": ["exact_bayes", "pbpf", "map"],
        },
    }


def _report(run_id: str, status: str, reason: str, rows: int, arrays: int) -> str:
    return (
        "# Phase-0 run report\n\n"
        f"- Run ID: {run_id}\n"
        f"- Direction: {_DIRECTION} (pbpf)\n"
        f"- Seed: {_SEED}\n"
        "- Profile: smoke\n"
        f"- Scientific status: {status}\n"
        f"- Status reason: {reason}\n"
        f"- Rows/events/arrays: {rows} / 0 / {arrays}\n\n"
        "This is a deterministic finite CPU Phase-0 mechanism experiment. "
        "It is not a 7B-model result or a coding-benchmark score. COMPLETE "
        "means that the artifacts are sealed; gates.json is authoritative "
        "for the scientific status.\n"
    )


def _create_native_run(parent: Path) -> Path:
    config = _tiny_config()
    digest = config_hash(config)
    run_id = f"pbpf_artifact_contract_smoke-seed{_SEED}-{digest[:8]}"
    raw = run_pbpf_phase0(config, _SEED)
    normalized = normalize_result("pbpf", _DIRECTION, config, _SEED, run_id, raw)
    manifest = {
        "schema_version": "phase0.run.v1",
        "run_id": run_id,
        "direction": _DIRECTION,
        "direction_alias": "pbpf",
        "output_identity": "pbpf_artifact_contract_smoke",
        "seed": _SEED,
        "profile": "smoke",
        "status": normalized.gates["status"],
        "config_sha256": digest,
        "source_configs": [
            {"path": "/registered/pbpf-artifact-contract.yaml", "sha256": "a" * 64}
        ],
        "git": {"commit": "b" * 40, "dirty": False},
        "environment": {
            "kind": "cpu_phase0",
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": "test",
            "pyyaml": "test",
            "model_backend": None,
            "model_checkpoint": None,
            "container_digest": "not_applicable",
        },
        "row_count": len(normalized.rows),
        "event_count": len(normalized.events),
        "array_keys": sorted(normalized.arrays),
    }
    run = parent / run_id
    with RunWriter(run, config, manifest) as writer:
        writer.write_metrics(normalized.metrics)
        writer.write_gates(normalized.gates)
        writer.write_events(normalized.events)
        writer.write_rows(normalized.rows)
        writer.write_npz(normalized.arrays)
        writer.write_report(
            _report(
                run_id,
                str(normalized.gates["status"]),
                str(normalized.gates["reason"]),
                len(normalized.rows),
                len(normalized.arrays),
            )
        )
        writer.complete()
    return run


def _read_rows(run: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (run / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _write_rows(run: Path, rows: list[dict[str, object]]) -> None:
    contents = "".join(_canonical_json(row) + "\n" for row in rows)
    (run / "rows.jsonl").write_text(contents, encoding="utf-8", newline="")


def _read_json(run: Path, name: str) -> dict[str, object]:
    return json.loads((run / name).read_text(encoding="utf-8"))


def _write_json(run: Path, name: str, value: object) -> None:
    (run / name).write_text(
        _canonical_json(value) + "\n", encoding="utf-8", newline=""
    )


def _read_arrays(run: Path) -> dict[str, np.ndarray]:
    with np.load(run / "arrays.npz", allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def _write_arrays(run: Path, arrays: dict[str, np.ndarray]) -> None:
    (run / "arrays.npz").write_bytes(_deterministic_npz_bytes(arrays))


def _reseal(run: Path) -> None:
    checksums = {
        name: sha256((run / name).read_bytes()).hexdigest()
        for name in sorted(_REQUIRED_PRECOMPLETION)
    }
    _write_json(run, "checksums.json", checksums)
    (run / "COMPLETE").write_text(
        sha256((run / "checksums.json").read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
        newline="",
    )


def _row_index(
    rows: list[dict[str, object]],
    *,
    episode: int,
    prefix: int,
    horizon: int | str,
    arm: str,
) -> int:
    return next(
        index
        for index, row in enumerate(rows)
        if (
            row["episode"],
            row["prefix"],
            row["horizon"],
            row["arm"],
        )
        == (episode, prefix, horizon, arm)
    )


def _update_numeric_row(
    arrays: dict[str, np.ndarray], rows: list[dict[str, object]], index: int
) -> None:
    columns = ("nll", "brier", "posterior_kl", "ess", "resampling", "hpd_true_inclusion")
    for column_index, column in enumerate(columns):
        value = rows[index][column]
        arrays["pbpf_row_numeric_present"][index, column_index] = value is not None
        arrays["pbpf_row_numeric"][index, column_index] = (
            0.0 if value is None else float(value)
        )


def _prefix_four_gate_cell(rows: list[dict[str, object]]) -> dict[str, object]:
    values: dict[str, object] = {
        "exact": {},
        "prior": {},
        "map": {},
        "particle_16": {},
        "hpd": {},
        "collapses": 0,
        "p16_collapse_ids": set(),
        "expected_episode_ids": set(),
    }
    for row in rows:
        if row["prefix"] != 4 or row["horizon"] != "all_remaining":
            continue
        episode = int(row["episode"])
        arm = str(row["arm"])
        if arm == "exact_bayes":
            values["exact"][episode] = float(row["nll"])
            values["hpd"][episode] = float(row["hpd_true_inclusion"])
            values["expected_episode_ids"].add(episode)
        elif arm == "prior":
            values["prior"][episode] = float(row["nll"])
        elif arm == "map":
            values["map"][episode] = float(row["nll"])
        elif arm == "particle_16":
            if row.get("diagnostic") == "PARTICLE_COLLAPSE":
                values["p16_collapse_ids"].add(episode)
            elif row["nll"] is not None:
                values["particle_16"][episode] = float(row["nll"])
        if row.get("diagnostic") == "PARTICLE_COLLAPSE":
            values["collapses"] += 1
    return _group_gate_inputs(
        values,
        outer_seed=_SEED,
        label="prefix=4/horizon=all_remaining",
        bootstrap_resamples=0,
    )


class PbpfArtifactContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._fixture_directory = tempfile.TemporaryDirectory()
        cls._baseline = _create_native_run(Path(cls._fixture_directory.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._fixture_directory.cleanup()

    def _copy_fixture(self, parent: Path) -> Path:
        target = parent / self._baseline.name
        shutil.copytree(self._baseline, target)
        return target

    def test_rejects_coordinated_row_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._copy_fixture(Path(directory))
            rows = _read_rows(run)
            arrays = _read_arrays(run)
            rows[0], rows[1] = rows[1], rows[0]
            arrays["pbpf_row_numeric"][[0, 1]] = arrays["pbpf_row_numeric"][[1, 0]]
            arrays["pbpf_row_numeric_present"][[0, 1]] = arrays[
                "pbpf_row_numeric_present"
            ][[1, 0]]
            for index, row in enumerate(rows):
                row["row_index"] = index
            _write_rows(run, rows)
            _write_arrays(run, arrays)
            _reseal(run)
            with self.assertRaises(RunIntegrityError):
                verify_run_dir(run)

    def test_rejects_resealed_array_signed_zero_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._copy_fixture(Path(directory))
            arrays = _read_arrays(run)
            matrix = arrays["pbpf_row_numeric"]
            zero_index = tuple(np.argwhere(matrix == 0.0)[0])
            matrix[zero_index] = -0.0
            _write_arrays(run, arrays)
            _reseal(run)
            with self.assertRaisesRegex(RunIntegrityError, "byte-for-byte"):
                verify_run_dir(run)

    def test_rejects_deterministic_arm_semantic_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._copy_fixture(Path(directory))
            rows = _read_rows(run)
            arrays = _read_arrays(run)
            index = _row_index(
                rows, episode=0, prefix=0, horizon=1, arm="exact_bayes"
            )
            rows[index]["posterior_kl"] = 0.125
            _update_numeric_row(arrays, rows, index)
            _write_rows(run, rows)
            _write_arrays(run, arrays)
            _reseal(run)
            with self.assertRaises(RunIntegrityError):
                verify_run_dir(run)

    def test_rejects_incoherent_ess_resampling_event_and_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._copy_fixture(Path(directory))
            rows = _read_rows(run)
            arrays = _read_arrays(run)
            for horizon in (1, "all_remaining"):
                index = _row_index(
                    rows, episode=0, prefix=1, horizon=horizon, arm="particle_16"
                )
                rows[index]["resampling"] = 1
                rows[index]["resampling_rate"] = 0.0
                rows[index]["resampling_events"] = []
                _update_numeric_row(arrays, rows, index)
            _write_rows(run, rows)
            _write_arrays(run, arrays)
            _reseal(run)
            with self.assertRaises(RunIntegrityError):
                verify_run_dir(run)

    def test_rejects_particle_posterior_diagnostic_drift_across_horizons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._copy_fixture(Path(directory))
            rows = _read_rows(run)
            arrays = _read_arrays(run)
            index = _row_index(
                rows, episode=0, prefix=1, horizon=1, arm="particle_16"
            )
            rows[index]["posterior_kl"] = float(rows[index]["posterior_kl"]) + 0.25
            _update_numeric_row(arrays, rows, index)
            _write_rows(run, rows)
            _write_arrays(run, arrays)
            _reseal(run)
            with self.assertRaises(RunIntegrityError):
                verify_run_dir(run)

    def test_rejects_rows_gates_and_npz_rewritten_as_one_consistent_story(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._copy_fixture(Path(directory))
            rows = _read_rows(run)
            arrays = _read_arrays(run)
            index = _row_index(
                rows,
                episode=0,
                prefix=4,
                horizon="all_remaining",
                arm="exact_bayes",
            )
            rows[index]["nll"] = float(rows[index]["nll"]) + 0.5
            _update_numeric_row(arrays, rows, index)

            cell = _prefix_four_gate_cell(rows)
            gates = _read_json(run, "gates.json")
            gate_inputs = gates["inputs"]
            gate_inputs["by_prefix_horizon"]["4"]["all_remaining"] = deepcopy(cell)
            gate_inputs["prefix_4"] = deepcopy(cell)
            for field in (
                "exact_vs_prior_nll_delta",
                "exact_mixture_vs_map_nll_delta",
                "p16_to_exact_nll_gap",
                "map_gap_fraction_closed",
            ):
                gate_inputs[field] = cell[field]
            for source, array_key in _GATE_ARRAYS.items():
                arrays[array_key] = np.asarray(cell[source], dtype="<f8")

            _write_rows(run, rows)
            _write_json(run, "gates.json", gates)
            _write_arrays(run, arrays)
            _reseal(run)
            with self.assertRaises(RunIntegrityError):
                verify_run_dir(run)


if __name__ == "__main__":
    unittest.main()
