"""End-to-end contract tests for the public Phase-0 command line."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from coding_opsd.cli import main
from coding_opsd.config import load_config


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
    return path


def _tiny_configs(root: Path, seed: int = 7) -> dict[str, Path]:
    common = {
        "runtime": {"profile": "smoke"},
        "seeds": [seed],
        "execution": {"container_digest_required": False},
    }
    jag = {
        **common,
        "schema_version": "0.1",
        "direction": "jag_tree",
        "output": {"identity": "jag_tiny"},
        "phase0": {
            "horizon": 2,
            "actions": 2,
            "reward_families": ["root_only"],
            "arms": ["uniform_tree"],
            "budgets": [2],
            "rollout_replications": 2,
            "oracle_scope": {
                "max_horizon": 2,
                "max_budget": 4,
                "max_frontier_nodes": 4,
                "max_states": 1000,
            },
        },
    }
    pbpf = {
        **common,
        "schema_version": "0.1",
        "direction": "predictive_belief_particle_filter",
        "output": {"identity": "pbpf_tiny"},
        "belief": {"particle_sweep": [2]},
        "phase0": {
            "episodes": {"test": 1},
            "candidates": 2,
            "prefixes": [0],
            "forecast_horizons": [1],
            "particle_sweep": [2],
            "arms": ["exact_bayes", "pbpf", "map"],
            "test_order_seed": 61030,
        },
    }
    goav = {
        **common,
        "schema_version": "0.2",
        "direction": "gradient_optimal_active_verification",
        "output": {"identity": "goav_tiny"},
        "acquisition": {
            "primary_inclusion_floor": 0.1,
            "solver": {"steps": 3, "learning_rate": 0.05, "initialization": ["uniform"]},
        },
        "oracle_noise_model": {
            "medium_noise": {
                "false_negative": 0.1,
                "false_positive": 0.2,
                "flip_icc": 0.3,
                "cluster_size": 1,
            }
        },
        "phase0": {
            "tasks_min": 1,
            "group_size": 2,
            "tests_per_task_min": 2,
            "subset_draws_per_group_design": 1,
            "primary_budget_fraction": 0.5,
            "epsilon_G": 0.01,
            "solver_steps": 3,
            "solver_restarts": 1,
            "arms": ["uniform_subset_aipw", "goav_exact_subset_aipw"],
        },
    }
    return {
        "jag": _write_yaml(root / "jag.yaml", jag),
        "pbpf": _write_yaml(root / "pbpf.yaml", pbpf),
        "goav": _write_yaml(root / "goav.yaml", goav),
    }


def _call(arguments: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(arguments)
    return code, stdout.getvalue(), stderr.getvalue()


def _formal_wrapper(root: Path, alias: str) -> Path:
    """Build a location-independent executable wrapper around a formal config."""

    repository = Path(__file__).resolve().parents[1]
    formal_names = {"jag": "jag_tree.yaml", "pbpf": "pbpf.yaml", "goav": "goav.yaml"}
    formal = repository / "configs" / formal_names[alias]
    registered = load_config(formal).data
    payload = {
        "base_config": os.path.relpath(formal, root),
        "runtime": {"profile": "formal"},
        "seeds": list(registered["statistics"]["training_seeds"]),
        "output": {"identity": f"{alias}_formal_test"},
    }
    return _write_yaml(root / f"{alias}-formal-wrapper.yaml", payload)


class CliTests(unittest.TestCase):
    def test_module_entrypoint_runs_three_directions_and_debug_controls_tracebacks(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        environment = {
            **os.environ,
            "PYTHONPATH": str(repository / "src"),
            "PYTHONHASHSEED": "0",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }

        def invoke(*arguments: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [sys.executable, "-m", "coding_opsd", *arguments],
                cwd=repository,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        help_result = invoke("--help")
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("run-all", help_result.stdout)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = _tiny_configs(root)
            for alias, config in configs.items():
                with self.subTest(alias=alias):
                    result = invoke(
                        "run",
                        alias,
                        str(config),
                        "--seed",
                        "7",
                        "--output",
                        str(root / "runs" / alias),
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    verified = invoke("verify", result.stdout.strip())
                    self.assertEqual(verified.returncode, 0, verified.stderr)
                    self.assertEqual(json.loads(verified.stdout)["status"], "INCOMPLETE")

            bad = yaml.safe_load(configs["jag"].read_text(encoding="utf-8"))
            bad["phase0"]["reward_families"] = ["unknown"]
            bad_path = _write_yaml(root / "bad.yaml", bad)
            concise = invoke("validate", str(bad_path))
            self.assertNotEqual(concise.returncode, 0)
            self.assertNotIn("Traceback", concise.stderr)
            debug = invoke("--debug", "validate", str(bad_path))
            self.assertNotEqual(debug.returncode, 0)
            self.assertIn("Traceback", debug.stderr)

    def test_help_resolve_and_validate_are_public_and_resolve_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _tiny_configs(root)["jag"]
            code, stdout, stderr = _call(["--help"])
            self.assertEqual(code, 0)
            self.assertIn("run-all", stdout)
            self.assertEqual(stderr, "")

            code, stdout, stderr = _call(["validate", str(config)])
            self.assertEqual((code, stderr), (0, ""))
            validation = json.loads(stdout)
            self.assertEqual(validation["direction"], "jag_tree")
            self.assertEqual(validation["status"], "VALID")

            target = root / "resolved.json"
            code, stdout, stderr = _call(["resolve", str(config), "--output", str(target)])
            self.assertEqual((code, stdout, stderr), (0, "", ""))
            resolved_bytes = target.read_bytes()
            self.assertTrue(resolved_bytes.endswith(b"\n"))
            json.loads(resolved_bytes)
            second, _, second_error = _call(["resolve", str(config), "--output", str(target)])
            self.assertNotEqual(second, 0)
            self.assertIn("already exists", second_error)
            self.assertEqual(target.read_bytes(), resolved_bytes)
            self.assertNotIn("Traceback", second_error)

    def test_three_tiny_runs_seal_and_verify_standard_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = _tiny_configs(root)
            output = root / "runs"
            for alias, config in configs.items():
                with self.subTest(alias=alias):
                    code, stdout, stderr = _call(
                        ["run", alias, str(config), "--seed", "7", "--output", str(output)]
                    )
                    self.assertEqual((code, stderr), (0, ""))
                    run_dir = Path(stdout.strip())
                    self.assertTrue(run_dir.is_dir())
                    self.assertEqual(
                        {entry.name for entry in run_dir.iterdir()},
                        {
                            "resolved_config.json",
                            "manifest.json",
                            "metrics.json",
                            "gates.json",
                            "events.jsonl",
                            "rows.jsonl",
                            "arrays.npz",
                            "report.md",
                            "checksums.json",
                            "COMPLETE",
                        },
                    )
                    gates = json.loads((run_dir / "gates.json").read_text(encoding="utf-8"))
                    self.assertEqual(gates["status"], "INCOMPLETE")
                    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                    self.assertEqual(manifest["schema_version"], "phase0.run.v1")
                    self.assertTrue(manifest["source_configs"])
                    self.assertEqual(len(manifest["source_configs"][0]["sha256"]), 64)
                    self.assertEqual(set(manifest["git"]), {"commit", "dirty"})
                    self.assertEqual(manifest["environment"]["kind"], "cpu_phase0")
                    self.assertIsNone(manifest["environment"]["model_backend"])
                    self.assertEqual(manifest["environment"]["container_digest"], "not_applicable")
                    verify_code, verify_stdout, verify_stderr = _call(["verify", str(run_dir)])
                    self.assertEqual((verify_code, verify_stderr), (0, ""))
                    self.assertEqual(json.loads(verify_stdout)["run_id"], run_dir.name)

    def test_metrics_and_gates_are_byte_identical_for_every_direction_across_output_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = _tiny_configs(root)
            for alias, config in configs.items():
                with self.subTest(alias=alias):
                    run_dirs = []
                    for name in ("first", "second"):
                        code, stdout, stderr = _call(
                            [
                                "run",
                                alias,
                                str(config),
                                "--seed",
                                "7",
                                "--output",
                                str(root / name / alias),
                            ]
                        )
                        self.assertEqual((code, stderr), (0, ""))
                        run_dirs.append(Path(stdout.strip()))
                    for artifact in ("metrics.json", "gates.json"):
                        self.assertEqual(
                            (run_dirs[0] / artifact).read_bytes(),
                            (run_dirs[1] / artifact).read_bytes(),
                        )

    def test_direction_seed_config_and_collision_fail_without_traceback_or_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _tiny_configs(root)["jag"]
            output = root / "runs"
            code, _, error = _call(
                ["run", "pbpf", str(config), "--seed", "7", "--output", str(output)]
            )
            self.assertNotEqual(code, 0)
            self.assertIn("direction", error)
            self.assertNotIn("Traceback", error)
            self.assertFalse(output.exists())

            code, _, error = _call(
                ["run", "jag", str(config), "--seed", "8", "--output", str(output)]
            )
            self.assertNotEqual(code, 0)
            self.assertIn("seed", error)
            self.assertFalse(output.exists())

            code, stdout, stderr = _call(
                ["run", "jag", str(config), "--seed", "7", "--output", str(output)]
            )
            self.assertEqual((code, stderr), (0, ""))
            run_dir = Path(stdout.strip())
            before = {
                entry.name: sha256(entry.read_bytes()).hexdigest()
                for entry in run_dir.iterdir()
                if entry.is_file()
            }
            second, _, second_error = _call(
                ["run", "jag", str(config), "--seed", "7", "--output", str(output)]
            )
            self.assertNotEqual(second, 0)
            self.assertIn("already exists", second_error)
            after = {
                entry.name: sha256(entry.read_bytes()).hexdigest()
                for entry in run_dir.iterdir()
                if entry.is_file()
            }
            self.assertEqual(before, after)

    def test_verify_rejects_mutation_and_run_all_preflights_then_runs_tiny_overlays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = _tiny_configs(root)
            mapping = {alias: path for alias, path in configs.items()}
            output = root / "all"
            with patch("coding_opsd.cli.SMOKE_CONFIG_PATHS", mapping):
                code, stdout, stderr = _call(
                    ["run-all", "--profile", "smoke", "--seed", "7", "--output", str(output)]
                )
            self.assertEqual((code, stderr), (0, ""))
            summaries = json.loads(stdout)
            self.assertEqual([item["direction_alias"] for item in summaries], ["jag", "pbpf", "goav"])
            self.assertTrue(all(item["status"] == "INCOMPLETE" for item in summaries))

            run_dir = Path(summaries[0]["run_dir"])
            metrics = run_dir / "metrics.json"
            metrics.write_text(metrics.read_text(encoding="utf-8") + " ", encoding="utf-8")
            verify_code, _, verify_error = _call(["verify", str(run_dir)])
            self.assertNotEqual(verify_code, 0)
            self.assertIn("checksum", verify_error)
            self.assertNotIn("Traceback", verify_error)

            before = sorted(str(path) for path in output.iterdir())
            with patch("coding_opsd.cli.SMOKE_CONFIG_PATHS", mapping):
                collision_code, _, collision_error = _call(
                    ["run-all", "--profile", "smoke", "--seed", "7", "--output", str(output)]
                )
            self.assertNotEqual(collision_code, 0)
            self.assertIn("already exists", collision_error)
            self.assertEqual(before, sorted(str(path) for path in output.iterdir()))

    def test_formal_preregistration_validates_but_is_not_directly_executable(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        formal = repository / "configs" / "jag_tree.yaml"
        code, stdout, stderr = _call(["validate", str(formal)])
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(json.loads(stdout)["status"], "VALID_PREREGISTRATION")
        with tempfile.TemporaryDirectory() as directory:
            run_code, _, run_error = _call(
                ["run", "jag", str(formal), "--seed", "101", "--output", directory]
            )
        self.assertNotEqual(run_code, 0)
        self.assertIn("executable overlay", run_error)

    def test_formal_profile_cannot_be_forged_from_any_smoke_overlay(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        smoke_paths = {
            "jag": repository / "configs" / "phase0" / "jag_smoke.yaml",
            "pbpf": repository / "configs" / "phase0" / "pbpf_smoke.yaml",
            "goav": repository / "configs" / "phase0" / "goav_smoke.yaml",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for alias, smoke in smoke_paths.items():
                with self.subTest(alias=alias):
                    overlay = _write_yaml(
                        root / f"{alias}-forged-formal.yaml",
                        {
                            "base_config": os.path.relpath(smoke, root),
                            "runtime": {"profile": "formal"},
                        },
                    )
                    code, _, error = _call(["validate", str(overlay)])
                    self.assertNotEqual(code, 0)
                    self.assertIn("formal scientific projection", error)

    def test_formal_projection_rejects_scientific_mutations_from_copied_configs(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        formal_names = {"jag": "jag_tree.yaml", "pbpf": "pbpf.yaml", "goav": "goav.yaml"}
        mutations = {
            "jag": (
                lambda value: value["phase0"].update(rollout_replications=8),
                lambda value: value["phase0"].update(arms=["uniform_tree"]),
                lambda value: value["phase0"]["gate"].update(relative_bias_max=0.5),
            ),
            "pbpf": (
                lambda value: value["phase0"]["episodes"].update(test=4),
                lambda value: value["phase0"].update(arms=["exact_bayes"]),
                lambda value: value["phase0"]["gate"].update(credible_set_nominal=0.8),
            ),
            "goav": (
                lambda value: value["phase0"].update(tasks_min=4),
                lambda value: value["phase0"].update(arms=["uniform_subset_aipw"]),
                lambda value: value["phase0"]["gate"].update(support_violations_max=1),
                lambda value: value["acquisition"]["solver"].update(steps=3),
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for alias, mutators in mutations.items():
                formal = repository / "configs" / formal_names[alias]
                for index, mutate in enumerate(mutators):
                    with self.subTest(alias=alias, mutation=index):
                        payload = load_config(formal).data
                        payload["runtime"] = {"profile": "formal"}
                        payload["seeds"] = list(payload["statistics"]["training_seeds"])
                        payload["output"] = {"identity": f"{alias}_copied_formal"}
                        mutate(payload)
                        candidate = _write_yaml(root / f"{alias}-copied-{index}.yaml", payload)
                        code, _, error = _call(["validate", str(candidate)])
                        self.assertNotEqual(code, 0)
                        self.assertIn("formal scientific projection", error)

    def test_formal_projection_is_type_and_signed_zero_sensitive(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        cases = (
            (
                "jag",
                "jag_tree.yaml",
                lambda value: value["execution"].update(container_digest_required=1),
            ),
            (
                "pbpf",
                "pbpf.yaml",
                lambda value: value["statistics"].update(bootstrap_resamples=10_000.0),
            ),
            (
                "goav",
                "goav.yaml",
                lambda value: value["phase0"]["gate"].update(
                    paired_seed_task_ci_lower_min=-0.0
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for alias, filename, mutate in cases:
                with self.subTest(alias=alias):
                    payload = load_config(repository / "configs" / filename).data
                    payload["runtime"] = {"profile": "formal"}
                    payload["seeds"] = list(payload["statistics"]["training_seeds"])
                    payload["output"] = {"identity": f"{alias}_typed_formal"}
                    mutate(payload)
                    candidate = _write_yaml(root / f"{alias}-typed-formal.yaml", payload)
                    code, _, error = _call(["validate", str(candidate)])
                    self.assertNotEqual(code, 0)
                    self.assertIn("formal scientific projection", error)

    def test_exact_formal_projection_validates_but_reference_runner_refuses_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for alias in ("jag", "pbpf", "goav"):
                with self.subTest(alias=alias):
                    wrapper = _formal_wrapper(root, alias)
                    code, stdout, stderr = _call(["validate", str(wrapper)])
                    self.assertEqual((code, stderr), (0, ""))
                    self.assertEqual(json.loads(stdout)["status"], "VALID")
                    seed = load_config(wrapper).data["seeds"][0]
                    with patch("coding_opsd.cli._execute") as execute:
                        run_code, _, run_error = _call(
                            [
                                "run",
                                alias,
                                str(wrapper),
                                "--seed",
                                str(seed),
                                "--output",
                                str(root / "runs"),
                            ]
                        )
                    self.assertNotEqual(run_code, 0)
                    self.assertIn("formal profile", run_error)
                    execute.assert_not_called()
            self.assertFalse((root / "runs").exists())

    def test_run_all_rejects_an_overlay_from_the_wrong_profile_before_compute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = _tiny_configs(root)
            payload = yaml.safe_load(configs["pbpf"].read_text(encoding="utf-8"))
            payload["runtime"]["profile"] = "formal"
            _write_yaml(configs["pbpf"], payload)
            output = root / "runs"
            with patch("coding_opsd.cli.SMOKE_CONFIG_PATHS", configs):
                code, _, error = _call(
                    ["run-all", "--profile", "smoke", "--seed", "7", "--output", str(output)]
                )
            self.assertNotEqual(code, 0)
            self.assertIn("profile", error)
            self.assertFalse(output.exists())

    def test_validate_rejects_runner_level_semantic_counterexamples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = _tiny_configs(root)
            mutations = [
                ("jag", lambda value: value["phase0"].update(reward_families=["not_a_family"])),
                (
                    "jag",
                    lambda value: value.update(
                        estimator={"primary_baseline": "current_outcome", "max_branching": 2}
                    ),
                ),
                (
                    "jag",
                    lambda value: value.update(
                        estimator={"primary_baseline": "zero", "max_branching": True}
                    ),
                ),
                (
                    "jag",
                    lambda value: value.update(
                        estimator={
                            "primary_baseline": "zero",
                            "max_branching": 2,
                            "branchable_depth_count": 3,
                        }
                    ),
                ),
                (
                    "jag",
                    lambda value: value["phase0"].update(covariance_audit_replications=1),
                ),
                (
                    "jag",
                    lambda value: value["phase0"]["oracle_scope"].update(max_budget=True),
                ),
                (
                    "jag",
                    lambda value: (
                        value["phase0"].update(
                            arms=["jag_learned"], learned_calibration_seeds=[7]
                        )
                    ),
                ),
                (
                    "jag",
                    lambda value: (
                        value.update(
                            estimator={
                                "primary_baseline": "zero",
                                "max_branching": 1,
                                "branchable_depth_count": 1,
                            }
                        ),
                        value["phase0"].update(budgets=[3]),
                    ),
                ),
                ("goav", lambda value: value["phase0"].update(solver_restarts=0)),
                (
                    "goav",
                    lambda value: value["oracle_noise_model"]["medium_noise"].update(cluster_size=0),
                ),
                (
                    "goav",
                    lambda value: value["oracle_noise_model"]["medium_noise"].update(flip_icc=1.0),
                ),
                (
                    "goav",
                    lambda value: value["phase0"].update(
                        coverage_tests_per_candidate_primary=99
                    ),
                ),
                (
                    "goav",
                    lambda value: value["phase0"].update(synthetic_score_rank=99),
                ),
                ("pbpf", lambda value: value["phase0"].update(prefixes=[64])),
                (
                    "pbpf",
                    lambda value: value["phase0"].update(
                        candidate_source_contamination=1.1
                    ),
                ),
            ]
            for index, (alias, mutate) in enumerate(mutations):
                with self.subTest(alias=alias, index=index):
                    payload = yaml.safe_load(configs[alias].read_text(encoding="utf-8"))
                    mutate(payload)
                    candidate = _write_yaml(root / f"bad-{alias}-{index}.yaml", payload)
                    code, _, error = _call(["validate", str(candidate)])
                    self.assertNotEqual(code, 0)
                    self.assertTrue(error.startswith("error:"))
                    self.assertNotIn("Traceback", error)

    def test_normalization_failure_creates_no_directory_and_status_exit_matrix_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _tiny_configs(root)["jag"]
            output = root / "runs"
            with patch(
                "coding_opsd.cli.normalize_result",
                side_effect=ValueError("normalization counterexample"),
            ):
                code, _, error = _call(
                    ["run", "jag", str(config), "--seed", "7", "--output", str(output)]
                )
            self.assertNotEqual(code, 0)
            self.assertIn("normalization counterexample", error)
            self.assertFalse(output.exists())

            for status, expected_code in (
                ("PASS", 0),
                ("FAIL", 0),
                ("INCOMPLETE", 0),
                ("INVALID", 1),
            ):
                with self.subTest(status=status), patch(
                    "coding_opsd.cli._execute",
                    return_value={
                        "direction_alias": "jag",
                        "direction": "jag_tree",
                        "run_dir": str(output / status.lower()),
                        "status": status,
                    },
                ):
                    status_code, _, status_error = _call(
                        ["run", "jag", str(config), "--seed", "7", "--output", str(output)]
                    )
                    self.assertEqual(status_code, expected_code)
                    self.assertEqual(status_error, "")

    def test_run_all_third_target_collision_prevents_every_runner_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = _tiny_configs(root)
            output = root / "runs"
            goav = load_config(configs["goav"])
            run_id = f"goav_tiny-seed7-{goav.sha256[:8]}"
            collision = output / run_id
            collision.mkdir(parents=True)
            marker = collision / "keep.txt"
            marker.write_text("unchanged\n", encoding="utf-8")
            with patch("coding_opsd.cli.SMOKE_CONFIG_PATHS", configs), patch(
                "coding_opsd.cli._execute"
            ) as execute:
                code, _, error = _call(
                    ["run-all", "--profile", "smoke", "--seed", "7", "--output", str(output)]
                )
            self.assertNotEqual(code, 0)
            self.assertIn("already exists", error)
            execute.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged\n")
            self.assertEqual({entry.name for entry in output.iterdir()}, {run_id})


if __name__ == "__main__":
    unittest.main()
