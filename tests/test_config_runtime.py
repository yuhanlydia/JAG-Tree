"""Contract tests for the Phase-0 shared foundation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from coding_opsd.adapters import (
    Candidate,
    CandidateBank,
    CandidateGroup,
    CandidateGroupRequest,
    GradientProvider,
    GradientRequest,
    GradientResult,
    PolicyBackend,
    PolicyScoreRequest,
    PolicyScoreResult,
    SandboxBackend,
    SandboxLimits,
    SandboxOutcome,
    SandboxRequest,
    SandboxResult,
)
from coding_opsd.config import ConfigError, config_hash, deep_merge, load_config
from coding_opsd.jag import run_jag_phase0
from coding_opsd.metrics import brier_score, categorical_nll, cosine_similarity, effective_sample_size, paired_bootstrap_ci
from coding_opsd.results import normalize_result
from coding_opsd.runtime import RunWriter, named_rng


class ConfigTests(unittest.TestCase):
    def test_deep_merge_replaces_lists_and_preserves_explicit_null(self) -> None:
        merged = deep_merge(
            {"nested": {"keep": 1, "list": [1, 2], "clear": "value"}},
            {"nested": {"list": [3], "clear": None}},
        )
        self.assertEqual(merged, {"nested": {"keep": 1, "list": [3], "clear": None}})

    def test_deep_merge_rejects_mapping_scalar_conflicts(self) -> None:
        with self.assertRaises(ConfigError):
            deep_merge({"nested": {"a": 1}}, {"nested": 3})
        with self.assertRaises(ConfigError):
            deep_merge({"nested": 3}, {"nested": {"a": 1}})

    def test_load_config_resolves_inheritance_and_removes_base_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.yaml").write_text("a: {x: 1, choices: [base]}\nclear: kept\n", encoding="utf-8")
            child = root / "child.yaml"
            child.write_text("base_config: base.yaml\na: {y: 2, choices: [child]}\nclear: null\n", encoding="utf-8")
            resolved = load_config(child)
            self.assertEqual(resolved.data, {"a": {"x": 1, "y": 2, "choices": ["child"]}, "clear": None})
            self.assertNotIn("base_config", resolved.data)
            self.assertEqual(resolved.sha256, config_hash(resolved.data))
            self.assertEqual(len(resolved.source_paths), 2)
            self.assertEqual(len(resolved.source_hashes), 2)

    def test_load_config_rejects_cycles_and_non_mapping_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.yaml").write_text("base_config: b.yaml\n", encoding="utf-8")
            (root / "b.yaml").write_text("base_config: a.yaml\n", encoding="utf-8")
            (root / "list.yaml").write_text("- not\n- a mapping\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(root / "a.yaml")
            with self.assertRaises(ConfigError):
                load_config(root / "list.yaml")

    def test_resolved_config_access_cannot_mutate_hashed_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text("nested: {items: [one, two]}\n", encoding="utf-8")
            resolved = load_config(config)
            exposed = resolved.data
            exposed["nested"]["items"].append("mutated")
            self.assertEqual(resolved.data, {"nested": {"items": ["one", "two"]}})
            self.assertEqual(resolved.sha256, config_hash(resolved.data))

    def test_hash_and_named_rng_are_stable(self) -> None:
        self.assertEqual(config_hash({"b": 1, "a": [2]}), config_hash({"a": [2], "b": 1}))
        expected = named_rng(17, "first").integers(0, 2**32, size=8)
        _ = named_rng(17, "other").integers(0, 2**32, size=8)
        actual = named_rng(17, "first").integers(0, 2**32, size=8)
        np.testing.assert_array_equal(expected, actual)


class RuntimeTests(unittest.TestCase):
    def test_run_writer_emits_atomic_standard_artifacts_and_refuses_completed_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "schema_version": "0.1",
                "direction": "jag_tree",
                "runtime": {"profile": "smoke"},
                "output": {"identity": "jag_test"},
                "seeds": [1],
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
            }
            digest = config_hash(config)
            run = Path(directory) / f"jag_test-seed1-{digest[:8]}"
            normalized = normalize_result(
                "jag",
                "jag_tree",
                config,
                1,
                run.name,
                run_jag_phase0(config, 1),
            )
            manifest = {
                "schema_version": "phase0.run.v1",
                "run_id": run.name,
                "direction": "jag_tree",
                "direction_alias": "jag",
                "output_identity": "jag_test",
                "seed": 1,
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
                "row_count": len(normalized.rows),
                "event_count": len(normalized.events),
                "array_keys": sorted(normalized.arrays),
            }
            with RunWriter(run, config, manifest) as writer:
                writer.write_metrics(normalized.metrics)
                writer.write_gates(normalized.gates)
                writer.write_events(normalized.events)
                writer.write_rows(normalized.rows)
                writer.write_npz(normalized.arrays)
                writer.write_report(
                    "# Phase-0 run report\n\n"
                    f"- Run ID: {run.name}\n"
                    "- Direction: jag_tree (jag)\n"
                    "- Seed: 1\n"
                    "- Profile: smoke\n"
                    "- Scientific status: INCOMPLETE\n"
                    f"- Status reason: {normalized.gates['reason']}\n"
                    f"- Rows/events/arrays: {len(normalized.rows)} / "
                    f"{len(normalized.events)} / {len(normalized.arrays)}\n\n"
                    "This is a deterministic finite CPU Phase-0 mechanism experiment. "
                    "It is not a 7B-model result or a coding-benchmark score. COMPLETE "
                    "means that the artifacts are sealed; gates.json is authoritative "
                    "for the scientific status.\n"
                )
                writer.complete()
            self.assertTrue((run / "COMPLETE").is_file())
            self.assertEqual(
                json.loads((run / "metrics.json").read_text(encoding="utf-8")),
                normalized.metrics,
            )
            self.assertEqual((run / "events.jsonl").read_text(encoding="utf-8"), "")
            with self.assertRaises(FileExistsError):
                RunWriter(run, config, manifest)

    def test_run_writer_exclusively_owns_incomplete_run_and_never_replaces_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "schema_version": "0.1",
                "direction": "jag_tree",
                "runtime": {"profile": "smoke"},
                "output": {"identity": "jag_incomplete"},
                "seeds": [1],
                "phase0": {"test_only_registration": True},
            }
            digest = config_hash(config)
            run = Path(directory) / f"jag_incomplete-seed1-{digest[:8]}"
            manifest = {
                "schema_version": "phase0.run.v1",
                "run_id": run.name,
                "direction": "jag_tree",
                "direction_alias": "jag",
                "output_identity": "jag_incomplete",
                "seed": 1,
                "profile": "smoke",
                "status": "INCOMPLETE",
                "config_sha256": digest,
                "source_configs": [{"path": "/registered/config.yaml", "sha256": "a" * 64}],
                "git": {"commit": None, "dirty": None},
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
                "array_keys": ["values"],
            }
            writer = RunWriter(run, config, manifest)
            writer.write_metrics({"score": 1.0})
            with self.assertRaises(FileExistsError):
                writer.write_metrics({"score": 2.0})
            self.assertEqual(json.loads((run / "metrics.json").read_text(encoding="utf-8")), {"score": 1.0})
            with self.assertRaises(FileExistsError):
                RunWriter(run, config, manifest)


class MetricTests(unittest.TestCase):
    def test_common_metrics_and_paired_bootstrap_are_deterministic(self) -> None:
        probabilities = np.array([[1.0, 0.0], [0.2, 0.8]])
        labels = np.array([0, 1])
        self.assertAlmostEqual(categorical_nll(probabilities, labels), -np.log(0.8) / 2)
        self.assertAlmostEqual(brier_score(probabilities, labels), 0.04)
        self.assertAlmostEqual(cosine_similarity([1, 0], [2, 0]), 1.0)
        self.assertAlmostEqual(effective_sample_size([1, 1, 0]), 2.0)
        first = paired_bootstrap_ci([1, 2, 3], [0, 0, 0], seed=9, resamples=1000)
        second = paired_bootstrap_ci([1, 2, 3], [0, 0, 0], seed=9, resamples=1000)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], 2.0)
        self.assertGreaterEqual(first[1], 2.0)
        with self.assertRaises(ValueError):
            paired_bootstrap_ci([1, 2], [0, 0, 0], seed=9)


class AdapterTests(unittest.TestCase):
    def test_fake_adapters_are_runtime_checkable(self) -> None:
        class Policy:
            def score(self, request: PolicyScoreRequest) -> PolicyScoreResult:
                return PolicyScoreResult(request.request_id, request.sample_ids, np.ones(len(request.sample_ids)))

        class Gradients:
            def gradients(self, request: GradientRequest) -> GradientResult:
                return GradientResult(request.request_id, request.sample_ids, request.block_ids, np.zeros((1, 1)))

        class Bank:
            def get_group(self, request: CandidateGroupRequest) -> CandidateGroup:
                return CandidateGroup(request.group_id, request.task_id, (Candidate("candidate-1", "hash", "code"),))

        class Sandbox:
            def evaluate(self, request: SandboxRequest) -> SandboxResult:
                return SandboxResult(request.request_id, (SandboxOutcome(request.candidate_id, "test-1", "PASS"),))

        self.assertIsInstance(Policy(), PolicyBackend)
        self.assertIsInstance(Gradients(), GradientProvider)
        self.assertIsInstance(Bank(), CandidateBank)
        self.assertIsInstance(Sandbox(), SandboxBackend)

        policy_request = PolicyScoreRequest("policy-1", ("sample-1",), ("context",))
        self.assertEqual(Policy().score(policy_request).sample_ids, ("sample-1",))
        gradient_request = GradientRequest("gradient-1", ("sample-1",), ("block-1",), {"reward": 1.0})
        self.assertEqual(Gradients().gradients(gradient_request).block_ids, ("block-1",))
        group_request = CandidateGroupRequest("group-1", "task-1")
        self.assertEqual(Bank().get_group(group_request).group_id, "group-1")
        sandbox_request = SandboxRequest("sandbox-1", "candidate-1", ("test-1",), SandboxLimits(1.0, 64))
        self.assertEqual(Sandbox().evaluate(sandbox_request).outcomes[0].candidate_id, "candidate-1")


if __name__ == "__main__":
    unittest.main()
