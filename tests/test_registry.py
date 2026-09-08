from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from jag_tree.cli import main
from jag_tree.config import ConfigError, ExperimentConfig, load_experiment, validate_experiment
from jag_tree.registry import ALLOCATOR_ARMS, DATASET_ROLES, MODELS, PROVENANCE_MODES


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_registry_contains_preregistered_models_roles_arms_and_modes() -> None:
    assert MODELS["qwen25_coder_7b"].hf_id == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert DATASET_ROLES == frozenset({"train", "dev", "frozen_audit", "public_eval", "locked_final"})
    assert {"uniform", "entropy", "value_variance", "gradient_only", "trace_score", "jag_no_cross", "jag_full", "oracle_moments"} <= ALLOCATOR_ARMS
    assert PROVENANCE_MODES == frozenset({"unavailable", "controlled_ablation"})


def test_recursive_resolution_is_frozen_and_canonical(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("model: {name: qwen25_coder_7b, revision: 0123456789abcdef0123456789abcdef01234567}\nformal: false\ndatasets: []\nallocator: uniform\n", encoding="utf-8")
    child = _write(tmp_path, "base_config: base.yaml\nallocator: jag_full\n")
    config = load_experiment(child)
    assert isinstance(config, ExperimentConfig)
    assert config.data["allocator"] == "jag_full"
    with pytest.raises(TypeError):
        config.data["allocator"] = "uniform"  # type: ignore[index]
    assert main(["plan", str(child)]) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["config"]["allocator"] == "jag_full"
    assert len(rendered["sha256"]) == 64


@pytest.mark.parametrize("field,value", [("allocator", "unknown"), ("model.name", "unknown")])
def test_unknown_registry_values_fail_closed(tmp_path: Path, field: str, value: str) -> None:
    model_name = value if field == "model.name" else "qwen25_coder_7b"
    allocator = value if field == "allocator" else "uniform"
    path = _write(tmp_path, f"model: {{name: {model_name}, revision: 0123456789abcdef0123456789abcdef01234567}}\nformal: false\ndatasets: []\nallocator: {allocator}\n")
    with pytest.raises(ConfigError, match="unknown"):
        validate_experiment(load_experiment(path))


def test_formal_config_rejects_mutable_revision_and_missing_roles(tmp_path: Path) -> None:
    path = _write(tmp_path, "model: {name: qwen25_coder_7b, revision: main}\nformal: true\ndatasets: [{name: taco, role: train, revision: main}]\nallocator: jag_full\ncontainer_digest: sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nseeds: [101, 202, 303]\n")
    with pytest.raises(ConfigError, match="immutable.*revision"):
        validate_experiment(load_experiment(path))


def _formal_config() -> dict[str, object]:
    revision = "a" * 40
    return {
        "stage": "online", "formal": True,
        "model": {"name": "qwen25_coder_7b", "revision": revision},
        "hardware": "h200_formal", "allocator": "jag_full",
        "arms": ["uniform", "jag_full"],
        "datasets": [{"name": "taco", "role": role, "revision": revision} for role in ("train", "dev", "frozen_audit", "public_eval", "locked_final")],
        "tasks": {"kind": "taco", "role": "train", "revision": revision},
        "container_digest": "sha256:" + "b" * 64, "seeds": [101, 202, 303],
        "generation": {"root_samples": 4, "branch_depths": [256, 512, 768], "children_per_branch": 3, "max_new_tokens": 2048},
        "budget": 8192,
        "training": {"warmup_updates": 15, "screening_updates": 200, "formal_updates": 800, "prompt_batch": 64, "group_cap": 8, "optimizer_epochs": 1},
        "gradient_audit": {"predictor_sha256": "e" * 64, "calibration_role": "calibration", "cross_fit_folds": 5},
        "prerequisites": {"e0_artifact_sha256": "c" * 64, "e1_artifact_sha256": "d" * 64, "e0_gate_status": "PASS", "e1_gate_status": "PASS"},
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: c["model"].update(name="qwen25_coder_1_5b"), "primary model"),
        (lambda c: c.update(hardware="cpu"), "h200_formal"),
        (lambda c: c["tasks"].update(revision="main"), "active task revision"),
        (lambda c: c.update(arms=["jag_full", "invented"]), "unknown allocator"),
        (lambda c: c["training"].update(formal_updates=100000), "formal_updates"),
        (lambda c: c["training"].update(updates=1), "updates override"),
        (lambda c: c["prerequisites"].update(e1_gate_status="INCOMPLETE"), "E1 gate"),
        (lambda c: c["gradient_audit"].update(calibration_role="frozen_audit"), "calibration role"),
    ],
)
def test_formal_online_validation_rejects_irrelevant_pins_and_profile_escape(tmp_path: Path, mutation, message: str) -> None:  # type: ignore[no-untyped-def]
    config = _formal_config()
    mutation(config)
    path = tmp_path / "formal.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        validate_experiment(load_experiment(path))


def test_registered_formal_online_contract_validates(tmp_path: Path) -> None:
    path = tmp_path / "formal.yaml"
    path.write_text(yaml.safe_dump(_formal_config()), encoding="utf-8")
    validate_experiment(load_experiment(path))
