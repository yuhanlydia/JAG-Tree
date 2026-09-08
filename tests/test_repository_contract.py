from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tomllib

import yaml

from jag_tree.cli import main


ROOT = Path(__file__).resolve().parents[1]


def test_required_experiment_model_benchmark_and_script_surface_exists() -> None:
    required = {
        "configs/experiments/smoke.yaml", "configs/experiments/frozen_16gb.yaml", "configs/experiments/frozen_24gb.yaml", "configs/experiments/online_h200.yaml",
        "configs/experiments/baseline_matrix.yaml", "configs/experiments/ablation_matrix.yaml",
        "configs/models/qwen25_coder_1_5b.yaml", "configs/models/qwen25_coder_7b.yaml", "configs/models/seed_coder_8b.yaml", "configs/models/deepseek_coder_6_7b.yaml",
        "configs/benchmarks/taco.yaml", "configs/benchmarks/public_evaluation.yaml",
        "scripts/run_smoke.sh", "scripts/run_frozen_7b.sh", "scripts/run_online.sh",
        "docs/EXPERIMENTS.md", "docs/BASELINES.md", "docs/DATA.md", "docs/ARTIFACTS.md",
    }
    assert not sorted(path for path in required if not (ROOT / path).is_file())


def test_registered_matrices_are_complete_and_provenance_modes_are_closed() -> None:
    baseline = yaml.safe_load((ROOT / "configs/experiments/baseline_matrix.yaml").read_text())
    controlled = {row["name"] for row in baseline["baseline_matrix"]["controlled"]}
    recipes = {row["name"] for row in baseline["baseline_matrix"]["recipes"]}
    assert controlled == {"uniform", "entropy", "value_variance", "gradient_only", "trace_score", "jag_no_cross", "jag_full", "oracle_moments"}
    assert recipes == {"base_checkpoint", "flat_grpo", "dapo", "treepo", "treerl_eptree", "tree_grpo", "treerpo", "vip"}
    assert {row["mode"] for rows in baseline["baseline_matrix"].values() for row in rows} <= {"unavailable", "controlled_ablation"}
    assert all(row.get("executable") is False for row in baseline["baseline_matrix"]["recipes"])
    public = yaml.safe_load((ROOT / "configs/benchmarks/public_evaluation.yaml").read_text())
    assert {row["kind"] for row in public["benchmarks"]} == {"livecodebench_v6", "bigcodebench_full", "bigcodebench_hard", "humaneval_plus", "mbpp_plus", "evaluator_held"}


def test_training_extra_is_directly_installable_without_self_dependency() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    training = project["optional-dependencies"]["training"]
    assert not any(requirement.startswith("jag-tree") for requirement in training)
    assert {requirement.split(">=")[0] for requirement in training} >= {"torch", "transformers", "peft", "bitsandbytes", "datasets"}


def test_scripts_are_valid_and_pass_explicit_output_and_seed() -> None:
    scripts = sorted((ROOT / "scripts").glob("*.sh"))
    subprocess.run(["bash", "-n", *map(str, scripts)], check=True)
    for script in scripts:
        text = script.read_text()
        assert "--output-root" in text
        assert "--seed" in text
        if script.name != "run_smoke.sh":
            assert "--sandbox container" in text
            assert "--allow-subprocess" not in text


def test_prepare_writes_create_once_canonical_plan(tmp_path: Path) -> None:
    output = tmp_path / "prepared.json"
    config = ROOT / "configs/experiments/smoke.yaml"
    assert main(["prepare", str(config), "--output", str(output)]) == 0
    assert output.is_file()
    assert main(["prepare", str(config), "--output", str(output)]) == 2


def test_readme_has_status_guard_and_no_empirical_superiority_claim() -> None:
    text = (ROOT / "README.md").read_text().lower()
    assert "no 7b result" in text
    assert "not evidence" in text
    forbidden = re.compile(r"jag-tree\s+(?:outperforms|beats|improves|achieves state-of-the-art)")
    assert forbidden.search(text) is None


def test_advertised_paths_have_no_implementation_placeholders_or_external_packages() -> None:
    files = subprocess.check_output(["git", "ls-files", "src/*.py", "configs/*.yaml", "scripts/*.sh", "README.md"], cwd=ROOT, text=True).splitlines()
    text = "\n".join((ROOT / file).read_text(errors="replace") for file in files)
    placeholders = "|".join(("TO" + "DO", "T" + "BD", "Not" + "Implemented"))
    external = "|".join(("coding" + "_" + "opsd", "p" + "bpf", "go" + "av"))
    assert not re.search(rf"\b(?:{placeholders})\b", text)
    assert not re.search(rf"\b(?:{external})\b", text, re.IGNORECASE)
