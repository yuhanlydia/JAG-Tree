# JAG-Tree 7B Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing Phase-0 monorepo snapshot into a standalone JAG-Tree package with auditable 7B candidate-bank and online-training entry points.

**Architecture:** Preserve the verified finite estimator, remove PBPF/GOAV surfaces, and add strict registries, immutable bank schemas, controlled allocators, optional Hugging Face/PEFT integration, matched-budget replay, statistics and launch configs. CPU CI uses fake backends and never downloads data or weights.

**Tech Stack:** Python 3.11/3.12, NumPy, SciPy, PyYAML; optional PyTorch, Transformers, PEFT, bitsandbytes, datasets, vLLM.

**Spec:** `docs/superpowers/specs/2026-09-08-jag-tree-7b-experiments-design.md`

## Global Constraints

- Package and CLI are `jag_tree` and `jag-tree`; PBPF/GOAV imports and configs are absent.
- CPU base installation remains lightweight and offline; GPU/benchmark dependencies are optional extras.
- Formal configs require full model/data/container revisions, three seeds, immutable outputs and matched budget tolerances.
- Baseline provenance modes are exactly `official_adapter`, `paper_spec_reimplementation`, and `controlled_ablation`.
- No command or README text may claim empirical improvement before formal artifacts exist.

---

### Task 1: Standalone package and strict experiment registry

**Files:** modify `pyproject.toml`, `README.md`; create `src/jag_tree/{__init__,__main__,cli,config,registry}.py`, `tests/test_registry.py`; delete direction-external `src/coding_opsd/{pbpf,goav}` and corresponding tests/configs after JAG equivalents are green.

**Interfaces:** produce `load_experiment(path) -> ExperimentConfig`, `validate_experiment(config) -> None`, and `jag-tree doctor|plan`.

- [ ] Write a failing test asserting `model=qwen25_coder_7b`, dataset roles, allocator names, provenance modes and rejection of `unknown`/mutable formal revisions.
- [ ] Run `pytest -q tests/test_registry.py` and confirm failure because `jag_tree.registry` does not exist.
- [ ] Implement frozen dataclasses and recursive YAML resolution; make `plan` emit canonical JSON plus SHA-256 without importing torch.
- [ ] Run the test and migrate Phase-0 JAG imports to `jag_tree`; run the full suite.
- [ ] Remove only PBPF/GOAV files from this split repository, run `rg 'coding_opsd|pbpf|goav' src tests configs README.md`, and commit `refactor: isolate JAG-Tree package`.

### Task 2: Immutable task/tree banks and budget ledgers

**Files:** create `src/jag_tree/{schema,manifest,bank,ledger,sandbox}.py`, `tests/test_bank_protocol.py`.

**Interfaces:** produce `TaskRecord`, `TreeNode`, `TreeBank`, `RunLedger`, `build_manifest(records, split_seed)`, `verify_bank(path)`, and `SandboxBackend` protocol.

- [ ] Write failing tests for lineage-group split disjointness, canonical hashes, tree parent/version validation, create-once banks, token/program/CPU budget comparisons and separate infrastructure failure.
- [ ] Run `pytest -q tests/test_bank_protocol.py` and confirm missing-module failure.
- [ ] Implement JSONL/NPZ bank read/write with atomic staging, checksums, normalized code/test fingerprints and three ledgers.
- [ ] Implement a deterministic fake sandbox and a subprocess sandbox that requires explicit opt-in; network-disabled container execution remains an external backend interface.
- [ ] Run bank tests and commit `feat: add immutable JAG experiment banks`.

### Task 3: Controlled estimators, allocators and gradient audits

**Files:** create/modify `src/jag_tree/{estimator,allocator,baselines,gradient,audit}.py`, `tests/test_jag_algorithms.py`.

**Interfaces:** produce `recursive_credit(tree)`, `allocate(nodes, moments, budget, arm)`, `score_sketch(model, batch, spec)`, and `audit_replay(bank, arm, seed)`.

- [ ] Write failing equation tests for unique-edge recursive credit, covariance transport, uniform/entropy/value/gradient/TRACE/no-cross/full/oracle scores, permutation invariance and outcome-leak rejection.
- [ ] Run the focused tests and verify the expected missing APIs.
- [ ] Port the already verified finite JAG implementation, then implement every controlled arm behind the same allocator interface and cost ledger.
- [ ] Add tiny-torch opt-in tests comparing registered-block score sketches with direct gradients; exact tests skip cleanly without torch.
- [ ] Run algorithm tests plus Phase-0 replay and commit `feat: implement JAG allocators and audits`.

### Task 4: Benchmark/model adapters and frozen-bank runner

**Files:** create `src/jag_tree/{models,benchmarks,rollout,phase1,statistics}.py`, `tests/test_phase1_runner.py`.

**Interfaces:** produce `PolicyBackend.generate_tree`, `TransformersPolicyBackend`, `load_tasks(spec)`, `run_frozen_audit(config, backend, sandbox)`, and `paired_bootstrap(rows, seed)`.

- [ ] Write failing fake-backend integration tests proving common candidate banks/common random numbers, exact task-level statistics, model revision recording and no data/model download during planning.
- [ ] Run the test and verify missing runner failure.
- [ ] Implement lazy optional imports, Qwen/DeepSeek/Seed chat templates, sequential 16/24GB generation, branch genealogy and registered score masks.
- [ ] Add adapters for JSONL plus optional TACO, EvalPlus, LiveCodeBench and BigCodeBench loaders; required fields fail closed.
- [ ] Run tests and commit `feat: add model-backed JAG frozen audits`.

### Task 5: One-update QLoRA objective and experiment launcher

**Files:** create `src/jag_tree/{loss,trainer,artifacts}.py`, `tests/test_training_contract.py`, `configs/{models,benchmarks,experiments}/`, `scripts/{run_smoke,run_frozen_7b,run_online}.sh`.

**Interfaces:** produce `jag_loss(logprobs, edge_mask, credit, weights)`, `train_one_update`, `run_experiment`, and immutable result artifacts.

- [ ] Write failing tests comparing `jag_loss` gradients with explicit REINFORCE sums and rejecting mismatched masks, repeated edges, extra optimizer epochs and budget drift.
- [ ] Implement the minimal torch/PEFT path with stop-before-download `--dry-run`, QLoRA profiles, checkpoint selection by dev token-AULC and failure gates.
- [ ] Add complete smoke/16GB/24GB/H200 configs and all registered baseline/ablation matrix configs; scripts must pass explicit output roots and seeds.
- [ ] Run `jag-tree plan` over every config and a fake-backend end-to-end smoke.
- [ ] Commit `feat: add staged JAG 7B experiment launcher`.

### Task 6: Reproducibility, documentation and CI

**Files:** modify `README.md`; create `docs/{EXPERIMENTS,BASELINES,DATA,ARTIFACTS}.md`, `.github/workflows/ci.yml`, `tests/test_repository_contract.py`.

**Interfaces:** document exact pull/install/doctor/prepare/run/verify commands and status semantics.

- [ ] Write a failing repository-contract test enumerating required configs, model/dataset/baseline matrices, shell syntax and README no-claim language.
- [ ] Complete documentation with official-versus-reimplementation provenance and expected GPU/CPU envelopes.
- [ ] Run `pytest -q`, `python -m compileall -q src tests`, `git diff --check`, `bash -n scripts/*.sh`, and all config dry-runs.
- [ ] Commit `docs: document reproducible JAG experiments`.

