# JAG-Tree 7B experiment repository implementation report

Date: 2026-09-08
Binding design: `docs/superpowers/specs/2026-09-08-jag-tree-7b-experiments-design.md`
Binding plan: `docs/superpowers/plans/2026-09-08-jag-tree-7b-experiments.md`
Starting commit: `89b6089`
Implementation commit range: `aad6f24..a418d65` (plus final report-only commits)

## Outcome

The repository is now a standalone `jag_tree` package with the `jag-tree` CLI. The verified finite JAG estimator/allocator was preserved under the new namespace. Direction-external package, config, test, example, and obsolete shared documentation surfaces were removed after the standalone registry and Phase-0 tests were green.

The base installation is offline and lightweight. Torch, Transformers, PEFT, bitsandbytes, datasets, and EvalPlus are optional extras and are imported only by their execution boundaries. No weights or benchmark datasets were downloaded during implementation or verification.

No empirical improvement is reported. Smoke and pilot artifacts are explicitly `INCOMPLETE` and cannot satisfy formal gates.

## Commits

| Commit | Plan task | Summary |
|---|---|---|
| `aad6f24` | 1 | Standalone package, strict registries/config resolution, `doctor`/`plan`, Phase-0 namespace migration, removal of external directions |
| `674696a` | 2 | Immutable task/tree banks, lineage manifests, three budget ledgers, sandbox protocol/backends |
| `a5db37b` | 3 | Recursive-credit public API, all controlled allocator arms, permutation/outcome controls, score sketches and replay audits |
| `663027b` | 4 | Lazy model and benchmark adapters, generation contracts, common-bank frozen audit, task-level bootstrap |
| `be94699` | 5 | Unique-edge loss, one-update QLoRA contract, immutable artifacts, staged launcher, configs and scripts |
| `a619a9d` | 6 | Reproducibility documentation, repository contract, offline Python 3.11/3.12 CI |
| `6a2a057` | hardening | Removed the optional-extra self-dependency and rejected candidate-bank reuse with a changed task manifest |
| `37af3b6` | review 1 | Correct recursive branching and incremental unique-edge accounting |
| `0f7616e` | review 1 | Deep-freeze arrays/manifests and bind exact bank/generation identity |
| `d2163b7` | review 1 | Strict formal gate plus content-derived cross-role deduplication |
| `7f67190` | review 1 | Execute task tests only in trusted fixtures or a bounded networkless container |
| `d6dfc07` | review 1 | Outcome-blind fixed predictor and measured-cost arm replay |
| `456003c` | review 1 | Support-positive stochastic audit and recursive unique-edge training |
| `2fa9df7` | review 1 | Fail-closed evidence, connected statistics, unavailable baseline demotion |
| `90f5b04` | review 2 | Close infrastructure, sandbox, selected-edge, immutable-array, formal-gate and evidence bypasses |
| `a2d4242` | review 2 | Bind/recompute gates, seal splits pre-generation and stream policy sketches |
| `89931b4` | review 3 | Nonce sandbox protocol, replay-weighted training, typed PASS evidence, EOS terminal propagation |
| `3aaeaa4` | final review | Remove stdout attestation, disable unsupported scientific PASS, correct oracle proposal and launcher contract |
| `a418d65` | final sandbox review | Disable real generated-code execution until a trusted external verifier exists |

## TDD red/green evidence

### Task 1

- RED: `python -m pytest -q tests/test_registry.py` failed during collection with `ModuleNotFoundError: No module named 'jag_tree'`.
- GREEN: `python -m pytest -q tests/test_registry.py tests/test_jag.py` passed: 42 tests and 17 subtests.
- Boundary check: `rg 'coding_opsd|pbpf|goav' src tests configs README.md` returned no matches after migration/removal.

### Task 2

- RED: `python -m pytest -q tests/test_bank_protocol.py` failed during collection with `ModuleNotFoundError: No module named 'jag_tree.bank'`.
- GREEN: focused bank tests passed 4/4; then the full suite passed 46 tests and 17 subtests.
- Covered lineage-role isolation, canonical AST/test identities, parent/depth/revision validation, create-once atomic banks, NPZ loading without pickle, checksum tamper rejection, separate ledgers, infrastructure outcomes, deterministic fake sandbox, and subprocess opt-in.

### Task 3

- RED: `python -m pytest -q tests/test_jag_algorithms.py` failed because `AllocationNode` and the unified allocator contract were absent.
- GREEN: focused algorithm plus Phase-0 tests passed 48 tests, with one optional Torch skip and 17 subtests; full suite passed 57 tests with one optional skip.
- Literal tests cover unique-edge recursion; uniform, entropy, value, gradient, trace, no-cross, full-joint and oracle scores; covariance cross terms; permutation invariance; and outcome-leak rejection.

### Task 4

- RED: `python -m pytest -q tests/test_phase1_runner.py` failed during collection with missing `jag_tree.benchmarks`.
- GREEN: focused runner tests passed 5/5; full suite passed 62 tests with one optional skip and 17 subtests.
- A second RED/GREEN cycle proved only leaf programs are executed. A final hardening cycle proved a changed task manifest cannot reuse an existing candidate bank.

### Task 5

- RED: `python -m pytest -q tests/test_training_contract.py` failed during collection with missing `jag_tree.artifacts`.
- GREEN: focused tests passed 4 tests with one optional Torch skip. Six experiment configs planned successfully; shell syntax passed; the fake backend smoke created and independently verified immutable artifacts.
- Additional RED/GREEN cycles added CLI `run --dry-run`, required a training-capable backend for online execution, and proved that two configured online updates use two distinct bank paths and exactly two backend update calls.

### Task 6

- RED: repository-contract tests reported missing `EXPERIMENTS`, `BASELINES`, `DATA`, and `ARTIFACTS` documents and stale generated namespace text; a separate RED cycle showed `prepare` was not a CLI command.
- GREEN: repository-contract tests passed after documentation, create-once preparation, pinned provenance, CI, and tracked-file boundary checks were implemented.
- A dependency-contract RED/GREEN cycle caught and removed the training extra's self-dependency.

## Final verification evidence

The following commands were run from the committed repository after all implementation commits:

```text
python -m pytest -q
  99 passed, 2 skipped, 17 subtests passed in 4.71s

python -m compileall -q src tests
  exit 0

git diff --check
  exit 0

bash -n scripts/*.sh
  exit 0

for config in configs/experiments/*.yaml; do python -m jag_tree plan "$config"; done
  6 configs planned, exit 0

python -m jag_tree doctor
  Python 3.12.13; torch=false; transformers=false

scripts/run_smoke.sh <temporary-output> 7
python -m jag_tree verify <temporary-output>/run-*
  offline smoke verified, exit 0

rg 'coding_opsd|pbpf|goav' src tests configs README.md
  no matches

placeholder scan over advertised source/config/script/document paths
  no TODO, TBD, or NotImplemented markers
```

The two skips are the explicitly optional real-Torch gradient-sketch and loss-gradient tests. They skip cleanly because Torch is not installed in the CPU base environment.

## Requirements coverage

- Closed registries include all four models, benchmark roles, eight allocator arms, executable controlled ablations, explicitly unavailable recipes, and five hardware profiles.
- Formal validation requires exact primary-model/hardware/tree/budget/update settings, full active model/task/data revisions, all dataset roles, a SHA-256 container digest, seeds 101/202/303, sealed calibration metadata, and verified E0/E1 prerequisites.
- JSONL plus lazy TACO, LiveCodeBench, BigCodeBench, HumanEval+, and MBPP+ adapters are implemented; missing normalized fields fail closed.
- Model configs use full observed Hugging Face commit SHAs. Public benchmark configs use full observed dataset commit SHAs.
- TreeRL, TreePO, TreeRPO, Tree-GRPO, GRPO, DAPO and VIP recipes are explicitly unavailable and rejected because this repository has no executable verified adapters for them.
- Online execution creates a new immutable bank per update. The Transformers backend builds recursive, globally weighted unique-edge QLoRA targets from original prompts, preserves the adapter/optimizer across updates, and saves immutable checkpoints.
- Frozen audits use support-positive stochastic proposals, common seeded uniform streams, explicit importance weights and measured token costs. Arm-specific rows retain MC gradient MSE/bias/cosine/SE and separate ledgers.
- CPU CI is offline, runs on Python 3.11/3.12, executes the full suite and compilation checks, plans all experiment configs, checks shell syntax, and verifies the fake-backend smoke.

## Concerns and unverified optional paths

1. GPU/model-backed execution was not run because the base environment intentionally lacks Torch/Transformers/PEFT and no large models or datasets were downloaded. The two optional Torch tests skipped. Model gradient geometry, QLoRA compatibility, checkpoint recovery, GPU-hours, and 7B memory envelopes remain unverified.
2. `configs/experiments/online_h200.yaml` is deliberately `formal: false`. No sealed cross-fit 7B calibration predictor or verified E0/E1 artifact exists in this CPU round; the loader and validation fail closed rather than fabricate one.
3. Container command construction is tested, but no immutable runtime image was supplied, so generated-code container execution was not exercised end-to-end here. The connected smoke uses only the explicitly fake sandbox/backend and remains `INCOMPLETE`.
4. Statistical utilities and replay evidence are connected to immutable rows, but no formal benchmark report aggregator or 50,000-draw GPU artifact was generated. Public benchmark interfaces were not downloaded or executed.
5. The first post-commit full-suite run hit the preserved Phase-0 wall-clock smoke guard once (10.56s versus 8s) under host contention. The exact test then passed in isolation and the full suite passed immediately afterward; no mathematical assertion failed, but the timing guard remains environment-sensitive.

## Review-round red/green evidence

- RED branching regression: registered 4x3x3x3 topology returned four leaves and cumulative-token totals; GREEN: 108 leaves with each conditional edge counted once.
- RED frozen audit: missing typed predictor import, reward-sensitive allocation and unavailable-edge oversubscription; GREEN: flipped outcomes preserve non-oracle proposals, budgets are measured, arms retain distinct laws/rows/ledgers.
- RED stochastic audit: seed produced identical deterministic selections and rows lacked proposal/importance fields; GREEN: independent seeds vary, CRN uniforms are recorded, and a 20,000-draw known finite target converges.
- RED recursive training: missing `recursive_training_targets`; GREEN: analytical uneven-tree global REINFORCE total is 2.25 and every edge mask column is used exactly once.
- RED evidence: PASS without gates and false official-adapter modes were accepted; GREEN: PASS fails closed without complete gates/exact inventory, and all unimplemented recipes are unavailable.
- RED connected smoke: direct launcher returned permission denied; GREEN: all launchers are executable and fake offline run plus independent verification exits zero.

## Review-round 2 boundary hardening

- Infrastructure failures now abort bank creation and cannot become zero rewards; generation and verification are recorded separately from hypothetical replay sampling.
- Replay reports `per_draw_gradient_mse` and the estimator-of-mean `gradient_mse = per_draw_gradient_mse / draw_count`; rows retain proposal weights and CRN uniforms.
- Online training is constrained to the selected replay edge set and its independently measured unique-edge ledger. Zero-budget online runs, unactivated non-plumbing online models, matrix-only recipes, and silently downgraded formal launch templates fail before training.
- Formal execution verifies complete PASS artifacts and their checksum files at configured paths; formal configs reject the `training.updates` alias.
- The test harness requires its post-test marker, resists `os._exit(0)`, and executes normalized TACO-style JSON input/output cases instead of treating a JSON expression as a test.
- Bank score arrays use immutable-bytes backing, so `setflags(write=True)` fails.
- Hierarchical bootstrap resamples a shared global seed vector across sampled tasks, preserving seed-correlated effects.
- Split/dedup manifests are constructed before any generation call; PASS gate claims are recomputed from bound evidence rows; normal score-sketch execution no longer retains dense gradients.

## Review-round 3 adversarial hardening

- Real generated-code execution is explicitly unavailable: subprocess and OCI backends return infrastructure failure without executing candidate code because no trusted external verifier is shipped. Exact caller-frame attacks against both string sentinels and randomized integer exit codes therefore cannot PASS. Only deterministic `FakeSandbox` fixtures may return PASS.
- Online policy coefficients aggregate every replay draw's multiplicity and target/proposal importance weight over its full genealogy path, then differentiate each resulting unique edge once. A CPU regression verifies the weighted coefficients sum to the registered trajectory estimator.
- Scientific PASS verification is explicitly unsupported in this CPU release. Every PASS artifact fails closed, preventing invented statistics, thresholds, identities, or tiny banks from being promoted without a registered formal evaluator.
- Frozen predictor covariance arrays use immutable bytes backing, and genealogy terminal callbacks prevent descendants after EOS.

## Final blocker verification

- Literal caller-frame `co_consts` attacks for both sentinel strings and integer exit codes are regression tests; real backends classify them as infrastructure failures without execution.
- Scientific PASS verification always fails with an explicit unsupported error. Consequently invented statistics/thresholds/configs/banks cannot activate E1/E2 in this release.
- Oracle trajectory proposals are proportional to full-information contribution magnitude under the fixed-draw design. The unequal-cost/equal-contribution regression produces equal 0.5 proposals and zero per-draw gradient MSE.
- Model launch documentation now supplies container image and config arguments; online launch requires an explicit formal overlay and still fails closed because PASS verification is unavailable.
- `git diff --check 89b6089` is clean. A wheel was built without dependency downloads, installed into an isolated target, and imported successfully.
