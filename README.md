# coding_opsd

Executable, deterministic CPU references for three statistically grounded coding-RL-with-verifiable-rewards (RLVR) ideas. The current code answers a narrow question first: do the proposed estimators, posterior approximations, and acquisition objectives behave correctly in controlled finite settings?

**Status (2026-09-07): Phase 0 reference implementation available. No formal Phase 0 result, 7B experiment, benchmark score, or policy-improvement claim is reported.** The checked-in smoke overlays are developer tests; their reports are always `INCOMPLETE` with respect to the preregistered scientific gates.

The three directions are:

1. **JAG-Tree** — genealogy-aware, joint value-gradient allocation for tree rollouts.
2. **PBPF** — predictive belief particle filtering from execution evidence.
3. **GOAV** — gradient-optimal active verification under noisy and costly tests.

Directions 2 and 3 remain separated in mechanism experiments: PBPF consumes a fixed, immutable test order, while GOAV chooses which audit subset to acquire and corrects the resulting selection bias. They are joined only after both isolated mechanisms pass their preregistered gates.

## What runs now

| Direction | Executable Phase 0 boundary | What this does not establish |
|---|---|---|
| JAG | Finite MDP with exact enumeration, recursive tree-gradient estimators, and matched-budget allocation arms | Correctness or benefit on language-model token trees |
| PBPF | Finite affine-program surrogate with exact latent enumeration and particle-filter comparisons under immutable evidence order | A learned semantic latent state, KV intervention, or code-repair gain |
| GOAV | Synthetic noisy-oracle panel with exact subset probabilities, AIPW estimation, and gradient-risk acquisition | Benefit on real generated tests, sandboxes, or online RL |

These are mathematical reference experiments, not miniature benchmark results. Formal counts, arms, and gates remain in `configs/{jag_tree,pbpf,goav}.yaml`; the small overlays under `configs/phase0/` only exercise the implementation quickly. A formal manifest can be validated, but `run` requires a concrete overlay that supplies `runtime.profile` and a safe `output.identity`; no formal-scale execution overlay is checked in.

## Install and inspect

Python 3.11 or 3.12 is supported.

```bash
python -m pip install -e .
python -m coding_opsd --help
```

Resolve an overlay and optionally write the canonical merged configuration. An output file is created exclusively rather than overwritten:

```bash
python -m coding_opsd resolve configs/phase0/jag_smoke.yaml
python -m coding_opsd resolve configs/phase0/jag_smoke.yaml --output /tmp/jag-resolved.json
```

Validate all three developer overlays without running an experiment:

```bash
python -m coding_opsd validate configs/phase0/jag_smoke.yaml
python -m coding_opsd validate configs/phase0/pbpf_smoke.yaml
python -m coding_opsd validate configs/phase0/goav_smoke.yaml
```

## Run Phase 0

Run one direction with an explicit seed and output parent:

```bash
python -m coding_opsd run jag configs/phase0/jag_smoke.yaml --seed 101 --output runs
python -m coding_opsd run pbpf configs/phase0/pbpf_smoke.yaml --seed 101 --output runs
python -m coding_opsd run goav configs/phase0/goav_smoke.yaml --seed 101 --output runs
```

The stable short aliases map exactly as follows: `jag` → `jag_tree`, `pbpf` → `predictive_belief_particle_filter`, and `goav` → `gradient_optimal_active_verification`. A direction/config mismatch is an error.

Run every checked-in developer overlay with one command:

```bash
python -m coding_opsd run-all --profile smoke --seed 101 --output runs
```

[`examples/run_all_phase0.sh`](examples/run_all_phase0.sh) is the equivalent repository-root-safe wrapper. It applies only the checked-in smoke overlays and makes no additional in-memory overrides.

Each single-direction `run` command prints its newly created run directory; `run-all` prints a JSON summary array. Under the chosen output parent, each immutable run ID is:

```text
{identity}-seed{seed}-{configsha8}
```

For example, `runs/jag_phase0_smoke-seed101-1a2b3c4d/`. The suffix is derived from the fully resolved configuration. A run never overwrites an existing directory; use a new output parent, seed, or configuration for a new run.

Verify a completed directory independently:

```bash
python -m coding_opsd verify runs/jag_phase0_smoke-seed101-1a2b3c4d
```

Replace the illustrative hash with the path printed by `run`. Verification checks the artifact contract, checksums, configuration/manifest identity, numeric-array readability without pickles, and direction-specific invariants such as logging a GOAV design before its sampled outcome. For executable smoke runs it also replays the finite experiment from the sealed configuration and seed, then compares the regenerated rows, events, gates, metrics, and arrays. Recomputing checksums alone therefore cannot legitimize a coordinated rewrite of the scientific evidence.

## Output contract and status

A completed run contains this flat artifact set:

| Artifact | Purpose |
|---|---|
| `resolved_config.json` | Canonical deep-merged configuration |
| `manifest.json` | Run identity, seed, direction, versions, and provenance |
| `metrics.json` | Aggregate mechanism metrics |
| `gates.json` | Gate values, completeness, and final status |
| `events.jsonl` | Ordered audit/event log |
| `rows.jsonl` | Per-replication or per-task analysis rows |
| `arrays.npz` | Numeric reference arrays, loadable with `allow_pickle=False` |
| `report.md` | Human-readable summary with scope warnings |
| `checksums.json` | Cryptographic digests for the evidence artifacts |
| `COMPLETE` | Final marker written only after required artifacts exist |

`gates.json` uses one of four statuses:

- `PASS`: all applicable formal gates are present and pass.
- `FAIL`: formal evidence is complete and at least one scientific gate fails.
- `INCOMPLETE`: the run is technically valid but cannot support a formal decision. Every developer smoke run has this status.
- `INVALID`: a configuration, mathematical invariant, execution, or evidence-integrity failure prevents scientific interpretation.

The process exits zero when a command completes normally, including a scientifically `FAIL` or `INCOMPLETE` run. Invalid input, a direction mismatch, an existing run-directory collision, failed integrity verification, or an execution failure exits nonzero. Scripts should therefore read `gates.json.status` for scientific decisions and use the process exit code for operational decisions.

## Reproducibility boundary

The runners derive labeled random streams from the outer seed, so results do not depend on Python hash iteration or incidental call order. Configurations are recursively merged and hashed canonically. Runs are create-once, checksummed, and independently verifiable. CI fixes `PYTHONHASHSEED=0` and single-thread BLAS settings, executes the unit suite and compilation checks, and validates all three smoke overlays without downloading a model or dataset.

This is not yet the shared model-backed harness required by gate G0. The package defines four runtime-checkable integration seams:

- `PolicyBackend` for identified policy-score requests;
- `GradientProvider` for gradients over registered parameter blocks;
- `CandidateBank` for immutable candidate groups;
- `SandboxBackend` for vector-valued test outcomes under explicit limits.

No implementation is currently provided for Transformers, vLLM, PEFT, a LoRA trainer, KV-cache injection or steering, benchmark ingestion, candidate generation, or Docker/container sandboxes. There are likewise no model weights, benchmark datasets, or cached model outputs in this repository.

Phase 1 must connect a frozen model and independently generated candidate bank, train only the small registered modules, and repeat exact/full-information audits. Phase 2 must add online LoRA RL, token/execution/GPU ledgers, a deterministic sandbox, leakage controls, and locked evaluation. Until those pieces pass G0, the project makes no 7B or coding-benchmark claim.

## Repository map

- [`src/coding_opsd/`](src/coding_opsd/): config/runtime core, typed integration seams, direction-specific mathematical operators, and Phase 0 runners.
- [`tests/`](tests/): deterministic unit, estimator, invariant, and CLI integration tests.
- [`configs/shared.yaml`](configs/shared.yaml): shared preregistration defaults.
- [`configs/jag_tree.yaml`](configs/jag_tree.yaml), [`configs/pbpf.yaml`](configs/pbpf.yaml), [`configs/goav.yaml`](configs/goav.yaml): unchanged formal experiment sizes, arms, and gates plus links to executable overlays.
- [`configs/phase0/`](configs/phase0/): tiny developer smoke overlays; never formal evidence.
- [`docs/00_shared_protocol.md`](docs/00_shared_protocol.md): common model, data, leakage, budget, statistics, and reporting rules.
- [`docs/01_jag_tree.md`](docs/01_jag_tree.md): JAG algorithm and experiment plan.
- [`docs/02_predictive_belief_particles.md`](docs/02_predictive_belief_particles.md): PBPF algorithm and experiment plan.
- [`docs/03_gradient_optimal_verification.md`](docs/03_gradient_optimal_verification.md): GOAV algorithm and experiment plan.
- [`docs/04_benchmarks_and_compute.md`](docs/04_benchmarks_and_compute.md): benchmark roles, visibility firewall, hardware tiers, and cost accounting.
- [`docs/05_execution_roadmap.md`](docs/05_execution_roadmap.md): ordered gates and go/stop decisions.
- [`docs/06_upstream_code_map.md`](docs/06_upstream_code_map.md): paper/code provenance, reuse boundaries, and exact fork points.
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml): CPU-only Python 3.11/3.12 verification.

## Research policy

The project does **not** treat “more tests,” easy-to-hard ordering, token entropy, KV-cache injection, or coder-tester competition as sufficient algorithmic novelty. The common target is reliable policy improvement under fixed generation and verification budgets.

Every eventual claim must report task-level confidence intervals and individual seeds; generated/scored tokens, sandbox CPU-seconds, GPU-hours, and wall time; the direction-specific mechanism metric; and an unpublished or strictly post-cutoff evaluation suite whose tasks, gold artifacts, tests, and per-task feedback were never exposed to training or checkpoint selection.

Closest reusable projects and exact reuse boundaries are catalogued in [`docs/06_upstream_code_map.md`](docs/06_upstream_code_map.md). Licences and upstream commits must be checked and pinned before code is copied; paper-only baselines must be labelled as reimplementations.
