# Three-Direction Phase-0 Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, deterministic Python reference implementation of the mathematical core and complete CPU Phase-0 experiments for JAG-Tree, PBPF, and GOAV, with stable adapter contracts for later 7B integrations.

**Architecture:** One dependency-light `coding_opsd` package owns configuration, deterministic runtime, metrics, artifacts, and adapter protocols. Each direction is an isolated plugin that exposes pure mathematical operators plus an end-to-end Phase-0 runner. The CLI resolves a checked-in formal or smoke overlay, creates an immutable run directory, and emits machine-readable metrics and a gate report without claiming that smoke-scale results satisfy formal scientific gates.

**Tech Stack:** Python 3.11+, NumPy, SciPy, PyYAML, standard-library `unittest`; optional future backends implement protocols without becoming Phase-0 dependencies.

**Spec:** `docs/00_shared_protocol.md`, `docs/01_jag_tree.md`, `docs/02_predictive_belief_particles.md`, `docs/03_gradient_optimal_verification.md`

## Global Constraints

- Phase 0 is CPU-only and imports no Transformers, vLLM, PEFT, Docker SDK, or benchmark package.
- Every stochastic component uses a named NumPy RNG derived from a stable SHA-256 label and outer seed.
- Lists replace lists during config inheritance; mappings merge recursively; explicit null overrides; inheritance cycles and mapping/scalar type changes fail closed.
- Primary estimators remain on-policy, unclipped, unnormalized, and one-step; no implementation silently changes the estimand.
- PBPF uses a fixed test order with all inputs known and outputs hidden; it never selects, reorders, skips, repeats, synthesizes, or adaptively stops tests.
- GOAV enumerates subsets in integer-bitmask order, computes exact first/second inclusion probabilities before revealing labels, and never clips or self-normalizes AIPW weights.
- Formal settings remain in the preregistration files. Checked-in smoke overlays are explicit scientific configs with their own hashes and cannot be reported as formal evidence.
- Phase-1/2 model generation and training remain external integrations; this repository supplies typed policy, gradient, candidate-bank, and sandbox contracts plus array-level estimator operators.

## Rulings that make the manifests executable

- JAG's CPU environment uses shared softmax parameters `logit(a|u)=theta[a] + W[a]·phi(u)` and four deterministic reward families over length-eight action paths. Parameter sharing prevents the registered value-gradient cross term from becoming structurally zero. Child draws are with replacement, matching the IID-child covariance assumption.
- JAG's learned Phase-0 allocator is an online ridge predictor trained only on completed earlier environment seeds; oracle moments never enter its features.
- PBPF uses 64 globally fixed affine-modulo-64 hypotheses `h(x)=(a*x+b) mod 64`, with eight unique odd multipliers and eight offsets. Programs are immutable 64-output tables, so normalized depth is one.
- PBPF marginalizes a candidate source hypothesis before the mutation kernel. Each source equals the task hypothesis with probability `0.20` and otherwise is uniform over its eight-member affine-operator family. This prevents the fully visible candidate table from making the registered prefix-4 mixture comparison structurally degenerate.
- PBPF's pinned bug prior is `[0.20, 0.12, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11]`. The eight kernels are correct, output shift +1, output shift -1, boundary flip, argument permutation, masked branch, index shift, and exception injection; each has an explicit finite site set.
- PBPF scores marginal categorical NLL per candidate-test. The prior gate compares the unconditional uniform-hypothesis predictive with the posterior already conditioned on observed candidate programs. MAP means the marginal-H MAP with smallest-ID tie breaking. Predictive probabilities use the registered `1e-8` floor followed by renormalization.
- GOAV uses additive unit audit costs in Phase 0. Its symmetric exploration distribution is iid Bernoulli at the registered budget fraction, mixed at `floor / budget_fraction`, making the exploration contribution to each marginal exactly the registered floor.
- GOAV enforces expected budget per group in Phase 0; a later global-batch solver may share the dual without changing the single-group operator. The deterministic 100% ceiling is explicitly exempt from full support.
- GOAV synthetic labels are independent of score geometry conditional on the task-success latent. One beta-distributed false-negative and false-positive rate is shared by every observation in a registered test cluster, which yields the requested within-cluster correlation model.

---

### Task 1: Package, configuration, runtime, metrics, and adapter contracts

**Files:**
- Create: `pyproject.toml`
- Create: `src/coding_opsd/{__init__,__main__,cli,config,runtime,metrics,adapters}.py`
- Create: `configs/phase0/{jag_smoke,pbpf_smoke,goav_smoke}.yaml`
- Test: `tests/test_config_runtime.py`

**Interfaces:**
- Produces `ResolvedConfig`, `load_config(path)`, `deep_merge(base, child)`, `config_hash(data)`, `named_rng(seed, label)`, `RunWriter`, common metrics, and four adapter protocols.
- `RunWriter` writes `resolved_config.json`, `manifest.json`, `metrics.json`, `gates.json`, `events.jsonl`, `report.md`, and `COMPLETE` through atomic replacement and refuses to overwrite completed runs.

- [ ] Write tests that demonstrate list replacement, explicit-null override, type-conflict/cycle rejection, stable hashing/RNG streams, atomic run completion, completed-run overwrite rejection, and fake adapter conformance.
- [ ] Run `python -m unittest tests.test_config_runtime -v`; verify failures are caused by the missing package.
- [ ] Implement only the APIs exercised by those tests and the CLI parser skeleton.
- [ ] Re-run the test module and confirm it passes.
- [ ] Commit the foundation and record the exact test output.

### Task 2: JAG-Tree finite-MDP implementation

**Files:**
- Create: `src/coding_opsd/jag/{__init__,mdp,estimator,allocator,experiment}.py`
- Test: `tests/test_jag.py`

**Interfaces:**
- `FiniteMDP` exposes path probabilities, rewards, score vectors, exact value/gradient, conditional moments, and seeded child sampling.
- `recursive_estimate(node, baseline)` implements the bottom-up value/gradient recursion.
- `joint_risk(covariance, accumulated_score)` includes both value-gradient cross terms.
- `greedy_allocate(nodes, budget)` uses squared ancestor transport and `b(b+1)` marginal cost; `integer_oracle` exhaustively solves tiny fixtures.
- `run_jag_phase0(config, seed)` evaluates registered arms and returns rows plus gate inputs.

- [ ] Write hand-derived tests for softmax path probabilities, exact gradient versus central finite differences, recursive unbiasedness on a two-step tree, the cross-covariance risk term, squared transport, and greedy/integer allocation on a literal fixture.
- [ ] Run `python -m unittest tests.test_jag -v`; verify expected missing-feature failures.
- [ ] Implement the MDP oracle, recursive estimator, allocation operators, and Phase-0 runner.
- [ ] Re-run the JAG tests and a smoke CLI run; confirm deterministic artifacts.
- [ ] Commit and record test/smoke output.

### Task 3: PBPF finite belief implementation

**Files:**
- Create: `src/coding_opsd/pbpf/{__init__,dsl,posterior,particles,experiment}.py`
- Test: `tests/test_pbpf.py`

**Interfaces:**
- `Program`, `Episode`, `Outcome`, mutation/inverse-mutation APIs, and immutable prefix views implement the finite model.
- `exact_posterior`, `sequential_update`, `predictive`, and `brute_force_h_marginal` expose exact inference.
- `ParticleFilter.initialize/observe/predict` uses log weights, ESS-triggered systematic resampling, and one independence-MH rejuvenation step.
- `randomized_hpd` returns inclusion probabilities whose posterior-weighted expected coverage is exactly the requested mass.
- `run_pbpf_phase0(config, seed)` emits fixed-prefix, non-teacher-forced NLL/Brier/coverage diagnostics.

- [ ] Write tests for 64 unique truth tables, mutation round trips, future-evidence isolation, posterior normalization, batch/sequential equality, brute-force/factorized equality, explicit predictive marginalization, candidate-permutation equivariance, systematic resampling, and randomized-HPD boundary behavior.
- [ ] Run `python -m unittest tests.test_pbpf -v`; verify expected missing-feature failures.
- [ ] Implement the finite DSL, exact inference, SMC, diagnostics, baselines, and Phase-0 runner.
- [ ] Re-run PBPF tests and a smoke CLI run; confirm fixed-prefix predictions never consume suffix outcomes.
- [ ] Commit and record test/smoke output.

### Task 4: GOAV exact subset-design implementation

**Files:**
- Create: `src/coding_opsd/goav/{__init__,subsets,estimator,noise,solver,experiment}.py`
- Test: `tests/test_goav.py`

**Interfaces:**
- `loo_influence(scores)`, `aipw_pseudolabel`, and `aipw_gradient` implement the fixed linear target.
- `enumerate_subsets`, `inclusion_probabilities`, `SubsetDesign`, and `sample_subset` expose auditable bitmask-ordered randomized designs.
- `design_risk` and `exact_realized_design_mse` implement analytic and explicit-enumeration forms.
- `poisson_neyman_design`, `score_design`, and `solve_goav` enforce exact expected cost and inclusion floors; the latter performs registered mirror steps/restarts and returns the lowest-risk feasible design.
- `oracle_joint_posterior` enumerates all 256 label vectors under the beta-binomial correlated-noise model.
- `run_goav_phase0(config, seed)` evaluates paired subset draws and returns nMSE, cosine, bias, ESS, cost, risk calibration, and gates.

- [ ] Write tests for bitmask order, exact `pi/pi2`, Frechet bounds, LOO equivalence, finite-group AIPW unbiasedness by full enumeration, analytic-risk equality to explicit MSE, Poisson off-diagonal independence, budget/floor feasibility, deterministic solver, and correlated-noise marginals.
- [ ] Run `python -m unittest tests.test_goav -v`; verify expected missing-feature failures.
- [ ] Implement subset arithmetic, estimators, designs/solver, oracle generator/posterior, and Phase-0 runner.
- [ ] Re-run GOAV tests and a smoke CLI run; persist the probability vector and hash before sampled labels.
- [ ] Commit and record test/smoke output.

### Task 5: CLI integration, reporting, documentation, and CI

**Files:**
- Modify: `src/coding_opsd/cli.py`
- Modify: `README.md`
- Modify: `configs/{jag_tree,pbpf,goav}.yaml`
- Create: `.github/workflows/ci.yml`
- Create: `examples/run_all_phase0.sh`
- Test: `tests/test_cli_integration.py`

**Interfaces:**
- `python -m coding_opsd validate CONFIG`, `resolve CONFIG`, `run {jag,pbpf,goav} CONFIG`, and `run-all --profile smoke` are stable commands.
- Every run prints its directory and produces a status in `{PASS, FAIL, INCOMPLETE, INVALID}`; smoke reports are always `INCOMPLETE` for formal gates.

- [ ] Write subprocess tests for help, resolve/validate, all three tiny runs, deterministic repeated metrics, invalid config exit status, and non-overwrite behavior.
- [ ] Run the integration tests and verify the expected failures.
- [ ] Wire all runners into the CLI; update README with exact install/run/output commands and scope; add explicit executable overlay references without weakening the original formal settings.
- [ ] Run the full suite, `compileall`, all three smoke experiments, config validation, and artifact verification.
- [ ] Review requirements line by line, run a whole-branch code review, fix Important/Critical findings, and push the verified tree to GitHub `main` as explicitly requested.
