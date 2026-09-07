# coding_opsd

Statistically grounded algorithms for coding RL with verifiable rewards (RLVR).

**Status (2026-09-07): research design and preregistered experiment plan. No experimental result is claimed yet.**

The YAML files are machine-readable preregistration manifests, not executable training configs yet. A future runner must deep-merge `shared.yaml`, validate a schema and fail on missing fields before these manifests are called runnable.

This repository studies three falsifiable directions:

1. **JAG-Tree** — genealogy-aware, joint value-gradient allocation for tree rollouts.
2. **PBPF** — predictive belief particle filtering from execution evidence.
3. **GOAV** — gradient-optimal active verification under noisy and costly tests.

Directions 2 and 3 are deliberately separated during mechanism experiments:

- PBPF learns the belief state while seeing a fixed, immutable test order.
- GOAV learns which test/audit to acquire and how to correct the resulting selection bias.
- They are joined only after both isolated mechanisms pass their preregistered gates.

The project does **not** treat “more tests”, “easy-to-hard tests”, token entropy, KV-cache injection, or coder-tester competition as sufficient algorithmic novelty. The common target is reliable policy improvement under fixed generation and verification budgets.

## Repository map

- [`docs/00_shared_protocol.md`](docs/00_shared_protocol.md): common model, data, leakage, budget, statistics, and reporting rules.
- [`docs/01_jag_tree.md`](docs/01_jag_tree.md): Direction 1 algorithm and full experiment plan.
- [`docs/02_predictive_belief_particles.md`](docs/02_predictive_belief_particles.md): Direction 2 algorithm and full experiment plan.
- [`docs/03_gradient_optimal_verification.md`](docs/03_gradient_optimal_verification.md): Direction 3 algorithm and full experiment plan.
- [`docs/04_benchmarks_and_compute.md`](docs/04_benchmarks_and_compute.md): benchmark roles, visibility firewall, hardware tiers, and cost accounting.
- [`docs/05_execution_roadmap.md`](docs/05_execution_roadmap.md): ordered gates and go/stop decisions.
- [`docs/06_upstream_code_map.md`](docs/06_upstream_code_map.md): paper/code provenance, reuse boundaries, and exact fork points.
- [`configs/shared.yaml`](configs/shared.yaml): common experimental defaults.
- [`configs/jag_tree.yaml`](configs/jag_tree.yaml), [`configs/pbpf.yaml`](configs/pbpf.yaml), [`configs/goav.yaml`](configs/goav.yaml): preregistration-style starting configurations.

## Recommended order

Run only the low-cost mechanism gates first:

1. JAG Phase 0 exact-gradient audit.
2. PBPF Phase 0 exact-posterior audit.
3. GOAV Phase 0 oracle-gradient audit.

Advance at most one or two successful directions to 7B online RL. A negative gate is a scientific result and stops expensive scaling.

## Main model policy

- Primary: `Qwen/Qwen2.5-Coder-7B-Instruct`.
- Cheap plumbing/mechanism checks: the matching 1.5B model or a finite tabular/DSL model.
- Replication only after success: `Qwen/Qwen3-8B`.
- Formal training: BF16 frozen base plus LoRA; QLoRA is permitted for 16/24GB pilots.

## Evidence standard

Every claimed improvement must report:

- task-level confidence intervals and individual seeds;
- generated tokens, scored/training tokens, sandbox CPU-seconds, GPU-hours, and wall time;
- the direction-specific mechanism metric, not only benchmark pass rate;
- an unpublished or strictly post-cutoff evaluation suite whose tasks, gold artifacts, tests and per-task feedback were never exposed to rollout, belief, tester or checkpoint selection.

## Closest reusable projects

- [TreePO](https://github.com/multimodal-art-projection/TreePO)
- [Tree-GRPO](https://github.com/AMAP-ML/Tree-GRPO)
- [TreeRPO](https://github.com/yangzhch6/TreeRPO)
- [VPO](https://github.com/ryanboldi/vpo)
- [UTRL](https://github.com/dgjun32/UTRL)
- [UpSkill](https://github.com/dshah02/upskill)
- [Random Soft Prompts](https://github.com/heejunkim00/RSP)
- [KV Cache Steering](https://github.com/MaxBelitsky/cache-steering)
- [CodeContests-O](https://github.com/cai-jianfeng/CodeContests-O)
- [EvalPlus](https://github.com/evalplus/evalplus)

Licences and exact upstream commits must be checked and pinned before code is copied. Paper-only baselines must be labelled as reimplementations.
