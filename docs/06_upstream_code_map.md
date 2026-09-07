# Upstream paper and code map

This file separates three things that are easy to blur:

- **infrastructure reused:** rollout, sandbox, data or cache plumbing;
- **baseline reproduced:** an upstream scientific method kept intact;
- **new code:** the estimator, belief or acquisition algorithm claimed here.

Pin an exact commit and archive its licence before importing code. If no official implementation is linked by the authors, label the result `paper-spec reimplementation` in tables and logs.

## Direction 1 — JAG-Tree

| Work | Status and role | Reuse | Our change |
|---|---|---|---|
| [TreePO](https://github.com/multimodal-art-projection/TreePO), [paper](https://arxiv.org/abs/2508.17445) | primary implementation base | segment decoding, genealogy, vLLM KV reuse, veRL/FSDP path | replace likelihood/fallback allocation and whitened tree advantage with preregistered JAG allocator and recursive estimator |
| [TreeRL](https://github.com/THUDM/TreeRL), [paper](https://arxiv.org/abs/2506.11902) | entropy-allocation baseline | EPTree high-entropy branching | port only allocation score into the common JAG rollout/estimator harness |
| [Tree-GRPO](https://github.com/AMAP-ML/Tree-GRPO) | agent-step tree task baseline | ReAct node expansion, intra/inter-tree objective | appendix comparison; do not mix its objective into allocator attribution |
| [TreeRPO](https://github.com/yangzhch6/TreeRPO), [paper](https://arxiv.org/abs/2506.05183) | fixed-tree/dense-credit baseline | bottom-up reward backup and sibling groups | reproduce as separate original-objective arm |
| [VIP](https://github.com/HieuNT91/VIP) | value-only allocation baseline | success/value predictor and warm-up schedule | feed its value variance into the same common allocator interface |
| [PAIR](https://arxiv.org/html/2608.11368v2) | theoretical/audit baseline; no official implementation identified | independent-prefix inclusion correction and frozen-population audit protocol | paper-spec reimplementation in a flat-prefix panel only; not applied unchanged to shared-prefix trees |
| [iGRPO](https://arxiv.org/abs/2602.09000) | two-stage refinement task baseline; no official method code identified | draft-then-refine protocol | optional appendix, not called a tree credit estimator |

Primary fork points in TreePO:

| Existing location | Added/replaced component |
|---|---|
| `recipe/treepo/vllm_rollout_tree.py::DataSampleTree` | stable genealogy, planned branching, predictor/version and cost fields |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | `jag` generation path; allocation before child outcomes |
| `verl/trainer/ppo/core_algos.py` | bottom-up `(V,g)` with lagged child-independent baselines, unique-edge weights and unclipped score loss |
| `verl/trainer/ppo/ray_trainer.py` | freeze rollout policy/allocator, one update, then fit next predictor |
| `verl/workers/actor/dp_actor.py` | one optimizer step after all rollout microbatches |
| new `recipe/jag_tree/{allocator,predictor,gradient_sketch,audit}.py` | joint covariance allocation, transported-risk prediction, score projections and exact-gradient audit |

## Direction 2 — PBPF

| Work | Status and role | Reuse | Our change |
|---|---|---|---|
| [UpSkill](https://github.com/dshah02/upskill), [paper](https://arxiv.org/abs/2602.22296) | multi-turn RL base / visible-skill baseline | rollout, reward adapters and iterative update loop | replace one visible strategy token with a predictive joint belief state; preserve visible-input parity |
| [KV Cache Steering](https://github.com/MaxBelitsky/cache-steering) | cache-interface base | precompute and layerwise cache modification | batched particle-conditioned low-rank deltas with shared immutable prompt cache |
| [RSP](https://github.com/heejunkim00/RSP), [paper](https://arxiv.org/abs/2605.11936) | random-latent control | matched-shape random soft prompt/evaluation | create same-norm random `ΔK,ΔV` and matched-parameter controls |
| [LaDi-RL](https://github.com/mk322/LaDi-RL) | deterministic latent and optional marginalization baseline | dual projection/scheduler patterns | no diffusion in v1; compare deterministic compression with particle posterior |
| [CrossBeam](https://github.com/google-research/crossbeam) | exact finite DSL environment | DSL operations and bottom-up enumeration | produce canonical task/bug hypotheses and exact posterior episodes |
| [CodeARC](https://github.com/Anjiang-Wei/CodeARC) | controlled interactive transfer | program-induction tasks and oracle protocol | enforce fixed hashed queries; use only for future-outcome prediction/repair |
| [EvalPlus](https://github.com/evalplus/evalplus) | execution and smoke evaluation | sandbox/oracles and expanded tests | expose categorical per-test/per-candidate rows without leaking expected outputs |

Primary fork points:

| Existing location | Added/replaced component |
|---|---|
| UpSkill `src/DAPO_math_dapo.py` | carry a versioned `BeliefState` rather than a single textual skill |
| UpSkill `UNSLOTH_rewards.py`, `flex_rewards_adapter.py` | return categorical outcome matrices |
| cache steering `src/steering/cache_steering.py` | particle batch, low-rank projector and no-op control |
| new `belief_pf/belief/*` | proposal, likelihood, log weights, ESS and systematic resampling |
| new `belief_pf/eval/*` | prefix-only future prediction, calibration and leakage tests |
| new `belief_pf/repair/*` | fixed-test-order repair and separated predictor/repair gradients |

## Direction 3 — GOAV

| Work | Status and role | Reuse | Our change |
|---|---|---|---|
| [VPO](https://github.com/ryanboldi/vpo) | primary implementation base and vector-reward comparison | per-test `sub_scores`, task harness and vendored veRL | retain the vector execution data but add clean-gradient risk, randomized audit and AIPW reward |
| [CodeContests-O](https://github.com/cai-jianfeng/CodeContests-O) | realistic test bank/executor | test generator, validator, parallel processor and candidate matrices | score tests by posterior gradient-risk reduction rather than raw feedback/kill rate |
| [UTRL](https://github.com/dgjun32/UTRL), [paper](https://arxiv.org/abs/2508.21107) | generated-test/kill-rate baseline | test generation and execution | preserve discrimination reward as baseline; add separately trained risk-reduction tester |
| [B4](https://github.com/ZJU-CTAG/B4), [paper](https://arxiv.org/abs/2409.08692) | Bayesian code/test selection baseline | `strategy.py:B4` on the common execution matrix | compare inference utility with training-gradient utility |
| [NC-GRPO](https://github.com/omarito101/GRPO_project) | global-noise correction baseline | corrected reward/advantage implementation | compare against joint instance-/policy-dependent noise plus audits |
| [BACE](https://arxiv.org/abs/2603.28653) | joint code/test belief baseline; no official code identified | paper-spec belief and anchors | reimplement behind the common belief interface |
| [Bayesian Control for Coding Agents](https://arxiv.org/abs/2606.24453) | inference-time value-of-information baseline; no official code identified | one-step/horizon acquisition | compare correctness utility with gradient-risk utility |
| [RobustTests](https://arxiv.org/abs/2608.24135) | fault-targeted test baseline; no official code identified | mutants, execution-vector deduplication and behavior selection | compare TSP/mutation proxies with realized gradient-risk reduction |

Primary fork points:

| Existing location | Added/replaced component |
|---|---|
| VPO `vpo_tasks/livecodebench.py` | physically separate cheap and trusted tests; store origin, cluster and cost |
| VPO `vpo/reward.py` | return cheap execution vectors and metadata; reveal trusted utility only for sampled audit IDs |
| veRL `core_algos.py` | linear leave-one-out `JE_AIPW` advantage |
| veRL `ray_trainer.py` | `cheap → joint residual model → 256-subset design → exact inclusion probabilities → sampled audit → Y_DR → update` order |
| new `joint_evidence/{belief,gradient_sketch,audit_policy,dr_reward}.py` | joint residual second moment, exact subset-design MSE, inclusion probabilities and correction |
| new `joint_evidence/{test_policy,ope,telemetry}.py` | optional test-content policy, off-policy evaluation and audit logs |

## Common provenance record

For every imported or reimplemented baseline, store:

```yaml
name: treepo
paper_url: https://arxiv.org/abs/2508.17445
repository_url: https://github.com/multimodal-art-projection/TreePO
upstream_commit: <full sha>
license_file_sha256: <sha256 or null>
integration_commit: <full sha>
mode: infrastructure_reuse  # baseline_reproduction | paper_spec_reimplementation
local_changes: <path to patch/notes>
```

Do not silently “clean up” a baseline's objective, prompts or budget. Keep an intact reproduction and a common-objective causal comparison as two separately named arms.
