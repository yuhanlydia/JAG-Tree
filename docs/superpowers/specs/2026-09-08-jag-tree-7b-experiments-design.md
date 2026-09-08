# JAG-Tree 7B Coding-RL Experiment Design

**Status:** preregistered implementation target; no 7B result or superiority claim exists yet.

## Claim and falsifiable hypotheses

JAG-Tree targets tree-RLVR in which sampled leaves share random ancestors. KV reuse is only a compute optimization; statistical dependence comes from shared stochastic genealogy. The method combines a recursive genealogy-aware policy-gradient estimator with allocation based on joint value-gradient covariance per unit cost.

- **H1 — credit:** recursive genealogy credit has lower bias/MSE than leaf-equal and sibling-only credit.
- **H2 — allocation:** joint value-gradient covariance has lower gradient MSE per unique token than uniform, entropy, value-only and gradient-only allocation.
- **H3 — learning:** the estimator gain improves token-normalized coding learning curves rather than merely producing more candidates.

If the learned allocator fails the frozen-bank mechanism gate, online 7B RL must not start.

## Repository boundary

This repository owns JAG algorithms, controlled allocator baselines, model/benchmark adapters, immutable candidate banks, tree replay, gradient audits, online QLoRA entry points, statistics and artifact verification. PBPF and GOAV code do not belong here. End-to-end upstream recipes are adapters with pinned provenance, not silently copied code.

## Models

| Role | Hugging Face ID | Scope |
|---|---|---|
| plumbing | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | CPU/schema tests and 16GB GPU smoke only |
| primary | `Qwen/Qwen2.5-Coder-7B-Instruct` | all controlled and formal experiments |
| architecture replication | `ByteDance-Seed/Seed-Coder-8B-Instruct` | JAG versus selected strongest baseline |
| cross-family stress | `deepseek-ai/deepseek-coder-6.7b-instruct` | frozen audit; online only if effective groups are non-degenerate |

Every downloaded model revision is resolved to a full commit SHA and stored in the run manifest. Results are reported per model, never averaged across models.

## Benchmarks and roles

| Dataset | Role |
|---|---|
| TACO | filtered RL train, dev, and disjoint 256-task frozen audit |
| LiveCodeBench v6 | public contest-code evaluation; not called sealed for post-2025 systems |
| BigCodeBench Full/Instruct and Hard | practical/API cross-domain evaluation; Hard is a subgroup |
| HumanEval+ and MBPP+ | cheap exposed compatibility smoke |
| evaluator-held post-cutoff set | one-shot locked final evidence after checkpoint/scaffold freeze |

Splits precede candidate generation. Exact/MinHash statement, normalized AST/token, test-I/O and source-lineage deduplication is mandatory. Public hidden tests are hidden-by-protocol, not genuinely unseen.

## Baselines and provenance classes

Controlled allocator arms share topology, reward, candidate bank, recursive loss and budgets: `uniform`, `entropy`, `value_variance`, `gradient_only`, `trace_score`, `jag_no_cross`, `jag_full`, and `oracle_moments`.

End-to-end recipe adapters are base checkpoint, flat GRPO, DAPO, TreePO, TreeRL/EPTree, Tree-GRPO, TreeRPO and VIP. TreeRL is the central official coding/tree baseline. TreePO and TreeRPO have official code but their published experiments are not coding results. TRACE/VIGOR/SALT/PAIR-style equations, if included, are labelled `paper_spec_reimplementation`; they are never labelled official reproductions without author code and pinned revisions.

Each entry records `paper_url`, `repository_url`, `upstream_commit`, `license_sha256`, `mode`, and local patch notes. Allowed modes are `official_adapter`, `paper_spec_reimplementation`, and `controlled_ablation`.

## Experiment stages

### E0 finite mechanism

Run root-only, suffix-only, entropy-distractor and covariance-reversal MDP families. Budgets are 64/128/256; 50 MDP seeds; formal sampling count 100,000 per cell. Gates: relative bias below 1% with absolute z-score at most 2; oracle JAG lowers MSE/token by at least 25%; learned JAG closes at least 70% of the oracle-uniform gap; full joint beats no-cross by at least 10% in covariance reversal.

### E1 frozen 7B tree bank

Use 256 calibration and 256 disjoint audit TACO tasks, four roots, branch depths 256/512/768, three children per branch and a maximum of 108 leaves. Replay matched budgets of 4,096/8,192/16,384 unique continuation tokens with common random numbers. Audit a fixed 256-dimensional Rademacher score sketch and exact registered coordinates from early/middle/late LoRA-B blocks.

Gates: relative bias below 2%; exact-block MSE/token at least 15% below the strongest unbiased adaptive baseline; improvement in at least three of four difficulty strata; predictor Spearman at least .40; sketch/exact ranking Spearman at least .70; oracle-gap closure at least 60%; two of three exact blocks agree; allocator overhead below 8%.

### E2 online 7B RL

All arms share 15 uniform-tree warm-up updates. Screening uses seed 17 for 200 updates. Formal comparison retains JAG plus the strongest non-JAG dev arm, uses fresh seeds 101/202/303, 800 updates, global prompt batch 64, group cap 8, response cap 2,048 and nominal budget 8,192 unique continuation tokens per prompt/update. One optimizer update follows each rollout batch. Checkpoints are selected only by TACO-dev token-normalized AULC.

Formal success requires at least 10% relative AULC improvement with a hierarchical paired 95% interval above zero, or at least 20% fewer tokens to the baseline endpoint; LiveCodeBench should improve about 2 points and BigCodeBench Full must not fall by more than .5 points. These thresholds are gates, not claimed results.

## Ablations

- Credit: full recursive, leaf-equal, sibling-only, no descendant gradient, no shared-prefix score, lagged versus sibling-LOO baseline, unique-edge versus duplicated-prefix weighting.
- Allocation: uniform, entropy, success variance, gradient only, no cross-covariance, no ancestor transport, no cost, root-only, prefix-only, joint.
- Predictor: oracle, lagged learned, shuffled, staleness 1/5/20, cross-fitting on/off, sketch dimension 64/128/256/512, exact-gradient audit.
- Tree/reward: segment 128/256/512, branch cap 2/4/8, early/middle/late branching, binary/full-suite versus pass fraction, shuffled genealogy.

Outcome-leaking allocation is a fault-injection test, not a valid baseline.

## Budgets, statistics and artifacts

Generation, verification and optimization ledgers are separate. Controlled arms match prompts and optimizer steps exactly, program executions exactly, cumulative unique tokens within 1%, and sandbox CPU seconds within 5%. Peak memory above 90%, infrastructure failures above 1%, or projected cost overrun above 25% aborts scaling.

The statistical unit is the base task. Formal online runs use three training seeds and 10,000 hierarchical paired bootstrap replicates; secondary hypotheses use Holm correction. Required outputs include task rows, seed curves, Pass@1, sampled Pass@8, normalized AULC, gradient bias/MSE/cosine, sibling ICC, signed effective sample size, all three cost ledgers, environment/model/data SHAs and artifact checksums.

## Hardware profiles

- `16gb`: 1.5B end-to-end smoke; 7B NF4 frozen generation, group 2, 1,024 response tokens, at most one QLoRA update.
- `24gb`: 7B NF4 QLoRA rank 16/32, microbatch 1, sequential group 4, 1,536 tokens, frozen audit and at most 20-update pilot.
- `4x24gb`: screening with separated rollout and training workers.
- `h200_formal`: BF16 LoRA rank 32/alpha 64, group 8, 2,048 tokens and formal seeds.

Local 16/24GB runs are pilots, not formal multi-seed evidence.

## Failure handling and tests

Config validation fails closed on unknown arms, mutable revisions, split overlap, unmatched budgets, missing container digests or unsupported formal hardware. Sandboxes disable network, distinguish wrong/exception/timeout/infrastructure failure and never overwrite runs. Unit tests cover every estimator equation, allocator permutation invariance, exact finite targets, manifest hashing, budget equality and tamper rejection. Integration smoke uses deterministic fake policy/sandbox backends; optional GPU tests are explicitly marked and never required in CPU CI.

