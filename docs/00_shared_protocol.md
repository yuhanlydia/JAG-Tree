# Shared experimental protocol

## 1. Common scientific question

All three directions ask a version of the same question:

> Under a fixed amount of generation and verification compute, can we obtain a policy update that is statistically more faithful to true program correctness?

The directions target different failure points:

| Direction | Object being learned | Mechanism metric |
|---|---|---|
| JAG-Tree | rollout genealogy and credit estimator | policy-gradient bias/MSE per generated token |
| PBPF | posterior over task/bug/candidate hypotheses | held-out execution predictive NLL/calibration |
| GOAV | evidence acquisition and corrected update | clean-gradient MSE per verification cost |

Final `pass@1` is necessary but insufficient. A direction is not supported if its proposed mechanism does not improve its mechanism metric.

## 2. Common model setup

| Role | Default | Notes |
|---|---|---|
| Main policy | `Qwen/Qwen2.5-Coder-7B-Instruct` | one common checkpoint for all main comparisons |
| Cheap pilot | Qwen2.5-Coder-1.5B-Instruct | plumbing only; not the main paper claim |
| Replication | `Qwen/Qwen3-8B` | only after the main result is frozen |
| Trainable parameters | LoRA rank 32, alpha 64 | rank 16/64 sensitivity; BF16 formal runs |
| Pilot precision | NF4 QLoRA on 16/24GB | frozen generation and small-head training first |
| Sampling | temperature 1.0, top-p 1.0 | no best-of, repetition penalty, or hidden reranking |
| Prompt limit | 2,048 tokens | truncate by a deterministic documented rule |
| Response limit | 2,048 primary, 4,096 sensitivity | thinking setting fixed across arms |
| Group size | 8 | 4 for smoke; 16 only as a scaling ablation |

For estimator theorems, the primary objective is on-policy, unclipped and unstandardized. PPO/GRPO clipping, reward whitening and multiple policy epochs are separate practical extensions and may not inherit the theorem.

## 3. Data roles

### Training

- Algorithmic code: TACO train, CodeContests train, or CodeContests-O after source/problem deduplication.
- Program repair: incorrect/correct CodeContests submissions and RunBugRun; repository-level SWE-smith is postponed.
- Test generation: CodeContests-O artifacts and UTRL-generated tests, with independent validity checking.

### Development and mechanism checks

- HumanEval+/MBPP+ via EvalPlus: smoke and regression only.
- CodeARC anonymized and CrossBeam finite DSL: belief/posterior experiments.
- TACO validation carved only from the official train split.
- B4 execution matrices: sanity reproduction for code-test selection.

### Conditional temporal and sealed evaluation

- Conditional secondary evidence: a LiveCodeBench/SWE-bench-Live time window strictly later than the latest data cutoff of every learning component, if such a public window exists and the complete protocol is frozen before benchmark-specific tuning.
- Sealed final evidence: unpublished contest tasks, CodeARC-style functions or repository issues whose statements, gold artifacts, tests and per-task feedback remain with an independent evaluator.
- Orthogonal evidence: genuinely private ARC-AGI-2 sets may test abstract induction, but are not coding-task substitutes.

Public “hidden” tests are only protocol-hidden and cannot rule out pretraining exposure. Once their content, target code, gold patch or outputs are used to train, debug, tune or select checkpoints, they become development data permanently. A private test suite cannot make a public task statement uncontaminated.

## 4. Leakage firewall

1. Split at the base-task level before producing candidates, mutants, tests or evidence prefixes.
2. Algorithmic tasks are grouped by source URL/problem lineage plus normalized solution AST similarity.
3. Repair tasks are grouped by repository, forks, commit ancestry and patch clones.
4. Program-induction tasks are grouped by target program family, normalized AST and full truth table.
5. The rollout process receives only prompt-visible fields.
6. The grader runs in a separate process/container and exposes only the allowed observation.
7. Locked tests, reference solutions, gold patches and expected outputs are unavailable to the policy, tester, belief model and acquisition model.
8. Freeze checkpoint, task IDs, agent/scaffold, prompt, tools/network policy, decoding seed, budgets, dataset commit, container/harness digest and all hyperparameters before the sealed run.
9. The independent evaluator holds keyed hashes for sealed artifacts and reveals all preregistered aggregates together after one run; only outcome-blinded infrastructure retries are allowed.

## 5. Three non-interchangeable budgets

### Token ledger

Report separately:

- prompt-prefill tokens;
- unique generated response tokens, including abandoned/pruned branches;
- actor/ref/belief scored tokens;
- backward tokens;
- test-generator tokens.

The primary token-matched comparison fixes unique generated response tokens. Also report a forward-equivalent total such as `rollout + auxiliary forward + 3 * train tokens`.

### Execution ledger

Use actual CPU core-seconds, not simply “number of tests”:

\[
\text{core-hours}=\frac{\sum_i \min(t_i,\text{timeout})\,n_{\text{cores},i}}{3600}.
\]

Record compilation, container startup, crashes and full timeout cost. Batch compatible tests for one candidate into one warm sandbox invocation.

### Wall-clock ledger

Fix GPU type/count, CPU quota, sandbox concurrency and deadline. Include queueing, compilation, evaluation, checkpointing and method-specific preprocessing.

Produce 0.25x, 0.5x, 1x and 2x resource curves where feasible. One run cannot generally be simultaneously token-, execution- and wall-clock-matched; report all three views.

## 6. Statistics

- Phase 0 Monte Carlo audits: report bias, variance and MSE with repeated random designs.
- Real-task comparisons: aggregate by base task, never by completion.
- Screening: two seeds are allowed, but no paper conclusion.
- Confirmation: at least three training seeds.
- Confidence intervals: 10,000 task-level paired bootstrap samples.
- Final learning curves: seed-by-task hierarchical bootstrap.
- Report individual seeds; do not hide an unstable run in the mean.
- Predeclare one primary metric and one primary benchmark per phase.

## 7. Common telemetry schema

Each rollout/tree/test event must include:

```text
run_id, method, seed, model_commit, policy_step, task_id, split,
prompt_hash, candidate_id, root_id, node_id, parent_id, depth,
sample_probability, audit_probability, test_probability,
generated_tokens, scored_tokens, train_tokens,
gpu_ms, cpu_ms, wall_ms, timeout, reward_source,
cheap_reward, trusted_reward, test_id, test_cluster,
belief_version, allocator_version, policy_version
```

Direction-specific arrays (gradient sketches, particle weights, execution vectors) are stored separately and linked by IDs.

## 8. Shared reproducibility tests

- Same seed plus same checkpoint reproduces candidate IDs and tree genealogy.
- vLLM sampled-token log probabilities agree with actor recomputation within declared tolerance.
- The policy cannot read grader-only files or use network access.
- Every method receives identical prompt IDs and global budget.
- Every dropped/truncated/unsafe output is counted and receives a deterministic reward.
- Resuming a job preserves RNG, optimizer, allocator/belief version and budget ledgers.
