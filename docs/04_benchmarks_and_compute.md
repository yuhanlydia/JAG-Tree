# Benchmarks, visibility and compute protocol

This document assigns each dataset one primary role. Public artifacts may be hidden from the rollout by protocol, but they are not thereby a truly unseen test set: pretraining exposure cannot be ruled out. Local inspection or adaptive use definitively makes them development data. A final audit requires unpublished evaluator-held tasks, or tasks strictly after every learning component's cutoff and frozen before benchmark-specific tuning.

## 1. Benchmark role matrix

Visibility classes are `V` (prompt-visible), `Hᴾ` (hidden from rollout by protocol but publicly downloadable), and `Hˢ` (genuinely private/server-held).

| Dataset | Primary use here | Directions | Visibility | 7B/8B? | Evidence tier | Main caution |
|---|---|---:|---:|---:|---:|---|
| [CrossBeam](https://github.com/google-research/crossbeam) | finite DSL posterior audit | D2 | V | optional | mechanism | synthetic and public |
| [CodeARC](https://github.com/Anjiang-Wei/CodeARC) | interactive program-induction transfer | D2 | V + Hᴾ | yes | public transfer | released data include target `code`; build new unpublished functions for final audit |
| [EvalPlus](https://github.com/evalplus/evalplus) | fast execution and mutation smoke tests | D2, D3 | V + Hᴾ | yes | dev-public | HumanEval+/MBPP+ are heavily exposed |
| [TACO](https://github.com/FlagOpen/TACO) | coding RL train/dev pool | all | V + Hᴾ | yes | train/dev | test data expose solutions and input/output tests |
| [CodeContests](https://github.com/google-deepmind/code_contests) | difficult train/dev pool and candidate matrix | D1, D2 | V + Hᴾ | yes | train/dev | `public`, `private` and `generated` tests are all downloadable |
| [CodeContests-O](https://github.com/cai-jianfeng/CodeContests-O) | generated/refined noisy test-generator bank and scalable execution | D3 | V + Hᴾ | yes | mechanism | generated feedback is not an oracle or an official DeepMind test bank |
| [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) | conditional temporal evaluation | all | V + Hᴾ | yes | conditional-temporal | release v6 ends 2025-04; unusable as post-cutoff evidence for later-cutoff systems |
| [RunBugRun](https://github.com/giganticode/run_bug_run) | executable short-program repair training/transfer | D2 | V + Hᴾ | yes | train/dev | group by source problem, submission/user lineage, language translation and shared test source |
| [SWE-smith](https://github.com/SWE-bench/SWE-smith) | later repository-level synthetic training | D2, D3 | V + Hᴾ | yes, costly | train | environment build failures can dominate reward |
| [SWE-bench Verified/Lite](https://github.com/SWE-bench/SWE-bench) | standard repository-repair development/reporting | D2, D3 | V + Hᴾ | yes, costly | dev-public | public fields include gold/test patches and test lists |
| [SWE-bench-Live](https://github.com/microsoft/swe-bench-live) | repository-level temporal replication | D2, D3 | V + Hᴾ | yes, costly | conditional secondary | field firewall required; Python `full` reaches 2025-08, while a 2026-08 Windows set is platform-specific |
| [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2) private sets | orthogonal abstract-induction audit | D2 only | Hˢ | yes | sealed but non-coding | private sets are server-held, but outputs are grids rather than source code |

The 7B question is therefore not “does the benchmark list a 7B score?” All coding suites above can execute outputs from a 7B model. What changes is statistical usefulness: EvalPlus is cheap but saturated/exposed; CodeContests/TACO produce training signal. Final coding evidence requires that the tasks themselves are unpublished or strictly post-cutoff, while gold artifacts, tests and per-task feedback remain isolated behind an independent grader. A private test suite cannot repair contamination of a public task statement or gold patch.

The cutoff is the latest data date across the base model, SFT/RL, teacher, reward/verifier/tester and retriever—not merely the base checkpoint's release date. The full agent protocol must be frozen before anyone on the training/model-selection team views the proposed temporal slice.

## 2. Split and contamination firewall

Every public dataset receives a committed manifest before training. The training team's manifest includes:

```text
dataset_name, dataset_revision, task_id, source_date, role,
statement_hash, permitted_code_hashes, permitted_test_hashes,
duplicate_cluster_id, manifest_seed, manifest_sha256
```

Required checks:

1. exact and MinHash statement deduplication;
2. normalized AST/token fingerprinting for reference and candidate code;
3. normalized test-input/output fingerprinting;
4. grouping all variants, repositories or bug mutations of one source task;
5. temporal filtering by the earliest public appearance, not repository commit date alone;
6. manual review of the nearest cross-split neighbors;
7. a signed record of who had access to locked artifacts.

For a sealed final set, the independent evaluator—not the training team—stores target/test hashes in a keyed-HMAC or escrow manifest and returns only deduplication certificates plus the final aggregate. Before scoring, freeze checkpoint, agent/scaffold, prompt, tools/network access, decoding seed, budget, task IDs, container and harness digest. Exclusion rules are preregistered and executed while outcomes remain blinded.

If anyone on the training/model-selection team opens a locked artifact or uses any result to tune a decision, the suite becomes dev. The evaluator runs the selected checkpoint and preregistered strongest baseline once and reveals all aggregates together. Only a preregistered, outcome-blinded infrastructure retry is allowed; any other second query destroys locked status. GOAV's adaptive cheap/generated tests are physically disjoint from final trusted-audit tests.

## 3. Standard task allocation

These counts are starting targets, adjusted only by the preregistered power calculation:

| Stage | Train/calibration | Development | Mechanism test | Locked test |
|---|---:|---:|---:|---:|
| Phase 0 finite/synthetic | 50,000 episodes | 5,000 | 5,000 | not applicable |
| Phase 1 frozen 7B | 1,000–4,000 tasks | 250–500 | 250–500 | no |
| Phase 2 online RL | 4,000 pilot, then 8,000–15,000 | 500–1,000 | periodic fixed public sample | 300+ unpublished or strictly post-cutoff tasks |

Report task counts after all filters, not before. For a small benchmark, use paired bootstrap over tasks and never treat multiple candidates/tests from the same task as independent samples.

## 4. Model policy

| Role | Model | Quantization/adaptation | Reason |
|---|---|---|---|
| exact/plumbing | finite model or Qwen2.5-Coder-1.5B | full/frozen/LoRA as needed | catches estimator errors cheaply |
| primary paper model | `Qwen/Qwen2.5-Coder-7B-Instruct` | BF16 base + LoRA r32, alpha64 | stable open coder and manageable gradients |
| architecture replication | `Qwen/Qwen3-8B` | BF16 base + LoRA | tests whether result survives a newer architecture |

Do not average 7B and 8B into one result. The 8B run starts only after the 7B mechanism and online gates pass, and it uses the same data manifest and metric definitions.

### KV-cache implication

Ignoring allocator overhead and batching, BF16 KV storage per token is approximately

\[
2\times n_{layers}\times n_{kv-heads}\times d_{head}\times2\ \text{bytes}.
\]

For the commonly used configurations, Qwen2.5-Coder-7B is about 56 KiB/token, while Qwen3-8B is about 144 KiB/token. Thus Qwen3-8B has roughly 2.6 times the KV pressure per cached token. PBPF must therefore report physical cache bytes and shared-prefix savings, not just nominal context length.

## 5. Three independent budget ledgers

No method is called “more efficient” from one denominator alone.

### 5.1 Generation ledger

- prompt tokens processed;
- unique continuation tokens generated;
- recomputed versus KV-reused prefix tokens;
- candidates, branches and repair rounds;
- padding/wasted tokens;
- sampled tokens discarded before training.

### 5.2 Verification ledger

- tests generated and compiled;
- cheap and trusted tests executed;
- sandbox CPU core-seconds;
- peak concurrent containers;
- timeout and infrastructure-failure counts;
- cache hit rate for identical program-test pairs.

### 5.3 Optimization ledger

- scored policy tokens;
- forward/backward FLOP estimate;
- GPU-hours by accelerator type;
- peak allocated and reserved memory;
- optimizer updates and epochs per sample;
- end-to-end wall time including sandbox waits.

Primary frontiers are quality versus generated tokens, quality versus verification CPU-seconds, and quality versus wall time. A method may dominate one frontier but lose another.

## 6. Hardware tiers

| Hardware | Permitted experiments | Excluded as formal evidence |
|---|---|---|
| 16GB GPU | finite audits, frozen 7B embeddings, small heads, 1.5B end-to-end, short 7B QLoRA smoke | sustained 7B online RL with large rollout groups |
| 24GB GPU | 7B QLoRA pilot, frozen-policy banks, one-batch gradient audits | multi-seed formal online RL |
| 4×24GB | medium screening, vLLM rollout plus LoRA training, ablation pruning | final claim if settings differ from H200 run |
| H200 141GB | BF16 LoRA formal RL, full telemetry, larger KV/tree/particle banks | none within this plan |

The execution farm is a separate resource. Adding GPUs will not remove a single-threaded Docker/test bottleneck. Before the 1% pilot, benchmark generator tokens/second and cached/uncached tests/second independently.

## 7. Cost calibration and envelopes

All values below are planning envelopes, not measured results. Replace them after the 1% calibration run.

| Direction | Dominant cost | Formal 7B envelope per arm/seed | Main scaling danger |
|---|---|---:|---|
| JAG-Tree | branched rollouts plus score sketches | 144–288 H200 GPU-hours | leaf count and long shared caches |
| PBPF | multiple candidates/repairs plus interface training | 70–220 H200 GPU-hours | particle × candidate cache materialization |
| GOAV | rollouts plus large test bank/audits | 144–480 H200 GPU-hours | sandbox CPU, not GPU |

A minimum two-arm confirmation (method plus preregistered strongest baseline, three seeds) is approximately 0.86–1.73K H200 GPU-hours for JAG, 0.42–1.32K for PBPF and 0.86–2.88K for GOAV. Confirming all three would therefore be roughly 2.1–5.9K GPU-hours before ablations; the roadmap advances at most two. A six-arm JAG study alone is about 2.6–5.2K. These are deliberately conservative planning products of per-arm ranges, not measured costs or commitments to spend.

## 8. One-percent calibration protocol

Before any formal run:

1. draw exactly 1% of the training manifest with the registered seed;
2. run baseline and method with profiling enabled;
3. measure output-length distribution, nonzero-reward fraction, branch/particle/audit utilization, GPU memory and sandbox throughput;
4. estimate hours with both mean and p95 task cost;
5. freeze batch size, concurrency and timeouts;
6. update only the cost envelope, not scientific thresholds;
7. preserve raw calibration logs even if the method is stopped.

Abort or redesign if p95 memory exceeds 90% of capacity, infrastructure failure exceeds 1%, or projected formal cost exceeds the approved envelope by 25%.

## 9. Statistical reporting

- three independent training seeds for any online claim;
- task-level paired bootstrap with at least 10,000 resamples;
- individual seed curves and area under the learning curve;
- predeclared primary comparison and all attempted arms;
- Holm correction for multiple secondary benchmark claims;
- non-inferiority margin fixed before observing locked results;
- failure counts and denominator definitions alongside percentages.

Pass@k sampling uses one frozen generation configuration and reports temperature, top-p, maximum tokens and number of samples. Reusing the same candidates across methods is encouraged for frozen-policy mechanism experiments, but online policies require fresh on-policy samples.
