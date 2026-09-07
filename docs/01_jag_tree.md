# Direction 1 — JAG-Tree

Joint value-gradient Adaptive Genealogy for tree-structured RLVR.

## 1. Claim to test

Existing tree rollouts save prefix computation and create additional suffixes, but the resulting leaves share random ancestors. This is not automatically biased. The problem appears when the estimator or adaptive allocation does not match the intended target.

JAG fixes the target as

\[
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[R(\tau)],
\qquad g=\nabla_\theta J(\theta),
\]

and tests three hypotheses:

- **H1 — credit:** a recursive genealogy estimator preserves shared-prefix credit that sibling-only contrasts can cancel.
- **H2 — allocation:** allocating roots and suffix branches by joint value-gradient covariance reduces gradient MSE per token relative to uniform, entropy and value-only allocation.
- **H3 — learning:** the estimator improvement produces better validation reward per generated token and per wall-clock second.

Exact deterministic KV reuse is not the claimed problem. Dependence comes from the sampled shared prefix, and bias depends on the estimand, allocation rule and weights.

## 2. Estimator

For a node/history \(u\), let edge \((u,v)\) be one sampled token segment and

\[
s_{uv}=\sum_{t\in(u,v)}\nabla_\theta\log\pi_\theta(a_t\mid h_t).
\]

At a terminal leaf \(\ell\):

\[
\widehat V_\ell=R_\ell,\qquad \widehat g_\ell=0.
\]

If node \(u\) has \(b_u\) children, selected before any of those child outcomes are observed:

\[
\widehat V_u=\frac1{b_u}\sum_v\widehat V_v,
\]

\[
\widehat g_u=\frac1{b_u}\sum_v
\left[\widehat g_v+(\widehat V_v-B_u)s_{uv}\right].
\]

The main estimator uses the same stop-gradient \(B_u\) for every child of \(u\), fitted only on prior batches/cross-fitting folds. Conditional on \(u\), it is independent of every newly sampled child; therefore the child contributions remain IID and their covariance scales as \(1/b_u\). The prompt is a synthetic root, so independently sampled roots use the same recursion.

A leave-one-sibling-out baseline remains an estimator ablation, but it creates a correlated U-statistic. It may not use the simple allocator below: its allocator must predict the complete \(b\)-dependent variance and use \([\operatorname{MSE}(b)-\operatorname{MSE}(b+1)]/c\).

The primary theorem/audit scope is:

- on-policy sampling;
- no PPO clipping or reward standardization;
- one optimizer step per rollout batch;
- \(b_u\ge1\);
- the allocator uses only the observed prefix plus a lagged/cross-fitted predictor;
- no reward-dependent hard pruning or forced-answer fallback.

Hard pruning with \(b_u=0\) requires recursive inclusion probabilities and is postponed.

## 3. Joint covariance allocator

For one child edge, define

\[
X_{uv}=
\begin{bmatrix}
\widehat V_v\\
P\left(\widehat g_v+(\widehat V_v-B_u)s_{uv}\right)
\end{bmatrix}.
\]

The allocator predicts the conditional single-child covariance

\[
\Sigma_u=\operatorname{Cov}(X_{uv}\mid u).
\]

Consequently the covariance after averaging \(b_u\) IID children is \(\Sigma_u/b_u\); \(b_u\) is not included inside the predicted \(\Sigma_u\).

Here \(P\) is a fixed 256-dimensional Johnson–Lindenstrauss/Rademacher projection over LoRA-gradient coordinates.

Let \(S_u=P\sum_{e\in\operatorname{path}(u)}s_e\) be the projected accumulated score from the prompt to \(u\). The local contribution to root-gradient MSE includes value-gradient cross terms:

\[
a_u=\operatorname{tr}\left[
\Sigma_{gg}+S_u\Sigma_{Vg}+\Sigma_{gV}S_u^\top
+\sigma_V^2S_uS_u^\top
\right].
\]

With predicted incremental cost \(\widehat c_u\), the greedy marginal value of one extra child is approximated by

\[
 w_u=\prod_{x\prec u}b_x^{-1},\qquad w_{root}=1,
\]

\[
\operatorname{score}(u)=
\frac{w_u^2a_u}{\widehat c_u\,b_u(b_u+1)}.
\]

The product is over strict ancestors between the synthetic root and \(u\). The square is required because the node estimator's error is transported to the root with weight \(w_u\).

Give each allowed root/node one child, then repeatedly allocate the remaining token budget to the largest score. Phase 0 also computes an integer dynamic-programming oracle. The learned allocator must close a substantial part of the oracle–uniform gap.

The live rollout does not compute a parameter gradient at every prospective branch. In Phase 0/1, `gradient_sketch.py` obtains \(Ps_e\) from actor score hooks with a fixed Rademacher matrix \(P_{ij}\in\{-1,+1\}/\sqrt{256}\) and frozen seed. These labels supervise a prefix predictor that outputs the already transported scalar \(\widehat a_u\). Phase 2 uses that lagged scalar predictor at rollout time; exact/projected scores are recomputed only in registered audits. This makes the online decision implementable without one backward pass per candidate node.

Raw token entropy is not an input in the main configuration. It is a baseline and optional diagnostic.

## 4. Upstream implementation base

Primary base: [TreePO](https://github.com/multimodal-art-projection/TreePO), because it already contains segment decoding, genealogy, KV prefix reuse, vLLM rollout and veRL/FSDP training.

Closest comparisons:

- [TreePO paper](https://arxiv.org/abs/2508.17445): heuristic tree sampling and nested subgroup advantages.
- [Tree-GRPO](https://github.com/AMAP-ML/Tree-GRPO): ReAct-step trees and intra/inter-tree advantages.
- [TreeRPO](https://github.com/yangzhch6/TreeRPO): fixed branching and bottom-up reward backup.
- [PAIR](https://arxiv.org/html/2608.11368v2): pairwise inclusion correction for conditionally independent prefixes; shared-prefix trees require a hierarchical estimand.
- [iGRPO](https://arxiv.org/abs/2602.09000): two-stage draft/self-feedback refinement; it is a task baseline, not a tree estimator.

### Planned TreePO modifications

| Location | Change |
|---|---|
| `recipe/treepo/vllm_rollout_tree.py::DataSampleTree` | add stable `node_id`, `parent_id`, `root_id`, segment boundaries, planned branching factor, predictor version, predicted moment/cost and RNG seed |
| vLLM tree rollout path | create a separate `jag` generator; decide \(b_u\) before sampling children; disable reward/length-dependent fallback and forced completion |
| `recipe/jag_tree/allocator.py` | implement root/node allocation, exploration floor, budget ledger and lagged predictor snapshot |
| `recipe/jag_tree/predictor.py` | predict value, remaining cost and low-rank-plus-diagonal joint covariance; train only on prior trees |
| `recipe/jag_tree/gradient_sketch.py` | fixed-seed Rademacher score projection, registered LoRA-block hooks and transported-\(a_u\) training labels; audit-time only |
| `verl/trainer/ppo/core_algos.py` | implement bottom-up values, LOO baselines, unique-edge weights and unclipped recursive score loss |
| `verl/trainer/ppo/ray_trainer.py` | add `JAG_RECURSIVE`; freeze policy and allocator for a rollout batch; update policy once; train next predictor afterwards |
| `verl/workers/actor/dp_actor.py` | accumulate all microbatches before one optimizer step; `ppo_epochs=1` |
| `recipe/jag_tree/audit.py` | exact-coordinate and projected gradient audit, finite-population replay and log-probability consistency checks |
| reward worker | binary full-suite reward, safe sandbox and per-task execution ledger |

Do not reuse TreePO's existing whitened tree advantage for the primary causal comparison.

## 5. Phase 0 — exact finite MDP

### Purpose

Prove the estimator implementation and show a setting in which the cross-covariance allocation matters before using an LLM.

### Environment

- finite horizon \(H=8\), four categorical actions per state;
- enumerate all \(4^8=65,536\) terminal paths;
- softmax logits are the policy parameters;
- 50 randomly parameterized MDP seeds;
- budgets: 64, 128 and 256 sampled node-actions;
- 100,000 sampled genealogies per method/seed/budget.

### Four diagnostic reward structures

1. **Root-only reward:** the first action determines reward; suffixes are irrelevant. Detects prefix cancellation.
2. **Suffix-only reward:** early history is irrelevant; a late action determines reward. Rewards suffix branching.
3. **Entropy distractor:** an early state has high policy entropy but identical downstream values. Tests entropy misallocation.
4. **Covariance reversal:** two nodes have equal reward variance but opposite value-gradient cross-covariance. Tests the genuinely new term.

### Arms

- flat IID score estimator with the same lagged baseline;
- uniform balanced tree with the same lagged baseline;
- leaf-equal naive adaptive tree;
- entropy allocation;
- value-variance-only allocation;
- gradient-only allocation;
- joint covariance without cross term;
- JAG with oracle integer DP;
- JAG with learned greedy predictor.

Sibling-LOO is reported only with its exact empirical \(b\)-dependent U-statistic variance, never with the simple JAG allocator. Negative controls deliberately use current-tree outcomes to add branches and mismatched top-p sampling/full-softmax scores.

### Primary metrics

- relative gradient bias;
- trace MSE per sampled node/token;
- gradient cosine to exact \(g\);
- allocation regret to oracle DP;
- coverage of predicted covariance.

### Phase 0 gate

- relative bias below 1% and bias z-score no larger than 2;
- oracle JAG lowers gradient MSE/token by at least 25% versus the best unbiased baseline;
- learned JAG closes at least 70% of the oracle–uniform gap;
- full joint covariance beats the no-cross-term version by at least 10% in the covariance-reversal family.
- nominal 90% covariance intervals cover the exact node contribution in at least 90% of registered audit cases.

Any unbiasedness failure stops the direction. If value-only matches full joint within 3%, remove the joint-covariance novelty claim.

## 6. Phase 1 — frozen 7B estimator audit

### Data and model

- frozen `Qwen2.5-Coder-7B-Instruct` after the common 15-update uniform warm start, so registered LoRA coordinates are non-degenerate;
- 256 TACO-train problems for predictor calibration;
- 64 disjoint audit problems, stratified into four base-policy success ranges;
- binary full-suite reward; per-test pass fraction is diagnostic only.

### Complete candidate bank

For each audit problem:

- four independent roots;
- candidate branch depths at generated token 256, 512 and 768;
- three available child continuations at every selected branch point;
- continue to EOS or 1,536 generated tokens;
- cap at 108 leaves per problem.

The evaluator generates this bank once. A replay randomly permutes available children and applies each frozen allocator under budgets of 4k, 8k and 16k unique generated tokens per prompt. Run 50,000 random replays without regenerating code.

### Gradient target

- policy parameters are LoRA coordinates, not all 7B weights;
- allocator uses a fixed 256-dimensional sketch;
- audit also computes exact gradients for three registered 16,384-coordinate `lora_B` blocks from early, middle and late layers; the nonzero warm-start adapter avoids zero-gradient `lora_A` coordinates at initialization;
- the target is the full-bank recursive gradient under the predeclared finite-population design.

### Baselines

- flat IID roots;
- uniform tree;
- TreePO likelihood heuristic with the same JAG recursive loss;
- entropy/EPTree allocator with the same JAG recursive loss;
- value-only/VIP-style allocator;
- gradient-only allocator;
- JAG without \(\Sigma_{Vg}\);
- oracle moment allocation.

PAIR is reproduced only in a separate independent-prefix panel; it is not treated as the same tree estimand.

### Metrics and gate

- relative bias below 2%, bias z-score no larger than 2;
- learned JAG reduces exact-block MSE/token by at least 15% over the best unbiased adaptive baseline;
- at least three of four success strata improve by 10%;
- predictor Spearman correlation with realized marginal gradient-MSE reduction at least 0.40;
- sketch-versus-exact marginal-risk ranking Spearman at least 0.70;
- at least 60% oracle–uniform gap closed;
- at least two of three exact coordinate blocks agree with the sketch conclusion;
- predictor/synchronization overhead below 8%;
- vLLM/actor chosen-token log-probability mean absolute error below \(10^{-4}\), p99 below \(10^{-3}\).

If oracle allocation wins but learned allocation does not, do not run online RL. If gains appear only in the sketch, kill the result.

## 7. Phase 2 — online 7B LoRA RL

### Training

- TACO train after source/AST and evaluation-overlap removal;
- start with a static preregistered 4,000-problem subset filtered only by source/AST overlap and harness validity—never by rollout success—then scale to 8,000–15,000;
- common 15-update uniform-tree warm-up;
- 200-update screening, then 600–800 updates for confirmed arms;
- global prompt batch 64, group cap 8;
- segment length 256; branch factor at most 4; depth at most 3; total leaves, not \(K^D\), is hard capped;
- about 8k unique generated tokens per prompt as the initial budget, reset after a measured 1% workload pilot;
- binary all-tests-pass reward;
- LoRA rank 32, alpha 64, learning rate \(10^{-6}\), AdamW, weight decay 0;
- one on-policy, unclipped optimizer step per rollout batch.

### Main arms

1. flat IID estimator with the lagged baseline and common unclipped loss;
2. uniform tree with the same recursive estimator;
3. TreePO likelihood allocation with the same recursive estimator;
4. entropy allocation with the same recursive estimator;
5. value-only allocation with the same recursive estimator;
6. full JAG allocation.

Original TreePO/TreeRPO/Tree-GRPO objectives are an appendix task comparison because their advantage and optimization targets differ. They never enter the primary allocator attribution.

### Evaluation

- every 10 updates: deterministic TACO validation;
- every 25 updates: small frozen-gradient audit;
- final: TACO official test as public secondary evidence, a strictly post-cutoff LiveCodeBench slice if available, and an unpublished evaluator-held contest set for the sealed claim;
- primary learning metric: normalized AUC of validation full-pass versus unique generated tokens;
- secondary: tokens and wall time to warm-start +5 absolute points, locked pass@1, pass@5.

### Essential ablations

- adaptive roots only, adaptive internal nodes only, both;
- segment 128/256/512;
- `b_max` 2/4/8 under a fixed total leaf cap;
- full/value-only/gradient-only/no-cross/diagonal covariance;
- predictor lag 1/5/20;
- projection dimension 64/256/1024;
- constant versus predicted execution/token cost;
- exploration fraction 0/0.1/0.2;
- binary full-pass versus test fraction;
- lagged predictor versus the deliberately biased same-tree allocator.

### Final success and kill criteria

The direction is supported only if all three links hold: Phase 0 unbiasedness, Phase 1 exact-block MSE reduction, and Phase 2 learning efficiency. The single Phase 2 primary gate is at least 10% higher normalized validation-AUC per generated token than the strongest same-objective allocator baseline, with the hierarchical paired 95% interval above zero, and no sealed-test degradation larger than 0.5 point.

Headline secondary evidence, Holm-corrected, is at least one of:

- at least 20% fewer generated tokens to a fixed validation threshold; or
- at least +1.0 absolute locked pass@1;

with a 95% interval excluding zero and no locked-benchmark degradation larger than 0.5 point.

Stop or narrow the claim if:

- uniform tree matches JAG: benefit is prefix amortization, not adaptive covariance;
- value-only matches full JAG: remove the cross-covariance contribution;
- Phase 1 improves MSE but Phase 2 does not improve learning AUC: gradient variance is not the training bottleneck;
- token-matched wins but wall-clock loses or overhead exceeds 10%: claim token efficiency only;
- the 25%-budget screening upper confidence bound is below uniform tree: terminate that arm;
- log-probability mismatch exceeds tolerance: fix the rollout-policy mismatch before continuing.

## 8. Compute envelope

| Tier | Work allowed |
|---|---|
| 16GB | Phase 0; 7B NF4 frozen audit with 12–16 tasks; 1.5B LoRA plumbing only |
| 24GB | 7B frozen audit with 24–32 tasks; two-arm 7B QLoRA pilot with short responses |
| 4x4090 | 512–2,000-group frozen/controller study and reduced 7B QLoRA screening |
| 1x8-H200 node | full Phase 1; one Phase 2 arm costs roughly 144–288 H200 GPU-hours after throughput calibration |

A mandatory two-arm, three-seed confirmation is approximately 0.86k–1.73k H200 GPU-hours. Only if it passes, the full six-arm causal study expands to roughly 2.6k–5.2k before extra ablations. Do not schedule either until the frozen estimator gate passes.
