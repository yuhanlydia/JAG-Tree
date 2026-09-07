# Direction 3 — GOAV

Gradient-Optimal Active Verification under costly, noisy tests.

## 1. The assumption being challenged

Coding RLVR usually treats a test result as a cheap, correct scalar reward and chooses extra tests using pass rate, entropy or mutation kill rate. Three problems are mixed together:

1. cheap tests can be incomplete, flaky or systematically wrong;
2. selectively running trusted tests changes which labels are observed;
3. a test that distinguishes programs need not reduce error in the policy-gradient direction that matters.

GOAV asks a sharper question:

> Given a cheap noisy outcome matrix and a small trusted-audit budget, which labels—or which new tests—should be acquired to minimize posterior policy-gradient mean-squared error per unit cost, while correcting the selection bias caused by that acquisition policy?

The first paper-worthy claim uses a fixed test pool. Learning test content is an extension and is attempted only after the fixed-pool estimator passes its gate.

## 2. Clean gradient target

For a group of \(K\) on-policy programs, let \(Y_i\) be the trusted binary utility and \(s_i=\nabla_\theta\log\pi_\theta(A_i\mid x)\) its score vector. With a leave-one-out group baseline,

\[
A_i^*=Y_i-\frac{1}{K-1}\sum_{j\ne i}Y_j,
\qquad
G^*=\frac1K\sum_i s_iA_i^*=LY.
\]

The influence of auditing label \(j\) is

\[
\ell_j=\frac1K\left(s_j-\frac1{K-1}\sum_{i\ne j}s_i\right),
\]

so \(L=[\ell_1,\ldots,\ell_K]\). This fixed linear target is the primary exact audit. PPO clipping, adaptive reward normalization and multiple optimizer epochs are excluded because they change the estimand. A practical clipped variant may be reported later, labelled as an approximation.

## 3. Corrected gradient-design objective

Let cheap outcomes, test metadata, coverage features and candidate relations be \(E\), and include the frozen gradient geometry \(L\) as an explicit conditioning variable. The cross-fitted joint outcome model is

\[
q(Y\mid E,L),\qquad \mu=\mathbb E[Y\mid E,L],\qquad
C=\mathbb E[(Y-\mu)(Y-\mu)^\top\mid E,L].
\]

For a frozen group, sample an audit **subset** \(S\subseteq\{1,\ldots,K\}\) before revealing any trusted label. Let \(D_i=\mathbb 1[i\in S]\),

\[
\pi_i=\Pr(i\in S\mid E,L),\qquad
\pi_{ij}=\Pr(i,j\in S\mid E,L),\qquad \pi_{ii}=\pi_i.
\]

With exact logged inclusion probabilities, use

\[
Y_i^{DR}=\mu_i+\frac{D_i}{\pi_i}(Y_i-\mu_i),
\qquad \widehat G_{DR}=LY^{DR}.
\]

The posterior expected design MSE under metric \(W\succeq0\) is

\[
R_{design}(d)=
\sum_{i,j}
\left(\frac{\pi_{ij}}{\pi_i\pi_j}-1\right)
C_{ij}\,\ell_i^\top W\ell_j,
\]

where \(d(S\mid E,L)\) is the subset-sampling distribution. This—rather than single-label entropy or Bayes information gain—is the primary GOAV objective.

For \(K=8\), enumerate all 256 subsets. Parameterize a full-support distribution

\[
d_\eta(S)\propto \exp\{u_\eta(S)-\lambda c(S)\},
\]

mix it with a symmetric exploration distribution, calculate every \(\pi_i,\pi_{ij}\) exactly, and optimize \(R_{design}\) subject to expected cost and \(\pi_i\ge\epsilon\). The primary point has expected trusted-label fraction 10%, hence \(\mathbb E|S|=0.8\) per group, with \(\epsilon=0.02\); the expected budget is enforced over the global prompt batch. A 5% sensitivity uses \(\epsilon=0.005\). Empty subsets are allowed.

The Phase 0 solver initializes subset logits from symmetric-uniform, independent Neyman and Bayes-VOI designs plus five fixed random starts. It runs 200 mirror-descent steps at learning rate `0.05`; a bisection-updated Lagrange multiplier enforces expected cost within `1e-4`, and infeasible floor violations invalidate the run. Return the feasible restart with lowest recomputed exact \(R_{design}\). Phase 1 trains an amortized score network to imitate this solver, but every reported design still recomputes \(\pi_i,\pi_{ij}\) exactly from its 256 probabilities.

Under independent Poisson audits, all design-dependent off-diagonal terms vanish and the Neyman relaxation reduces to

\[
\pi_i=\operatorname{clip}\left(
\sqrt{\frac{C_{ii}\ell_i^\top W\ell_i}{\lambda c_i}},
\epsilon,1\right).
\]

This is a strong baseline. It is not called joint-covariance GOAV.

For reference, the single-label Bayes posterior-mean risk reduction

\[
\Delta_j^{Bayes}=
\frac{(L C_{:j})^\top W(LC_{:j})}{C_{jj}}
\]

assumes the estimator updates every posterior mean after observing \(Y_j\). It is included as a VOI heuristic baseline, not substituted into the AIPW design-MSE formula.

## 4. What the correction guarantees

For a fixed realized group \((Y,E,L)\), exact design propensities and \(\pi_i>0\) give

\[
\mathbb E_D[\widehat G_{DR}\mid Y,E,L]=G^*.
\]

This is finite-group design unbiasedness; it does not require a correct outcome model. If propensities are wrong, even a correct superpopulation conditional mean does not recover that particular realized \(LY\). Standard double-robust consistency applies only to a superpopulation target under MAR, many independent groups, and an outcome model conditioning on the complete action/group/\(L\) information. These two claims are reported separately.

The propensity is known from the randomized design, not fitted. It is stored before audit execution along with the full 256-subset distribution hash. Clipping, self-normalization and replacing \(\pi\) with a perturbed value are explicitly biased ablations. Mandatory diagnostics are design bias, ESS, weight quantiles, expected/realized cost and support violations.

## 5. Learned test content extension

Only after fixed-pool GOAV works, a tester proposes executable test inputs \(t\). Three quantities remain separate.

Test content targets posterior **imputation** risk, distinct from the audit-design variance:

\[
R_{imp}(E)=\operatorname{tr}\{WL C(E,L)L^\top\}.
\]

Before execution, the selection score is

\[
v_{pre}(t)=\frac{R_{imp}(E)-
\mathbb E_{z_t\sim q}[R_{imp}(E,z_t)]}
{c_{gen}(t)+c_{exec}(t)}.
\]

Tests are sampled—not top-k chosen—with logged selection propensity. After execution, use two conditionally independent trusted-audit subset draws from one fixed uniform full-support **evaluation design**, independent of the proposed test and of the pre/post outcome models. Its exact \(\pi_i^{eval}\) are frozen beforehand. Each replicate produces an unbiased \(\widehat G_{DR}^{(r)}\) for the same frozen \(G^*\). For \(m(E)=L\mu(E,L)\), define

\[
\widehat R_{2rep}(E)=
\left(m(E)-\widehat G_{DR}^{(1)}(E)\right)^\top W
\left(m(E)-\widehat G_{DR}^{(2)}(E)\right).
\]

Conditional independence gives

\[
\mathbb E[\widehat R_{2rep}(E)\mid Y,E,L]
=\|m(E)-G^*\|_W^2.
\]

The detached tester reward is

\[
r_{train}(t)=
\frac{\widehat R_{2rep}(E)-\widehat R_{2rep}(E,z_t)}
{c_{gen}(t)+c_{exec}(t)}.
\]

Use the same two evaluation-design audit replicates for the pre/post contrast, recompute the AIPW pseudo-label with each state's \(\mu\), and charge both costs. Offline/Phase 0 evaluation alone may use full \(Y\) to report external realized reduction; those labels live in an evaluator sidecar and never become tester/belief features. Clipping a negative/high-variance cross-product reward is a biased ablation, not the primary estimator.

Compilation, type validity, timeout and duplicate detection are feasibility constraints, not scientific reward. The tester generates eight candidates and samples one or two. Generated tests never grade themselves, and a tester-controlled belief cannot be used as the sole evaluator of its own reward.

## 6. Code base and exact modification surface

Use [VPO](https://github.com/ryanboldi/vpo) and its vendored veRL stack as the primary base. Reuse [CodeContests-O](https://github.com/cai-jianfeng/CodeContests-O) for scalable test execution and [UTRL](https://github.com/dgjun32/UTRL) for the learned-test baseline. [B4](https://github.com/ZJU-CTAG/B4) and [NC-GRPO](https://github.com/omarito101/GRPO_project) are comparison methods.

Planned upstream modifications:

- `vpo_tasks/livecodebench.py`: expose `cheap_tests`, `trusted_tests`, test origin and cost separately.
- `vpo/reward.py`: return cheap candidate-by-test outcomes and metadata; a separate audit service returns \(Y_i\) only when \(D_i=1\).
- veRL `core_algos.py`: add the fixed-target `JE_AIPW` advantage operator.
- veRL `ray_trainer.py`: execute cheap tests, call a frozen/cross-fitted \(q,C\), construct all 256 subset probabilities, persist \(\pi_i,\pi_{ij}\), sample one subset, reveal only selected trusted outcomes, then compute `Y_DR`.
- CodeContests-O: reuse generator, validator and parallel executor while withholding evaluator-side labels from the policy.
- UTRL: reuse generation and execution code for kill-rate and learned-tester baselines.

New modules:

```text
joint_evidence/
  belief.py
  gradient_sketch.py
  audit_policy.py
  dr_reward.py
  test_policy.py
  ope.py
  telemetry.py
```

`audit_policy.py` enumerates the subset distribution and validates inclusion probabilities by summation. `gradient_sketch.py` computes \(L\) on registered LoRA blocks and a fixed Rademacher projection. The same projection seed is used for all arms within a run; exact registered-block gradients audit the design-risk ranking.

## 7. Evidence and noise construction

The realistic offline bank contains 1,500 tasks split 1,000/250/250 by problem identity, three frozen policy checkpoints, \(K=8\) candidates per task and 48 tests per task:

- 12 official/curated tests;
- 12 UTRL-generated tests;
- 12 fault-targeted tests;
- 8 metamorphic tests;
- 4 duplicate or deliberately corrupted controls.

Store the full candidate-by-test outcome matrix, execution time, timeout/exception, coverage signature and test origin. Trusted full-suite \(Y\) is stored only in an offline evaluator sidecar; the trainer receives selected labels.

There are two explicitly separate panels.

In the **synthetic-oracle panel**, real model score geometries \(L\) may be reused, but labels are synthetic: for task/group \(g\), draw \(p_g\sim\operatorname{Beta}(2,6)\) and \(Y_{gi}\sim\operatorname{Bernoulli}(p_g)\), with a registered logit offset for each frozen policy checkpoint. For test cluster \(c\), draw positive/negative flip rates from Beta distributions with the registered means and concentration \(\kappa=1/\rho-1\), so the induced within-cluster flip ICC is \(\rho\); \(\rho=0\) fixes the rate. Then draw \(E\) conditional on \(Y\). Quadrature/enumeration over \(p_g\) and cluster rates yields oracle \(\mu,C\).

In the **real-suite panel**, \(Y\) is the evaluator-side result of a registered full suite and \(E\) is the real cheap outcome matrix. No oracle posterior is claimed; \(\mu,C\) are model-based and cross-fitted. This panel tests transfer, not identification of a Bayes posterior.

The primary medium-noise point is

\[
P(E=0\mid Y=1)=0.10,\quad
P(E=1\mid Y=0)=0.20,\quad \rho=0.60,
\]

with cluster size four. Additional noise is injected only for controlled mechanism claims and is separated from naturally weak tests:

- symmetric and asymmetric outcome flips;
- coverage sizes \(m\in\{1,2,4,8\}\);
- false-negative-biased tests;
- instance-dependent noise based on code length or exception class;
- policy-dependent noise based on checkpoint;
- correlated errors with target ICC \(\rho\in\{0,0.3,0.6,0.9\}\);
- shared systematic errors that fool all candidates in one semantic cluster.

Calibration anchors include trusted positive/negative programs and trusted tests from a task-disjoint calibration split. A full suite may label that calibration split or the offline evaluator sidecar only; its result, test content and coverage signature never enter online features. Repeat key experiments with 1% and 5% anchor corruption.

Marginal ECE is insufficient for \(C\). Report joint log/energy score, pairwise covariance RMSE, correlation calibration by predicted-covariance bins and conditional audit calibration, in addition to Brier/ECE.

## 8. Phase 0 — oracle-gradient audit

### 8.1 Setup

- 256–512 tasks;
- \(K=8\) on-policy candidates;
- 1.5B model for exhaustive plumbing, then 7B LoRA registered-block gradients;
- 32–48 tests per task;
- four primary cheap coverage observations per candidate, taken from the fixed pool as one correlated cluster; `1`, `2`, and `8` are registered coverage sensitivities;
- rank-one synthetic score geometry with one task-permuted dominant candidate, embedded by the fixed 256-dimensional Rademacher sketch for the primary mechanism stratum;
- expected trusted-audit fractions `5%`, `10%`, `20%`, `40%`, and `100%`;
- primary point 10% with inclusion floor `0.02`; 5% sensitivity uses floor `0.005`;
- 200 independent subset draws per task/group/design;
- fixed test pool, no PBPF, no test generator.

Run both panels. Synthetic-oracle generates \(Y,E\) from the fully specified model and isolates the design objective. Real-suite takes \(Y=1\) to mean passing the evaluator-side registered suite—not semantic proof of correctness—and uses cross-fitted model-based \(\mu,C\). Full \(Y\) never reaches the trainer.

### 8.2 Arms

1. cheap reward only;
2. uniform full-support subset design plus Horvitz–Thompson;
3. uniform full-support subset design plus AIPW;
4. entropy-scored subset design plus AIPW;
5. mutation-kill-rate subset design plus AIPW;
6. independent-Poisson Neyman allocation plus AIPW;
7. single-label Bayes-VOI subset heuristic plus AIPW;
8. GOAV exact subset-design objective plus AIPW;
9. deterministic top-k without correction, as a known biased control;
10. 100% trusted audit ceiling.

### 8.3 Metrics

For \(\widehat G\), report normalized MSE

\[
\operatorname{nMSE}=\frac{\sum_g\|\widehat G_g-G_g^*\|^2}
{\sum_g(\|G_g^*\|^2+\epsilon_G)},
\]

where \(\epsilon_G\) is 1% of the dev median squared gradient norm and is frozen before test. Also report standardized design bias, variance, cosine with \(G^*\), coordinate/sign agreement and one-step improvement on an independent trusted minibatch. Joint-model metrics, design-objective calibration, audit ESS, weight p50/p95/p99, support violations and expected/realized cost are mandatory.

Standardized design bias is \(\|\mathbb E_D\widehat G-G^*\|/\sqrt{\|G^*\|^2+\epsilon_G}\), with the expectation estimated from the 200 registered subset draws.

### 8.4 Gate

At the unique primary point—10% expected audit, inclusion floor `0.02`, false-negative `0.10`, false-positive `0.20`, ICC `0.60`, cluster size four—require:

- at least 20% lower normalized gradient MSE than the best uniform/entropy AIPW baseline;
- at least `+0.05` gradient cosine;
- standardized design bias below 5% of the stabilized oracle-gradient norm;
- effective sample size at least 30% of nominal audits and zero support violations;
- sketch-to-exact risk-ranking Spearman at least `0.70`.

The nMSE point reduction must be at least 20% and its pooled seed×task paired 95% interval must have lower bound above zero; cosine and bias are simultaneous safety gates. The primary gate applies to synthetic-oracle. If it fails, stop the direction. Real-suite model-based GOAV must then show at least 10% nMSE reduction with a paired interval above zero before Phase 1.

## 9. Phase 1 — frozen-policy active audit

Build the 1,500-task bank above from CodeContests-O, with EvalPlus used for a smaller transparent mechanism replication. Split every learned component by task and policy checkpoint; no candidate or test outcome from evaluation enters fitting.

The frozen bank has evaluator-side full \(Y\), so Phase 1 directly replays each audit design rather than relying on off-policy evaluation. For every design and group, persist all 256 subset probabilities and verify with 100,000 Monte Carlo draws that empirical \(\pi_i,\pi_{ij}\) differ from their exact sums by at most `0.005`.

Compare GOAV with random, entropy, coverage, disagreement, kill rate, B4-style strategy, NC-GRPO-style noise correction and full audit. Evaluate every selector on held-out task families, checkpoints and unseen noise regimes.

Advance if:

- trusted-gradient risk reduction per CPU-second is at least 15% above the strongest baseline;
- predicted versus realized risk reduction has Spearman correlation at least `0.30`;
- empirical first- and second-order inclusion probabilities match exact values within `0.005`, with zero support violations;
- normalized gradient MSE improves at least 10% on an unseen noise regime.

## 10. Phase 2 — online 7B RL

### 10.1 Fixed-pool stage

- model: Qwen2.5-Coder-7B-Instruct, LoRA rank 32, alpha 64;
- prompt batch 32 or 64, \(K=8\), response limit 2,048 tokens with 4,096 sensitivity;
- eight cheap tests per group, each executed on all \(K\) candidates;
- primary expected trusted-audit budget 10% with floor `0.02`; 5%/20% are sensitivities with feasible registered floors;
- 800–1,200 updates, three seeds;
- first 100–200 updates use a frozen belief and fixed test pool;
- primary update is one on-policy unclipped step; a clipped practical variant is secondary;
- train on TACO/CodeContests, select checkpoints on a disjoint public dev suite, and query the unpublished or strictly post-cutoff grader once.

Primary causal arms all use the same one-step on-policy, unclipped, linear-LOO loss: cheap imputation only, uniform-subset AIPW, entropy-subset AIPW, independent-Poisson Neyman AIPW, GOAV subset-design AIPW and 100% audit. Original NC-GRPO, clipped GRPO and VPO are separately labelled practical task baselines and do not identify the acquisition effect. The trainer sees \(Y_i\) only for \(D_i=1\).

### 10.2 Learned-test stage

Run only if fixed-pool GOAV passes. Alternate four coder updates with one tester update. The tester generates eight tests; the acquisition policy samples one or two. Freeze the tester for the first 200 coder updates and snapshot it every 50 updates to detect non-stationary collusion.

Add these arms:

- UTRL-style kill-rate tester;
- information-gain tester that ignores gradients;
- GOAV tester with risk-reduction reward;
- GOAV tester with shuffled gradient sketches;
- GOAV tester graded by its own test, as a deliberately invalid control.

Measure semantic validity, duplication, execution cost, evaluator-side trusted risk reduction, adversarial false positives and held-out pass@1. The unique primary final comparison is GOAV fixed-pool versus the strongest uniform/entropy/Neyman AIPW arm at 10% expected trusted audit and matched **total verification CPU core-seconds**. Require a point gain of at least `+1.5` absolute sealed pass@1 points and a paired hierarchical 95% interval above zero.

The learned-test extension must additionally improve evaluator-side gradient-risk reduction per total verification second by at least 15% over the UTRL/label-information-gain tester. A 30% total-verification-cost reduction within a `0.5` pass@1 non-inferiority margin versus full audit is a Holm-corrected secondary claim, not an alternative primary gate.

## 11. Essential ablations

- diagonal \(C\) versus full/low-rank joint residual second moment;
- exact subset design versus independent-Poisson Neyman design versus Bayes-VOI heuristic;
- random versus learned gradient projection;
- outcome entropy versus gradient risk at matched propensity;
- AIPW versus inverse propensity only versus imputation only;
- exact and deliberately perturbed inclusion probabilities;
- feasible `(budget, floor)` pairs `(0.05,0.005)`, `(0.10,0.02)`, `(0.20,0.02)`, `(0.40,0.05)`;
- marginally calibrated but covariance-miscalibrated versus jointly calibrated outcome models;
- one-step versus batch acquisition;
- test cost ignored versus modeled;
- fixed tests versus learned test content;
- policy snapshot lag 0/1/5/20 updates;
- independent versus correlated cheap-test errors.

## 12. Failure interpretation

| Observation | Interpretation | Action |
|---|---|---|
| Oracle-model subset GOAV does not beat random/Neyman | joint design-risk objective is not useful | stop |
| Oracle works, learned GOAV fails | covariance/calibration failure | do not train online |
| Selector gains vanish after AIPW | earlier gain was selection bias | reject raw-selector claim |
| Gradient MSE falls but learning does not improve | local target does not predict training dynamics | narrow to estimator result |
| Tester wins only on its own tests | reward hacking/collusion | reject learned-test claim |
| Entropy matches GOAV | gradient geometry adds no value | prefer simpler baseline |

## 13. Expected paper contribution

Joint outcome modeling, randomized survey design and AIPW are existing ingredients. The proposed contribution is their estimator-specific composition: optimize the exact inclusion-probability design MSE of a linear coding policy gradient under correlated residuals, then test whether the same external risk defines a useful test-content policy. Learned test generation is claimed only if it adds value beyond the fixed-pool estimator under a grader it cannot influence.
