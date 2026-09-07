# Direction 2 — PBPF

Predictive Belief Particle Filtering for test-time program repair.

## 1. The assumption being challenged

Most iterative coding agents compress execution history into one textual transcript, one hidden state, or one current patch. That silently assumes the observed failures identify a nearly unique task/bug explanation. In practice, a short prefix of tests is often compatible with several explanations that imply different repairs.

PBPF keeps a distribution over explanations and asks a falsifiable question:

> Does a calibrated posterior over task, bug and candidate hypotheses predict unseen execution outcomes—and improve repair—better than a single summary state at the same visible context and execution budget?

This is not a claim that “KV cache contains useful information.” KV injection is only one controlled interface. The scientific claim is about posterior uncertainty and future-outcome prediction.

PBPF must be evaluated under a fixed test order. If it chooses tests, its effect becomes confounded with Direction 3.

## 2. Probabilistic model

For specification/context \(s\), candidate programs \(A_{1:G}\), latent task hypothesis \(H\), candidate source hypotheses \(J_{1:G}\), candidate-specific bug classes \(C_{1:G}\), mutation sites \(M_{1:G}\), tests \(x_{1:T}\), and execution outcomes \(E_{g,t}\), the finite synthetic model is

\[
p(H,J,C,M,A,E\mid s,x)=p(H\mid s)
\prod_{g=1}^{G}p(J_g\mid H)p(C_g\mid J_g)p(M_g\mid J_g,C_g)
p(A_g\mid J_g,C_g,M_g)
\prod_{g,t}p(E_{g,t}\mid H,A_g,x_t).
\]

Phase 0 fixes \(p(H\mid s)\) to uniform over the 64 canonical hypotheses. To prevent a fully observed deterministic mutant from disclosing \(H\) before any test executes, each candidate source \(J_g\) equals \(H\) with probability `0.20`; otherwise it is sampled uniformly from the eight hypotheses in the same affine-operator family. This `0.80` source-contamination rate is part of the checked-in manifest. The model fixes the categorical bug prior, takes \(M_g\) uniformly over valid edit sites, and makes \(p(A_g\mid J_g,C_g,M_g)\) the deterministic mutation kernel. The execution likelihood still compares \(A_g(x_t)\) with \(H(x_t)\). After the full program \(A_g\) is observed, \(J_g,C_g,M_g\) are marginalized rather than inserted redundantly into that likelihood.

The product over candidates is the exact synthetic data generator. It is only a modeling approximation for correlated samples from a real LLM and is tested against an exchangeable set encoder in Phase 1.

The online posterior after a fixed prefix is

\[
q_t(H,J_{1:G},C_{1:G},M_{1:G})=q(H,J_{1:G},C_{1:G},M_{1:G}\mid s,A_{1:G},x_{1:T},E_{:,1:t}).
\]

The protocol is input-known/output-hidden: canonical test inputs \(x_{1:T}\) are visible from the start, while assertions, expected values and future execution outcomes are not. Only the next manifest test may be executed at each round.

The outcome alphabet is `PASS`, `WRONG`, `EXCEPTION`, and `TIMEOUT`; when permitted, `WRONG` additionally carries a normalized output-difference embedding. The principal proper-scoring target is the posterior predictive distribution

\[
\hat p(E_{:,t+1:T}\mid E_{:,1:t})
=\sum_{m=1}^{P}w_t^{(m)}
p(E_{:,t+1:T}\mid z_t^{(m)},A,x_{t+1:T}).
\]

In Phase 0 each particle is the discrete static state \(z^{(m)}=(H,J_{1:G},C_{1:G},M_{1:G})\). With initial proposal \(r_0\),

\[
w_0^{(m)}\propto
\frac{p(H^{(m)}\mid s)\prod_g p(J_g^{(m)}\mid H^{(m)})p(C_g^{(m)},M_g^{(m)},A_g\mid J_g^{(m)})}
{r_0(z^{(m)}\mid s,A)},
\]

and the fixed-order Bayes update is

\[
w_t^{(m)}\propto w_{t-1}^{(m)}
\prod_g p(E_{g,t}\mid H^{(m)},A_g,x_t).
\]

After resampling, one manifest-pinned Gibbs/Metropolis rejuvenation step leaves the exact \(q_t\) invariant. A full-support categorical decoder \(q_\phi(H,J,C,M\mid z_{1:P},w)\), with an explicit probability floor, supports finite synthetic posterior cross-entropy/KL even when an empirical particle set misses a state.

On realistic tasks, \(H,J,C,M\) are not identifiable labels. The model is therefore called an **amortized particle predictive belief**, not a calibrated semantic posterior. Each 256-dimensional neural particle follows proposal \(r_\phi(z_t\mid z_{t-1},o_t)\), transition prior \(p_\phi(z_t\mid z_{t-1})\), and likelihood \(p_\phi(E_t\mid z_t,A,x_t)\), with log-weight increment

\[
\Delta\log w_t=\log p_\phi(E_t\mid z_t,A,x_t)
+\log p_\phi(z_t\mid z_{t-1})
-\log r_\phi(z_t\mid z_{t-1},o_t).
\]

The base implementation uses \(P=8\), log-space weights, an evidence encoder, and GRU/MLP proposal, transition and likelihood networks. Discrete ancestor indices are stop-gradient; proposal/likelihood learning comes from the supervised fixed-prefix objective, not an unreported straight-through resampling estimator. Log pre-resampling ESS, unique-ancestor ratio and post-rejuvenation unique-state ratio. Systematic resampling occurs only when effective sample size

\[
\operatorname{ESS}=1/\sum_m(w_t^{(m)})^2 < P/2.
\]

In the finite Phase-0 reference, resampling is suppressed when the positive-weight particles already represent every exact finite-support \(H\) state. Dropping a represented low-mass state in that case can create avoidable particle extinction at the next deterministic observation. This support check uses enumeration only in the synthetic reference and is not an available Phase-1 operation.

Split/merge moves are permanently excluded from version 1. Any future split/merge study receives a new preregistration rather than being triggered adaptively from these results.

## 3. Training objective

For a sampled cutoff \(k\), the primary loss predicts the entire suffix from the same fixed prefix:

\[
\mathcal L_{\text{PBPF}}(k)
=-\sum_{\tau>k,g}\log
\hat p(E_{g,\tau}\mid s,A,x_{1:T},E_{:,1:k})
+\lambda_{syn}\operatorname{KL}(p^*_{syn}\|q_\epsilon)
+\lambda_C\mathcal L_C.
\]

- Future-outcome negative log likelihood is always present.
- The synthetic posterior KL is used only where the exact posterior is known.
- Bug-class supervision is auxiliary and is removed on datasets without trustworthy labels.
- Repair reward is not used to define or tune the belief quality metric.

The suffix is not teacher-forced while computing this metric. A separate prequential score may reveal \(E_{:,k+1:\tau-1}\) before predicting \(E_{:,\tau}\), but is named and reported separately. The model is trained with randomly sampled cut points, so an outcome can never be both an input and its own target.

## 4. Belief-to-policy interfaces

All interfaces receive the same visible task, candidate code, test inputs and observed outcomes.

| Interface | What the repair model receives | Role |
|---|---|---|
| Transcript | Raw observations in text | strong ordinary baseline |
| Single state | One posterior mean/MAP vector | tests whether multimodality matters |
| Soft prompt | Learned belief tokens | expressive upper bound |
| KV mixture | One low-rank \(\Delta K_m,\Delta V_m\) and decoder forward per particle; probabilities mixed by \(w_m\) | primary controlled interface |
| Oracle text | Ground-truth synthetic hypothesis | diagnostic ceiling only |

For the KV-mixture arm, the visible token IDs are identical to the transcript baseline. Each particle is projected separately, each forward produces \(p_m\), and prediction is mixed in probability space as \(\sum_mw_mp_m\). Projecting \(\sum_mw_mz_m\) and running once is named the posterior-mean baseline, not a particle mixture. During repair, each generated continuation samples one particle index from \(w\); the rollout budget is allocated before outcomes. Add an eight-member deterministic ensemble with identical forward count, total latent capacity and decoder budget. Report the full quality–FLOPs/latency frontier. Raw hidden hypotheses are never appended as text in the primary comparison.

## 5. Code base and exact modification surface

The recommended base is [UpSkill](https://github.com/dshah02/upskill) for multi-turn coding RL, with cache mechanics adapted from [KV Cache Steering](https://github.com/MaxBelitsky/cache-steering). [RSP](https://github.com/heejunkim00/RSP) and [LaDi-RL](https://github.com/mk322/LaDi-RL) supply representation baselines, not the posterior algorithm.

Upstream touch points:

- `src/DAPO_math_dapo.py`: replace the single state passed between turns with a `BeliefState` handle.
- `UNSLOTH_rewards.py` and `flex_rewards_adapter.py`: return the per-test categorical outcome vector, not only aggregate pass rate.
- `src/steering/cache_steering.py::{precompute_kv_cache,steer_kv_cache,steer_kv_cache_layer}`: add batched particle-conditioned low-rank deltas and an unmodified-cache control.
- RSP's `generate_random_embedding.py` and `eval.py`: reuse for random-latent and matched-parameter controls.
- LaDi-RL's `model_dual_proj.py` and `fm_scheduler.py`: reuse only as a deterministic latent compression baseline.

New modules planned for implementation:

```text
belief_pf/
  data/{crossbeam_episodes,codearc_adapter,codecontests_matrix,split_manifest}.py
  belief/{state,proposal,likelihood,particle_filter}.py
  kv/{dynamic_cache_adapter,lowrank_projector}.py
  repair/{rollout,reward}.py
  eval/{prefix_forecast,metrics,leakage_checks}.py
```

Every episode artifact stores dataset revision, split manifest hash, immutable test-order hash, candidate hashes and execution-container digest.

## 6. Phase 0 — exact-posterior mechanism experiment

### 6.1 Environment

Construct a finite typed scalar/list DSL based on [CrossBeam](https://github.com/google-research/crossbeam):

- normalized AST depth at most 4;
- 64 input assignments per task;
- 64 task hypotheses after truth-table canonicalization;
- \(G=4\) candidate programs;
- eight bug classes: correct, operator swap, constant \(\pm1\), comparison boundary, argument/order, missing branch/filter, index \(\pm1\), and aggregator/type/exception.

Ignoring edit-site multiplicity, the nominal \((H,J,C)\) space has \(64\times(8\times8)^4\) states. The exact implementation enumerates the 64 values of \(H\), marginalizes each candidate's \((J,C,M)\) explanations independently, and matches explicit brute force on audit-sized fixtures before using that factorization.

Create 50,000/5,000/5,000 train/dev/test episodes. Split by normalized AST plus operator multiset, and remove any cross-split pair with the same 64-input truth table.

### 6.2 Prefix protocol

Use fixed, hashed test orders and prefixes \(t\in\{0,1,2,4,8,16\}\). Score prediction of the next one, next eight and all remaining outcomes. Expected outputs are hidden from the inference model until an outcome is revealed.

### 6.3 Arms

1. Exact Bayesian posterior.
2. PBPF with \(P\in\{1,4,8,16,32\}\).
3. Single MAP hypothesis.
4. Deterministic GRU state with matched parameter count.
5. Bag of observed outcomes, order removed.
6. Shuffled evidence.
7. Random soft prompt / random cache delta.

### 6.4 Unit-level invariants

- posterior probabilities sum to one within `1e-8`;
- sequential and batch Bayes agree within `1e-8`;
- factorized and brute-force posteriors agree within `1e-8`;
- the predictive distribution agrees with posterior marginalization within `1e-6`;
- permuting candidate IDs and applying the inverse permutation leaves scores unchanged;
- no target outcome is reachable from the prefix data loader.

### 6.5 Metrics and gate

Report fixed-prefix future NLL per test, prequential NLL separately, Brier score, ECE, full-support posterior KL to exact Bayes, 90% credible-set coverage for the \(H\) marginal, ESS, resampling rate and particles per correct equivalence class.

Advance only if:

- the exact posterior predictive at prefix 4 improves over the prior predictive by at least `0.05` nats/test;
- the exact posterior mixture beats its exact MAP point prediction by at least `0.02` nats/candidate-test and has a paired 95% interval above zero, establishing that retaining uncertainty matters;
- the one-sided 95% upper bound on the \(P=16\)-to-exact NLL gap is below `0.02` nats/test;
- PBPF closes at least 80% of the exact-MAP-to-exact-mixture NLL gap;
- a randomized 90% HPD set includes its boundary state with the probability needed to make posterior mass exactly `0.90`; the test-set 95% interval for its \(H\)-marginal coverage must contain `0.90`. Exact Bayes is never temperature-calibrated;
- calibrated-on-dev PBPF beats the matched deterministic state on held-out AST/operator families with a paired hierarchical 95% interval above zero.

If exact prediction does not beat the prior, the task supplies no useful evidence. If the exact mixture does not beat exact MAP, the multimodality claim stops even if a neural representation later helps.

## 7. Phase 1 — frozen 7B future-outcome prediction

### 7.1 Data

Use two layers:

- controlled OOD transfer: CrossBeam plus [CodeARC](https://github.com/Anjiang-Wei/CodeARC);
- realistic joint candidate errors: CodeContests, or MBPP+ for a smoke run, with \(G=8\) total candidates: six frozen model samples and two registered bug-template mutants.

Split base-task/program families first; every candidate, mutant, test and prefix inherits that split. Candidates are generated once with Qwen2.5-Coder-7B-Instruct at temperature `0.8`, top-p `0.95` and an independent candidate-bank seed, then frozen and shared across arms. CodeARC is held-out mechanism transfer, never mixed into fitting. Test order is a stable hash of task ID, test ID and its own preregistered seed. Changing the seed is a separate replicate, never an adaptive choice.

### 7.2 Training

- Freeze the 7B coder.
- Train an evidence encoder, proposal, likelihood and interface with fewer than 20M parameters.
- Primary \(P=8\); ablate \(P\in\{1,4,16,32\}\).
- Sample prefix length uniformly and score only unseen suffixes.
- On 16GB, attach to the last four transformer layers; on 24GB/H200, compare half/all registered layers.
- Keep visible context length identical across all representation arms. If raw transcript exceeds it, apply the same deterministic truncation before every arm.

### 7.3 Baselines and ablations

- transcript-only and transcript plus aggregate pass rate;
- MAP and posterior mean;
- deterministic UpSkill state;
- RSP random latent;
- LaDi deterministic latent;
- eight-member deterministic ensemble with matched forward passes and latent capacity;
- weights reset to uniform after every observation;
- no resampling;
- candidate-independent bug posterior;
- outcomes shuffled within task;
- cache delta attached to random layers;
- same cache delta for all particles;
- raw posterior text and learned soft prompt as upper bounds.

### 7.4 Gate

The single primary gate is at least 5% relative macro-task fixed-prefix future-NLL reduction versus the best matched-compute deterministic state selected on dev, with the base-task-family paired hierarchical 95% interval above zero. Brier must not worsen; ECE and classwise/adaptive calibration are diagnostic.

For NLL, let \(g=L_{base}-L_{PBPF}>0\) and define control attenuation as

\[
1-\frac{L_{base}-L_{control}}{g}.
\]

Both shuffled-evidence and matched-norm RSP controls must attenuate at least 80% of the gain; if \(g\le0\), the gate fails automatically.

Failure means the belief representation is not predictive enough; do not start repair RL.

## 8. Phase 2 — fixed-test iterative repair

All canonical input-only representations in the **fixed interaction manifest** are visible from the start, but at round \(t\) the agent may execute only the next test, after which it updates \(q_t\) and proposes a repair. This train/dev manifest is not a final locked grader. The agent cannot generate, choose, skip, repeat, reorder or adaptively stop tests; expected outputs and full assertions are never visible. Every arm executes the same candidates, tests and four rounds. Status-only evidence is primary, and any output-difference feature is a separately permissioned ablation.

The actor reward and belief loss are separate:

\[
R_{actor}=\operatorname{passrate}_{remaining}
-\beta\,\operatorname{patchsize}
-\gamma\,\mathbb 1[timeout].
\]

The fixed-prefix \(\mathcal L_{PBPF}\) trains only the belief modules and KV projector; it is stopped at actor LoRA. \(R_{actor}\) trains only actor LoRA and is stopped at belief modules/KV projector. The primary coefficients are \(\beta=0.02\) for `min(changed_tokens/256,1)` and \(\gamma=0.10\); a zero-patch-penalty arm is secondary. Prediction is also audited on a frozen candidate bank so it cannot look better merely because the actor changed its data distribution.

Each slot has `(lineage_id, version, code_hash)`. A patch creates a new version in one slot. Task-level \(H\) evidence from ancestors remains available, while the new program receives a fresh candidate-specific bug state; an ancestor's pass/fail is never counted as an execution of the patched code. Previously revealed inputs may be rerun only if every arm does so and the cost is charged.

Start with eight candidates, choose one active slot by highest predicted remaining-suite success with generation log probability as the fixed tie-break, and replace only that slot after each repair. Every representation arm trains the same-capacity success head and uses this selector; only its state input differs. After four rounds, return exactly one program using the same registered selector. This defines pass@1 and prevents an unreported best-of-eight oracle.

### 8.1 Configuration

- model: Qwen2.5-Coder-7B-Instruct;
- LoRA: rank 32, alpha 64, BF16 formal run;
- \(G=8\) candidates and at most 4 repair rounds;
- 2,048 response tokens primary; 4,096-token sensitivity;
- train on TACO/CodeContests; use protocol-hidden EvalPlus only as public secondary evidence, then evaluate on a temporally valid and/or unpublished evaluator-held coding suite;
- three seeds, with one fixed 1.5B plumbing run first.

### 8.2 Comparisons

1. transcript-only repair;
2. aggregate-pass-rate state;
3. deterministic latent state;
4. PBPF posterior mean;
5. PBPF particle mixture through soft prompts;
6. PBPF particle mixture through low-rank KV deltas;
7. oracle hypothesis on synthetic tasks only.

Match generated tokens, number of executions and maximum visible tokens. Report held-out pass@1 after each repair round, future NLL, calibration, patch size, timeout rate, GPU-hours and CPU core-seconds. Executions-to-first-correct is descriptive because the primary protocol always runs four rounds.

Advance only if PBPF obtains at least `+3.0` absolute held-out-pass@1 points at matched execution cost, with a paired hierarchical 95% interval above zero, while retaining the Phase 1 prediction advantage. Execution reduction is secondary and Holm-corrected rather than an alternative success gate.

## 9. Failure interpretation

| Observation | Interpretation | Action |
|---|---|---|
| Exact posterior gives no predictive gain | ambiguity is not useful in the constructed task | stop |
| Exact works, particles fail | inference approximation problem | improve proposal; do not train RL |
| Particles predict but repairs do not improve | posterior is sufficient for forecasting but not action | publish mechanism result or stop |
| Text works but KV does not | interface failure, not belief failure | use text/soft prompt; narrow the claim |
| Gain disappears with equal context length | benefit was extra information/tokens | reject claim |
| PBPF helps only when it chooses tests | confounded with active verification | transfer to Direction 3 only after isolated audit |

## 10. Expected paper contribution

The publishable result is not “an agent remembers test history.” It is one of two clean outcomes:

1. an exact-posterior result on finite tasks plus a calibrated predictive particle belief on realistic tasks predicts future program behavior and causally improves fixed-test repair; or
2. a strong negative result showing that predictive belief can be learned but collapses to a single sufficient state for these coding tasks.
