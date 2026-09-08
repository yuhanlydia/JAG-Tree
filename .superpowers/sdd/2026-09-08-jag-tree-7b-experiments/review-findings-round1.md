# Independent review findings — round 1 disposition

Date: 2026-09-08

No 7B, GPU, benchmark, or empirical-superiority evidence was produced in this round. `online_h200.yaml` remains `formal: false` and all CPU smoke output remains globally `INCOMPLETE`.

| Finding | Disposition | Evidence |
|---|---|---|
| Recursive branching ignored `children_per_branch`; cumulative prefixes double-counted | Fixed | `build_genealogy` now creates 4x3^3=108 leaves and stores only each conditional edge; regression test checks exact token sum. |
| Frozen allocation leaked sealed outcomes and substituted token IDs for gradients | Fixed | Non-oracle arms require an immutable calibration predictor; policy backend must return finite registered score arrays for every edge; flipped reward tests preserve proposals. |
| Audit seed unused and subset error mislabeled MSE | Fixed | Support-positive stochastic trajectory proposals use seeded common uniform streams, target/proposal probabilities, importance weights, repeated estimates, draw count, MC MSE/bias/cosine/SE/z. A 20,000-draw finite-target test checks variation and convergence. |
| Allocation requested unavailable children / ignored measured token costs | Fixed | Replay draw count is bounded by the maximum measured path cost; every trajectory ledger is measured and cannot exceed budget. |
| Arms were identical and outputs discarded | Fixed | Literal arm risks produce distinct proposal laws; `FrozenAuditResult.arm_replays` retains rows, metrics, calibration identity and separate ledgers. Oracle outcome access is isolated to `oracle_moments`. |
| Training was leaf-equal and nonrecursive | Fixed | `recursive_training_targets` assigns one row per unique edge, recursive subtree values, global task/branch transport, and explicit independent or sibling-LOO controls; uneven-tree test checks the analytical REINFORCE total. |
| Training/audit causal objectives differed; adapter and optimizer reset | Fixed | Both score sketches and training use `causal_edge_logprob`; training conditions on the original stored chat prompt plus ancestor edge IDs. LoRA model and optimizer persist across updates and create immutable adapter/optimizer checkpoints. GPU execution remains unverified. |
| Training budget comparison was the same object twice | Fixed | Actual edge tokens and optimizer step are recorded after execution into a distinct measured ledger and compared to the registered expected ledger. GPU-hours remain zero until a GPU runner supplies measurement, so formal evidence is ineligible. |
| Generated programs ran unsandboxed and exit status replaced tests | Demoted fail-closed | Real subprocess and OCI backends do not execute candidate code and always return infrastructure failure because a trusted external verifier is not shipped. Only deterministic `FakeSandbox` fixtures may PASS. |
| Formal validation allowed invalid models/profiles/revisions/arms/update counts and skipped gates | Fixed | Formal contract pins active task/model/data revisions, primary 7B/H200 profile, exact seeds/tree/budget/update schedule, closed arms, calibration role/SHA/cross-fit count, and E0+E1 PASS identities before E2. Runtime seed must be registered. |
| Cross-role duplicates passed under different lineage labels | Fixed | Manifest uses connected components over declared lineage plus normalized statement similarity, AST/code fingerprints and test-I/O fingerprints before split-role validation. |
| Baselines/evaluators claimed adapters without executable dispatch | Demoted | Every unimplemented recipe is `mode: unavailable`, `executable: false`, with a reason; unavailable recipes are rejected by experiment validation. No official-adapter claim remains. Public benchmark loaders remain lazy interfaces and were not executed. |
| Metrics and evidence statuses were disconnected/permissive | Fixed with formal limitation | Runs retain replay ledgers and MC gradient metrics; statistics expose task×seed Pass@1/8, ICC/ESS and Holm correction. PASS artifacts require `evidence_complete`, all nonempty gates PASS, zero infrastructure failures and exact checksum inventory. Current launchers emit `INCOMPLETE`; no formal report aggregator is claimed. |
| Candidate bank identity was incomplete/mutable | Fixed | Bank arrays and nested manifest structures are read-only; exact payload inventory, task manifest, policy revision, template version and generation identity are verified on reuse. |
| Genuine cross-fit calibration training artifact | Not supported in CPU round | The Transformers path only loads a sealed calibration-role artifact bound to the policy and formal config requires predictor SHA/cross-fit metadata. No fitter or 7B calibration artifact was fabricated; therefore E1/E2 remain ineligible until an independently generated calibration artifact is verified. |

## Round-2 adversarial follow-up

| Finding | Disposition | Evidence |
|---|---|---|
| Infrastructure result scored as wrong | Fixed | Bank creation raises and writes no sealed reward/bank on infrastructure failure. |
| Online ignored replay selection | Fixed | Runner passes the configured arm's selected edge IDs; model batches only those IDs and compares a separately measured ledger. |
| Per-draw MSE mislabeled estimator MSE | Fixed | Both quantities are named; estimator-mean MSE is per-draw MSE divided by draw count. |
| Task tests bypassed by process exit or JSON-as-expression | Fixed | PASS requires a post-test marker; `os._exit(0)` is not PASS; JSON input/output tests execute each case and compare normalized output. |
| NumPy write flag could be re-enabled | Fixed | Arrays are reconstructed over immutable bytes; regression calls `setflags(write=True)` and expects failure. |
| Formal aliases/downgrade/gates bypassed | Fixed | `training.updates` is forbidden; formal activation cannot execute as nonformal; formal execution verifies checksum-bound PASS artifacts. |
| Zero-budget or unsupported online/matrix execution | Fixed | Positive allocation and activated formal gates are preconditions for non-plumbing online work; matrix-only configs have no dispatch and are rejected. |
| Invented empty PASS artifact | Fixed | PASS requires exact registered gates, nonempty rows, config/bank bindings, matching row digest, zero infrastructure failures and exact inventory. |
| Bootstrap erased common seed effects | Fixed | Seeded hierarchical bootstrap uses a shared seed resampling vector across task clusters. |
| GPU generation EOS/tokenization/LoRA/checkpoint-resume audit | Not supported | No GPU path was executed or promoted to evidence. Non-plumbing online execution is gated; formal E1/E2 remains ineligible. These require hardware-backed validation before support can be advertised. |
| Dense exact gradients retained during sketching | Fixed | Non-exact operation streams registered gradient chunks directly into the sketch and returns no dense gradient/projection. |
| Split manifest built after generation | Fixed | Content/lineage connected components are sealed and validated before the first backend generation call. |
| PASS gate labels not recomputed | Fixed within advertised artifact schema | Verifier reconstructs the exact registered gate map from checksum-bound evidence rows and rejects discrepancies. Scientific threshold production remains unsupported without formal artifacts. |

## Round-3 adversarial follow-up

| Finding | Disposition | Evidence |
|---|---|---|
| Public success sentinel/exit-code caller-frame bypass | Fixed by disabling execution | No same-interpreter marker, secret, or exit status is used. Real backends return infrastructure failure before candidate execution; literal string-constant and integer-constant frame exploits cannot PASS. |
| Separate globals/locals broke valid TACO programs | Fixed | Candidate execution uses the same namespace for globals and locals; a function reading a module-global variable passes an I/O case. |
| Selected-edge training omitted inclusion correction/multiplicity | Fixed | Replay rows include path IDs and importance weights; online training aggregates every draw onto unique edges with `weight/N`. CPU test covers repeated paths and nonuniform weights. |
| Arbitrary row could substantiate PASS | Fixed by demotion | Scientific PASS verification is unsupported in this CPU release and every PASS artifact fails closed. No generic row schema can claim formal evidence. |
| Predictor covariance write flag escape | Fixed | Covariance is backed by immutable bytes; re-enabling writes raises. |
| Branching continued after EOS | Fixed | Genealogy accepts a terminal predicate and removes terminal edges from the frontier; Transformers binds tokenizer EOS. |
