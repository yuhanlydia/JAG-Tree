# SDD ledger — plan: docs/superpowers/plans/2026-09-08-jag-tree-7b-experiments.md

| Tasks/interfaces | Producer → consumer | Preflight finding |
|---|---|---|
| 1 → 2–6 | config/registry → all runners | consistent |
| 2 → 3–5 | immutable bank/ledger → audit and trainer | consistent |
| 3 → 4–5 | estimator/allocator → frozen and online runner | consistent |
| 4 → 5 | backend/benchmark/statistics → trainer | consistent |
| 5 → 6 | configs/scripts/artifacts → docs/CI | consistent |
| 1 | tests, files and interfaces | consistent |
| 2 | tests, files and interfaces | consistent |
| 3 | tests, files and interfaces | consistent |
| 4 | tests, files and interfaces | consistent |
| 5 | tests, files and interfaces | consistent |
| 6 | tests, files and interfaces | consistent |

## Independent review fix round 1

| Slice | Red evidence | Green evidence | Commit |
|---|---|---|---|
| Full recursive genealogy and unique-edge accounting | 4 leaves and 32 stored tokens for the registered 4x3x3x3 tree | 108 leaves and 296 incremental edge tokens | `37af3b6` |
| Immutable bank identity | mutable arrays/manifests and surplus files accepted | arrays/manifests deep-frozen; exact inventory and generation identity verified | `0f7616e` |
| Formal validation and split leakage | invalid profile/revision/arms/update escape and cross-role duplicate accepted | strict formal contract and content-derived connected-component rejection | `d2163b7` |
| Generated-code isolation | unrestricted local execution | trusted-fixture subprocess or network-disabled resource-limited immutable container | `7f67190` |
| Frozen stochastic audit | reward-derived moments, token-ID gradients, discarded outputs, oversubscription | sealed predictor, actual score interface, support-positive CRN replay, measured costs, importance weights, MC metrics and retained arm ledgers | `d6dfc07`, `456003c` |
| Recursive training | leaf-equal response-only centered reward and reinitialized optimizer | unique conditional edge log-probabilities, recursive/global transport, independent/LOO controls, persistent adapter/optimizer checkpoints | `456003c` |
| Evidence/statistics/adapters | permissive PASS and false adapter labels | exact artifact inventory, fail-closed PASS, Pass@1/8+ICC/ESS+Holm utilities, unimplemented recipes explicitly unavailable | `2fa9df7` |
| Review-2 adversarial boundaries | infrastructure-as-wrong, selected-edge bypass, task-test escape, mutable arrays, alias/formal downgrade, empty PASS | abort/quarantine, selected-only training, marker+I/O harness, immutable backing, verified formal artifacts, evidence-bound PASS | final hardening commit |
