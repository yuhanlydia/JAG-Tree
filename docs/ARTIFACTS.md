# Artifact and statistics contract

Runs and banks are written into sibling staging directories and atomically renamed only after all required files exist. Destinations are create-once. Verification recomputes every recorded SHA-256 and rejects a missing completion marker, malformed JSON/JSONL/NPZ, count mismatch, invalid tree genealogy, or coordinated task-manifest mismatch.

A result directory contains:

- `result.json`: status, config identity, seed, model revision, bank path, and online update losses;
- `rows.jsonl`: task-level rows, with update index for online runs;
- `manifest.json`: source configuration paths and canonical config SHA-256;
- `checksums.json`: SHA-256 for the preceding evidence files;
- `COMPLETE`: written last.

The frozen bank is independently verified before replay. `jag-tree verify RUN` verifies result artifacts; `verify_bank(PATH)` verifies candidate banks. Neither command upgrades `INCOMPLETE` evidence to `PASS`.

Scientific `PASS` verification is intentionally unsupported in this CPU release because no formal stage evaluator is shipped. `PASS` artifacts fail closed. Formal execution therefore cannot proceed until a later, registered evaluator can recompute complete stage statistics and verify prerequisite identities.

Required formal analysis includes per-task rows, individual seed curves, Pass@1, sampled Pass@8, token-normalized AULC, gradient bias/MSE/cosine, sibling ICC, signed effective sample size, the three cost ledgers, model/data/environment commits, and artifact checksums. Formal online inference uses 10,000 hierarchical paired bootstrap replicates over base tasks and Holm correction for secondary hypotheses.

The registered success rule is at least 10% relative token-normalized AULC improvement with a hierarchical paired 95% interval above zero, or at least 20% fewer tokens to the baseline endpoint. LiveCodeBench is a secondary target and BigCodeBench Full has a non-degradation guard. These are preregistered thresholds, not reported results.

Keep raw logs and immutable artifacts under a run-specific output root. Never edit evidence in place; launch a new identity and preserve the invalidated directory for audit.
