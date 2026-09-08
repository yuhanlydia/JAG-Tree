# Data and evaluation protocol

Task splitting happens before candidate generation. The manifest groups exact/MinHash-normalized statements, normalized AST/token code, test I/O, source lines, and source lineage; a lineage group may occupy only one role.

| Dataset | Role |
|---|---|
| TACO | filtered train, dev, and disjoint 256-task frozen audit |
| LiveCodeBench v6 | public contest-code evaluation; not described as sealed |
| BigCodeBench Full and Hard | practical/API evaluation and registered hard subgroup |
| HumanEval+ and MBPP+ | inexpensive exposed compatibility smoke |
| evaluator-held post-cutoff set | one locked query after checkpoint and scaffold freeze |

Pinned public revisions are in `configs/benchmarks/`. Remote adapters import `datasets` only inside `load_tasks`; JSONL remains the offline boundary. Every normalized row requires task ID, statement, starter code, tests, source, lineage group, and role. Missing fields abort ingestion.

Candidate banks contain create-once `tasks.jsonl`, `nodes.jsonl`, `scores.npz`, `manifest.json`, checksums, and a completion marker. Parent depth, task identity, model revision, template version, and checksum mismatches invalidate the bank. Public hidden tests are hidden only by protocol and are never called genuinely unseen.

Generation, verification, and optimization ledgers remain separate. Controlled comparisons match programs exactly, optimizer steps exactly, cumulative unique continuation tokens within 1%, and sandbox CPU seconds within 5%. Wrong answer, exception, timeout, and infrastructure failure are distinct outcomes.

Formal task rows preserve model/data commits, source lineage, prompt/template version, reward, tree identity, unique-token cost, program executions, sandbox CPU seconds, optimizer costs, and environment/container identity. The base task—not a leaf or execution—is the statistical unit.
