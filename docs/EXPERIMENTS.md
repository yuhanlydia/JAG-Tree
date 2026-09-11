# Experiment execution

JAG-Tree separates planning, data preparation, frozen replay, and online optimization. A successful command is an operational result; it is not a scientific success claim.

## Install and inspect

```bash
git clone https://github.com/yuhanlydia/JAG-Tree.git
cd JAG-Tree
git pull --ff-only
python -m pip install -e '.[test]'
jag-tree doctor
```

The base install contains NumPy, SciPy, PyYAML, and the CPU tests. Install `.[models]`, `.[benchmarks]`, or `.[training]` only on the worker that needs that boundary. Optional imports occur when the adapter executes, not when it is imported or planned.

## Plan and prepare

Planning validates registries, recursively resolves `base_config`, prints canonical JSON, and performs no model or dataset download:

```bash
jag-tree plan configs/experiments/smoke.yaml
jag-tree prepare configs/experiments/frozen_24gb.yaml --output prepared/frozen-24gb.json
```

`prepare` uses exclusive creation. An existing output is an error. Before a remote run, the operator must compare every model and dataset commit in the prepared JSON with the intended preregistration and record the actual container digest in the sealed run overlay. `online_h200.yaml` is a preregistered launch template and is not formal evidence by itself.

## Run and verify

The offline integration smoke is deterministic and uses no downloaded weights or data:

```bash
scripts/run_smoke.sh runs/smoke 7
jag-tree verify runs/smoke/run-<config-sha-prefix>-seed7
```

The model-backed launchers require optional dependencies, local/container execution permission, and the pinned data/model revisions:

The `subprocess` and OCI backends return an infrastructure failure because no external test-completion verifier is shipped. Dedicated pilot workers can explicitly select `trusted-local`, which runs candidates as an unprivileged process with resource limits. It is unavailable for formal runs.

For the outcome-blind structural pilot predictor and the small rank-2 LoRA gradient sketch:

```bash
PYTHON=.venv/bin/python scripts/run_frozen_7b.sh runs/screen16 17 trusted-local configs/pilots/jag_screen16.yaml
```

This uses `--sandbox trusted-local --allow-pilot-predictor`. The predictor has constant value and gradient variances and zero cross covariance, so `uniform`, `value_variance`, and `gradient_only` coincide and this pilot cannot establish a learned joint-moment advantage. The sketch covers one late adapter tensor, not the formal multi-block audit. The default container behavior remains fail-closed.

The screen JSONL is local input. To reproduce the September 2026 worker subset, download `ALL/train-00000-of-00009.parquet` from `BAAI/TACO` at revision `6e7429e9bcfb0e8d7aebfc719d98518f770b3985`, then run:

```bash
.venv/bin/python scripts/prepare_taco_screen.py /root/jag-data/taco-train-00000.parquet /root/jag-data/taco-screen16.jsonl
```

The script checks the source SHA-256, selects tasks before generation, preserves full tests, and writes a create-once provenance sidecar. It requires PyArrow from the dataset dependencies.

```bash
scripts/run_frozen_7b.sh runs/frozen-24gb 17 python@sha256:<64-hex-digest> configs/experiments/frozen_24gb.yaml
scripts/run_online.sh runs/online-h200 101 python@sha256:<64-hex-digest> configs/experiments/formal-overlay.yaml
```

Frozen arms replay one candidate bank with common random numbers. Online execution creates a distinct immutable bank for every configured update and calls exactly one optimizer step for that bank. The H200 template schedules 800 updates; screening and formal runs must use separate output roots and configurations. A path collision aborts rather than overwriting evidence.

## Stages and gates

- E0 uses exact finite MDPs at budgets 64, 128, and 256. The formal sampling count is 100,000 per cell over 50 MDP seeds.
- E1 uses 256 calibration and 256 disjoint TACO audit tasks, four roots, branch depths 256/512/768, three children, a 108-leaf cap, and 4,096/8,192/16,384 unique-token replays.
- E2 uses 15 shared uniform warm-up updates, seed 17 screening for 200 updates, then seeds 101/202/303 for 800 formal updates. Prompt batch is 64, group cap 8, response cap 2,048, and nominal unique-token budget 8,192 per prompt/update.

The frozen mechanism gate must pass before online 7B training. `online_h200.yaml` is intentionally non-executable as formal evidence; `run_online.sh` therefore requires an explicit sealed formal overlay. In this CPU release scientific `PASS` verification is unsupported, so non-plumbing online execution fails closed. Config validation rejects unknown arms, unsupported hardware, mutable formal revisions, missing formal roles, missing container digests, and the registered formal seed mismatch.

## Status semantics

- `PASS`: complete formal evidence satisfies every registered primary gate.
- `FAIL`: complete formal evidence violates at least one registered scientific gate.
- `INCOMPLETE`: technically valid smoke, pilot, or partial evidence that cannot support a formal decision.
- `INVALID`: integrity, configuration, infrastructure, or invariant failure prevents interpretation.

The checked-in smoke and pilot configs produce `INCOMPLETE`. Process exit status reports operational success or failure; consumers must read `result.json.status` for scientific status.

## Hardware envelopes

| Profile | Intended envelope | Scientific role |
|---|---|---|
| CPU | registry, schema, fake backends, exact finite tests | CI only |
| 16GB | 1.5B smoke; 7B NF4 frozen generation, group 2, 1,024 tokens, at most one update | pilot |
| 24GB | 7B BF16 LoRA rank 32, microbatch 1, sequential group 4, 1,024 response tokens, at most 20 updates | pilot/frozen audit |
| 4×24GB | separated rollout/training workers | screening |
| H200 | BF16 LoRA rank 32/alpha 64, group 8, 2,048 tokens | formal after sealing |

Peak memory above 90%, infrastructure failures above 1%, or projected cost overrun above 25% abort scaling.
