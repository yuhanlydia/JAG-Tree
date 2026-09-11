# JAG-Tree

JAG-Tree is a standalone research implementation of genealogy-aware credit and joint value-gradient allocation for tree reinforcement learning with verifiable rewards.

**Status:** the finite Phase-0 mathematics is implemented and tested. Two local real-7B pilots have completed; they are not evidence of general empirical improvement. No formal 7B result, benchmark superiority, or policy-improvement claim is made.

The [2026-09-11 pilot report](reports/2026-09-11/REPORT.md) records two 16-task runs on an RTX 5090, verified candidate banks, and six-arm comparisons. Fixed-bank JAG gradient MSE was worse than uniform in the 192-token configuration and lower in the 1024-token configuration. Both remain `INCOMPLETE` scientific evidence and used an outcome-blind structural pilot predictor.

## Lightweight installation

Python 3.11 and 3.12 are supported. The base install is CPU-only and does not download models or datasets.

```bash
python -m pip install -e '.[test]'
jag-tree doctor
jag-tree plan configs/experiments/smoke.yaml
```

Model, benchmark, and training integrations are lazy optional extras:

```bash
python -m pip install -e '.[models,benchmarks]'
```

The CLI has a non-executing planning boundary. `jag-tree plan` recursively resolves YAML, validates closed registries and immutable formal revisions, then prints canonical JSON and its SHA-256 without importing Torch.

## Experiment stages

- Phase 0 checks the estimator and allocator on exactly enumerable finite MDPs.
- Phase 1 replays controlled arms over immutable frozen candidate banks.
- Phase 2 performs one optimizer update per rollout batch through optional QLoRA integrations.

Formal gates are preregistered in the design documents. Smoke and pilot runs are never formal evidence. See `docs/EXPERIMENTS.md`, `docs/BASELINES.md`, `docs/DATA.md`, and `docs/ARTIFACTS.md` for the complete reproducibility contract.
