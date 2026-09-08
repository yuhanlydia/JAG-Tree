# Baselines and provenance

Controlled allocator arms use the same topology, candidate bank, reward, recursive credit, prompts, optimizer-step count, and budgets. Only the allocation score changes.

| Arm | Score source |
|---|---|
| `uniform` | constant |
| `entropy` | prefix entropy |
| `value_variance` | value variance |
| `gradient_only` | gradient-covariance trace |
| `trace_score` | registered trace predictor |
| `jag_no_cross` | transported joint risk without value-gradient covariance |
| `jag_full` | transported joint value-gradient covariance per cost |
| `oracle_moments` | full-information moments, used only as an upper-bound mechanism arm |

Executable entries in this repository are `controlled_ablation`. A recipe without implemented dispatch is marked `unavailable` and is rejected if supplied to an experiment runner.

`configs/experiments/baseline_matrix.yaml` lists the intended comparison names, but every end-to-end external recipe is currently `unavailable` because no verified executable adapter is shipped. The matrix is metadata, not an executable benchmark claim.

An adapter never silently copies upstream code. Before execution, verify the recorded commit and license hash, apply only documented local patches, and store the patch diff in the run manifest. Results from different model families are reported separately.

Outcome-leaking allocation, duplicated-prefix weighting, and shuffled genealogy are fault injections. They are not valid baselines and cannot be selected as a winning arm.
