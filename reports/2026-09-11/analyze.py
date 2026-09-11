"""Post-run conditional analysis of the sealed pilot candidate bank."""
from pathlib import Path
from hashlib import sha256
import json
import sys
import numpy as np
from jag_tree.bank import verify_bank
from jag_tree.artifacts import verify_artifacts
from jag_tree.models import TransformersPolicyBackend
from jag_tree.audit import _risk

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
response_cap = int(sys.argv[2]) if len(sys.argv) > 2 else 192
run = next(root.glob('run-*'))
result = verify_artifacts(run)
bank = verify_bank(result['bank_path'])
backend = TransformersPolicyBackend('qwen25_coder_7b', result['model_revision'], allow_pilot_predictor=True)
predictor = backend.frozen_predictor(bank.nodes)
by_id = {n.node_id: n for n in bank.nodes}
children = {}
for n in bank.nodes:
    if n.parent_id is not None:
        children.setdefault(n.parent_id, []).append(n)
roots = [n for n in bank.nodes if n.parent_id is None]
leaves = [n for n in bank.nodes if n.reward is not None]
replay_rows = [json.loads(line) for line in (run / 'rows.jsonl').read_text().splitlines()]
paths, scores, costs, target_p = [], [], [], []
for leaf in leaves:
    path = []
    node = leaf
    probability = 1 / len(roots)
    while node.parent_id is not None:
        path.append(node)
        probability /= len(children[node.parent_id])
        node = by_id[node.parent_id]
    paths.append(path)
    costs.append(sum(n.unique_tokens for n in path))
    scores.append(sum((bank.score_arrays[n.node_id] for n in path)))
    target_p.append(probability)
scores, costs, target_p = np.asarray(scores), np.asarray(costs), np.asarray(target_p)
reward = np.asarray([n.reward for n in leaves])
np.testing.assert_allclose(target_p.sum(), 1)
target_gradient = (target_p[:, None] * reward[:, None] * scores).sum(axis=0)
dim = scores.shape[1]
arms = {}
for arm, evidence in result['arm_evidence'].items():
    if arm == 'oracle_moments':
        priorities = target_p * reward * np.linalg.norm(scores, axis=1)
    else:
        priorities = np.asarray([max(0, _risk(arm, predictor.predict(n.node_id), s))
                                 for n, s in zip(leaves, scores)]) / costs
    floor = max(priorities.max(initial=0) * 1e-6, 1e-12)
    q = (priorities + floor) / (priorities + floor).sum()
    q_by_id = dict(zip((n.node_id for n in leaves), q))
    for row in replay_rows:
        if row.get('arm') == arm:
            np.testing.assert_allclose(row['proposal_probability'], q_by_id[row['leaf_id']], rtol=1e-12)
    draws = int(evidence['metrics']['draw_count'])
    per_draw = ((target_p**2 / q) * reward**2 * (scores**2).sum(axis=1)).sum() - (target_gradient**2).sum()
    exact_mse = max(0, float(per_draw / (dim * draws)))
    # Independent repeated sampling checks the closed-form bank-conditional MSE.
    rng = np.random.default_rng(20260911)
    contributions = target_p[:, None] / q[:, None] * reward[:, None] * scores
    replicate_mse = []
    for _ in range(32):
        choices = rng.choice(len(leaves), size=(256, draws), p=q)
        estimates = contributions[choices].mean(axis=1)
        replicate_mse.extend(np.mean((estimates - target_gradient)**2, axis=1))
    mc_mse = float(np.mean(replicate_mse))
    mc_se = float(np.std(replicate_mse, ddof=1) / np.sqrt(len(replicate_mse)))
    if abs(mc_mse - exact_mse) > 6 * mc_se + 1e-10:
        raise ValueError(f'Monte Carlo disagrees with exact conditional MSE for {arm}')
    arms[arm] = dict(exact_conditional_gradient_mse=exact_mse, draw_count=draws,
                     validation_mc_mse=mc_mse, validation_mc_se=mc_se,
                     sampled_gradient_mse=evidence['metrics']['gradient_mse'],
                     sampled_cosine=evidence['metrics']['gradient_cosine'],
                     replay_sampling_tokens=evidence['ledger']['replay_sampling_tokens'])
for row in arms.values():
    row['exact_mse_ratio_to_uniform'] = row['exact_conditional_gradient_mse'] / arms['uniform']['exact_conditional_gradient_mse'] if arms['uniform']['exact_conditional_gradient_mse'] else None
tasks = []
for task in bank.tasks:
    selected = [n for n in leaves if n.task_id == task.task_id]
    tasks.append(dict(task_id=task.task_id, leaves=len(selected), passed=sum(n.reward == 1 for n in selected),
                      at_response_cap=sum(sum(e.unique_tokens for e in paths[leaves.index(n)]) >= response_cap for n in selected)))
report = dict(status='INCOMPLETE', interpretation='Completed real 7B pilot; conditional on one 16-task bank and one late rank-2 LoRA sketch; no formal or training claim.',
              bank_manifest_sha256=sha256((bank.path / 'manifest.json').read_bytes()).hexdigest(),
              task_count=len(bank.tasks), edges=len(bank.score_arrays), leaves=len(leaves),
              passed_leaves=int(reward.sum()), solved_tasks=sum(t['passed'] > 0 for t in tasks),
              unique_generated_tokens=sum(n.unique_tokens for n in bank.nodes),
              target_reward=float(target_p @ reward),
              dimension=dim, arms=arms, tasks=tasks)
(root / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
