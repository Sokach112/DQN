"""Reproduce assignment pilot figures or additional checkpoint evaluations."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
out=args.output
reports=[json.loads(p.read_text()) for p in sorted(out.glob('*.json'))]
reports=[r for r in reports if 'config' in r and 'rows' in r]
fig,axs=plt.subplots(2,2,figsize=(11,7))
for report in reports:
 if report['name']=='breakout':continue
 for ax,key,kind,title in zip(axs.flat,['network/loss','q_value_mean','reward_mean','episode_rewards_mean'],['train','train','train','test'],['MSE (logged minibatch)','Mean Q (logged minibatch)','Training return (rolling episodes)','Greedy evaluation return (3 games)']):
  rows=[r for r in report['rows'] if r['kind']==kind and key in r]
  ax.plot([r['step'] for r in rows],[r[key] for r in rows],label=report['name'],marker='.' if kind=='test' else None);ax.set_title(title);ax.set_xlabel('Transitions');ax.grid(alpha=.2)
axs[0,0].legend();fig.tight_layout();fig.savefig(out/'pong-curves.png',dpi=150)
lines=['# Assignment baseline and pilot experiments','', 'All runs: seed 0, CPU, 20,000 transitions, 5,000 warmup transitions, 10,000 replay capacity, batch 32, one update per four transitions, three evaluation games per checkpoint. Paper2013 CNN, ReLU, MSE, vanilla target DQN, Adam; target copied every 2,500 updates. These are short pilots, not convergence benchmarks or exact historical-paper reproductions.','', '| Run | Learning rate | Epsilon duration | Eval 0 / 10k / 20k |','|---|---:|---:|---|']
for r in reports:
 a=r['config']['algorithm'];scores=[x['episode_rewards_mean'] for x in r['rows'] if x['kind']=='test'];lines.append(f"| {r['name']} | {a['learning_rate']} | {a['epsilon_timestep']} | {scores} |")
lines+=['','## Interpretation','Two tuning directions were tested independently: halving the learning rate and shortening epsilon decay. Three evaluations on one seed and a short budget cannot establish convergence or a reliable ranking. Loss/Q curves are logged single-minibatch measurements rather than averages across every update. Breakout scores use its own reward scale and should not be compared numerically with Pong scores.','', '## Remaining requirements','- Demonstrate sustained Pong improvement and the requested +21 evaluation score.','- Extend tuning comparisons with matched longer budgets and more seeds; explain reliable differences rather than minibatch noise.','- Reach comparable-to-paper performance in the additional game; running it is not sufficient.','- Optional softmax policy remains to be implemented or its omission justified under the epsilon-greedy allowance.','- Keep checkpoint/experiment metadata and configs with the report.']
(out/'PILOT_SUMMARY.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
