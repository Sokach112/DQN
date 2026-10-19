"""Integrate and independently verify the Kaggle paper-scale Pong run.

The run was produced outside this working tree (Kaggle, clone of
github.com/Sokach112/DQN) and is stored in ``results/``. This script:

1. parses ``log.txt`` (periodic greedy evaluations + rolling training reward),
2. re-evaluates the saved checkpoint on fresh reset seeds, recording every
   complete game score (not just mean/std), and
3. writes the combined evidence (JSON + PNG) under an output directory.

It never trains. Example::

    python scripts/kaggle_results_analysis.py \
      --log ../results/log.txt --checkpoint ../results/newest_model.pth \
      --output outputs/assignment-2026-10-19-kaggle-pong --games 100
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from agent.atari_agent import AtariDQNAgent
from env.make_envs import make_vec_envs


ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


def parse_log(path: Path) -> dict:
    """Extract periodic test scores and rolling training rewards from the log."""
    text = ANSI.sub('', path.read_text(encoding='utf-8', errors='replace'))
    tests, train = [], []
    kind = None
    step = None
    metrics = {}
    for line in text.splitlines():
        if 'Testing Info' in line:
            kind = 'test'
            metrics = {}
            continue
        if 'Training Info' in line:
            if kind == 'test' and step is not None and 'episode_rewards_mean' in metrics:
                tests.append((step, metrics['episode_rewards_mean'], metrics.get('episode_rewards_std', float('nan'))))
            kind = 'train'
            metrics = {}
            continue
        match = re.search(r'interaction_step\s*[│|]\s*(\d+)', line)
        if match:
            step = int(match.group(1))
            continue
        for key in ('episode_rewards_mean', 'episode_rewards_std', 'reward_mean'):
            match = re.search(key + r'\s*[│|]\s*([-\d\.eE+]+)', line)
            if match:
                value = float(match.group(1))
                metrics[key] = value
                if kind == 'train' and key == 'reward_mean':
                    train.append((step, value))
    if kind == 'test' and step is not None and 'episode_rewards_mean' in metrics:
        tests.append((step, metrics['episode_rewards_mean'], metrics.get('episode_rewards_std', float('nan'))))
    return {
        'tests': [{'step': s, 'mean': m, 'std': sd} for s, m, sd in tests],
        'train_reward': [{'step': s, 'mean': m} for s, m in train],
        'runtime_line': next((l for l in text.splitlines() if re.search(r'\d+/\d+ \[.*?<', l) and ('epoch/s' in l)), None),
        'raw_head': text.splitlines()[:40],
    }


def evaluate_games(agent, cfg, episodes: int, seeds: list[int]) -> dict:
    """Run ``episodes`` complete games per seed and return every score."""
    env = make_vec_envs(cfg, False, seed=0)
    observation_space, action_space = env.observation_space, env.action_space
    env.close()
    results = {}
    for seed in seeds:
        # A fresh factory per seed mirrors ``evaluate_atari`` and avoids relying
        # on re-seeding an existing vector environment.
        env = make_vec_envs(cfg, False, seed=seed)
        try:
            state = env.reset()
            scores = []
            while len(scores) < episodes:
                actions = agent.select_action(state, deterministic=True)
                state, _, _, infos = env.step(actions)
                for info in infos:
                    if 'episode' in info and len(scores) < episodes:
                        scores.append(float(info['episode']['r']))
        finally:
            env.close()
        results[int(seed)] = scores
        print(f'seed {seed}: {episodes} games mean {np.mean(scores):.3f} '
              f'std {np.std(scores):.3f} perfect {sum(s >= 21 for s in scores)}', flush=True)
    all_scores = [s for scores in results.values() for s in scores]
    return {
        'seeds': {str(k): v for k, v in results.items()},
        'all_scores': all_scores,
        'games': len(all_scores),
        'mean': float(np.mean(all_scores)),
        'std': float(np.std(all_scores)),
        'min': float(np.min(all_scores)),
        'max': float(np.max(all_scores)),
        'perfect_games': int(sum(s >= 21 for s in all_scores)),
        'perfect_fraction': float(np.mean([s >= 21 for s in all_scores])),
        'score_histogram': {str(k): int(v) for k, v in zip(*np.unique(all_scores, return_counts=True))},
    }


def main() -> int:
    """Parse the log, evaluate the checkpoint, and persist the combined evidence."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--game', default='PongNoFrameskip-v4')
    parser.add_argument('--games', type=int, default=100, help='Complete games in total.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--reference-lines', nargs='+', type=float, default=None,
                        help='Horizontal reference scores (default: Pong 20/21, Breakout 168/225).')
    parser.add_argument('--reference-labels', nargs='+', default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    device = ('cuda' if torch.cuda.is_available() else
              ('mps' if torch.backends.mps.is_available() else 'cpu')) if args.device == 'auto' else args.device
    if args.reference_lines is None:
        args.reference_lines = [168.0, 225.0] if 'Breakout' in args.game else [20.0, 21.0]
    if args.reference_labels is None:
        args.reference_labels = (['paper average (168)', 'paper best episode (225)']
                                 if 'Breakout' in args.game else
                                 ['paper average (20)', 'paper best episode (21)'])
    cfg = OmegaConf.load(ROOT / 'config' / 'config.yaml')['env']
    cfg.name = args.game
    cfg.num_testing_envs = 1

    log = parse_log(args.log)
    print(f"test points: {len(log['tests'])}; best test mean "
          f"{max(t['mean'] for t in log['tests']):.2f} at step "
          f"{max(log['tests'], key=lambda t: t['mean'])['step']}")
    print(f"21.0 points: {[t['step'] for t in log['tests'] if abs(t['mean'] - 21) < 1e-9]}")

    # Environment spaces then checkpoint.
    env = make_vec_envs(cfg, False, seed=0)
    observation_space, action_space = env.observation_space, env.action_space
    env.close()
    agent = AtariDQNAgent(observation_space, action_space, device)
    agent.load(str(args.checkpoint))
    print(f'checkpoint architecture={agent.network.architecture} activation={agent.network.activation_name}')

    per_seed = max(1, args.games // len(args.seeds))
    verification = evaluate_games(agent, cfg, per_seed, args.seeds)
    verification.update({
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': __import__('hashlib').sha256(args.checkpoint.read_bytes()).hexdigest(),
        'device': str(device),
        'architecture': agent.network.architecture,
        'activation': agent.network.activation_name,
        'game': args.game,
        'episodes_per_seed': per_seed,
    })
    (args.output / 'verification.json').write_text(json.dumps({
        'log': log,
        'verification': verification,
    }, indent=2))

    steps = [t['step'] for t in log['tests']]
    means = [t['mean'] for t in log['tests']]
    stds = [t.get('std', np.nan) for t in log['tests']]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(steps, means, marker='.', color='tab:blue')
    axes[0].fill_between(steps, np.array(means) - np.nan_to_num(stds), np.array(means) + np.nan_to_num(stds), alpha=.2)
    for line, label in zip(args.reference_lines, args.reference_labels):
        axes[0].axhline(line, color='black' if line == max(args.reference_lines) else 'green',
                        ls='--' if line == max(args.reference_lines) else ':', lw=1, label=label)
    axes[0].set_xlabel('decisions (4 frames each)')
    axes[0].set_ylabel('Pong reward')
    axes[0].set_title('Kaggle run: greedy evaluation (10 games / 50k decisions)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    rewards = [t['mean'] for t in log['train_reward']]
    axes[1].plot([t['step'] for t in log['train_reward']], rewards, color='tab:orange')
    axes[1].set_xlabel('decisions')
    axes[1].set_title('Training reward (rolling episode mean, clipped)')
    axes[1].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / 'kaggle_pong_curve.png', dpi=140)
    plt.close(fig)

    scores = np.array(verification['all_scores'])
    values, counts = np.unique(scores, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(values, counts, width=.8, color='tab:blue')
    for line, label in zip(args.reference_lines, args.reference_labels):
        ax.axvline(line, color='black' if line == max(args.reference_lines) else 'green',
                   ls='--' if line == max(args.reference_lines) else ':', lw=1, label=label)
    ax.set_xlabel('Pong score (reward)')
    ax.set_ylabel('games')
    ax.set_title(f"Fresh-seed evaluation ({args.game}): {verification['games']} complete games "
                 f"(mean {verification['mean']:.2f})")
    ax.set_xticks(sorted(set(values)))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / 'score_distribution.png', dpi=140)
    plt.close(fig)

    print(json.dumps({k: verification[k] for k in
                      ('games', 'mean', 'std', 'min', 'max', 'perfect_games',
                       'perfect_fraction', 'score_histogram')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
