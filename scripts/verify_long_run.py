"""Independent evaluation of saved long-run checkpoints on fresh reset seeds.

This is intentionally separate from the training runner: it never selects a
checkpoint, it just measures the saved policies on seeds that were not used for
periodic evaluation, so the reported numbers are not the selection metric.

Example::

    python scripts/verify_long_run.py outputs/assignment-2026-10-18-long/pong-tuned-s0 \
        --episodes 20 --seeds 555 777
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from agent.atari_agent import AtariDQNAgent
from env.make_envs import make_vec_envs
from omegaconf import OmegaConf
from utils.evaluate import evaluate_atari


def main() -> int:
    """Evaluate every requested checkpoint with several fresh reset seeds."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--checkpoints', nargs='+', default=['best_model.pth', 'newest_model.pth'])
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seeds', nargs='+', type=int, default=[555, 777])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    if args.device != 'auto':
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    config = json.loads((args.run_dir / 'config.json').read_text())
    cfg = OmegaConf.create(config['config'])
    cfg.env.num_testing_envs = 1

    env = make_vec_envs(cfg.env, False, seed=0)
    observation_space, action_space = env.observation_space, env.action_space
    env.close()

    results = {}
    for name in args.checkpoints:
        path = args.run_dir / name
        if not path.exists():
            print(f'skip missing {path}')
            continue
        agent = AtariDQNAgent(observation_space, action_space, device,
                              activation=config['config']['algorithm']['activation'],
                              architecture=config['config']['algorithm']['architecture'])
        agent.load(str(path))
        results[name] = {}
        for seed in args.seeds:
            metrics = evaluate_atari(agent, args.episodes, cfg.env, seed=seed).metrics
            results[name][seed] = metrics
            print(f'{name} seed {seed}: mean {metrics["episode_rewards_mean"]:.3f} '
                  f'std {metrics["episode_rewards_std"]:.3f}', flush=True)
        means = [results[name][s]['episode_rewards_mean'] for s in args.seeds]
        results[name]['mean_of_seed_means'] = float(np.mean(means))
        print(f'{name}: mean of seed means {results[name]["mean_of_seed_means"]:.3f}', flush=True)

    destination = args.output or (args.run_dir / 'verification.json')
    destination.write_text(json.dumps({
        'checkpoints': args.checkpoints,
        'episodes_per_seed': args.episodes,
        'seeds': args.seeds,
        'device': device,
        'results': results,
    }, indent=2))
    print(f'wrote {destination}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
