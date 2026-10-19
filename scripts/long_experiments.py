"""Longer Atari DQN experiments with incremental results and early stopping.

This runner is deliberately independent of Hydra: it builds the production
``DQN``/``AtariDQNAgent``/``ReplayBuffer`` components from ``config/config.yaml``
and overrides only the fields exposed on the command line. It is intended for
runs that are long enough that a crash must not lose the measurement history.

Every evaluation is appended immediately (JSONL plus an atomically rewritten
report). Checkpoints are written under ``checkpoints/`` and the best evaluation
also updates ``best_model.pth``.

Example (paper-scale Pong on Apple MPS, early stop once solved)::

    python long_experiments.py --game PongNoFrameskip-v4 --steps 1500000 \
        --tag pong-tuned-s0 --device mps --buffer-size 100000 \
        --epsilon-timestep 100000 --end-epsilon 0.01 \
        --target-update-interval 1000 --test-interval 50000 \
        --test-episodes 10 --stop-score 20.5 --stop-patience 2

The paper reports 10M *emulator* frames for Atari, which is 2.5M decisions at
frame-skip 4. ``--steps`` counts decisions (vector steps), matching
``interaction_step`` in the rest of this repository.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from agent.atari_agent import AtariDQNAgent
from algorithm.offpolicy_algorithm.dqn import DQN
from env.make_envs import make_vec_envs
from memory.memory import ReplayBuffer
from utils.evaluate import evaluate_atari
from utils.utils import get_best_device, set_seed


def parse_args() -> argparse.Namespace:
    """Parse experiment overrides."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--game', default='PongNoFrameskip-v4')
    parser.add_argument('--tag', required=True, help='Run directory name under --output.')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'assignment-2026-10-18-long')
    parser.add_argument('--steps', type=int, default=1_000_000,
                        help='Decisions (vector steps) to collect; must be a multiple of 4.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'mps', 'cuda'))
    parser.add_argument('--torch-threads', type=int, default=1)

    parser.add_argument('--buffer-size', type=int, default=100_000,
                        help='Replay transitions; ~56.5 KB each (states+next_states, 4x84x84 uint8).')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-starts', type=int, default=20_000)
    parser.add_argument('--start-epsilon', type=float, default=1.0)
    parser.add_argument('--end-epsilon', type=float, default=0.01)
    parser.add_argument('--epsilon-timestep', type=int, default=100_000,
                        help='Decisions over which epsilon anneals (paper: 1M frames = 250k decisions).')
    parser.add_argument('--target-update-interval', type=int, default=1000,
                        help='Gradient steps between hard target copies.')
    parser.add_argument('--architecture', default='paper2013', choices=('paper2013', 'nature2015'))
    parser.add_argument('--activation', default='relu', choices=('relu', 'leaky_relu'))
    parser.add_argument('--loss', default='mse', choices=('mse', 'huber'))
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--double-dqn', action='store_true', help='Enable Double DQN targets (default: vanilla target DQN).')

    parser.add_argument('--test-interval', type=int, default=50_000,
                        help='Decisions between greedy evaluations; 0 disables periodic evaluation.')
    parser.add_argument('--test-episodes', type=int, default=10)
    parser.add_argument('--eval-seed-offset', type=int, default=100,
                        help='Evaluation env seed = seed + offset (held fixed across checkpoints).')
    parser.add_argument('--train-log-interval', type=int, default=10_000)
    parser.add_argument('--save-interval', type=int, default=50_000,
                        help='Decisions between checkpoint saves; 0 only saves at evaluation time.')
    parser.add_argument('--final-test-episodes', type=int, default=20,
                        help='Extra independent evaluation at the end; 0 disables.')

    parser.add_argument('--stop-score', type=float, default=None,
                        help='Stop once the evaluation mean reaches this score on --stop-patience consecutive evals.')
    parser.add_argument('--stop-patience', type=int, default=2)
    parser.add_argument('--max-seconds', type=float, default=None,
                        help='Wall-clock guard: stop cleanly after this many training seconds.')
    args = parser.parse_args()
    if args.steps < 1000 or args.steps % 4:
        parser.error('--steps must be a multiple of four and at least 1000')
    if args.test_interval <= 0 and args.train_log_interval <= 0 and args.save_interval <= 0:
        parser.error('at least one of the interval options must be positive')
    return args


def main() -> int:
    """Build components, train, and persist an incremental report."""
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    out = args.output / args.tag
    if out.exists() and any(out.iterdir()):
        print(f'error: {out} is not empty; choose a new --tag/--output', file=sys.stderr)
        return 2
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_best_device() if args.device == 'auto' else torch.device(args.device)

    cfg = OmegaConf.load(ROOT / 'config' / 'config.yaml')
    # The hydra section contains Hydra-only interpolations (${now:...}); this
    # runner does not use Hydra's config store and sets its own output directory.
    cfg.pop('hydra', None)
    cfg.log_dir = str(out)
    cfg.env.name = args.game
    cfg.env.num_training_envs = 1
    cfg.env.num_testing_envs = 1
    cfg.total_epoch = args.steps // 4
    cfg.interact_per_epoch = 4
    cfg.algorithm.buffer_size = args.buffer_size
    cfg.algorithm.learning_rate = args.learning_rate
    cfg.algorithm.gamma = args.gamma
    cfg.algorithm.batch_size = args.batch_size
    cfg.algorithm.learning_starts = args.learning_starts
    cfg.algorithm.start_epsilon = args.start_epsilon
    cfg.algorithm.end_epsilon = args.end_epsilon
    cfg.algorithm.epsilon_timestep = args.epsilon_timestep
    cfg.algorithm.epsilon_schedule = 'linear'
    cfg.algorithm.target_update_interval = args.target_update_interval
    cfg.algorithm.architecture = args.architecture
    cfg.algorithm.activation = args.activation
    cfg.algorithm.loss = args.loss
    cfg.algorithm.optimizer = args.optimizer
    cfg.algorithm.double_dqn = bool(args.double_dqn)
    cfg.algorithm.use_target = True
    cfg.algorithm.target_update_method = 'hard'
    cfg.test_interval = args.test_interval
    cfg.test_episodes = args.test_episodes
    cfg.train_log_interval = args.train_log_interval
    cfg.save_interval = args.save_interval
    cfg.train_action_deterministic = False

    meta = {
        'args': vars(args) | {'output': str(args.output)},
        'device': str(device),
        'torch': torch.__version__,
        'numpy': np.__version__,
        'python': sys.version,
        'git': git_describe(),
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'config': OmegaConf.to_container(cfg, resolve=True),
    }
    (out / 'config.json').write_text(json.dumps(meta, indent=2, default=str))
    (out / 'results.jsonl').write_text('')
    print(f'[{args.tag}] device={device} steps={args.steps} buffer={args.buffer_size} '
          f'lr={args.learning_rate} eps={args.start_epsilon}->{args.end_epsilon} '
          f'over {args.epsilon_timestep} decisions', flush=True)

    training_env = make_vec_envs(cfg.env, True, seed=args.seed)
    report = {'tag': args.tag, 'game': args.game, 'seed': args.seed, 'meta': meta,
              'rows': [], 'elapsed_seconds': 0.0, 'stopped_early': False,
              'stop_reason': None, 'final_evaluation': None}
    agent = None
    try:
        agent = AtariDQNAgent(training_env.observation_space, training_env.action_space, device,
                              activation=args.activation, architecture=args.architecture)
        target = AtariDQNAgent(training_env.observation_space, training_env.action_space, device,
                               activation=args.activation, architecture=args.architecture)
        buffer = ReplayBuffer(training_env.observation_space, training_env.action_space, args.buffer_size, 1)
        algorithm = DQN(training_env, None, buffer, agent, _JsonLogger(), device,
                        str(out / 'newest_model.pth'), cfg, target)
        run_training(algorithm, args, out, report)
    finally:
        training_env.close()

    # Final independent evaluation with fresh reset seeds (never used for selection).
    if agent is not None and args.final_test_episodes > 0:
        try:
            finals = []
            for offset in (args.eval_seed_offset, args.eval_seed_offset + 1):
                result = evaluate_atari(agent, args.final_test_episodes, cfg.env, seed=args.seed + offset)
                finals.append(result.metrics)
            report['final_evaluation'] = {
                'episodes_per_seed': args.final_test_episodes,
                'seeds': [args.seed + args.eval_seed_offset, args.seed + args.eval_seed_offset + 1],
                'runs': finals,
                'mean_of_means': float(np.mean([r['episode_rewards_mean'] for r in finals])),
            }
        finally:
            agent.save(str(out / 'newest_model.pth'))
        write_report(report, out)

    print(f'[{args.tag}] done in {report["elapsed_seconds"]:.0f}s '
          f'({args.steps / max(report["elapsed_seconds"], 1e-9):.0f} steps/s) '
          f'best_eval={report.get("best_eval_mean")} final={report["final_evaluation"]}', flush=True)
    return 0


def run_training(algorithm: DQN, args: argparse.Namespace, out: Path, report: dict) -> None:
    """Replicate the production training loop with incremental persistence."""
    algorithm.initialize()
    report['best_eval_mean'] = None
    started = time.monotonic()
    last_eval = None
    consecutive_hits = 0
    save_interval = max(args.save_interval, 0)
    for _ in range(algorithm.total_epoch):
        batch, _ = algorithm.interact_with_envs()
        log_now = algorithm.train_log_condition()
        algorithm._collect_update_metrics = log_now
        result = algorithm.update(batch)
        if log_now:
            row = {'kind': 'train', 'step': algorithm.interaction_step,
                   'updates': algorithm.gradient_step,
                   'epsilon': float(algorithm.epsilon), **result.metrics}
            _append_row(report, out, row)
        if algorithm.test_condition() and args.test_interval > 0:
            eval_result = evaluate_atari(algorithm.agent, args.test_episodes, algorithm.args.env,
                                         seed=args.seed + args.eval_seed_offset)
            last_eval = eval_result.metrics
            row = {'kind': 'test', 'step': algorithm.interaction_step,
                   'updates': algorithm.gradient_step, **eval_result.metrics}
            _append_row(report, out, row)
            mean = last_eval['episode_rewards_mean']
            if report['best_eval_mean'] is None or mean > report['best_eval_mean']:
                report['best_eval_mean'] = mean
                algorithm.agent.save(str(out / 'best_model.pth'))
            algorithm.agent.save(str(out / 'checkpoints' / f'step_{algorithm.interaction_step:07d}.pth'))
            print(f'[{args.tag}] step={algorithm.interaction_step} updates={algorithm.gradient_step} '
                  f'eval={mean:.2f} (best {report["best_eval_mean"]:.2f}) eps={algorithm.epsilon:.3f} '
                  f't={time.monotonic() - started:.0f}s', flush=True)
            if args.stop_score is not None and mean >= args.stop_score:
                consecutive_hits += 1
                if consecutive_hits >= args.stop_patience:
                    report['stopped_early'] = True
                    report['stop_reason'] = f'eval mean >= {args.stop_score} for {consecutive_hits} evaluations'
                    break
            else:
                consecutive_hits = 0
        elif save_interval and algorithm.save_condition():
            algorithm.agent.save(str(out / 'newest_model.pth'))
        if args.max_seconds is not None and time.monotonic() - started >= args.max_seconds:
            report['stopped_early'] = True
            report['stop_reason'] = f'max-seconds ({args.max_seconds}) reached'
            break
    algorithm.agent.save(str(out / 'newest_model.pth'))
    report['elapsed_seconds'] = time.monotonic() - started
    report['last_eval'] = last_eval
    write_report(report, out)


class _JsonLogger:
    """Minimal logger used by DQN; persistence happens in run_training."""

    def log_train(self, *args, **kwargs):
        """No-op: rows are appended by run_training."""

    def log_test(self, *args, **kwargs):
        """No-op: rows are appended by run_training."""

    def close(self):
        """No-op."""


def _append_row(report: dict, out: Path, row: dict) -> None:
    report['rows'].append(row)
    with (out / 'results.jsonl').open('a') as handle:
        handle.write(json.dumps(row) + '\n')
    write_report(report, out)


def write_report(report: dict, out: Path) -> None:
    """Atomically rewrite the cumulative JSON report."""
    tmp = out / 'report.json.tmp'
    tmp.write_text(json.dumps(report, indent=2, default=str))
    os.replace(tmp, out / 'report.json')


def git_describe() -> str:
    """Best-effort source revision for provenance."""
    try:
        import subprocess
        return subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return 'unknown'


if __name__ == '__main__':
    raise SystemExit(main())
