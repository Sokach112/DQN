"""Summarize and plot ``long_experiments.py`` runs.

Reads one or more ``report.json`` files (or a directory of run subdirectories),
prints a compact table, and writes ``<run>/curve.png`` for each run. Also reports
measured throughput and the projected wall time for paper-scale budgets, so the
unfinished parts of the assignment can be planned with real numbers.

Examples::

    python scripts/long_analysis.py outputs/assignment-2026-10-18-long/pong-tuned-s0
    python scripts/long_analysis.py outputs/assignment-2026-10-18-long --pattern report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_reports(paths: list[Path]) -> list[dict]:
    """Expand directories to report.json files and load them."""
    reports = []
    for path in paths:
        if path.is_dir():
            candidates = sorted(path.glob('report.json')) or sorted(path.glob('*/report.json'))
            reports.extend(json.loads(p.read_text()) for p in candidates if p.stat().st_size > 0)
        elif path.is_file():
            reports.append(json.loads(path.read_text()))
        else:
            raise FileNotFoundError(path)
    return reports


def summarize(report: dict) -> dict:
    """Extract headline numbers from one report."""
    rows = report.get('rows', [])
    tests = [r for r in rows if r.get('kind') == 'test']
    final = report.get('final_evaluation') or {}
    final_mean = final.get('mean_of_means')
    return {
        'tag': report.get('tag'),
        'game': report.get('game'),
        'seed': report.get('seed'),
        'steps': rows[-1]['step'] if rows else 0,
        'updates': rows[-1].get('updates', 0) if rows else 0,
        'evals': len(tests),
        'best_eval': report.get('best_eval_mean'),
        'last_eval': report.get('last_eval', {}).get('episode_rewards_mean') if report.get('last_eval') else None,
        'final_eval': final_mean,
        'stopped_early': report.get('stopped_early'),
        'stop_reason': report.get('stop_reason'),
        'seconds': report.get('elapsed_seconds', 0.0),
        'steps_per_second': (rows[-1]['step'] / report['elapsed_seconds']) if rows and report.get('elapsed_seconds') else None,
        'tests': tests,
    }


def plot_report(report: dict, destination: Path) -> None:
    """Write a two-panel learning-curve figure next to the report."""
    rows = report.get('rows', [])
    tests = [r for r in rows if r.get('kind') == 'test']
    train = [r for r in rows if r.get('kind') == 'train']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if tests:
        axes[0].plot([r['step'] for r in tests], [r['episode_rewards_mean'] for r in tests], marker='.')
        axes[0].fill_between([r['step'] for r in tests],
                             [r['episode_rewards_mean'] - r.get('episode_rewards_std', 0) for r in tests],
                             [r['episode_rewards_mean'] + r.get('episode_rewards_std', 0) for r in tests],
                             alpha=.2)
    axes[0].axhline(20, color='green', ls='--', lw=1, label='paper average (20)')
    axes[0].axhline(21, color='darkgreen', ls=':', lw=1, label='paper best episode (21)')
    axes[0].set_title('Greedy evaluation (10 complete games)')
    axes[0].set_xlabel('decisions (4 frames each)')
    axes[0].set_ylabel('Pong reward')
    axes[0].legend(loc='lower right', fontsize=8)
    axes[0].grid(alpha=.2)
    loss = [r for r in train if 'network/loss' in r]
    if loss:
        axes[1].plot([r['step'] for r in loss], [r['network/loss'] for r in loss], label='logged minibatch MSE')
        q = [r for r in train if 'q_value_mean' in r]
        if q:
            axes[1].plot([r['step'] for r in q], [r['q_value_mean'] for r in q], label='mean Q')
        axes[1].set_xlabel('decisions')
        axes[1].set_title('Training diagnostics')
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=.2)
    fig.suptitle(f"{report.get('tag')} — {report.get('game')} seed {report.get('seed')}")
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def projection(steps_per_second: float | None) -> str:
    """Format paper-scale budget projections from a measured throughput."""
    if not steps_per_second:
        return 'throughput unknown'
    lines = ['Budget projections at the measured rate (decisions, emulator frames = 4x):']
    for decisions, label in ((1_000_000, '1M decisions = 4M frames'),
                             (2_500_000, '2.5M decisions = 10M frames (paper 2013 budget)'),
                             (6_250_000, '6.25M decisions = 25M frames')):
        hours = decisions / steps_per_second / 3600
        lines.append(f'  - {label}: {hours:.1f} h')
    return '\n'.join(lines)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    reports = load_reports(args.paths)
    if not reports:
        print('no reports found')
        return 1
    header = f"{'tag':24s} {'game':22s} {'seed':>4s} {'steps':>9s} {'best':>7s} {'final':>7s} {'steps/s':>8s} {'early':>6s}"
    print(header)
    print('-' * len(header))
    summaries = []
    for report in reports:
        s = summarize(report)
        summaries.append(s)
        print(f"{str(s['tag']):24s} {str(s['game']):22s} {s['seed']!s:>4s} {s['steps']:>9,d} "
              f"{fmt(s['best_eval']):>7s} {fmt(s['final_eval']):>7s} {fmt(s['steps_per_second'], 1):>8s} "
              f"{str(s['stopped_early']):>6s}")
        if not args.no_plots:
            run_dir = report.get('meta', {}).get('args', {}).get('output')
            tag = report.get('tag')
            if run_dir and tag:
                plot_report(report, Path(run_dir) / tag / 'curve.png')
    rates = [s['steps_per_second'] for s in summaries if s['steps_per_second']]
    if rates:
        print('\n' + projection(float(np.median(rates))))
    return 0


def fmt(value, digits=2):
    """Format an optional number."""
    return '-' if value is None else f'{value:.{digits}f}'


if __name__ == '__main__':
    raise SystemExit(main())
