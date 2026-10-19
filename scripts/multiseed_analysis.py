"""Summarize multi-seed ``long_experiments.py`` runs into CSV, Markdown, and PNG.

Reads ``<root>/<variant>-s<seed>/report.json`` for each matched run created by
``long_experiments.py`` and writes ``multiseed.csv``, ``SUMMARY.md``, and
``multiseed.png`` into the same root. It never trains and never edits reports.

Usage (from the repository root)::

    .venv/bin/python scripts/multiseed_analysis.py \
        --root outputs/assignment-2026-10-18-long/multiseed
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

DEFAULT_VARIANTS = ('fast-eps', 'slow-eps', 'low-lr')
DEFAULT_SEEDS = (0, 1, 2)
PERIODIC_STEPS = (50_000, 100_000, 150_000, 200_000)
EVAL_INTERVAL = 25_000


def parse_args() -> argparse.Namespace:
    """Parse output root and run grid."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path,
                        default=Path('outputs/assignment-2026-10-18-long/multiseed'),
                        help='Directory containing <variant>-s<seed> run directories.')
    parser.add_argument('--variants', nargs='+', default=list(DEFAULT_VARIANTS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def load_runs(root: Path, variants, seeds) -> list[dict]:
    """Load available reports; missing/incomplete runs are reported, not fatal."""
    runs = []
    for variant in variants:
        for seed in seeds:
            run_dir = root / f'{variant}-s{seed}'
            report_path = run_dir / 'report.json'
            if not report_path.exists():
                print(f'warning: missing report {report_path}')
                continue
            try:
                report = json.loads(report_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                print(f'warning: unreadable report {report_path}: {error}')
                continue
            runs.append(summarize_run(variant, seed, run_dir, report))
    return runs


def summarize_run(variant: str, seed: int, run_dir: Path, report: dict) -> dict:
    """Extract the columns and curve points needed for aggregation."""
    rows = report.get('rows') or []
    tests = [row for row in rows if row.get('kind') == 'test']
    train = [row for row in rows if row.get('kind') == 'train']
    last_step = max((int(row.get('step') or 0) for row in rows), default=0)
    eval_by_step = {int(row['step']): float(row['episode_rewards_mean'])
                    for row in tests if row.get('episode_rewards_mean') is not None}
    final = report.get('final_evaluation') or {}
    elapsed = float(report.get('elapsed_seconds') or 0.0)
    args = ((report.get('meta') or {}).get('args')) or {}
    return {
        'variant': variant,
        'seed': seed,
        'tag': report.get('tag', f'{variant}-s{seed}'),
        'dir': str(run_dir),
        'steps_completed': last_step,
        'best_eval_mean': report.get('best_eval_mean'),
        'final_eval_mean': tests[-1].get('episode_rewards_mean') if tests else None,
        'final_eval_mean_of_means': final.get('mean_of_means'),
        'steps_per_sec': (last_step / elapsed) if elapsed > 0 else None,
        'stopped_early': bool(report.get('stopped_early')),
        'stop_reason': report.get('stop_reason'),
        'elapsed_seconds': elapsed,
        'eval_by_step': eval_by_step,
        'test_rows': tests,
        'train_rows': train,
        'args': args,
        'final_runs': final.get('runs') or [],
        'game': report.get('game'),
    }


def nearest_eval(eval_by_step: dict[int, float], target: int) -> float | None:
    """Return the evaluation at ``target`` (or one interval away), else None."""
    if not eval_by_step:
        return None
    step = min(eval_by_step, key=lambda candidate: abs(candidate - target))
    if abs(step - target) > EVAL_INTERVAL:
        return None
    return eval_by_step[step]


def mean_std(values) -> tuple[float, float, int] | None:
    """Mean and sample standard deviation (ddof=1) of the finite values."""
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return mean, std, len(finite)


def format_stat(stat, digits: int = 2) -> str:
    """Render ``mean ± std (n=...)`` or an em dash when absent."""
    if stat is None:
        return '—'
    mean, std, count = stat
    if count == 1:
        return f'{mean:.{digits}f} (n=1)'
    return f'{mean:.{digits}f} ± {std:.{digits}f}'


def write_csv(runs: list[dict], path: Path) -> None:
    """One row per run with the requested columns."""
    columns = ('variant', 'seed', 'steps_completed', 'best_eval_mean',
               'final_eval_mean', 'final_eval_mean_of_means', 'steps_per_sec',
               'stopped_early')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for run in runs:
            writer.writerow({
                'variant': run['variant'],
                'seed': run['seed'],
                'steps_completed': run['steps_completed'],
                'best_eval_mean': _round(run['best_eval_mean']),
                'final_eval_mean': _round(run['final_eval_mean']),
                'final_eval_mean_of_means': _round(run['final_eval_mean_of_means']),
                'steps_per_sec': _round(run['steps_per_sec'], 1),
                'stopped_early': run['stopped_early'],
            })


def _round(value, digits: int = 3):
    """Round optional numeric values for the CSV."""
    if value is None:
        return ''
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ''
    return f'{number:.{digits}f}' if math.isfinite(number) else ''


def build_summary(runs: list[dict], variants, seeds) -> str:
    """Create the Markdown summary text."""
    lines: list[str] = []
    add = lines.append
    found_variants = [variant for variant in variants
                      if any(run['variant'] == variant for run in runs)]
    add('# Multi-seed Pong DQN tuning summary')
    add('')
    add('Generated by `multiseed_analysis.py` from each run\'s `report.json`.')
    add('')
    add('## Scope and caveats')
    add('')
    add('These are **1M-frame (200,000-decision) CPU runs, not paper-scale** '
        '(the DQN paper reports 10M emulator frames = 2.5M decisions). Only '
        '**3 seeds** per variant are available, so means and ±1 sample standard '
        'deviations are noisy estimates, not reliable rankings. The final '
        '20-game independent evaluation still has substantial sampling error '
        'and is not used for checkpoint selection.')
    add('')
    add('## Protocol')
    add('')
    add('- Game: `PongNoFrameskip-v4`; CPU only, 1 PyTorch thread per run; '
        'three variants run concurrently per seed batch.')
    add('- Budget: `--steps 200000` decisions per run (~1M emulator frames at '
        'frame-skip 4), seeds 0/1/2, replay 50,000 transitions, batch 32, '
        'learning starts 20,000, target update interval 1,000, gamma 0.99, '
        'MSE loss, Adam, vanilla target DQN, paper2013 + ReLU architecture.')
    add('- Exploration: linear epsilon from 1.0 to 0.01 over the variant\'s '
        '`epsilon_timestep`; periodic greedy evaluation every 25,000 decisions '
        '(10 episodes, seed+100); final independent evaluation 20 games '
        '(2 seeds × 10 episodes, seed offsets +100/+101).')
    add('')
    add('| Variant | Learning rate | Epsilon timestep |')
    add('| --- | --- | --- |')
    for variant in found_variants:
        sample = next(run for run in runs if run['variant'] == variant)
        args = sample['args']
        add(f"| {variant} | {_fmt(args.get('learning_rate'))} | "
            f"{_fmt(args.get('epsilon_timestep'), 0)} |")
    add('')
    if len(found_variants) * len(seeds) != len(runs):
        add('## Missing runs')
        add('')
        expected = {(variant, seed) for variant in found_variants for seed in seeds}
        present = {(run['variant'], run['seed']) for run in runs}
        for variant, seed in sorted(expected - present):
            add(f'- Missing: `{variant}-s{seed}` (not completed or no report).')
        add('')
    add('## Per-variant results (mean ± sample std across seeds)')
    add('')
    header = '| Variant | 50k | 100k | 150k | 200k | Final 20-game mean_of_means |'
    add(header)
    add('| --- | --- | --- | --- | --- | --- |')
    for variant in found_variants:
        variant_runs = [run for run in runs if run['variant'] == variant]
        cells = []
        for target in PERIODIC_STEPS:
            stat = mean_std(nearest_eval(run['eval_by_step'], target) for run in variant_runs)
            cells.append(format_stat(stat))
        final_stat = mean_std(run['final_eval_mean_of_means'] for run in variant_runs)
        cells.append(format_stat(final_stat))
        add(f'| {variant} | ' + ' | '.join(cells) + ' |')
    add('')
    add('Cells show the mean ± sample standard deviation across seeds at the '
        'nearest periodic evaluation (within 25k decisions). Blanks (—) mean '
        'the run did not reach that point.')
    add('')
    add('## Per-seed detail')
    add('')
    add('| Variant | Seed | Steps | Best periodic eval | Last periodic eval | Final 20-game mean_of_means | Steps/s | Stopped early |')
    add('| --- | --- | --- | --- | --- | --- | --- | --- |')
    for run in runs:
        add(f"| {run['variant']} | {run['seed']} | {run['steps_completed']} | "
            f"{_cell(run['best_eval_mean'])} | {_cell(run['final_eval_mean'])} | "
            f"{_cell(run['final_eval_mean_of_means'])} | "
            f"{_cell(run['steps_per_sec'], 1)} | {run['stopped_early']} |")
    add('')
    add('## Anomalies')
    add('')
    anomalies = []
    target_steps = max(PERIODIC_STEPS)
    for run in runs:
        if run['stopped_early']:
            anomalies.append(f"- `{run['tag']}` stopped early: {run['stop_reason']}.")
        if run['steps_completed'] < target_steps:
            anomalies.append(f"- `{run['tag']}` completed {run['steps_completed']} of "
                             f"{target_steps} decisions.")
        if run['final_eval_mean_of_means'] is None:
            anomalies.append(f"- `{run['tag']}` has no final independent evaluation.")
        if not run['eval_by_step']:
            anomalies.append(f"- `{run['tag']}` recorded no periodic evaluations.")
    if anomalies:
        lines.extend(anomalies)
    else:
        add('No missing runs, early stops, incomplete budgets, or missing final evaluations.')
    add('')
    add('## Seed-spread check and interpretation')
    add('')
    if len(found_variants) < 2:
        add('Not enough variants completed for pairwise comparisons.')
    else:
        add('Pairwise differences at two checkpoints. "Seed spread" is '
            '`sqrt(std_a² + std_b²)` from the per-variant tables; a difference '
            'is called distinguishable only when its magnitude exceeds that '
            'combined spread. This is a deliberately conservative, low-power '
            'test with 3 seeds.')
        add('')
        add('| Comparison | Checkpoint | Mean A | Mean B | Δ (A−B) | Seed spread | Exceeds seed spread? |')
        add('| --- | --- | --- | --- | --- | --- | --- |')
        comparisons = _comparisons(runs, found_variants)
        for row in comparisons:
            add('| ' + ' | '.join(row) + ' |')
        add('')
        add('### Which tuning direction differed?')
        add('')
        for line in _interpretation(comparisons):
            add(f'- {line}')
        add('')
        add('Interpretation is limited by the 3 seeds and 200k-decision budget: '
            'differences that do not exceed the seed spread are **inconclusive**, '
            'not evidence of equivalence, and differences that do exceed it are '
            'still single-grid-point observations that would need more seeds and '
            'longer runs to confirm.')
    return '\n'.join(lines) + '\n'


def _comparisons(runs: list[dict], variants) -> list[list[str]]:
    """Build pairwise comparison rows at 200k and at the final evaluation."""
    rows = []
    pairs = []
    if 'fast-eps' in variants and 'slow-eps' in variants:
        pairs.append(('epsilon decay: fast-eps vs slow-eps', 'fast-eps', 'slow-eps'))
    if 'fast-eps' in variants and 'low-lr' in variants:
        pairs.append(('learning rate: low-lr vs fast-eps', 'low-lr', 'fast-eps'))
    for label, variant_a, variant_b in pairs:
        for checkpoint, key in (('200k periodic', 'periodic'), ('final 20-game', 'final')):
            runs_a = [run for run in runs if run['variant'] == variant_a]
            runs_b = [run for run in runs if run['variant'] == variant_b]
            if key == 'periodic':
                stat_a = mean_std(nearest_eval(run['eval_by_step'], 200_000) for run in runs_a)
                stat_b = mean_std(nearest_eval(run['eval_by_step'], 200_000) for run in runs_b)
            else:
                stat_a = mean_std(run['final_eval_mean_of_means'] for run in runs_a)
                stat_b = mean_std(run['final_eval_mean_of_means'] for run in runs_b)
            if stat_a is None or stat_b is None:
                rows.append([label, checkpoint, '—', '—', '—', '—', '—'])
                continue
            mean_a, std_a, _ = stat_a
            mean_b, std_b, _ = stat_b
            delta = mean_a - mean_b
            spread = math.hypot(std_a, std_b)
            exceeds = abs(delta) > spread
            rows.append([label, checkpoint, f'{mean_a:.2f}', f'{mean_b:.2f}',
                         f'{delta:+.2f}', f'{spread:.2f}',
                         'yes' if exceeds else 'no'])
    return rows


def _interpretation(comparisons: list[list[str]]) -> list[str]:
    """Turn comparison rows into short plain-language statements."""
    lines = []
    for row in comparisons:
        label, checkpoint, mean_a, mean_b, delta, spread, exceeds = row
        if '—' in (mean_a, mean_b):
            lines.append(f'{label} ({checkpoint}): incomplete data, no comparison.')
            continue
        direction = 'higher' if float(delta) > 0 else 'lower' if float(delta) < 0 else 'equal'
        if exceeds == 'yes':
            lines.append(
                f'{label} ({checkpoint}): mean difference {delta} (A {direction}); '
                f'it exceeds the combined seed spread ({spread}), so this grid '
                f'point differs beyond seed noise.')
        else:
            lines.append(
                f'{label} ({checkpoint}): mean difference {delta}; it is within '
                f'the combined seed spread ({spread}), so the difference is '
                f'not distinguishable from seed noise here.')
    if any(row[-1] == 'yes' for row in comparisons):
        lines.append('At least one contrast exceeded seed spread, but with only '
                     '3 seeds the sign can still change with more seeds.')
    else:
        lines.append('No contrast exceeded the combined seed spread: the flagging '
                     'from the earlier 20k pilots is not reproduced as a reliable '
                     'difference at this budget.')
    return lines


def _fmt(value, digits: int = 0) -> str:
    """Format config values compactly."""
    if value is None:
        return '—'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f'{number:.{digits}f}' if digits else f'{number:g}'


def _cell(value, digits: int = 2) -> str:
    """Format an optional per-run value for Markdown tables."""
    if value is None:
        return '—'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f'{number:.{digits}f}' if math.isfinite(number) else '—'


def plot_curves(runs: list[dict], variants, path: Path) -> None:
    """Plot per-variant mean eval-mean curve with a ±1 std band across seeds."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(variants), 2)))
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for index, variant in enumerate(variants):
        variant_runs = [run for run in runs if run['variant'] == variant]
        steps = sorted({step for run in variant_runs for step in run['eval_by_step']})
        if not steps:
            continue
        means, stds = [], []
        for step in steps:
            values = [run['eval_by_step'][step] for run in variant_runs
                      if step in run['eval_by_step']]
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        steps_array = np.array(steps, dtype=float)
        means_array = np.array(means, dtype=float)
        stds_array = np.array(stds, dtype=float)
        color = colors[index]
        axis.plot(steps_array, means_array, marker='o', markersize=4,
                  label=f'{variant} (n={len(variant_runs)} seeds)', color=color)
        if len(variant_runs) > 1:
            axis.fill_between(steps_array, means_array - stds_array,
                              means_array + stds_array, alpha=0.18, color=color,
                              label=f'{variant} ±1 std')
    axis.set_xlabel('Training decisions (vector steps; ~4 decisions per frame-skip step)')
    axis.set_ylabel('Greedy evaluation mean reward (Pong, 10 episodes)')
    axis.set_title('Pong DQN multi-seed learning curves (CPU, 200k decisions)')
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> int:
    """Load reports, write CSV/Markdown/PNG, and print a short summary."""
    args = parse_args()
    root = args.root
    runs = load_runs(root, args.variants, args.seeds)
    if not runs:
        print(f'error: no complete reports found under {root}', flush=True)
        return 2
    root.mkdir(parents=True, exist_ok=True)
    write_csv(runs, root / 'multiseed.csv')
    (root / 'SUMMARY.md').write_text(build_summary(runs, args.variants, args.seeds))
    plot_curves(runs, args.variants, root / 'multiseed.png')
    print(f'wrote {root / "multiseed.csv"}')
    print(f'wrote {root / "SUMMARY.md"}')
    print(f'wrote {root / "multiseed.png"}')
    print(f'runs summarized: {len(runs)}')
    for run in runs:
        print(f"  {run['variant']}-s{run['seed']}: steps={run['steps_completed']} "
              f"final20={_cell(run['final_eval_mean_of_means'])} "
              f"steps/s={_cell(run['steps_per_sec'], 1)} stopped_early={run['stopped_early']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
