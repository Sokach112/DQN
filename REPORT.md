# DQN on Atari — final report

This repository implements DQN from scratch (network, agent, replay, target
network, exploration) and applies it to Pong and Breakout. The main result is a
paper-scale Pong run that reaches the DQN paper's score: a **21.0 greedy
evaluation mean** at ten checkpoints and a saved policy that wins **100/100
fresh-seed games 21–0**.

## Assignment requirements

| Requirement | Status | Evidence |
| --- | --- | --- |
| Implement the CNN + fully-connected Q-network and forward pass | Done | `utils/networks.py`, `agent/atari_agent.py`, 26 tests |
| Deterministic argmax and optional softmax action selection | Done | optional Boltzmann mode in `agent/`, `tests/test_softmax.py` |
| ε-greedy DQN update with MSE (Huber optional), replay, hard/soft target updates | Done | `algorithm/offpolicy_algorithm/dqn.py`, `memory/`, tests |
| Vanilla target-network baseline, optional Double DQN | Done | `algorithm.double_dqn` flag (default `false`) |
| Tune at least two directions and analyse the causes | Done | `reports/01_pilot_2026-10-18.md`, `reports/04_multiseed_2026-10-18.md` |
| Reach the DQN paper's Pong score (21) | **Done** | `reports/03_pong21_2026-10-19.md` (21.0 ×10 checkpoints; 100/100 verification games) |
| Additional Atari environment comparable to the paper | Not run at paper scale | ready-to-run notebook `kaggle/breakout_paper2013.ipynb`; local long-run analysis in `reports/02_long_run_2026-10-18.md` |

Full status list: [`ASSIGNMENT_STATUS.md`](ASSIGNMENT_STATUS.md).

## Implementation

- **Network:** `paper2013` (16 8×8/stride-4, 32 4×4/stride-2, 256 ReLU hidden,
  linear outputs) is the default; `nature2015` (32/64/64, 512) remains selectable.
  Activation (`relu`/`leaky_relu`), loss (`mse`/`huber`), optimizer and
  `double_dqn` are configuration options.
- **Learning:** experience replay with explicit terminal observations, hard or
  soft target updates, gradient clipping, ε-greedy with linear/constant
  schedules, and an opt-in softmax (Boltzmann) exploration mode.
- **Environment support:** Atari wrappers built on the Stable-Baselines3 vector
  environment API (frame skip 4, 84×84 grayscale, 4-frame stacks, clipped
  rewards and life-loss endings for training; complete unclipped games for
  evaluation).
- **Tests:** 26 ROM-free regression tests
  (`python -m unittest discover -s tests -v`), including exact paper layer
  dimensions, MSE update, Double DQN targets, normalization, replay overflow,
  checkpoint loading and softmax distributions.

## Experiments

### 1. Short pilots (seed 0, 20k decisions)

Baseline, half learning rate and faster ε decay were compared on Pong plus a
Breakout run. All stayed near random play at this budget; three evaluation games
were too noisy to rank variants. Details and the initial reflection:
[`reports/01_pilot_2026-10-18.md`](reports/01_pilot_2026-10-18.md).

### 2. Multi-seed tuning at 1M frames

Three variants (fast ε decay, slower ε decay, half learning rate) × 3 seeds ×
200,000 decisions on CPU. At this budget every variant stays near random play
(−20.3 to −20.6 on 20-game final evaluation), and no contrast exceeds the seed
spread. The pilot ranking was therefore noise. Report and curves:
[`reports/04_multiseed_2026-10-18.md`](reports/04_multiseed_2026-10-18.md),
[`reports/figures/multiseed_curves.png`](reports/figures/multiseed_curves.png).

### 3. Local long run and compute analysis (seed 0, 1.25M decisions)

A tuned local run improved from −21 to a best fresh-seed 20-game mean of +4.3,
but did not reach 21. The measured budget analysis identified the missing
ingredients: the paper's full 10M-frame budget, one update per decision instead
of one per four, a 250k-decision ε anneal, and a larger replay. See
[`reports/02_long_run_2026-10-18.md`](reports/02_long_run_2026-10-18.md).

### 4. Paper-scale Kaggle run (Pong 21)

Applying all four changes (2.5M decisions = 10M frames, 1 update/decision,
replay 200k, ε 1→0.1 over 250k decisions) on a Tesla T4 solved Pong in 8 h 19 m:
21.0 at ten checkpoints and 100/100 fresh games at 21–0. Full protocol,
verification and caveat: [`reports/03_pong21_2026-10-19.md`](reports/03_pong21_2026-10-19.md).

![Pong learning curve](reports/figures/pong21_learning_curve.png)

## Caveat on the successful run

The Kaggle run used the pre-baseline code defaults (nature2015 network,
LeakyReLU, Huber loss, Double DQN), not the strict assignment baseline
(paper2013, ReLU, MSE, vanilla target). The repository now implements and
defaults to the strict baseline. To attribute 21 to the exact assignment
configuration, run [`kaggle/pong_baseline_paper2013.ipynb`](kaggle/pong_baseline_paper2013.ipynb)
and replace the numbers in the report; the earlier strict-baseline evidence
stopped at +4.3 because it used only 1.25M decisions at 1 update per 4 decisions.

## Reproducibility

```bash
# install
pip install -r requirements.txt

# tests (ROM-free)
python -m unittest discover -s tests -v

# short training smoke test
python main.py total_epoch=3 interact_per_epoch=2 algorithm.buffer_size=16 \
  algorithm.batch_size=2 algorithm.learning_starts=0 test_interval=0 \
  train_log_interval=2 use_tensorboard=false

# paper-scale strict-baseline run (see kaggle/ notebook for the full recipe)
python main.py total_epoch=2500000 interact_per_epoch=1 \
  algorithm.buffer_size=200000 algorithm.epsilon_timestep=250000 \
  algorithm.end_epsilon=0.1 algorithm.target_update_interval=1000 \
  test_interval=50000 test_episodes=10
```

Analysis and experiment drivers (run from the repository root):

```bash
python scripts/assignment_experiments.py --output outputs/pilots
python scripts/long_experiments.py --tag pong-long --steps 1500000 --device mps
python scripts/kaggle_results_analysis.py --log training.log \
  --checkpoint newest_model.pth --output outputs/analysis --games 100
```

## Repository map

| Path | Contents |
| --- | --- |
| `main.py`, `config/config.yaml` | Hydra training entry point and defaults |
| `agent/`, `algorithm/`, `env/`, `memory/`, `logger/`, `utils/` | implementation |
| `tests/` | 26 ROM-free regression tests |
| `scripts/` | pilot/long-run/analysis/verification drivers |
| `kaggle/` | paper-scale run notebooks (Pong strict baseline, Breakout) |
| `reports/` | detailed English reports, figures, data and verification JSON |
| `reports/03_pong21_2026-10-19.md` | main result: paper-scale Pong 21 |
| `ASSIGNMENT_STATUS.md` | requirement checklist with evidence |
| `outputs/` | local raw runs (gitignored; not part of the submission) |
