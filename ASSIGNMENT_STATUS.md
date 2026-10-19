# Assignment status — 2026-10-19

- [x] Paper2013 network option and default (16/32 convolutions, 256 hidden units, ReLU).
- [x] MSE used by the default update; explicit Huber alternative.
- [x] Vanilla target-network DQN baseline; optional Double DQN.
- [x] Preserve old checkpoint architecture/activation behavior.
- [x] Test two tuning directions independently in short matched pilots.
- [x] Record configurations, checkpoints, curves and initial reflection.
- [x] Run an initial additional-environment experiment (Breakout).
- [x] Repeat tuning with multiple training seeds and longer budgets.
  - 3 variants (fast-eps, slow-eps, low-lr) × 3 seeds × 200,000 decisions
    (800k emulator frames, 10× the pilot budget) on CPU. At this budget the
    pilot ranking does not reproduce; all variants remain near random play
    (−20.3 to −20.6 on 20-game final evaluation) and differences are within the
    seed spread. See `reports/04_multiseed_2026-10-18.md`.
- [x] Demonstrate sustained Pong improvement and the required +21 score.
  - Paper-scale Kaggle run (Tesla T4, 8 h 20 m): 2,500,000 decisions
    (10M emulator frames), one gradient update per decision (2.5M updates),
    replay 200k transitions, ε 1.0 → 0.1 over 250k decisions.
  - Greedy evaluation reached **21.0 ± 0.0** (all 10 games won 21–0) at ten
    checkpoints (800k, 1.35M, 1.45M, 1.8M, 2.0M, 2.1M, 2.2M, 2.25M, 2.4M,
    2.45M decisions); final checkpoint 20.5 ± 0.5.
  - The saved checkpoint was independently re-evaluated on fresh seeds:
    **100/100 complete games won 21–0** (mean 21.0, std 0.0).
  - Caveat: the Kaggle clone predates the assignment-baseline changes, so this
    run used the old defaults (nature2015 CNN, LeakyReLU, Huber loss, Double
    DQN), not the strict baseline (paper2013, ReLU, MSE, vanilla target). A
    ready-to-run strict-baseline notebook is prepared at
    [`kaggle/pong_baseline_paper2013.ipynb`](kaggle/pong_baseline_paper2013.ipynb).
  - Evidence: `reports/03_pong21_2026-10-19.md`.
- [ ] Reach comparable-to-paper performance on Breakout or another Atari game.
  - Not yet run at paper scale. A ready-to-run Kaggle notebook with the same
    2.5M-decision/1-update-per-decision recipe, evaluation every 250k
    decisions, and paper reference scores 168 (average) / 225 (best) is at
    [`kaggle/breakout_paper2013.ipynb`](kaggle/breakout_paper2013.ipynb).
- [x] Implement optional softmax action selection or explicitly justify its
  omission under the epsilon-greedy allowance.
  - Optional Boltzmann (`action_selection: softmax`, `softmax_temperature`) is
    implemented with tests; `epsilon_greedy` remains the default and the
    assignment's §2 epsilon-greedy allowance is cited as the justification for
    not making softmax the baseline.

See `reports/01_pilot_2026-10-18.md` for the initial pilot results,
`reports/02_long_run_2026-10-18.md` for the local long-budget
experiments and paper-scale analysis, and
`reports/03_pong21_2026-10-19.md` for the successful
paper-scale run and its independent verification.

The consolidated submission report is [`REPORT.md`](REPORT.md); detailed
reports and curated evidence are under [`reports/`](reports/).
