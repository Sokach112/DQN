# Kaggle runbooks

Ready-to-run notebooks for the two remaining assignment experiments. Both
assume the current working tree has been **committed and pushed** to
`github.com/Sokach112/DQN` branch `MyBranch`; the notebooks clone the repo and
then assert that the assignment-baseline defaults are present.

## Prerequisites

```bash
cd rlfromscratch
git add -A
git commit -m "assignment baseline, long-run tooling, and verification scripts"
git push origin MyBranch
```

Kaggle notebook settings: **Accelerator = GPU (T4 x2)**, **Internet = On**.

## 1. `pong_baseline_paper2013.ipynb` (recommended first)

Strict assignment baseline (`paper2013` CNN, ReLU, MSE, vanilla target DQN) at
the paper-scale recipe that already produced 21.0 on the old-default code:

- 2,500,000 decisions = 10,000,000 emulator frames
- 1 gradient update per decision (2,500,000 updates)
- replay 200,000 transitions, ε 1.0 → 0.1 over 250,000 decisions
- greedy evaluation every 50,000 decisions on 10 complete games
- estimated 8–10 h on a T4; short smoke test included first

Why run it: the successful Kaggle run used the pre-baseline code defaults
(`nature2015` + LeakyReLU + Huber + Double DQN). This notebook is the strict
configuration-compliance run. Its output (log + checkpoint) can be verified
locally with `kaggle_results_analysis.py`.

## 2. `breakout_paper2013.ipynb` (assignment §4.3 additional environment)

Same budget and network, on `BreakoutNoFrameskip-v4`; paper reference scores are
168 (average) / 225 (best episode). Evaluation every 250,000 decisions with 5
games to bound the cost of long Breakout episodes; estimated 8–11 h.

## After a run

1. Download the notebook's `*_artifacts.tgz` from the Kaggle output panel.
2. Copy the log, checkpoint and analysis folder into
   `rlfromscratch/outputs/assignment-2026-10-19-kaggle-pong/source/` (Pong) or a
   new directory under `outputs/` (Breakout).
3. Re-verify locally (CPU/MPS is enough for evaluation):

```bash
python kaggle_results_analysis.py \
  --log <training.log> --checkpoint <checkpoint.pth> \
  --output outputs/<run> --game <game> --games 100
```

## Notes

- Kaggle GPU sessions have a quota and a maximum duration; the Pong run above
  used 8 h 20 m. If the notebook approaches the limit, reduce
  `total_epoch` (e.g. `2000000`) and report the reduced budget explicitly.
- The strict baseline is smaller than the network that produced the solved Pong
  policy, so allow for slower learning; report the measured curve rather than
  assuming the previous result transfers.
