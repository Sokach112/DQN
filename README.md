# README

## Environment Configuratiom

1. **Python Version**  
   This project uses **Python 3.10**.

2. **Install PyTorch**  
   Select installation commands according to your device:
   
   - **With GPU（CUDA）**：
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
     > Replace`cu118` with the CUDA version that matches your GPU.

   - **CPU-only**：
     ```bash
     pip install torch torchvision torchaudio
     ```


3. **Dependencies Installation**  

   ```bash
   pip install -r requirements.txt
   ``` 

## Running training

Run from the project root after installing PyTorch and `requirements.txt`:

```bash
python main.py
```

The Atari ROMs must be available to ALE. The default game is
`PongNoFrameskip-v4`; supported game IDs are listed in `env/make_envs.py`.
The environment factory uses Stable-Baselines3 vector environments (not
Gymnasium's vector API). Training uses clipped rewards and life-loss endings;
evaluation uses complete games and original rewards. FIRE resets start games
that require them.

A short run that exercises collection, learning, and checkpoint writing:

```bash
python main.py total_epoch=3 interact_per_epoch=2 algorithm.buffer_size=16 \
  algorithm.batch_size=2 algorithm.learning_starts=0 test_interval=0 \
  train_log_interval=2 use_tensorboard=false
```

`seed` seeds the libraries and environments; each vector member receives a
separate seed. Exact reproducibility still depends on device and dependencies.
CUDA is preferred, followed by Apple MPS and CPU.

## Configuration and counters

- `total_epoch`: number of collect/update cycles. Each cycle gathers
  `interact_per_epoch` vector steps and performs at most one gradient update.
- `interaction_step`: individual transitions across all training environments.
  Logging, evaluation, saving, warmup, and epsilon use this counter. An interval
  triggers when crossed, even when a batch skips its exact multiple. Set logging,
  evaluation, or saving intervals to zero to disable their periodic trigger.
  Evaluation first runs after its interval; a final checkpoint is always saved.
- `algorithm.learning_starts`: minimum transitions before training. Replay must
  also contain at least `batch_size` transitions.
- `algorithm.epsilon_schedule`: `linear` or `constant`. Linear interpolation runs
  from `start_epsilon` to `end_epsilon` over `epsilon_timestep` transitions.
  The old `epsilon_schedular` key is accepted when the new key is absent.
  `train_action_deterministic=true` explicitly disables exploration.
- `use_target=true`: Double DQN, with online action selection and target-network
  evaluation. Without a target, ordinary max-Q bootstrapping uses the online net.
- `target_update_interval`: completed gradient updates between hard copies.
  Soft updates run after every gradient update with `target_update_tau`.

Replay capacity counts transitions, rounded down to whole vector steps. Explicit
current and next observations preserve terminal states across automatic resets.
At `(4, 84, 84)` uint8, the default 10,000 transitions require approximately
565 MB just for observations; one million would require about 56.5 GB. Increase
capacity deliberately for longer experiments. This implementation prioritizes
correctness and clarity over frame-deduplicated storage.

## Code map and data contracts

| File | Responsibility |
| --- | --- |
| `main.py` | Hydra configuration, component assembly, and resource cleanup |
| `config/config.yaml` | Training, environment, logging, and exploration defaults |
| `env/make_envs.py` | Atari preprocessing, frame stacking, monitoring, seeded vector factories |
| `agent/atari_agent.py` | Pixel normalization, batched actions, Q-values, checkpoints |
| `utils/networks.py` | Convolutional network taking normalized channel-first images |
| `algorithm/offpolicy_algorithm/baseoffpolicy.py` | Collection, counters, schedules, training lifecycle |
| `algorithm/offpolicy_algorithm/dqn.py` | Huber TD loss, gradient clipping, Double DQN targets |
| `memory/memory.py` | Circular replay with uniform sampling |
| `utils/evaluate.py` | Exact episode-count evaluation and cleanup |
| `logger/logger.py`, `utils/result.py` | Console/backend logging, scalar metrics, timing |
| `tests/test_regressions.py` | ROM-independent regression suite |

Collection batches are dictionaries with leading `(time, environments)` axes.
Actions can be `(time, environments)` or include a final singleton axis.
Sampling returns a dictionary whose leading axis is minibatch size; reward and
terminal vectors are reshaped to columns before TD loss calculation, avoiding
accidental broadcasting. `dones` masks true termination/life loss; time-limit
truncations bootstrap from `info['terminal_observation']`, never the next game's
reset observation.

Agents accept uint8 images or already-normalized float images. Byte inputs are
normalized exactly once by the agent; direct network calls expect floats in
`[0, 1]`. Evaluation restores the previous agent mode and closes its environments
on success or failure. Optional logging packages are imported only when enabled.

`newest_model.pth` contains model weights and space metadata. `agent.load(path)`
loads a trusted checkpoint onto the agent's current device. These checkpoints do
not include replay, optimizer state, counters, or RNG state, so they are not full
training-resume snapshots.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic image environments and do not require ROMs, GPU access, or
online logging accounts. They cover replay overflow and successors, vector
collection, truncation masks, TD updates, target-copy timing, normalization,
exploration shape, checkpoint loading, evaluation cleanup, metrics, and wrappers.

## Diagnosing flat Atari scores

A low TD loss and increasing update count do not establish policy improvement.
Before another long run, inspect an existing trusted checkpoint:

```bash
python diagnose.py --checkpoint /path/to/newest_model.pth --steps 2000 \
  --device auto --output diagnostic_report.json
```

This runs bounded random and greedy rollouts without training. The JSON records
completed game scores, action frequencies, observation range/dtype, frame changes,
Q-value variability, and the gap between the two highest Q-values. It also records
source/checkpoint hashes for identifying the code and model being inspected. Omit
`--checkpoint` for an untrained control. The environment comes from the local
`config/config.yaml`; it must match the checkpoint's game. A rollout may finish
before an episode, producing an empty score list; that is not a zero score.

One dominant action may explain poor play but does not establish the cause.
Changing images and finite Q-values likewise do not establish successful learning.
`tests/test_learning.py` checks exact Double DQN targets and basic visual learning;
it is explicitly not a Pong convergence benchmark.

### Checkpoint destination in Colab

The optional Hydra override `+save_pth=/content/drive/MyDrive/dqn_checkpoints/`
now selects a checkpoint directory; `newest_model.pth` is appended. An explicit
filename such as `+save_pth=/content/drive/MyDrive/dqn_checkpoints/pong.pth` is
also accepted. The resolved destination is printed at startup. Without this
override, checkpoints remain under `log_dir`.

Earlier versions silently ignored the added `save_pth` field. For those runs,
look for `outputs/**/newest_model.pth` under the Colab project directory before
ending the runtime. With the default save interval, the first periodic save is
at 500,000 transitions, and a final save occurs on successful training completion.
Loading weights is separate: passing `save_pth` never resumes a training run.

### Visual-feature collapse and activation choice

A failed Pong checkpoint inspected on 2026-10-12 chose action 3 on every one of
5,000 greedy steps. Its third convolution's ReLU output was all zeros across
128 sampled game observations. Backpropagation on those observations produced
zero gradients in all three convolutional weight tensors. Thus the downstream
layers could fit constant values without responding to the ball or paddle.
This identifies the checkpoint's failure mechanism, not when or why it began.

New training runs now default to `algorithm.activation=leaky_relu` (negative slope
0.01); `algorithm.activation=relu` remains available for controlled comparisons.
LeakyReLU retains a nonzero derivative for negative inputs. On the failed weights,
a local activation-only probe restored nonzero gradients to all three convolutions.
That is a gradient-flow safeguard, not a validated Pong performance improvement.

Activation choice is saved in checkpoint metadata. Legacy checkpoints without
this field load with ReLU, preserving their original policy even when the current
training default is LeakyReLU. Loading does not silently repair or convert them.
Training also logs `network/encoder_gradient_norm` (after gradient clipping) and
`td_error_abs`. A persistently zero encoder gradient warrants an activation audit.
The checkpoint diagnostic now includes per-layer activation sparsity, variation
across game observations, and a gradient probe that does not update model weights.

## Controlled learner comparison

`compare_dqn.py` isolates the custom update from collection/preprocessing:

```bash
python compare_dqn.py --mode parity
python compare_dqn.py --mode train --backend both --steps 25000 --episodes 3
python compare_dqn.py --mode native --steps 25000 --episodes 3
```

All commands default to CPU, one PyTorch thread, seed 0, 50,000 replay transitions,
LeakyReLU, and the same initial weights. Outputs go to `outputs/comparison`.
Use a different `--output` directory to retain independent experiments. Existing
reports/checkpoints in the chosen directory are replaced. Atari ROMs are required.

- **Parity:** identical real minibatches through the production custom update and
  the installed SB3 `DQN.train`; compare predictions, losses, and weights over
  50 updates, including five target copies.
- **Train:** independent rollouts with a shared collector and replay contract.
  This intentionally isolates learner differences; it cannot independently test
  the project's collection/replay implementation.
- **Native:** SB3's own collector and replay on the project's environment wrappers.
  The network/initial weights/Adam options are matched. SB3 retains uniform random
  warmup, begins updating at 10,004 rather than 10,000 transitions, and copies its
  target at environment-step boundaries. These small differences are recorded in
  the report; this control is not claimed to be numerically identical.

The comparison uses **vanilla target-network DQN** for both implementations,
selected by the new `algorithm.double_dqn=false` option. The normal training
configuration keeps `double_dqn=true`. Setting `use_target=false` is not an
appropriate substitute: that removes the target network entirely.

The budget controls total transitions, not epochs. Warmup is 10,000 transitions,
epsilon decays from 1 to .1 over 400,000 transitions, and training uses one update
per four transitions. Final evaluation uses separately seeded complete games and
preserves the training NumPy/Torch CPU RNG states. Three evaluation games and a
single seed are a bounded diagnostic, not a convergence or reliability benchmark.
SB3-native checkpoints retain their usual `.zip` format; shared-run custom
checkpoints use `.pth`.

Audit a trusted checkpoint's image sensitivity without training:

```bash
python checkpoint_sensitivity.py --checkpoint /path/to/newest_model.pth
```

This saves a 256-image regional-occlusion/frame-order report and a sample frame
stack. Regional removal creates synthetic inputs; a changed action does not
establish correct tracking, and unchanged actions do not prove the region is
ignored. The reported centered Q-values remove the per-state mean to distinguish
changes in overall predictions from changes in relative action preferences.

`python compare_wrappers.py` independently compares 2,000 identical actions with
stock SB3 AtariWrapper/VecTransposeImage/VecFrameStack. Its report excludes reset
padding and does not exercise the project's 5,000-decision time-limit boundary.
This verifies normal rollout preprocessing without claiming equivalence for all
termination cases.
