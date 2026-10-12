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
