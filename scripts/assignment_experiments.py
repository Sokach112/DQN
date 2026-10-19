"""Run four bounded assignment experiments with fixed seed and saved configs."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
from utils.evaluate import evaluate_atari
from utils.utils import set_seed
from memory.memory import ReplayBuffer
from env.make_envs import make_vec_envs
from algorithm.offpolicy_algorithm.dqn import DQN
from agent.atari_agent import AtariDQNAgent
from omegaconf import OmegaConf
import torch
import sys
import json
import time
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

parser = argparse.ArgumentParser(
    description='Assignment pilots: baseline, two one-factor variants, and Breakout.')
parser.add_argument('--output', type=Path, default=root /
                    'outputs'/'assignment-pilots')
parser.add_argument('--steps', type=int, default=20000)
args = parser.parse_args()
if args.steps < 10000 or args.steps % 4:
 parser.error('steps must be a multiple of four, at least 10000')
out = args.output
if out.exists() and any(out.iterdir()):
 parser.error('use a new output directory to preserve previous runs')
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)


class Log:
 """Collect train and evaluation metrics for reproducible JSON reports."""

 def __init__(self):
  """Start an empty metric history."""
  self.rows = []

 def log_train(self, epoch, step, grad, result):
  """Record one training measurement."""
  self.rows.append(dict(kind='train', step=step,
                   updates=grad, **result.metrics))

 def log_test(self, epoch, step, grad, result):
  """Record one greedy evaluation measurement."""
  self.rows.append(dict(kind='test', step=step,
                   updates=grad, **result.metrics))


for name, game, lr, eps in [('baseline', 'PongNoFrameskip-v4', 1e-4, 400000), ('learning_rate', 'PongNoFrameskip-v4', 5e-5, 400000), ('exploration', 'PongNoFrameskip-v4', 1e-4, 100000), ('breakout', 'BreakoutNoFrameskip-v4', 1e-4, 400000)]:
 set_seed(0)
 cfg = OmegaConf.load(root/'config/config.yaml')
 cfg.log_dir = str(out/name)
 del cfg.hydra
 cfg.env.name = game
 cfg.env.num_training_envs = cfg.env.num_testing_envs = 1
 cfg.total_epoch = args.steps//4
 cfg.interact_per_epoch = 4
 cfg.algorithm.buffer_size = 10000
 cfg.algorithm.learning_starts = 5000
 cfg.algorithm.learning_rate = lr
 cfg.algorithm.epsilon_timestep = eps
 cfg.algorithm.target_update_interval = 2500
 cfg.test_interval = 10000
 cfg.test_episodes = 3
 cfg.train_log_interval = 1000
 cfg.save_interval = 0
 env = make_vec_envs(cfg.env, True, seed=0)
 log = Log()
 start = time.monotonic()
 try:
  agent = AtariDQNAgent(env.observation_space, env.action_space,
                        'cpu', activation='relu', architecture='paper2013')
  target = AtariDQNAgent(env.observation_space, env.action_space,
                         'cpu', activation='relu', architecture='paper2013')
  assert agent.network.conv1.out_channels == 16 and agent.network.conv2.out_channels == 32 and agent.network.fc1.out_features == 256
  algorithm = DQN(env, None, ReplayBuffer(env.observation_space, env.action_space,
                  10000, 1), agent, log, 'cpu', str(out/name/'model.pth'), cfg, target)
  log.log_test(-1, 0, 0, evaluate_atari(agent, 3, cfg.env, seed=0))
  algorithm.run()
  report = {'name': name, 'config': OmegaConf.to_container(
      cfg, resolve=True), 'seconds': time.monotonic()-start, 'rows': log.rows}
  (out/(name+'.json')).write_text(json.dumps(report, indent=2))
  print(name, [r for r in log.rows if r['kind'] == 'test'], flush=True)
 finally:
  env.close()
