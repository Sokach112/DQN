"""Reproduce assignment pilot figures or additional checkpoint evaluations."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
from utils.evaluate import evaluate_atari
from env.make_envs import make_vec_envs
from agent.atari_agent import AtariDQNAgent
from omegaconf import OmegaConf
import torch
import sys
import json
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
out = args.output
torch.set_num_threads(1)
report = {}
for name in ['baseline', 'learning_rate', 'exploration', 'breakout']:
 cfg = OmegaConf.create(json.loads((out/(name+'.json')).read_text())['config'])
 cfg.env.num_testing_envs = 1
 env = make_vec_envs(cfg.env, False)
 agent = AtariDQNAgent(env.observation_space, env.action_space, 'cpu')
 agent.load(out/name/'model.pth')
 env.close()
 report[name] = {}
 for seed in (100, 200):
  report[name][seed] = evaluate_atari(agent, 10, cfg.env, seed).metrics
 (out/'heldout-evaluation.json').write_text(json.dumps(report, indent=2))
 print(name, report[name], flush=True)
