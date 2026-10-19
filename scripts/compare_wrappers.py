"""Compare project preprocessing to SB3 AtariWrapper under identical actions.

The project repeats the first frame on reset; SB3 VecFrameStack zero-pads older
frames. Compare full stacks only after four steps since the last reset. The
project's 5,000-decision time limit is not exercised by this short random rollout.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
from pathlib import Path
import numpy as np
import gymnasium as gym
from omegaconf import OmegaConf
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from env.make_envs import make_vec_envs


def make_reference():
    """Use SB3's stock Atari preprocessing with its default matching settings."""
    return AtariWrapper(gym.make('PongNoFrameskip-v4',frameskip=1))


def main():
    """Compare rewards, endings, and aligned observations for 2,000 steps."""
    cfg=OmegaConf.load(Path(__file__).resolve().parent.parent/'config/config.yaml');cfg.env.num_training_envs=1
    custom=make_vec_envs(cfg.env,True,seed=0)
    reference=VecFrameStack(VecTransposeImage(DummyVecEnv([make_reference])),n_stack=4)
    reference.seed(0)
    rng=np.random.default_rng(0)
    report={'steps':2000,'seed':0,'reward_mismatches':0,'done_mismatches':0,
      'newest_frame_mismatches':0,'aligned_stack_mismatches':0,'aligned_stack_checks':0,
      'reset_padding_difference_expected':True,
      'scope':'Random-action rollout; excludes project time-limit boundary and reset stack padding.'}
    try:
        custom.reset();reference.reset();age=0
        for _ in range(2000):
            action=rng.integers(6,size=1)
            left,lr,ld,_=custom.step(action)
            right,rr,rd,_=reference.step(action)
            report['reward_mismatches']+=int(not np.array_equal(lr,rr))
            report['done_mismatches']+=int(not np.array_equal(ld,rd))
            report['newest_frame_mismatches']+=int(not np.array_equal(left[:,-1],right[:,-1]))
            age=0 if ld[0] or rd[0] else age+1
            if age>=4:
                report['aligned_stack_checks']+=1
                report['aligned_stack_mismatches']+=int(not np.array_equal(left,right))
        output=Path('outputs/comparison');output.mkdir(parents=True,exist_ok=True)
        (output/'wrappers.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:custom.close();reference.close()


if __name__=='__main__':main()
