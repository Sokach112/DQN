"""Bounded observation-ablation audit of a trusted Atari checkpoint.

Regions are image coordinates, not object detectors. Occlusion creates synthetic
inputs, so changes indicate sensitivity, not correct ball tracking or causality.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from agent.atari_agent import AtariDQNAgent
from env.make_envs import make_vec_envs
from utils.utils import set_seed


def sensitivity(checkpoint, output, steps=5000, seed=0):
    """Compare Q-values/action advantages under regional occlusion and frame reversal."""
    set_seed(seed);torch.set_num_threads(1)
    cfg=OmegaConf.load(Path(__file__).resolve().parent.parent/'config/config.yaml');cfg.env.num_testing_envs=1
    env=make_vec_envs(cfg.env,False,seed=seed)
    try:
        agent=AtariDQNAgent(env.observation_space,env.action_space,'cpu')
        agent.load(checkpoint);agent.eval()
        state=env.reset();samples=[]
        for step in range(steps):
            if step%max(1,steps//256)==0 and len(samples)<256:samples.append(state[0].copy())
            state,_,_,_=env.step(np.array([np.random.randint(env.action_space.n)]))
    finally:env.close()
    images=np.stack(samples)
    # Estimate background independently for each image row/channel, preserving
    # Atari's horizontal bands rather than introducing black rectangles.
    background=np.empty_like(images)
    for n in range(len(images)):
        for c in range(images.shape[1]):
            for y in range(images.shape[2]):
                background[n,c,y,:]=np.bincount(images[n,c,y],minlength=256).argmax()
    def values(batch):
        """Evaluate small batches to bound intermediate convolution memory."""
        with torch.no_grad():
            return np.concatenate([agent.q_values(batch[i:i+32]).numpy() for i in range(0,len(batch),32)]).astype(np.float64)
    base=values(images);centered=base-base.mean(1,keepdims=True)
    report={'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
      'seed':seed,'samples':len(images),'activation':agent.network.activation_name,
      'scope':'Random-policy observations; synthetic regional occlusion, not semantic ball/paddle labels.',
      'baseline':{'mean_std_q_across_images':float(base.std(0).mean()),
         'mean_std_action_advantage_across_images':float(centered.std(0).mean()),
         'mean_action_q_range':float(np.ptp(base,axis=1).mean()),
         'greedy_action_counts':np.bincount(base.argmax(1),minlength=base.shape[1]).tolist()},'ablations':{}}
    regions={'top_16_rows':(slice(0,16),slice(None)),
      'lower_68_rows':(slice(16,None),slice(None)),
      'left_playfield_strip':(slice(16,None),slice(0,16)),
      'center_playfield':(slice(16,None),slice(16,68)),
      'right_playfield_strip':(slice(16,None),slice(68,None))}
    for name in [*regions,'reverse_frame_order']:
        changed=images.copy()
        if name=='reverse_frame_order':changed=changed[:,::-1].copy()
        else:
            ys,xs=regions[name];changed[:,:,ys,xs]=background[:,:,ys,xs]
        q=values(changed);adv=q-q.mean(1,keepdims=True)
        report['ablations'][name]={'mean_abs_q_change':float(np.abs(q-base).mean()),
           'mean_abs_action_advantage_change':float(np.abs(adv-centered).mean()),
           'greedy_action_change_fraction':float(np.mean(q.argmax(1)!=base.argmax(1)))}
    output.mkdir(parents=True,exist_ok=True)
    (output/'sensitivity.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(10,3))
    for i,ax in enumerate(axes):
        ax.imshow(images[len(images)//2,i],cmap='gray',vmin=0,vmax=255)
        ax.axhline(15.5,color='orange',linewidth=.7);ax.set_title(f'Frame {i+1}');ax.axis('off')
    fig.suptitle('Actual checkpoint inputs; orange line marks region split')
    fig.tight_layout();fig.savefig(output/'input-frames.png',dpi=130);plt.close(fig)
    print(json.dumps(report,indent=2))


def main():
    """Run the audit without changing weights or collecting training data."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/comparison'))
    parser.add_argument('--steps',type=int,default=5000)
    args=parser.parse_args()
    if args.steps<1:parser.error('steps must be positive')
    sensitivity(args.checkpoint,args.output,args.steps)


if __name__=='__main__':main()
