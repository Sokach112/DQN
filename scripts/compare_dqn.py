"""Controlled comparison of the production DQN update and installed SB3 DQN.

Both use the same architecture, initial weights, Adam settings, environment
wrappers, replay implementation, collection loop, and transition schedules.
Only the update implementation differs. This isolates the learner, not the SB3
rollout stack. Vanilla target-network DQN is used by both; production defaults
remain Double DQN. Run parity first, then independent bounded training runs.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import torch
import stable_baselines3 as sb3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.logger import configure
from omegaconf import OmegaConf
from agent.atari_agent import AtariDQNAgent
from algorithm.offpolicy_algorithm.dqn import DQN
from env.make_envs import make_vec_envs
from memory.memory import ReplayBuffer
from utils.networks import AtariDQNNetwork
from utils.utils import set_seed


class MatchedFeatures(BaseFeaturesExtractor):
    """The project's convolution/512-unit feature stack inside an SB3 policy."""
    def __init__(self, observation_space, activation='leaky_relu'):
        """Construct identically shaped layers; weights are copied explicitly."""
        super().__init__(observation_space, features_dim=512)
        source = AtariDQNNetwork(observation_space.shape, 6, activation)
        for name in ('conv1','conv2','conv3','fc1','activation'):
            setattr(self,name,getattr(source,name))

    def forward(self, x):
        """Accept SB3-normalized images and emit 512 hidden features."""
        for layer in (self.conv1,self.conv2,self.conv3):
            x=self.activation(layer(x))
        return self.activation(self.fc1(x.flatten(1)))


def parameter_pairs(custom, reference):
    """Yield corresponding online tensors for explicit equality checks/copies."""
    for name in ('conv1','conv2','conv3','fc1','fc2'):
        left=getattr(custom.network,name)
        right=reference.q_net.q_net[0] if name=='fc2' else getattr(reference.q_net.features_extractor,name)
        yield left.weight,right.weight
        yield left.bias,right.bias


def sample_tensors(batch, device):
    """Convert the shared raw-byte replay sample to SB3's minibatch contract."""
    return ReplayBufferSamples(
        observations=torch.as_tensor(batch['states'],device=device),
        actions=torch.as_tensor(batch['actions'],device=device),
        next_observations=torch.as_tensor(batch['next_states'],device=device),
        dones=torch.as_tensor(batch['dones'],device=device).reshape(-1,1),
        rewards=torch.as_tensor(batch['rewards'],device=device).reshape(-1,1))


class FixedSample:
    """Supply one explicit minibatch to the unmodified SB3 train method."""
    def __init__(self, batch, device):
        """Cache the tensor form for exactly one update."""
        self.batch=sample_tensors(batch,device)

    def sample(self, batch_size, env=None):
        """Return the chosen sample without consuming additional random numbers."""
        return self.batch


def build(seed, capacity, device, activation):
    """Create identically initialized learners and their shared environment API."""
    set_seed(seed)
    cfg=OmegaConf.load(Path(__file__).resolve().parent.parent/'config/config.yaml')
    cfg.env.num_training_envs=cfg.env.num_testing_envs=1
    cfg.algorithm.architecture="nature2015"
    cfg.algorithm.loss="huber"
    cfg.algorithm.double_dqn=False
    cfg.algorithm.target_update_interval=2500
    cfg.algorithm.learning_starts=10000
    cfg.algorithm.epsilon_timestep=400000
    cfg.algorithm.activation=activation
    cfg.algorithm.buffer_size=capacity
    env=make_vec_envs(cfg.env,True,seed=seed)
    agent=AtariDQNAgent(env.observation_space,env.action_space,device,activation)
    target=AtariDQNAgent(env.observation_space,env.action_space,device,activation)
    replay=ReplayBuffer(env.observation_space,env.action_space,capacity,1)
    custom=DQN(env,None,replay,agent,None,device,'unused.pth',cfg,target)
    reference=sb3.DQN('CnnPolicy',env,learning_rate=1e-4,buffer_size=1,
        gamma=.99,batch_size=32,device=device,seed=seed,
        policy_kwargs={'features_extractor_class':MatchedFeatures,
          'features_extractor_kwargs':{'activation':activation}, 'net_arch':[],
          'optimizer_class':torch.optim.Adam,'optimizer_kwargs':{'eps':1e-8}},
        verbose=0)
    reference.set_logger(configure(folder=None,format_strings=[]))
    with torch.no_grad():
        for left,right in parameter_pairs(agent,reference):right.copy_(left)
    reference.q_net_target.load_state_dict(reference.q_net.state_dict())
    return cfg,env,custom,reference


def reference_update(reference, batch, device, step, target_interval):
    """Call installed SB3's real optimizer update, then match target timing."""
    reference.replay_buffer=FixedSample(batch,device)
    reference.train(gradient_steps=1,batch_size=len(batch['states']))
    if step % target_interval==0:
        reference.q_net_target.load_state_dict(reference.q_net.state_dict())


def parity(output, seed, device, activation):
    """Compare predictions and optimizer results on 50 identical real minibatches."""
    cfg,env,custom,reference=build(seed,1024,device,activation)
    rng=np.random.default_rng(seed)
    try:
        state=env.reset()
        signs=[]
        for _ in range(1024):
            action=rng.integers(env.action_space.n,size=1)
            successor,reward,done,infos=env.step(action)
            terminal=successor.copy()
            mask=done.copy()
            if done[0]:
                terminal[0]=infos[0]['terminal_observation']
                mask[0]=not infos[0].get('TimeLimit.truncated',False)
            custom.buffer.add(dict(states=state[None],next_states=terminal[None],actions=action[None],rewards=reward[None],dones=mask[None]))
            signs.append(float(reward[0]));state=successor
        custom.target_update_interval=10
        errors=[]
        for step in range(1,51):
            batch=custom.buffer.sample(32)
            with torch.no_grad():
                input_tensor=torch.as_tensor(batch['states'],device=device)
                prediction_error=(custom.agent.q_values(batch['states'])-reference.q_net(input_tensor)).abs().max().item()
            with patch.object(custom.buffer,'sample',return_value=batch):
                result=custom._update_policy()
            reference_update(reference,batch,device,step,10)
            errors.append({'step':step,'prediction_max_abs_error_before':prediction_error,
                'loss_abs_error':abs(result.metrics['network/loss']-float(reference.logger.name_to_value['train/loss'])),
                'parameter_max_abs_error_after':max((a-b).abs().max().item() for a,b in parameter_pairs(custom.agent,reference))})
        report={'sb3_version':sb3.__version__,'device':device,'activation':activation,'seed':seed,
                'scope':'Identical initialization, 50 identical real replay batches, vanilla DQN, hard target every 10 updates.',
                'reward_counts':{'positive':sum(r>0 for r in signs),'negative':sum(r<0 for r in signs),'zero':sum(r==0 for r in signs)},'updates':errors,
                'passed':max(e['parameter_max_abs_error_after'] for e in errors)<1e-5}
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'parity_passed':report['passed'],'max_parameter_error':max(e['parameter_max_abs_error_after'] for e in errors)},indent=2),flush=True)
        if not report['passed']:raise RuntimeError('Parity check failed; inspect report before training')
    finally:env.close()


def evaluate(q_function, env_args, seed, episodes):
    """Evaluate complete games using separate envs and preserve training RNGs."""
    numpy_state=np.random.get_state();torch_state=torch.random.get_rng_state()
    env=make_vec_envs(env_args,False,seed=seed)
    scores=[];actions=np.zeros(env.action_space.n,dtype=int)
    try:
        state=env.reset()
        while len(scores)<episodes:
            with torch.no_grad():action=q_function(state).argmax(1).cpu().numpy()
            actions+=np.bincount(action,minlength=len(actions))
            state,_,_,infos=env.step(action)
            if 'episode' in infos[0]:scores.append(float(infos[0]['episode']['r']))
        return {'scores':scores,'mean':float(np.mean(scores)),'action_fractions':(actions/actions.sum()).tolist()}
    finally:
        env.close();np.random.set_state(numpy_state);torch.random.set_rng_state(torch_state)


def train(backend, output, seed, steps, capacity, device, activation, episodes):
    """Train independently with matched collection/replay/update/evaluation rules."""
    cfg,env,custom,reference=build(seed,capacity,device,activation)
    # Dedicated action RNG prevents model construction or evaluation from affecting exploration.
    rng=np.random.default_rng(seed)
    updates=0;reward_counts=np.zeros(3,dtype=int);start=time.monotonic()
    q_function=custom.agent.q_values if backend=='custom' else lambda x:reference.q_net(torch.as_tensor(x,device=device))
    report={'backend':backend,'seed':seed,'steps':steps,'activation':activation,
      'algorithm':'vanilla DQN with target','scope':'Shared wrappers, collector and replay; independent trajectories after policies diverge.',
      'sb3_version':sb3.__version__,'config':OmegaConf.to_container(cfg.algorithm,resolve=True),
      'evaluation':[],'progress':[]}
    try:
        state=env.reset()
        report['evaluation'].append({'step':0,**evaluate(q_function,cfg.env,seed+100,episodes)})
        for step in range(1,steps+1):
            epsilon=1-.9*min(1.,(step-1)/400000)
            if rng.random()<epsilon:action=rng.integers(env.action_space.n,size=1)
            else:
                with torch.no_grad():action=q_function(state).argmax(1).cpu().numpy()
            successor,reward,done,infos=env.step(action)
            terminal=successor.copy();mask=done.copy()
            if done[0]:
                terminal[0]=infos[0]['terminal_observation'];mask[0]=not infos[0].get('TimeLimit.truncated',False)
            custom.buffer.add(dict(states=state[None],next_states=terminal[None],actions=action[None],rewards=reward[None],dones=mask[None]))
            reward_counts[int(np.sign(reward[0]))+1]+=1
            state=successor
            if step>=10000 and step%4==0:
                updates+=1;batch=custom.buffer.sample(32)
                if backend=='custom':
                    with patch.object(custom.buffer,'sample',return_value=batch):result=custom._update_policy()
                    loss=result.metrics['network/loss']
                else:
                    reference_update(reference,batch,device,updates,2500)
                    loss=float(reference.logger.name_to_value['train/loss'])
            if step%5000==0 or step==steps:
                progress={'step':step,'updates':updates,'epsilon':1-.9*min(1.,step/400000),'reward_counts_negative_zero_positive':reward_counts.tolist(),'seconds':time.monotonic()-start}
                if updates:progress['last_minibatch_loss']=loss
                report['progress'].append(progress)
                output.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps({'backend':backend,**progress}),flush=True)
            if step%25000==0 or step==steps:
                report['evaluation'].append({'step':step,**evaluate(q_function,cfg.env,seed+100,episodes)})
                print(json.dumps({'backend':backend,'evaluation':report['evaluation'][-1]}),flush=True)
                if backend=='custom':custom.agent.save(output.with_suffix('.pth'))
                else:reference.save(output.with_suffix('.zip'))
                output.write_text(json.dumps(report,indent=2)+'\n')
    finally:env.close()


def native_train(output, seed, steps, capacity, device, activation, episodes):
    """Run SB3's own collector/replay as a second, explicitly near-matched control.

    SB3 samples random actions throughout warmup, starts updates strictly after
    learning_starts, and copies targets on environment-step boundaries. These
    documented differences are retained rather than hidden by a shared collector.
    """
    from stable_baselines3.common.buffers import ReplayBuffer as SB3ReplayBuffer
    from stable_baselines3.common.callbacks import BaseCallback
    cfg,env,custom,reference=build(seed,1,device,activation)
    reference.replay_buffer=SB3ReplayBuffer(capacity,env.observation_space,env.action_space,
                                          device=device,n_envs=1,handle_timeout_termination=True)
    reference.learning_starts=10000
    reference.target_update_interval=10000
    reference.exploration_schedule=lambda progress:1-.9*min(1.,(1-progress)*steps/400000)
    report={'backend':'sb3_native','seed':seed,'steps':steps,'activation':activation,
      'algorithm':'vanilla DQN with target','sb3_version':sb3.__version__,
      'scope':'SB3 native collector/replay/train, same initialization/architecture/wrappers/hyperparameters.',
      'differences':['Uniform random warmup; custom uses scheduled epsilon during warmup.',
         'First update at 10004 rather than 10000.',
         'Target copied every 10000 environment transitions rather than every 2500 completed updates.'],
      'evaluation':[],'progress':[]}
    q_function=lambda x:reference.q_net(torch.as_tensor(x,device=device))
    start=time.monotonic()

    class Progress(BaseCallback):
        """Persist progress without affecting the native training procedure."""
        def _on_step(self):
            """Log at bounded intervals; the callback precedes the current update."""
            if self.num_timesteps%5000==0:
                row={'step':self.num_timesteps,'updates_before_current_rollout_update':reference._n_updates,
                     'seconds':time.monotonic()-start,'epsilon':reference.exploration_rate}
                report['progress'].append(row)
                output.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps({'backend':'sb3_native',**row}),flush=True)
            return True
    try:
        report['evaluation'].append({'step':0,**evaluate(q_function,cfg.env,seed+100,episodes)})
        reference.learn(total_timesteps=steps,callback=Progress(),log_interval=None)
        report['updates']=reference._n_updates
        report['evaluation'].append({'step':reference.num_timesteps,**evaluate(q_function,cfg.env,seed+100,episodes)})
        reference.save(output.with_suffix('.zip'))
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'backend':'sb3_native','evaluation':report['evaluation'][-1]}),flush=True)
    finally:env.close()


def main():
    """Run a reproducible parity test or matched independent learning runs."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('parity','train','native'),default='parity')
    parser.add_argument('--backend',choices=('custom','sb3','both'),default='both')
    parser.add_argument('--steps',type=int,default=100000)
    parser.add_argument('--buffer-size',type=int,default=50000)
    parser.add_argument('--episodes',type=int,default=3)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--activation',choices=('relu','leaky_relu'),default='leaky_relu')
    parser.add_argument('--output',type=Path,default=Path('outputs/comparison'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    if min(args.steps,args.buffer_size,args.episodes)<=0:parser.error('budgets must be positive')
    if args.mode=='parity':parity(args.output/'parity.json',args.seed,args.device,args.activation)
    elif args.mode=='native':
        native_train(args.output/'sb3_native.json',args.seed,args.steps,args.buffer_size,args.device,args.activation,args.episodes)
    else:
        for backend in ('custom','sb3') if args.backend=='both' else (args.backend,):
            train(backend,args.output/f'{backend}.json',args.seed,args.steps,args.buffer_size,args.device,args.activation,args.episodes)


if __name__=='__main__':main()
