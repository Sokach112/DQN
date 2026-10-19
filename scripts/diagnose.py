"""Bounded Atari checkpoint audit; does not train or overwrite model weights."""

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
from utils.utils import set_seed, get_best_device


def inspect_visual_path(agent, observations):
    """Measure hidden activations and visual gradients without updating weights.

    The gradient probe minimizes the mean Q-value, not a replay TD loss. A zero
    encoder gradient plus an all-zero hidden ReLU output identifies a blocked
    visual path on these observations; it cannot establish when collapse began.
    """
    samples = np.asarray(observations)
    layer_report = {}
    handles = []
    names = [name for name in ("conv1", "conv2", "conv3", "fc1")
             if hasattr(getattr(agent.network, name), "weight")]

    def capture(name):
        """Build a hook that measures a hidden layer after its activation."""
        def hook(module, inputs, output):
            """Record sparsity and variation across the sampled observations."""
            # MPS has no float64 support; float32 is sufficient for these diagnostics.
            values = agent.network.activation(output.detach()).float()
            layer_report[name] = {
                'zero_fraction': float((values == 0).float().mean()),
                'mean_std_across_states': float(values.std(dim=0, unbiased=False).mean()),
            }
        return hook

    try:
        for name in names:
            handles.append(getattr(agent.network, name).register_forward_hook(capture(name)))
        with torch.no_grad():
            qs = agent.q_values(samples).float().cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    with torch.enable_grad():
        output = agent.q_values(samples).mean()
        parameters = [getattr(agent.network, name).weight for name in names]
        gradients = torch.autograd.grad(output, parameters)
    return {
        'sample_count': len(samples), 'activation': agent.network.activation_name,
        'layers': layer_report,
        'q_std_by_action': np.std(qs, axis=0).tolist(),
        'gradient_norm_by_layer': {name: float(grad.norm()) for name, grad in
            zip(names, gradients)},
    }


def audit(checkpoint=None, steps=2000, seed=0, device='cpu'):
    """Compare random/greedy rollouts and record observation and policy statistics.

    A checkpoint must be trusted: loading includes pickled Gym space metadata.
    Fixed step budgets can end mid-game; only completed game scores are reported.
    """
    if steps <= 0:
        raise ValueError('steps must be positive')
    set_seed(seed)
    cfg = OmegaConf.load(Path(__file__).resolve().parent.parent / 'config/config.yaml')
    cfg.env.num_testing_envs = 1
    envs = make_vec_envs(cfg.env, False, seed=seed)
    try:
        agent = AtariDQNAgent(envs.observation_space, envs.action_space, device)
        if checkpoint:
            agent.load(checkpoint)
        agent.eval()
        report = {'checkpoint': str(checkpoint) if checkpoint else None,
                  'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None,
                  'environment': cfg.env.name, 'seed': seed, 'device': str(device),
                  'steps_per_policy': steps,
                  'action_meanings': envs.env_method('get_action_meanings'),
                  'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path('agent/atari_agent.py'), Path('env/make_envs.py'), Path('algorithm/offpolicy_algorithm/dqn.py')]}}
        for policy in ('random', 'greedy'):
            envs.seed(seed)
            states = envs.reset()
            counts = np.zeros(envs.action_space.n, dtype=int)
            qs, gaps, changes, stack_changes, scores = [], [], [], [], []
            samples = []
            lo, hi = 255., 0.
            for step in range(steps):
                if policy == "random" and step % max(1, steps // 128) == 0 and len(samples) < 128:
                    samples.append(states[0].copy())
                lo, hi = min(lo,float(states.min())), max(hi,float(states.max()))
                with torch.no_grad():
                    q = agent.q_values(states).cpu().numpy()[0]
                if not np.isfinite(q).all():
                    raise ValueError('Non-finite Q-values')
                qs.append(q)
                sorted_q = np.sort(q)
                gaps.append(float(sorted_q[-1]-sorted_q[-2]))
                stack_changes.append(float(np.mean(states[:,1:] != states[:,:-1])))
                action = int(np.random.randint(envs.action_space.n)) if policy == 'random' else int(q.argmax())
                counts[action] += 1
                next_states, _, dones, infos = envs.step(np.array([action]))
                successor = infos[0]['terminal_observation'][None] if dones[0] else next_states
                changes.append(float(np.mean(states != successor)))
                if 'episode' in infos[0]:
                    scores.append(float(infos[0]['episode']['r']))
                states = next_states
            report[policy] = dict(completed_episode_scores=scores,
                action_fractions=(counts/steps).tolist(), observation_dtype=str(states.dtype),
                observation_range=[lo,hi], mean_changed_pixel_fraction=float(np.mean(changes)),
                mean_adjacent_frame_difference_fraction=float(np.mean(stack_changes)),
                mean_q_by_action=np.mean(np.asarray(qs,dtype=np.float64),axis=0).tolist(),
                std_q_by_action=np.std(np.asarray(qs,dtype=np.float64),axis=0).tolist(), mean_top_two_q_gap=float(np.mean(gaps)))
            if policy == "random":
                report["visual_path"] = inspect_visual_path(agent, samples)
        return report
    finally:
        envs.close()


def main():
    """Write an audit report for a trusted checkpoint or an untrained control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='diagnostic_report.json')
    args = parser.parse_args()
    device = get_best_device() if args.device == 'auto' else args.device
    report = audit(args.checkpoint, args.steps, args.seed, device)
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
