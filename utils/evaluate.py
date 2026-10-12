"""Deterministic evaluation on complete, unclipped Atari episodes."""
import numpy as np
from utils.result import Result
from env.make_envs import make_vec_envs


def evaluate_atari(agent, test_episodes, env_args, seed=0):
    """Evaluate exactly ``test_episodes`` games and restore the agent's mode.

    Evaluation environments are always closed, including on policy errors.
    """
    if test_episodes <= 0:
        raise ValueError("test_episodes must be positive")
    envs = make_vec_envs(env_args, False, scale=False, seed=seed)
    was_training = agent.training
    rewards = []
    try:
        agent.eval()
        states = envs.reset()
        with Result('evaluation') as result:
            while len(rewards) < test_episodes:
                actions = agent.select_action(states, deterministic=True)
                states, _, _, infos = envs.step(actions)
                for info in infos:
                    if 'episode' in info and len(rewards) < test_episodes:
                        rewards.append(info['episode']['r'])
        result.add_metric('episode_rewards_mean', np.mean(rewards))
        result.add_metric('episode_rewards_std', np.std(rewards))
        return result
    finally:
        agent.train(was_training)
        envs.close()
