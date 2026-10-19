"""Regression checks runnable without Atari ROMs or external services."""
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import gymnasium as gym
from omegaconf import OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv
from agent.atari_agent import AtariDQNAgent
from algorithm.offpolicy_algorithm.dqn import DQN
from memory.memory import ReplayBuffer
from env.make_envs import TransposeImage, FrameStack
from utils.evaluate import evaluate_atari
from utils.result import Result


class ImageEnv(gym.Env):
    """Small deterministic image environment with time-limit endings."""
    observation_space = gym.spaces.Box(0, 255, (4, 84, 84), np.uint8)
    action_space = gym.spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, np.uint8), {}

    def step(self, action):
        assert np.asarray(action).ndim == 0
        return np.full(self.observation_space.shape, 255, np.uint8), 1., False, True, {'episode': {'r': 7.}}


class RegressionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_replay_overflow_and_successors(self):
        space = gym.spaces.Box(0, 255, (1,), np.uint8)
        buffer = ReplayBuffer(space, gym.spaces.Discrete(2), 6, 2)
        states = np.arange(20, dtype=np.uint8).reshape(10, 2, 1)
        buffer.add(dict(states=states, next_states=states+100, actions=np.zeros((10, 2)), rewards=np.zeros((10, 2)), dones=np.zeros((10, 2))))
        self.assertEqual(len(buffer), 6)
        sample = buffer.sample(100)
        np.testing.assert_array_equal(sample['next_states'], sample['states']+100)
        self.assertGreaterEqual(sample['states'].min(), 14)
        self.assertEqual(buffer.states.shape, (3, 2, 1))

    def test_collection_update_and_schedule(self):
        cfg = OmegaConf.load('config/config.yaml')
        cfg.env.num_training_envs = 2
        cfg.interact_per_epoch = 2
        cfg.algorithm.batch_size = 2
        cfg.algorithm.learning_starts = 0
        cfg.algorithm.epsilon_timestep = 4
        cfg.algorithm.target_update_interval = 2
        envs = DummyVecEnv([ImageEnv, ImageEnv])
        self.addCleanup(envs.close)
        agent = AtariDQNAgent(envs.observation_space, envs.action_space, 'cpu')
        target = AtariDQNAgent(envs.observation_space, envs.action_space, 'cpu')
        buffer = ReplayBuffer(envs.observation_space, envs.action_space, 8, 2)
        algo = DQN(envs, None, buffer, agent, None, 'cpu', '/tmp/unused.pth', cfg, target)
        algo.initialize()
        batch, _ = algo.interact_with_envs()
        self.assertEqual(batch['states'].shape, (2, 2, 4, 84, 84))
        self.assertTrue((batch['next_states'] == 255).all())
        self.assertFalse(batch['dones'].any())
        self.assertEqual(algo.interaction_step, 4)
        self.assertAlmostEqual(algo.epsilon, .1)
        algo.update(batch, True)
        self.assertEqual(algo.gradient_step, 1)
        self.assertFalse(all(torch.equal(a,b) for a,b in zip(agent.parameters(), target.parameters())))
        algo._update_policy()
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(agent.parameters(), target.parameters())))
        self.assertTrue(algo._interval_due(3, '_test'))
        self.assertFalse(algo._interval_due(3, '_test'))

    def test_normalization_exploration_and_checkpoint(self):
        agent = AtariDQNAgent(ImageEnv.observation_space, ImageEnv.action_space, 'cpu')
        pixels = np.full((5, 4, 84, 84), 255, np.uint8)
        torch.testing.assert_close(agent.q_values(pixels), agent.q_values(torch.ones((5, 4, 84, 84))))
        agent.epsilon = 1.
        self.assertEqual(agent.select_action(pixels).shape, (5,))
        with tempfile.TemporaryDirectory() as directory:
            path = directory+'/nested/model.pth'
            agent.save(path)
            agent.load(path)

    def test_evaluation_restores_mode_and_closes(self):
        agent = AtariDQNAgent(ImageEnv.observation_space, ImageEnv.action_space, 'cpu')
        envs = DummyVecEnv([ImageEnv, ImageEnv])
        with patch('utils.evaluate.make_vec_envs', return_value=envs), patch.object(envs, 'close', wraps=envs.close) as close:
            result = evaluate_atari(agent, 3, None)
            self.assertEqual(result.metrics['episode_rewards_mean'], 7.)
            self.assertTrue(agent.training)
            close.assert_called_once()

    def test_evaluation_error_cleanup(self):
        agent = AtariDQNAgent(ImageEnv.observation_space, ImageEnv.action_space, 'cpu')
        agent.eval()
        envs = DummyVecEnv([ImageEnv])
        with patch('utils.evaluate.make_vec_envs', return_value=envs), patch.object(envs, 'close', wraps=envs.close) as close, patch.object(agent, 'select_action', side_effect=RuntimeError('policy failed')):
            with self.assertRaises(RuntimeError):
                evaluate_atari(agent, 1, None)
            self.assertFalse(agent.training)
            close.assert_called_once()

    def test_empty_replay_and_metric_validation(self):
        buffer = ReplayBuffer(ImageEnv.observation_space, ImageEnv.action_space, 1, 1)
        with self.assertRaises(ValueError):
            buffer.sample(1)
        with self.assertRaises(ValueError):
            ReplayBuffer(ImageEnv.observation_space, ImageEnv.action_space, 1, 2)
        result = Result('test')
        with self.assertRaises(ValueError):
            result.add_metric('empty', np.array([]))
        result.add_metric('one', 1)
        with self.assertRaises(ValueError):
            result.add_metric('one', 2)

    def test_metrics(self):
        result = Result('test')
        result.add_metric('int', 1)
        result.add_metric('array', np.array([1, 3]))
        result.add_metric('tensor', torch.tensor([2, 4]))
        self.assertEqual(result.metrics, {'int': 1., 'array': 2., 'tensor': 3.})

    def test_wrapper_dtype(self):
        env = ImageEnv()
        env.observation_space = gym.spaces.Box(0, 255, (84, 84, 1), np.uint8)
        wrapper = TransposeImage(env)
        self.assertEqual(wrapper.observation_space.dtype, np.uint8)
        self.assertEqual(FrameStack(wrapper, 4).observation_space.shape, (4, 84, 84))


if __name__ == '__main__':
    unittest.main()
