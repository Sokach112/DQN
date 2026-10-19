"""Numerical TD-target and visual learning checks, independent of Atari ROMs."""
import unittest
from unittest.mock import patch
import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv
from agent.atari_agent import AtariDQNAgent
from algorithm.offpolicy_algorithm.dqn import DQN
from memory.memory import ReplayBuffer
from test_regressions import ImageEnv


class LearningTests(unittest.TestCase):
    def make_algorithm(self, actions=3):
        """Build the production learner on CPU with a small replay buffer."""
        torch.set_num_threads(1)
        torch.manual_seed(7)
        np.random.seed(7)
        envs = DummyVecEnv([ImageEnv])
        self.addCleanup(envs.close)
        space = gym.spaces.Discrete(actions)
        agent = AtariDQNAgent(envs.observation_space, space, 'cpu')
        target = AtariDQNAgent(envs.observation_space, space, 'cpu')
        cfg = OmegaConf.load('config/config.yaml')
        cfg.algorithm.loss = 'huber'
        cfg.algorithm.double_dqn = True
        cfg.algorithm.batch_size = 4
        cfg.algorithm.learning_rate = 0.001
        buffer = ReplayBuffer(envs.observation_space, space, 8, 1)
        return DQN(envs, None, buffer, agent, None, 'cpu', '/tmp/unused.pth', cfg, target)

    def test_double_dqn_target_and_terminal_mask(self):
        """Online argmax selects target values; terminals suppress bootstrap."""
        algo = self.make_algorithm()
        batch = dict(states=np.zeros((2,4,84,84), np.uint8),
                     next_states=np.ones((2,4,84,84), np.uint8),
                     actions=np.array([[0],[1]]), rewards=np.array([1.,2.],np.float32),
                     dones=np.array([0.,1.],np.float32))
        current = torch.tensor([[0.],[0.]], requires_grad=True)
        online = torch.tensor([[1.,3.,2.],[4.,1.,2.]])
        target = torch.tensor([[10.,20.,30.],[40.,50.,60.]])
        expected = torch.tensor([[1.+algo.gamma*20.],[2.]])
        with patch.object(algo.buffer, 'sample', return_value=batch), patch.object(algo.agent,'get_q',return_value=current), patch.object(algo.agent,'q_values',return_value=online), patch.object(algo.target_agent,'q_values',return_value=target):
            result = algo._update_policy()
        self.assertAlmostEqual(result.metrics['target_mean'], expected.mean().item(), places=5)
        self.assertAlmostEqual(result.metrics['network/loss'], torch.nn.functional.smooth_l1_loss(current,expected).item(), places=5)

    def test_vanilla_target_uses_target_max(self):
        """Vanilla DQN selects the maximum target value, unlike Double DQN."""
        algo = self.make_algorithm()
        algo.double_dqn = False
        batch = dict(states=np.zeros((2,4,84,84), np.uint8),
                     next_states=np.ones((2,4,84,84), np.uint8),
                     actions=np.array([[0],[1]]), rewards=np.array([1.,2.],np.float32),
                     dones=np.array([0.,1.],np.float32))
        current = torch.tensor([[0.],[0.]], requires_grad=True)
        target = torch.tensor([[10.,20.,30.],[40.,50.,60.]])
        expected = torch.tensor([[1.+algo.gamma*30.],[2.]])
        with patch.object(algo.buffer,'sample',return_value=batch), patch.object(algo.agent,'get_q',return_value=current), patch.object(algo.target_agent,'q_values',return_value=target):
            result = algo._update_policy()
        self.assertAlmostEqual(result.metrics['target_mean'], expected.mean().item(), places=5)
        self.assertAlmostEqual(result.metrics['network/loss'], torch.nn.functional.smooth_l1_loss(current,expected).item(), places=5)

    def test_learns_opposite_actions_from_images(self):
        """Actual CNN/replay/optimizer must learn two opposing visual choices.

        This verifies basic learnability, not Atari convergence or temporal credit.
        """
        algo = self.make_algorithm(actions=2)
        states = np.zeros((4,1,4,84,84),np.uint8)
        states[:2,:,:,:,8:32] = 255
        states[2:,:,:,:,52:76] = 255
        algo.buffer.add(dict(states=states, next_states=states.copy(),
            actions=np.array([[0],[1],[0],[1]]), rewards=np.array([[1],[-1],[-1],[1]],np.float32),
            dones=np.ones((4,1),np.float32)))
        for _ in range(100):
            algo._update_policy()
        np.testing.assert_array_equal(algo.agent.select_action(states[[0,2],0],deterministic=True),[0,1])
        with torch.no_grad():
            predictions = algo.agent.get_q(states[:,0],np.array([0,1,0,1])).flatten()
        self.assertLess(torch.abs(predictions-torch.tensor([1.,-1.,-1.,1.])).mean().item(), .2)


if __name__ == '__main__':
    unittest.main()
