"""Assignment architecture, MSE, and checkpoint compatibility checks."""
import unittest
import tempfile
import numpy as np
import torch
from omegaconf import OmegaConf
from agent.atari_agent import AtariDQNAgent
from utils.networks import AtariDQNNetwork
from test_regressions import ImageEnv
import test_learning
from unittest.mock import patch

class AssignmentTests(unittest.TestCase):
    def test_paper_architecture_checkpoint(self):
        """Check exact layer dimensions and loading into a legacy-size agent."""
        torch.set_num_threads(1)
        agent=AtariDQNAgent(ImageEnv.observation_space,ImageEnv.action_space,'cpu',architecture='paper2013')
        net=agent.network
        self.assertEqual((net.conv1.out_channels,net.conv2.out_channels,net.fc1.in_features,net.fc1.out_features),(16,32,2592,256))
        x=np.zeros((2,4,84,84),np.uint8)
        with tempfile.TemporaryDirectory() as d:
            agent.save(d+'/model.pth')
            other=AtariDQNAgent(ImageEnv.observation_space,ImageEnv.action_space,'cpu')
            other.load(d+'/model.pth')
            torch.testing.assert_close(agent.q_values(x),other.q_values(x))
    def test_baseline_configuration(self):
        """Assignment defaults explicitly choose MSE and vanilla target DQN."""
        cfg=OmegaConf.load('config/config.yaml').algorithm
        self.assertEqual((cfg.architecture,cfg.activation,cfg.loss),('paper2013','relu','mse'))
        self.assertTrue(cfg.use_target)
        self.assertFalse(cfg.double_dqn)

    def test_mse_actual_update(self):
        """A terminal target of 2 and prediction 0 must give MSE 4, not Huber 1.5."""
        helper=test_learning.LearningTests()
        try:
            algo=helper.make_algorithm()
            algo.loss_name='mse'
            batch=dict(states=np.zeros((1,4,84,84),np.uint8),next_states=np.zeros((1,4,84,84),np.uint8),actions=np.zeros((1,1),np.int64),rewards=np.array([2.],np.float32),dones=np.array([1.],np.float32))
            with patch.object(algo.buffer,'sample',return_value=batch), patch.object(algo.agent,'get_q',return_value=torch.zeros((1,1),requires_grad=True)):
                result=algo._update_policy()
            self.assertAlmostEqual(result.metrics['network/loss'],4.)
        finally:
            helper.doCleanups()
