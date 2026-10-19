"""Dead-feature gradients and checkpoint activation compatibility."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from agent.atari_agent import AtariDQNAgent
from test_regressions import ImageEnv
from scripts.diagnose import inspect_visual_path


class ActivationTests(unittest.TestCase):
    def make_agent(self, activation):
        """Construct a reproducible small Atari agent."""
        torch.set_num_threads(1)
        torch.manual_seed(9)
        return AtariDQNAgent(ImageEnv.observation_space, ImageEnv.action_space, 'cpu', activation)

    def test_negative_convolution_preserves_gradient_with_leaky_relu(self):
        """Force a dead third layer and verify the safeguard restores its gradient."""
        agent = self.make_agent('relu')
        with torch.no_grad():
            agent.network.conv3.weight.zero_()
            agent.network.conv3.bias.fill_(-1)
        pixels = np.full((2,4,84,84),128,np.uint8)
        relu = inspect_visual_path(agent,pixels)
        self.assertEqual(relu['layers']['conv3']['zero_fraction'],1.)
        self.assertEqual(relu['gradient_norm_by_layer']['conv3'],0.)
        agent.network.set_activation('leaky_relu')
        leaky = inspect_visual_path(agent,pixels)
        self.assertGreater(leaky['gradient_norm_by_layer']['conv3'],0.)
        self.assertTrue(all(p.grad is None for p in agent.parameters()))

    def test_activation_roundtrip_and_legacy(self):
        """New metadata round-trips; old weights always retain ReLU semantics."""
        pixels = np.full((2,4,84,84),128,np.uint8)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'agent.pth'
            for activation in ('relu','leaky_relu'):
                source = self.make_agent(activation)
                source.save(path)
                loaded = self.make_agent('leaky_relu' if activation=='relu' else 'relu')
                loaded.load(path)
                self.assertEqual(loaded.network.activation_name,activation)
                torch.testing.assert_close(source.q_values(pixels),loaded.q_values(pixels))
            source = self.make_agent('relu')
            metadata = source._get_other_parameters()
            del metadata['activation']
            torch.save({'state_dict':source.state_dict(),'data':metadata},path)
            loaded = self.make_agent('leaky_relu')
            loaded.load(path)
            self.assertEqual(loaded.network.activation_name,'relu')
            torch.testing.assert_close(source.q_values(pixels),loaded.q_values(pixels))
