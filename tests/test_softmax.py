"""Optional softmax/Boltzmann action selection, runnable without Atari ROMs."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agent.atari_agent import AtariDQNAgent
from test_regressions import ImageEnv


class SoftmaxActionSelectionTests(unittest.TestCase):
    """Exercise both exploration modes through the public agent API."""

    num_actions = ImageEnv.action_space.n
    states = np.zeros((3, 4, 84, 84), np.uint8)

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        self.agent = AtariDQNAgent(ImageEnv.observation_space, ImageEnv.action_space, 'cpu')
        self.agent.epsilon = 0.0

    def q_rows_patch(self, rows):
        """Stub q_values with one Q row per observation (no network call)."""
        q_values = torch.tensor(rows, dtype=torch.float32).reshape(-1, self.num_actions)
        return patch.object(self.agent, 'q_values', return_value=q_values)

    def draw(self, q_row, samples):
        """Sample ``samples`` actions from one stubbed Q row.

        The network is stubbed, so the observation batch is only a placeholder;
        the returned actions come from the repeated stubbed Q row.
        """
        q_values = torch.tensor(q_row, dtype=torch.float32).reshape(1, -1).expand(samples, -1)
        with patch.object(self.agent, 'q_values', return_value=q_values):
            return self.agent.select_action(self.states)

    @staticmethod
    def boltzmann(q_row, temperature):
        """Reference softmax probabilities from the assignment definition."""
        scaled = np.asarray(q_row, np.float64) / temperature
        scaled = scaled - scaled.max()
        probabilities = np.exp(scaled)
        return probabilities / probabilities.sum()

    def test_default_mode_is_unchanged_epsilon_greedy(self):
        """Defaults stay epsilon-greedy: epsilon=0 is plain argmax."""
        self.assertEqual(self.agent.action_selection, 'epsilon_greedy')
        self.assertEqual(self.agent.softmax_temperature, 1.0)
        with self.q_rows_patch([[0.5, 2.0, 1.0], [3.0, 0.1, 2.0], [1.0, 1.0, 1.0]]):
            actions = self.agent.select_action(self.states)
            deterministic = self.agent.select_action(self.states, deterministic=True)
        np.testing.assert_array_equal(actions, [1, 0, 0])
        np.testing.assert_array_equal(deterministic, [1, 0, 0])
        self.assertEqual(actions.shape, (3,))
        self.assertTrue(np.issubdtype(actions.dtype, np.integer))

    def test_default_mode_still_explores_uniformly(self):
        """epsilon=1 in epsilon-greedy mode ignores Q ranks and is uniform."""
        self.agent.epsilon = 1.0
        np.random.seed(1)
        actions = self.draw([100.0, 0.0, 0.0], 6000)
        counts = np.bincount(actions, minlength=self.num_actions) / len(actions)
        np.testing.assert_allclose(counts, np.full(self.num_actions, 1.0 / self.num_actions), atol=0.03)

    def test_softmax_distribution_matches_boltzmann(self):
        """Sampled frequencies approximate exp(Q/alpha) proportions."""
        self.agent.configure_exploration('softmax', 1.0)
        expected = np.array([1.0, 2.0, 5.0]) / 8.0
        np.random.seed(2)
        actions = self.draw(np.log(expected), 20000)
        counts = np.bincount(actions, minlength=self.num_actions) / len(actions)
        np.testing.assert_allclose(counts, expected, atol=0.02)

    def test_softmax_temperature_changes_concentration(self):
        """A lower temperature sharpens the distribution around the best action."""
        q_row = [0.0, 1.0, 2.0]
        self.agent.configure_exploration('softmax', 0.5)
        np.random.seed(3)
        sharp = self.draw(q_row, 20000)
        self.agent.configure_exploration('softmax', 5.0)
        np.random.seed(3)
        flat = self.draw(q_row, 20000)
        sharp_counts = np.bincount(sharp, minlength=self.num_actions) / len(sharp)
        flat_counts = np.bincount(flat, minlength=self.num_actions) / len(flat)
        np.testing.assert_allclose(sharp_counts, self.boltzmann(q_row, 0.5), atol=0.02)
        np.testing.assert_allclose(flat_counts, self.boltzmann(q_row, 5.0), atol=0.02)

    def test_softmax_ignores_epsilon(self):
        """Softmax mode replaces the epsilon switch; no uniform mixing occurs."""
        self.agent.configure_exploration('softmax', 1.0)
        self.agent.epsilon = 0.9
        np.random.seed(4)
        actions = self.draw([0.0, 0.0, 10.0], 5000)
        counts = np.bincount(actions, minlength=self.num_actions) / len(actions)
        self.assertGreater(counts[2], 0.99)
        self.assertLess(counts[0], 0.01)
        self.assertLess(counts[1], 0.01)

    def test_deterministic_ignores_softmax_mode(self):
        """deterministic=True is greedy argmax in softmax mode, regardless of epsilon."""
        self.agent.configure_exploration('softmax', 1.0)
        self.agent.epsilon = 0.9
        expected = [1, 0, 2]
        with self.q_rows_patch([[0.2, 5.0, 1.0], [7.0, 0.0, 2.0], [1.0, 2.0, 3.0]]):
            for _ in range(25):
                np.testing.assert_array_equal(
                    self.agent.select_action(self.states, deterministic=True), expected)

    def test_configure_round_trip(self):
        """Configuration switches modes and restores epsilon-greedy defaults."""
        self.agent.configure_exploration('softmax', 0.25)
        self.assertEqual((self.agent.action_selection, self.agent.softmax_temperature), ('softmax', 0.25))
        self.agent.configure_exploration('epsilon_greedy')
        self.assertEqual((self.agent.action_selection, self.agent.softmax_temperature), ('epsilon_greedy', 1.0))

    def test_invalid_configuration_raises_value_error(self):
        """Unknown modes and non-positive/non-finite temperatures are rejected."""
        for mode in ('greedy', 'greedy_softmax', '', 'softmax ', None, 1):
            with self.assertRaises(ValueError):
                self.agent.configure_exploration(mode)
        for temperature in (0.0, -0.5, float('nan'), float('inf'), None, 'warm'):
            with self.assertRaises(ValueError):
                self.agent.configure_exploration('softmax', temperature)
        self.assertEqual((self.agent.action_selection, self.agent.softmax_temperature), ('epsilon_greedy', 1.0))
        with self.assertRaises(ValueError):
            self.agent.configure_exploration('epsilon_greedy', 0.0)


if __name__ == '__main__':
    unittest.main()
