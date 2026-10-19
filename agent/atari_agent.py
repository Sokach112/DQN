"""Atari Q-value agent, observation normalization, and portable checkpoints."""

from pathlib import Path
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from gymnasium.spaces import Space, Discrete
from typing import Union

from utils.networks import AtariDQNNetwork
from utils.utils import atari_state_preprocess_function
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs


ACTION_SELECTION_MODES = ("epsilon_greedy", "softmax")


def to_correct_device_tensor(input, device, dtype=torch.float32) -> torch.Tensor:
    """Convert an array or tensor to the requested device and dtype."""
    if isinstance(input, np.ndarray):
        return torch.tensor(input, dtype=dtype, device=device)
    elif isinstance(input, torch.Tensor):
        input = input.to(device=device, dtype=dtype)
        return input
    else:
        raise TypeError("input must be np array or torch tensor")


class AgentBase(nn.Module, ABC):
    def __init__(self, observation_space: Space, action_space: Space, device):
        """Initialize the component and retain its configuration."""
        super(AgentBase, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

    @abstractmethod
    def select_action(self, states: np.ndarray | VecEnvObs, deterministic: bool) -> np.ndarray:
        """Choose the action according to the states agent observed."""
        pass

    def _get_other_parameters(self):
        """Return space and device metadata stored with checkpoints."""
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device
        )

    def save(self, pth_file: str):
        """Save model weights and space metadata, creating parent directories."""
        Path(pth_file).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(),
                   "data": self._get_other_parameters()}, pth_file)

    def load(self, path: str):
        """Load a trusted checkpoint onto the current device; optimizer state is not restored."""
        pth_file = torch.load(path, map_location=self.device, weights_only=False)
        data = pth_file['data']
        self._restore_metadata(data)
        self.load_state_dict(pth_file['state_dict'])
        self.to(self.device)

    def _restore_metadata(self, data):
        """Restore checkpoint space metadata while retaining the current device."""
        self.observation_space = data['observation_space']
        self.action_space = data['action_space']


class AtariDQNAgent(AgentBase):
    def __init__(self, observation_space: Space, action_space: Space, device, activation="relu", architecture="nature2015"):
        """Initialize the component and retain its configuration."""
        super(AtariDQNAgent, self).__init__(
            observation_space, action_space, device)
        if not isinstance(action_space, Discrete) or action_space.start != 0:
            raise ValueError("Atari requires zero-based Discrete actions")
        self.num_actions = action_space.n
        self.observation_shape = observation_space.shape

        self.network = AtariDQNNetwork(
            observation_space.shape, self.num_actions, activation=activation, architecture=architecture)  # type: ignore
        self.to(device)
        self.epsilon = 0.0
        self.action_selection = "epsilon_greedy"
        self.softmax_temperature = 1.0

    def configure_exploration(self, action_selection="epsilon_greedy", softmax_temperature=1.0):
        """Validate and store the training-time exploration rule.

        ``epsilon_greedy`` (the default and assignment baseline) keeps the
        existing uniform-random exploration switch. ``softmax`` samples actions
        from a Boltzmann distribution over Q-values; it replaces the epsilon
        switch rather than being combined with it. ``select_action(...,
        deterministic=True)`` always returns the greedy argmax in either mode.

        Args:
            action_selection: ``"epsilon_greedy"`` or ``"softmax"``.
            softmax_temperature: Positive finite Boltzmann temperature
                ``alpha`` (only used by ``softmax`` mode).

        Raises:
            ValueError: If the mode is unknown or the temperature is not a
                positive finite number. Attributes are unchanged on failure.
        """
        if action_selection not in ACTION_SELECTION_MODES:
            raise ValueError(
                f"action_selection must be one of {ACTION_SELECTION_MODES}, "
                f"got {action_selection!r}")
        try:
            temperature = float(softmax_temperature)
        except (TypeError, ValueError):
            raise ValueError(
                "softmax_temperature must be a positive finite number") from None
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(
                "softmax_temperature must be a positive finite number")
        self.action_selection = action_selection
        self.softmax_temperature = temperature

    def _softmax_action_probabilities(self, q_values: torch.Tensor) -> np.ndarray:
        """Return numerically stable ``softmax(q / temperature)`` as a NumPy array."""
        scaled = q_values.detach().to(device="cpu", dtype=torch.float32).numpy()
        scaled = scaled / self.softmax_temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        probabilities = np.exp(scaled)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def _sample_softmax(self, q_values: torch.Tensor) -> np.ndarray:
        """Sample one action per row from the Boltzmann policy (inverse CDF)."""
        cumulative = np.cumsum(self._softmax_action_probabilities(q_values), axis=1)
        uniform = np.random.random((len(cumulative), 1))
        actions = (uniform > cumulative).sum(axis=1)
        return np.minimum(actions, self.num_actions - 1)

    def select_action(self, state, deterministic=False):
        """Return one action per observation using the configured exploration rule.

        ``deterministic=True`` always returns the greedy argmax action. Otherwise
        ``epsilon_greedy`` mode (default) takes the argmax and replaces a uniform
        random subset (probability ``epsilon``) with uniform random actions,
        exactly as before. ``softmax`` mode samples from
        ``exp(Q(s, a) / softmax_temperature)`` normalized over actions; epsilon
        is ignored there because Boltzmann sampling replaces the epsilon switch.
        """
        with torch.no_grad():
            q_values = self.q_values(state)
            if deterministic:
                return q_values.argmax(dim=1).cpu().numpy()
            if self.action_selection == "softmax":
                return self._sample_softmax(q_values)
            if self.action_selection != "epsilon_greedy":
                raise ValueError(
                    f"action_selection must be one of {ACTION_SELECTION_MODES}, "
                    f"got {self.action_selection!r}")
            actions = q_values.argmax(dim=1).cpu().numpy()
        explore = np.random.random(len(actions)) < self.epsilon
        actions[explore] = np.random.randint(self.num_actions, size=int(explore.sum()))
        return actions

    def _get_other_parameters(self):
        """Include activation semantics so checkpoints reproduce their policy."""
        data = super()._get_other_parameters()
        data['architecture'] = self.network.architecture
        data['activation'] = self.network.activation_name
        return data

    def _restore_metadata(self, data):
        """Use ReLU for legacy checkpoints that predate activation metadata."""
        super()._restore_metadata(data)
        architecture = data.get('architecture', 'nature2015')
        if self.network.architecture != architecture:
            self.network = AtariDQNNetwork(self.observation_space.shape, self.action_space.n, architecture=architecture).to(self.device)
        self.network.set_activation(data.get('activation', 'relu'))

    def q_values(self, states):
        """Return (batch, actions) Q-values, normalizing byte pixels once.

        Floating observations are expected to already be in [0, 1].
        """
        states = atari_state_preprocess_function(self.observation_space, states)
        states = to_correct_device_tensor(states, self.device)
        if states.ndim == 3:
            states = states.unsqueeze(0)
        return self.network(states)

    def get_q(self, states: torch.Tensor, actions: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute Q(s, a) for the given state-action pairs.

        Args:
            states: Batch of states, shape (B, C, 84, 84)
            actions: Batch of actions, shape (B,) or (B, 1)

        Returns:
            q_values: Q(s, a) for each pair, shape (B, 1)
        """
        all_q = self.q_values(states)

        # Convert actions to tensor
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(
                actions, dtype=torch.long, device=self.device)
        else:
            actions = actions.to(self.device).long()

        # Handle shape (B,) or (B, 1)
        actions = actions.view(-1)  # (B,)

        # Gather Q(s, a) for the chosen actions
        q_values = all_q.gather(1, actions.unsqueeze(1))  # (B, 1)

        return q_values

    def get_max_q(self, states: torch.Tensor):
        """
        Compute max_a Q(s, a) for the given states.

        Args:
            states: Batch of states, shape (B, C, 84, 84)

        Returns:
            max_q: max_a Q(s, a) for each state, shape (B, 1)
        """
        return self.q_values(states).max(dim=1, keepdim=True)[0]
