import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from gymnasium.spaces import Space
from typing import Union

from utils.networks import AtariDQNNetwork
from utils.utils import atari_state_preprocess_function


def to_correct_device_tensor(input, device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(input, np.ndarray):
        return torch.tensor(input, dtype=dtype, device=device)
    elif isinstance(input, torch.Tensor):
        input = input.to(device)
        return input
    else:
        raise TypeError("input must be np array or torch tensor")


class AgentBase(nn.Module, ABC):
    def __init__(self, observation_space: Space, action_space: Space, device):
        super(AgentBase, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

    @abstractmethod
    def select_action(self, states: np.ndarray, deterministic: bool) -> np.ndarray:
        """Choose the action according to the states agent observed."""
        pass

    def _get_other_parameters(self):
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device
        )

    def save(self, pth_file: str):
        torch.save({"state_dict": self.state_dict(),
                   "data": self._get_other_parameters()}, pth_file)

    def load(self, path: str):
        pth_file = torch.load(path, weights_only=False)
        self.load_state_dict(pth_file['state_dict'])
        data = pth_file['data']
        self.observation_space = data['observation_space']
        self.action_space = data['action_space']
        self.device = data['device']


class AtariDQNAgent(AgentBase):
    def __init__(self, observation_space: Space, action_space: Space, device):
        super(AtariDQNAgent, self).__init__(
            observation_space, action_space, device)
        self.num_actions = action_space.shape
        self.observation_shape = observation_space.shape
        self.network = AtariDQNNetwork(
            observation_space.shape, self.num_actions)  # type: ignore

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: Input state, shape (C, 84, 84) or (B, C, 84, 84)
            deterministic: If True, always choose the greedy action (epsilon=0).
                          If False, use epsilon-greedy exploration.

        Returns:
            action: The selected action as an integer array, shape (1,)
        """
        # Ensure state is a tensor on the correct device
        state = to_correct_device_tensor(state, self.device)

        # Add batch dimension if not present
        if state.dim() == 3:
            state = state.unsqueeze(0)  # (1, C, 84, 84)

        # Compute Q-values (no gradient needed for action selection)
        with torch.no_grad():
            q_values = self.network(state)  # (1, num_actions)

        # epsilon-greedy action selection
        if deterministic:
            action = q_values.argmax(dim=1)
        else:
            # epsilon is typically managed by the training loop (e.g., linearly
            # annealed from 1.0 to 0.1), so we assume the caller has set
            # self.epsilon. If not, default to a small value.
            epsilon = getattr(self, 'epsilon', 0.1)
            if np.random.random() < epsilon:
                action = torch.randint(
                    0, q_values.shape[1], (1,), device=self.device)
            else:
                action = q_values.argmax(dim=1)

        return action.cpu().numpy()

    def get_q(self, states: torch.Tensor, actions: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute Q(s, a) for the given state-action pairs.

        Args:
            states: Batch of states, shape (B, C, 84, 84)
            actions: Batch of actions, shape (B,) or (B, 1)

        Returns:
            q_values: Q(s, a) for each pair, shape (B, 1)
        """
        states = to_correct_device_tensor(states, self.device)

        # Ensure batch dimension
        if states.dim() == 3:
            states = states.unsqueeze(0)

        # Compute all Q-values
        all_q = self.network(states)  # (B, num_actions)

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
        states = to_correct_device_tensor(states, self.device)

        if states.dim() == 3:
            states = states.unsqueeze(0)

        all_q = self.network(states)  # (B, num_actions)
        max_q = all_q.max(dim=1, keepdim=True)[0]  # (B, 1)

        return max_q
