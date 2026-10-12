"""Vector-step replay storage and uniformly sampled transition dictionaries."""


import numpy as np
import gymnasium as gym
from utils.utils import atari_state_preprocess_function

def get_action_dim(action_space: gym.spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, gym.spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, gym.spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        # Number of discrete actions
        return int(action_space.nvec.size)
    elif isinstance(action_space, gym.spaces.MultiBinary):
        # Number of binary actions
        return int(np.prod(action_space.shape))
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


class ReplayBuffer:
    """Circular replay storing explicit successors, including terminal observations.

    Capacity is measured in transitions and rounded down to whole vector steps.
    Batches have leading dimensions (time, environments).
    """

    def __init__(self, observation_space, action_space, buffer_size, num_envs):
        """Allocate storage for at most ``buffer_size`` transitions."""
        if num_envs <= 0 or buffer_size < num_envs:
            raise ValueError("buffer_size must be at least num_envs > 0")
        self.buffer_size = buffer_size // num_envs
        self.num_envs = num_envs
        self.action_dim = get_action_dim(action_space)
        self.observation_space = observation_space
        self.action_space = action_space
        shape = (self.buffer_size, num_envs)
        self.states = np.empty((*shape, *observation_space.shape), dtype=observation_space.dtype)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty((*shape, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.empty(shape, dtype=np.float32)
        self.dones = np.empty(shape, dtype=np.float32)
        self.pos = 0
        self.full = False

    def __len__(self):
        """Return the number of sampleable transitions."""
        return (self.buffer_size if self.full else self.pos) * self.num_envs

    def add(self, batch):
        """Copy a time-major batch, retaining the newest data on overflow."""
        count = len(batch['states'])
        arrays = {}
        for key in ('states', 'next_states', 'actions', 'rewards', 'dones'):
            storage = getattr(self, key)
            value = np.asarray(batch[key])
            if key == 'actions' and value.shape == (count, self.num_envs):
                value = value[..., None]
            if value.shape != (count, *storage.shape[1:]):
                raise ValueError(f"Invalid {key} shape: {value.shape}")
            arrays[key] = value
        for i in range(count):
            for key, value in arrays.items():
                getattr(self, key)[self.pos] = value[i]
            self.pos += 1
            if self.pos == self.buffer_size:
                self.full = True
                self.pos = 0

    def _get_indices(self, batch_size):
        """Draw occupied time indices uniformly with replacement."""
        if batch_size <= 0 or not len(self):
            raise ValueError("Sampling requires a positive batch size and nonempty buffer")
        return np.random.randint(self.buffer_size if self.full else self.pos, size=batch_size)

    def _sample_from_indices(self, batch_indices):
        """Pair time indices with random environments and return copied arrays."""
        env_indices = np.random.randint(self.num_envs, size=len(batch_indices))
        return {key: getattr(self, key)[batch_indices, env_indices]
                for key in ('states', 'actions', 'rewards', 'next_states', 'dones')}

    def sample(self, batch_size):
        """Return a dictionary of transitions sampled with replacement."""
        return self._sample_from_indices(self._get_indices(batch_size))
