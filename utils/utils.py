"""Device selection, reproducibility, and Atari observation/action conversion."""

import torch
import numpy as np
import gymnasium as gym
import random
from typing import Union

def get_best_device():

    """Choose the CUDA device with most free memory, otherwise MPS or CPU."""
    if not torch.cuda.is_available():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using {device}.")
    else:
        num_gpus = torch.cuda.device_count()
        max_free = -1
        best_gpu = 0

        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)

            total_memory = props.total_memory

            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)

            free_memory, total_memory = torch.cuda.mem_get_info(i)

            print(f"GPU {i}: total {total_memory/1e6:.0f}MB, free {free_memory/1e6:.0f}MB")

            if free_memory > max_free:
                max_free = free_memory
                best_gpu = i

        device = torch.device(f"cuda:{best_gpu}")
        print(f"Using GPU {best_gpu} with approx {max_free/1e6:.0f} MB free memory.")
    return device

def set_seed(seed):
    """Seed Python, NumPy, and PyTorch and configure deterministic cuDNN."""
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def atari_state_preprocess_function(observation_space:gym.spaces.Space, states:Union[np.ndarray,torch.Tensor]):
    """Normalize observations to 0~1.

    :param gym.Env env: the environment to wrap.
    """
    if isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 3:
        if isinstance(states, torch.Tensor) and states.dtype == torch.uint8:
            return states.float() / 255.0
        if isinstance(states, np.ndarray) and states.dtype == np.uint8:
            return states.astype(np.float32) / 255.0
    return states


def atari_to_useful_action(actions: np.ndarray):
    """Convert action arrays shaped (N,) or (N, 1) to a vector."""
    actions = np.asarray(actions)
    if actions.ndim == 1:
        return actions
    if actions.ndim == 2 and actions.shape[1] == 1:
        return actions[:, 0]
    raise ValueError("Expected actions shaped (N,) or (N, 1)")

if __name__ == "__main__":
    print(get_best_device())
