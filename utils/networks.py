"""Convolutional Q network for normalized channel-first Atari observations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional


class AtariDQNNetwork(nn.Module):
    def __init__(self, input_shape: Union[tuple, list], num_actions: int):
        """
        input_shape: the shape of the input figure (C, H, W), usually (4, 84, 84)
        num_actions: Atari's action space is Discrete
        """
        super(AtariDQNNetwork, self).__init__()

        # Unpack input shape
        in_channels, in_height, in_width = input_shape

        # --- Convolutional layers ---
        # Layer 1: 32 filters, 8x8 kernel, stride 4
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=8,
            stride=4,
        )

        # Layer 2: 64 filters, 4x4 kernel, stride 2
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
        )

        # Layer 3: 64 filters, 3x3 kernel, stride 1
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
        )

        # --- Compute the flattened size after convolutions ---
        # We need this to size the first fully-connected layer.
        # Do a dummy forward pass with a zero tensor of the input shape.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, in_height, in_width)
            dummy = F.relu(self.conv1(dummy))
            dummy = F.relu(self.conv2(dummy))
            dummy = F.relu(self.conv3(dummy))
            self._flat_features = dummy.view(1, -1).shape[1]

        # --- Fully-connected layers ---
        # Hidden layer: 512 units
        self.fc1 = nn.Linear(self._flat_features, 512)

        # Output layer: one Q-value per action
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        # x: (batch, 4, 84, 84)
        # --- Convolutional stack ---
        """Map normalized (batch, channels, height, width) tensors to Q-values."""
        x = F.relu(self.conv1(x))  # (B, 32, 20, 20)
        x = F.relu(self.conv2(x))  # (B, 64, 9, 9)
        x = F.relu(self.conv3(x))  # (B, 64, 7, 7)

        # Flatten: (B, 64, 7, 7) → (B, 64*7*7) = (B, 3136)
        x = x.view(x.size(0), -1)

        # --- Fully-connected stack ---
        x = F.relu(self.fc1(x))    # (B, 512)
        x = self.fc2(x)            # (B, num_actions)

        return x
