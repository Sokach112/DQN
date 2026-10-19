"""Convolutional Q network for normalized channel-first Atari observations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional


class AtariDQNNetwork(nn.Module):
    def __init__(self, input_shape: Union[tuple, list], num_actions: int, activation="relu", architecture="nature2015"):
        """
        input_shape: the shape of the input figure (C, H, W), usually (4, 84, 84)
        num_actions: Atari's action space is Discrete
        """
        super(AtariDQNNetwork, self).__init__()

        if architecture not in ("nature2015", "paper2013"):
            raise ValueError("Unknown architecture")
        self.architecture = architecture
        paper = architecture == "paper2013"
        self.set_activation(activation)

        # Unpack input shape
        in_channels, in_height, in_width = input_shape

        # --- Convolutional layers ---
        # Layer 1: 32 filters, 8x8 kernel, stride 4
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=16 if paper else 32,
            kernel_size=8,
            stride=4,
        )

        # Layer 2: 64 filters, 4x4 kernel, stride 2
        self.conv2 = nn.Conv2d(
            in_channels=16 if paper else 32,
            out_channels=32 if paper else 64,
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

        if paper:
            self.conv3 = nn.Identity()

        # --- Compute the flattened size after convolutions ---
        # We need this to size the first fully-connected layer.
        # Do a dummy forward pass with a zero tensor of the input shape.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, in_height, in_width)
            dummy = self.activation(self.conv1(dummy))
            dummy = self.activation(self.conv2(dummy))
            if not paper:
                dummy = self.activation(self.conv3(dummy))
            self._flat_features = dummy.view(1, -1).shape[1]

        # --- Fully-connected layers ---
        # Hidden layer: 512 units
        self.fc1 = nn.Linear(self._flat_features, 256 if paper else 512)

        # Output layer: one Q-value per action
        self.fc2 = nn.Linear(256 if paper else 512, num_actions)

    def set_activation(self, name):
        """Select the hidden activation without changing parameter names/shapes.

        ReLU preserves legacy checkpoints; LeakyReLU retains a gradient for
        negative preactivations to reduce the risk of a dead visual pathway.
        """
        if name not in ("relu", "leaky_relu"):
            raise ValueError("activation must be relu or leaky_relu")
        self.activation_name = name
        self.activation = (nn.ReLU() if name == "relu"
                           else nn.LeakyReLU(negative_slope=0.01))

    def forward(self, x):
        # x: (batch, 4, 84, 84)
        # --- Convolutional stack ---
        """Map normalized (batch, channels, height, width) tensors to Q-values."""
        x = self.activation(self.conv1(x))  # (B, 32, 20, 20)
        x = self.activation(self.conv2(x))  # (B, 64, 9, 9)
        if self.architecture != "paper2013":
            x = self.activation(self.conv3(x))

        # Flatten: (B, 64, 7, 7) → (B, 64*7*7) = (B, 3136)
        x = x.view(x.size(0), -1)

        # --- Fully-connected stack ---
        x = self.activation(self.fc1(x))    # (B, 512)
        x = self.fc2(x)            # (B, num_actions)

        return x
