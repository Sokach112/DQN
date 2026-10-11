import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class AtariDQNNetwork(nn.Module):
    def __init__(self, input_shape:Union[tuple, list], num_actions):
        """
        input_shape: the shape of the input figure (C, H, W), usually (4, 84, 84)
        num_actions: Atari's action space is Discrete
        """
        super(AtariDQNNetwork, self).__init__()
        # TODO: initialize layers according to the paper

    def forward(self, x):
        # x: (batch, 4, 84, 84)
        # TODO: input is a batch of figure, do forward propogation to calculate the Q for each action