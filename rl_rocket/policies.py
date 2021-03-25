from typing import Optional, Union, List, Tuple
from pprint import pprint

import gym
import torch
from torch import nn


class PolicyEstimatorNet(nn.Module):
    # Network for Reinforce
    def __init__(self, state_dim, action_dim, net = None):
        super().__init__()
        self.in_features = state_dim
        self.out_features = 2 * action_dim  # mean and std for each dim
        if net is None:
            self.net = nn.Sequential(
                nn.Linear(self.in_features, 256),
                nn.ReLU(),
                # TODO: find a suitable architecture
                nn.Linear(256, self.out_features),
                # nn.Softmax(),  # TODO: Softmax macht hier keinen Sinn oder? wir haben ja means und std's, keine Wahrscheinlichkeiten
                )
        else:
            self.net = net

    def forward(self, state: torch.Tensor):
        out = self.net(state)
        means = out[:self.out_features // 2]
        std = out[self.out_features // 2:]
        return means, std
