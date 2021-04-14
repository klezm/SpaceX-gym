from typing import Optional, Union, List, Tuple
from pprint import pprint

import gym
import torch
from torch import nn
import torch.nn.functional as F

class PolicyEstimatorNet(nn.Module):
    # Network for Reinforce
    def __init__(self, state_dim, action_dim, net = None, hidden_size = 256, n_hidden = 0, clip_std = 0):
        super().__init__()
        self.in_features = state_dim
        self.out_features = 2 * action_dim  # mean and std for each dim
        self.clip_std = clip_std # set zero to deactivate
        if net is None:
            self.net = nn.Sequential(
                nn.Linear(self.in_features, hidden_size),
                nn.ReLU(),
                *[
                     nn.Linear(hidden_size, hidden_size),
                     nn.ReLU(),
                 ] * n_hidden,
                # TODO: find a suitable architecture
                nn.Linear(hidden_size, self.out_features),
                # nn.ReLU(),
                # nn.Linear(self.out_features, self.out_features),
                # nn.Softmax(),  # TODO: Softmax macht hier keinen Sinn oder? wir haben ja means und std's, keine Wahrscheinlichkeiten
                )
            # for i in range(n_hidden):
            #     self.net.add_module(f'lin{i}', nn.Linear(self.out_features, self.out_features))
            #     self.net.add_module(f'relu{i}', nn.ReLU())
        else:
            self.net = net

    def forward(self, state: torch.Tensor):
        out = self.net(state)
        means = out[:self.out_features // 2]
        std = out[self.out_features // 2:]

        # This part was added to make it possible for the std to be zero
        # (otherwise it can only converge but never reach 0)
        std = torch.abs(std)
        std = self.clip_std + F.relu(std - self.clip_std) # sets all values below 0.1 to zero
        return means, std
