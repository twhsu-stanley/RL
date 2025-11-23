import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy_Net(nn.Module):
    def __init__(self, dim_states, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(dim_states, 16)
        self.layer2 = nn.Linear(16, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    
    def get_distribution(self, state, with_grad=False):
        """Softmax policy distribution"""
        if with_grad:
            action_dist = torch.distributions.Categorical(torch.softmax(self.forward(state), dim=-1))
        else:
            with torch.no_grad():
                action_dist = torch.distributions.Categorical(torch.softmax(self.forward(state), dim=-1))
        return action_dist
    
    def get_action(self, state):
        """Get action from the policy distribution"""
        action = self.get_distribution(state, with_grad=False).sample()
        return action