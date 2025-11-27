import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_Net(nn.Module):
    """Q-network for DQN"""
    
    def __init__(self, dim_states, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(dim_states, 12)
        self.layer2 = nn.Linear(12, 8)
        self.layer3 = nn.Linear(8, n_actions)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x